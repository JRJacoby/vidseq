# Graph Cut Segmentation

## Problem

SAM2-based propagation is the only way to produce training masks in vidseq. It works well but is ML-dependent — it requires GPU inference, accumulates drift over long sequences, and its failure modes (hallucination, loss of track) can be hard to diagnose. For some videos, especially grayscale or low-contrast footage, a simpler algorithm with more direct user control would be preferable.

## Solution

Add spatio-temporal graph cuts as a parallel method for producing training masks. The user paints foreground/background seeds on video frames using a brush tool, then runs a single graph cut solve across a 150-frame chunk. The algorithm uses pixel intensity similarity to propagate labels from seeded pixels to unseeded ones, both spatially within frames and temporally across frames. Results are written to the same `tracker_masks.h5` storage as SAM2 masks, so the downstream pipeline (cropping, alignment, PCA) works without changes.

No ML models, no GPU, no TCP worker. Pure CPU algorithm using PyMaxflow.

## Intended User Flow

1. **Select graph cut tool.** User clicks a "Graph Cut" tool button in the sidebar (same pattern as training range or working range tool selection).
2. **Place chunk on data track.** User clicks on the data track to place a 150-frame region starting at the clicked frame. Renders as a distinct-colored range bar (teal/cyan). Only one region at a time; clicking again moves it.
3. **Paint seeds.** User navigates to frames within the chunk and paints foreground (left-drag, green) and background (right-drag, red) seeds using an adjustable-size brush. Seeds accumulate across frames — navigate away and back, they're still there.
4. **Run.** User clicks "Run." The backend builds a 3D graph over the full chunk at native resolution, solves maxflow, writes binary masks to `tracker_masks.h5`.
5. **Inspect and refine.** User scrubs through the chunk to check results. Where the mask is wrong, they paint corrective seeds and run again. Each run re-solves the full chunk with all accumulated seeds and overwrites the masks.
6. **Move on.** When satisfied, the user either places a new chunk elsewhere or exits graph cut mode. Masks are already persisted.

## Design

### Data Track — Graph Cut Regions

A new marker type on the DataTrack: the "graph cut region."

**Placement:** Requires selecting the graph cut tool first (sidebar button), then clicking on the data track. This is a single-click-to-place interaction (not drag-to-create like training/working ranges), since the region is always a fixed size. The region anchors at the clicked frame and extends 150 frames forward (clicking frame 200 creates region 200–349). Clamped to video length — if the region would extend past the last frame, it's truncated. Clicking again moves the region. Frame navigation uses the video timeline scrub bar, not the data track (data track clicks always place/move the region while the tool is active).

**Rendering:** A colored range bar in `rgba(6, 182, 212, 0.4)` (teal/cyan), visually distinct from training ranges (green `rgba(34, 197, 94, 0.5)`), masked ranges (blue `rgba(59, 130, 246, 0.3)`), and working range (orange `rgba(245, 158, 11, 0.4)`). Same rendering pattern as existing range types (`trainingRangeStyles` etc.).

**Deletion:** Delete/Backspace removes the region and exits graph cut mode, same as working range deletion.

**Ephemeral:** The region is frontend-only state — not persisted to the database. It scopes the current graph cut session; the masks it produces are what get persisted.

**Chunk size:** Fixed at 150 frames. Defined as a named constant in both frontend (`GRAPHCUT_CHUNK_SIZE` in `useGraphCut.ts`) and backend (`GRAPHCUT_CHUNK_SIZE` in `graphcut_service.py`). Could become configurable later.

### Brush Tool & Seed Painting

When a graph cut region is active, the video overlay switches from point-click mode to brush mode.

**Interaction:**
- Left-click drag = foreground seeds (rendered green, semi-transparent)
- Right-click drag = background seeds (rendered red, semi-transparent). The overlay must call `preventDefault()` on the `contextmenu` event to suppress the browser context menu during right-click painting.
- Brush size adjustable via sidebar slider (pixel radius in display pixels). Default: 5px, range: 1–50px.
- Painting is disabled outside the graph cut region's frame bounds

**Coordinate mapping:** The overlay canvas maps display coordinates to native video pixel coordinates using the same scaling logic as `getNormalizedCoords()` in VideoOverlay, but producing integer pixel coordinates instead of [0,1] normalized values. The brush radius is similarly scaled from display pixels to native pixels based on the video-to-display scale factor. All seed coordinates stored and sent to the backend are in native pixel space.

**Seed visibility:** A toggle to hide/show seed overlays, same pattern as the existing mask and prompt visibility toggles. Allows inspecting the mask underneath without seed clutter.

**Seed storage (frontend):** Seeds are held in memory as a map of `{frame_idx: Array}` where each array records painted pixel positions and labels. Sparse representation — only painted pixels are tracked.

**Seed serialization:** Sent to the backend as `{frame_idx: [{x, y, label}]}` — a list of seed points per frame. The backend reconstructs dense seed arrays from these sparse lists.

**Lifecycle:** Seeds are ephemeral working state. They live only in the frontend composable. Exiting graph cut mode or moving the region discards them. The masks in H5 are the durable output.

### Graph Cut Solver (Backend)

A new service module: `vidseq/services/graphcut_service.py`. Runs in the FastAPI process via `asyncio.to_thread()` — no TCP/GPU worker involvement. The route handler awaits `asyncio.to_thread(graphcut_service.run_graphcut, ...)` to avoid blocking the event loop during the CPU-intensive solve.

**Input:** Video ID, video path, video dimensions, chunk start frame, chunk end frame, sparse seed dict.

**Algorithm:**

1. **Read frames.** Use `VideoFrameSource` (extracted to `vidseq/services/video_io.py` — a small utility shared with `segmentation_commands.py` to avoid importing GPU worker modules from FastAPI) for sequential reads of the 150-frame chunk at native resolution.

2. **Build 3D graph.** PyMaxflow grid graph with H x W x T nodes. Each pixel is 6-connected:
   - 4 spatial neighbors (up, down, left, right) within the same frame
   - 2 temporal neighbors (same pixel location in previous and next frame)

3. **Edge weights.** All edges (spatial and temporal) use the same weight function:
   ```
   weight = exp(-beta * ||I_p - I_q||^2)
   ```
   Where `||.||` is Euclidean distance in color space (for 3-channel grayscale this reduces to intensity difference up to a constant factor absorbed by beta). `beta = 1 / (2 * mean(||I_p - I_q||^2))` computed over all neighbor pairs in the chunk (standard Boykov auto-beta formula). This auto-scales to the image's color/intensity statistics.

4. **Terminal edges (seeds).** Foreground seed pixels get source capacity = `1 + max_edge_weight * 6` (guarantees the seed can never be overruled by its neighbors), sink capacity = 0. Background seed pixels get the inverse. Non-seed pixels get zero unary cost — their labeling is entirely determined by edge weights propagating from seeds.

5. **Solve.** `graph.maxflow()` — single global solve across the full 3D volume.

6. **Extract masks.** `graph.get_grid_segments()` returns a boolean array (H x W x T). Convert to per-frame uint8 binary masks.

7. **Write to H5.** Write masks to `tracker_masks.h5` using the existing `tracker_masks()` context manager from `array_storage.py`. Only masks are written — no logits (graph cuts produces binary labels, not continuous logits, so `tracker_logits.h5` is not touched).

**No downsampling.** The target videos are small enough (under 640x480) that native resolution is tractable.

**Memory estimate:** For a 640x480x150 chunk: ~46M nodes, ~276M edges. At ~48 bytes/node + ~32 bytes/edge, roughly ~11 GB. For smaller videos (e.g., 320x240x150): ~11.5M nodes, ~2.5 GB. If this becomes a concern, the chunk size is the knob to turn.

**Error handling:**
- Chunk extends past video end: clamp `end_frame` to `num_frames - 1` (match frontend clamping).
- No seeds provided: return 400 error — at least one foreground and one background seed required.
- All seeds same label: return 400 error — graph cut needs both source and sink terminals.
- Wrap the solve in try/except to catch OOM and return a 500 with a useful message rather than crashing the server.

**Overwrite behavior:** Running graph cut on frames that already have SAM2 masks silently overwrites them. This is intentional — the two methods are interchangeable ways to fill the same mask slots. Conditioning frames (SAM2 concept) are not affected; they remain in the database but their associated masks may be overwritten.

### REST Endpoint

**`POST /projects/{project_id}/videos/{video_id}/graphcut-masks`**

Request body:
```json
{
  "start_frame": 200,
  "end_frame": 349,
  "seeds": {
    "205": [{"x": 120, "y": 80, "label": 1}, {"x": 300, "y": 200, "label": 2}],
    "210": [{"x": 115, "y": 82, "label": 1}]
  }
}
```

Where `label` 1 = foreground, 2 = background. Coordinates are in native pixel space (integers).

Response:
```json
{
  "frames_processed": 150
}
```

Route lives in `vidseq/api/routes/graphcut.py`, registered in `server.py` with `app.include_router(graphcut.router, prefix="/api", tags=["graphcut"])`. Route calls `graphcut_service.run_graphcut()` via `asyncio.to_thread()` — follows the route → service layering convention.

### Frontend Composable

New composable: `frontend/src/composables/useGraphCut.ts`

**State managed:**
- `graphcutRegion: { start: number, end: number } | null` — current chunk bounds
- `seeds: Map<number, Array<{x: number, y: number, label: number}>>` — accumulated seeds per frame
- `brushSize: number` — current brush radius in display pixels
- `seedsVisible: boolean` — toggle for seed overlay visibility
- `isRunning: boolean` — loading state during solve

**Methods:**
- `placeRegion(frameIdx: number, numFrames: number)` — set chunk from frameIdx to min(frameIdx + 149, numFrames - 1)
- `paintSeed(frameIdx: number, x: number, y: number, label: number)` — add seed point
- `runGraphCut(clearMaskCache: Function)` — POST to API, call the provided `clearMaskCache(start, end)` from `useSegmentation` on success to invalidate cached masks for the chunk frames
- `clearSeeds()` — reset all seeds
- `exitGraphCutMode()` — clear region and seeds

### Component Modifications

**DataTrack.vue:**
- New graph cut region rendering (teal/cyan range bar)
- Click handler for placing the region when graph cut tool is active
- Emit event for region placement

**VideoOverlay.vue:**
- Brush painting mode (mousemove + mousedown tracking for continuous strokes)
- Render seed points as colored pixels (green fg, red bg) with semi-transparency
- Respect `seedsVisible` toggle
- Map display coordinates to native pixel coordinates for seed recording

**VideoDetail.vue:**
- Graph cut tool button in sidebar (entering graph cut mode deactivates all other tools — positive point, negative point, bounding box, marking mode, working range, keypoint labeling — and vice versa)
- Brush size slider
- Run button
- Seed visibility toggle
- Wire up the composable to overlay and data track

### What We're NOT Building

- No new database tables — seeds are ephemeral frontend state
- No TCP/GPU worker involvement — CPU-only in FastAPI process
- No new H5 files — writes to existing `tracker_masks.h5`
- No changes to conditioning frames or SAM2 memory system
- No changes to downstream pipeline (cropping, alignment, PCA)
- No optical flow — temporal edges use same-pixel-location connections only
- No color model (GMMs) — edge weights from raw intensity difference only
- No downsampling — native resolution throughout

### File Layout

| File | Purpose |
|------|---------|
| `vidseq/services/graphcut_service.py` | Graph cut solver (new) |
| `vidseq/api/routes/graphcut.py` | REST endpoint (new) |
| `vidseq/schemas/graphcut.py` | Request/response schemas (new) |
| `frontend/src/composables/useGraphCut.ts` | Frontend state & API (new) |
| `vidseq/services/video_io.py` | VideoFrameSource extracted from segmentation_commands (new) |
| `frontend/src/services/api.ts` | Add `createGraphcutMasks()` API call |
| `frontend/src/components/DataTrack.vue` | Graph cut region rendering + placement |
| `frontend/src/components/VideoOverlay.vue` | Brush painting mode + seed rendering |
| `frontend/src/components/VideoDetail.vue` | Tool button, controls, wiring |

### Dependencies

- **PyMaxflow** (`pip install PyMaxflow`) — Python/C++ wrapper for Kolmogorov's maxflow library. GPL v3 license, acceptable for internal research use.
- All other dependencies (OpenCV, numpy, h5py) already in the project.
