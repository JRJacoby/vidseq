# Box Prompting for Video Detail Screen

**Date:** 2026-04-05
**Branch:** feat/associated-videos (or new branch)
**Status:** Design

## Summary

Add bounding box prompting to the VideoDetail screen as an alternative to point prompting for creating initial segmentation masks. SAM2 was trained with box prompts 25% of the time as initial prompts, making this fully in-distribution. Box prompts create conditioning frames that anchor propagation, just like point prompts do today.

## Constraints

- **Box is initial-prompt only.** SAM2's training used corrective clicks (not corrective boxes) for refinement. The current API enforces `clear_old_points=True` when a box is provided. Boxes always start a new mask; refinement uses points.
- **Correction points after a box are supported.** After a box creates a mask, subsequent point prompts refine it via the existing `refine_mask` path. This is in-distribution (training mixes prompt types).
- **Multiple conditioning frames.** Boxes on frame 1 and frame 50 both become conditioning frames. Propagation from frame 1 uses frame 50's conditioning memory as an anchor.

## End-to-End Flow

1. User clicks "Bounding Box" tool button in VideoDetail toolbar
2. User drags a rectangle on the canvas (mousedown → mousemove → mouseup)
3. VideoOverlay emits `box-complete` with normalized [0,1] coords `{ x1, y1, x2, y2 }`
4. `useSegmentation.handleBoxComplete()` calls `api.submitBoxPrompt()`
5. `POST /projects/{id}/videos/{id}/box-prompt/{frame_idx}` → `segmentation_service.submit_box_prompt()`
6. TCP command `add_box_prompt` → `handle_add_box_prompt()` → `segmentor.add_box_prompt()`
7. SAM2 `track_step()` with point labels [2, 3] (box top-left, bottom-right convention)
8. Frame becomes a conditioning frame; mask PNG returned to frontend
9. User can add correction points via existing point prompt flow (`refine_mask`)
10. User can draw boxes on other frames (each becomes a conditioning frame)
11. Propagation uses all conditioning frames as anchors

## Layer-by-Layer Design

### 1. Frontend: VideoOverlay.vue — Box Drawing Interaction

**New interaction mode:** When `activeTool === 'bounding_box'`:
- `mousedown`: Record start point in normalized coords, begin drag
- `mousemove`: Update rectangle endpoint, re-render with dashed outline
- `mouseup`: Finalize rectangle, emit `'box-complete'` event with `{ x1, y1, x2, y2 }` normalized [0,1]

**Rendering:**
- During drag: dashed rectangle outline (same approach as pending point's dashed circle)
- Completed box prompts: solid rectangle outline in blue (`#3b82f6`) to distinguish from detector bbox (tomato red) and OBB (cyan)
- Box prompts stored in `localPrompts` alongside points; rendered in the existing prompt drawing loop with a type check

**New emit:**
```typescript
(e: 'box-complete', box: { x1: number; y1: number; x2: number; y2: number }): void
```

**Cursor:** Reuses existing `tool-active` class (crosshair cursor, pointer-events enabled).

### 2. Frontend: useSegmentation.ts — State and Handlers

**ToolType extension:**
```typescript
export type ToolType = 'none' | 'positive_point' | 'negative_point' | 'bounding_box'
```

**Prompt union type:**
```typescript
export type LocalPrompt =
  | { x: number; y: number; type: 'positive_point' | 'negative_point' }
  | { x1: number; y1: number; x2: number; y2: number; type: 'bounding_box' }
```

The `localPrompts` map values become `Array<LocalPrompt>`.

**New toggle:**
```typescript
const toggleBoundingBoxTool = () => {
    activeTool.value = activeTool.value === 'bounding_box' ? 'none' : 'bounding_box'
}
```

**New handler — `handleBoxComplete`:**
- Adds box to `localPrompts` for rendering
- Calls `submitBoxPrompt(projectId, videoId, frameIdx, { x1, y1, x2, y2 })`
- Receives mask PNG blob, creates ImageBitmap, updates `currentMask` and `maskCache`
- On error, rolls back box from `localPrompts`
- Same `isSegmenting` guard as `handlePointComplete`

**Correction points after box:** No special handling. Once the box creates a mask, `has_tracker_mask` is true for that frame. Subsequent point prompts go through `handlePointComplete` → `submitPrompt` → `refine_mask`. This already works.

**New exports from composable:**
- `toggleBoundingBoxTool`
- `handleBoxComplete`

### 3. Frontend: api.ts — New API Function

```typescript
export async function submitBoxPrompt(
    projectId: number,
    videoId: number,
    frameIdx: number,
    box: { x1: number; y1: number; x2: number; y2: number }
): Promise<Blob> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/${videoId}/box-prompt/${frameIdx}`,
        {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(box),
        }
    )
    // ... error handling, return blob (same pattern as submitPrompt)
}
```

### 4. Frontend: VideoDetail.vue — Toolbar Button

New "Bounding Box" tool button alongside existing positive/negative point buttons. Wires up:
- `toggleBoundingBoxTool` click handler
- `@box-complete="handleBoxComplete"` event from VideoOverlay
- Active state styling (same pattern as point tool buttons)

### 5. Backend: Schema — BoxPromptRequest

**File:** `vidseq/schemas/segmentation.py`

```python
class BoxPromptRequest(BaseModel):
    """Request to submit a bounding box prompt for segmentation.

    All coordinates are normalized [0, 1].
    """
    x1: float
    y1: float
    x2: float
    y2: float
```

### 6. Backend: Route — POST /box-prompt/{frame_idx}

**File:** `vidseq/api/routes/segmentation/inference.py`

```python
@router.post("/projects/{project_id}/videos/{video_id}/box-prompt/{frame_idx}")
async def submit_box_prompt(
    project_id: int,
    frame_idx: int,
    request: BoxPromptRequest,
    video: Video = Depends(get_video),
    session: AsyncSession = Depends(get_project_session),
):
    """Submit a bounding box prompt to create an initial segmentation mask."""
    mask = await segmentation_service.submit_box_prompt(
        session=session,
        project_id=project_id,
        video_id=video.id,
        frame_idx=frame_idx,
        x1=request.x1, y1=request.y1,
        x2=request.x2, y2=request.y2,
    )
    mask_png = segmentation_service.mask_to_png(mask)
    return Response(content=mask_png, media_type="image/png")
```

### 7. Backend: Service — submit_box_prompt()

**File:** `vidseq/services/segmentation_service.py`

New `submit_box_prompt()` function:
- Calls `segmentation_tcp_client.add_box_prompt(project_id, video_id, frame_idx, x1, y1, x2, y2)`
- Creates `ConditioningFrame` DB record (same pattern as `submit_prompt`)
- Updates mask presence index via `frame_data_service.set_has_tracker_mask()`
- Saves confidence score via `frame_data_service.save_score()`

Simpler than `submit_prompt` — no has_existing_mask branching. Box is always an initial prompt.

### 8. Backend: TCP Client — add_box_prompt()

**File:** `vidseq/services/segmentation_tcp_client.py`

New method on `SegmentationService` class + module-level wrapper:
```python
def add_box_prompt(self, project_id, video_id, frame_idx, x1, y1, x2, y2):
    # Sends: {"type": "add_box_prompt", "video_id": ..., "frame_idx": ...,
    #         "x1": ..., "y1": ..., "x2": ..., "y2": ...}
    # Returns: (mask, score) — same as add_point_prompt
```

Coordinates are passed as normalized [0,1] — the TCP command handler converts to pixels.

### 9. Backend: TCP Command Handler — handle_add_box_prompt()

**File:** `vidseq/services/segmentation_commands.py`

New handler, mirrors `handle_add_prompt`:
1. Extract `video_id, frame_idx, x1, y1, x2, y2` from params
2. Convert normalized coords to pixel coords: `px1 = x1 * resources.width`, etc.
3. Open `tracker_masks` and `tracker_logits` H5 files
4. Call `segmentor.add_box_prompt(video_id, frame_idx, box=(px1, py1, px2, py2), frames=resources.frame_source, masks=mask_data)`
5. Write mask + logits to H5
6. Return `{"type": "add_box_prompt_result", "status": "ok", "mask_rle": ..., "score": ...}`

Register `"add_box_prompt": handle_add_box_prompt` in the command dispatch dict.

### 10. Backend: Streaming Segmentor — add_box_prompt()

**File:** `vidseq/services/segmentation_model/streaming_segmentor.py`

New method, follows `add_point_prompt` structure:

```python
def add_box_prompt(
    self,
    video_id: str,
    frame_idx: int,
    box: tuple[float, float, float, float],  # (x1, y1, x2, y2) in pixel coords
    frames,   # Indexable frame source
    masks,    # Indexable mask source
) -> tuple[np.ndarray, np.ndarray, float]:
```

Implementation:
1. Get session state (`frame_dims`, `output_dict`, `cond_frame_indices`)
2. Call `_set_memory_frame()` to prepare memory for arbitrary frame access
3. Get frame from source
4. Scale box coords from pixel space to 1024x1024 input space:
   ```python
   scaled_x1 = x1 * self.INPUT_SIZE / orig_w
   scaled_y1 = y1 * self.INPUT_SIZE / orig_h
   # ... same for x2, y2
   ```
5. Create `point_inputs` with SAM2 box convention:
   ```python
   point_coords = tensor([[[scaled_x1, scaled_y1], [scaled_x2, scaled_y2]]])  # (1, 2, 2)
   point_labels = tensor([[2, 3]])  # box top-left, box bottom-right
   ```
6. Get image features via `_get_image_features()` and `_prepare_backbone_features()`
7. Determine `is_init_cond_frame = len(output_dict["cond_frame_outputs"]) == 0`
8. Call `predictor.track_step()` with `point_inputs`
9. Extract high-res mask, threshold, resize to original dims
10. Extract low-res logits for potential refinement
11. Add `frame_idx` to `cond_frame_indices`
12. Store compact output to `cond_frame_outputs[frame_idx]`
13. Return `(mask, logits, score)`

This is intentionally separate from `propagate_with_box()`, which was designed for the detector pipeline (takes a pre-read frame, has `add_as_conditioning`/`use_memory_with_prompt` params, no memory setup). The new method follows the interactive prompt pattern.

## What This Design Does NOT Include

- **Box refinement** — out of distribution for SAM2. Only points refine.
- **Box + points in a single request** — SAM2 supports this, but we keep the interaction simple: draw box first, then add correction points as separate requests.
- **Resizing/editing a drawn box** — draw a new one (which replaces via the existing reset-on-new-prompt behavior) or reset the frame.
