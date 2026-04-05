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

**Template changes:** Add `@mousemove` and `@mouseup` event bindings to the canvas element (currently only `@mousedown` exists). These are only active when `activeTool === 'bounding_box'`.

**Type changes:** Move `LocalPrompt` definition from VideoOverlay.vue to `useSegmentation.ts` as a union type (see Section 2). Update VideoOverlay's `prompts` prop to accept `LocalPrompt[]`. Update the `render()` function to branch on `prompt.type`:
- Point prompts (`positive_point` / `negative_point`): existing circle + crosshair rendering
- Box prompts (`bounding_box`): solid rectangle outline in blue (`#3b82f6`)

**Rendering during drag:** Dashed rectangle outline (same approach as pending point's dashed circle). Track as `pendingBox` ref, cleared when `activeTool` changes (same as `pendingPoint`).

**Rendering completed box prompts:** Solid rectangle outline in blue (`#3b82f6`) to distinguish from detector bbox (tomato red) and OBB (cyan). Boxes render in array order alongside points — since a box replaces prior prompts on the frame (see Section 2), the array will contain `[box, correction_point, ...]` which gives correct visual layering.

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
1. **Clear existing prompts for this frame** — `localPrompts.value.delete(currentFrameIdx.value)`. A box always starts a fresh mask, so any previous points or boxes for this frame are stale.
2. Add box to `localPrompts` for rendering
3. Call `submitBoxPrompt(projectId, videoId, frameIdx, { x1, y1, x2, y2 })`
4. Receive mask PNG blob, create ImageBitmap, update `currentMask` and `maskCache`
5. On error, roll back box from `localPrompts`
6. Same `isSegmenting` guard as `handlePointComplete`

**Correction points after box:** No special handling. Once the box creates a mask, `has_tracker_mask` is true for that frame. Subsequent point prompts go through `handlePointComplete` → `submitPrompt` → `refine_mask`. This already works. The logits needed for refinement are written to H5 by the box prompt handler (see Section 9).

**New exports from composable** (also update the `UseSegmentationReturn` interface):
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
- Returns `(mask, score)` — TCP client decodes mask RLE, same as `add_point_prompt`
- Upserts `ConditioningFrame` DB record (same check-then-insert pattern as `submit_prompt` lines 274-281)
- Updates mask presence index via `frame_data_service.set_has_tracker_mask()`
- Saves confidence score via `frame_data_service.save_score()`

Simpler than `submit_prompt` — no has_existing_mask branching. Box is always an initial prompt. The segmentor handles clearing old conditioning state internally (see Section 10).

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
1. Extract `video_id, frame_idx, x1, y1, x2, y2` from params (all normalized [0,1])
2. Convert normalized coords to pixel coords: `px1 = x1 * resources.width`, etc.
3. Open `tracker_masks` and `tracker_logits` H5 files
4. Call `segmentor.add_box_prompt(video_id, frame_idx, box=(px1, py1, px2, py2), frames=resources.frame_source, masks=mask_data)`
5. Write **both mask and logits** to H5 — logits are required for subsequent point refinement via `refine_mask`
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
2. **Pop any existing conditioning output for this frame** — `output_dict["cond_frame_outputs"].pop(frame_idx, None)`. Same pattern as `refine_mask` (line 708). Without this, old mask memory for the same frame would contaminate the fresh box prompt via self-attention.
3. Call `_set_memory_frame()` to prepare memory for arbitrary frame access
4. Get frame from source
5. Scale box coords from pixel space to 1024x1024 input space:
   ```python
   scaled_x1 = x1 * self.INPUT_SIZE / orig_w
   scaled_y1 = y1 * self.INPUT_SIZE / orig_h
   # ... same for x2, y2
   ```
6. Create `point_inputs` with SAM2 box convention:
   ```python
   point_coords = tensor([[[scaled_x1, scaled_y1], [scaled_x2, scaled_y2]]])  # (1, 2, 2)
   point_labels = tensor([[2, 3]])  # box top-left, box bottom-right
   ```
7. Get image features via `_get_image_features()` and `_prepare_backbone_features()`
8. Determine `is_init_cond_frame = len(output_dict["cond_frame_outputs"]) == 0`
9. Call `predictor.track_step()` with `point_inputs`
10. Extract high-res mask, threshold, resize to original dims
11. Extract low-res logits for potential refinement
12. Add `frame_idx` to `cond_frame_indices`
13. Store compact output to `cond_frame_outputs[frame_idx]`
14. Return `(mask, logits, score)`

**`is_init_cond_frame` note:** This uses the same logic as `add_point_prompt` — `True` only when no other conditioning frames exist. This means a box on frame 50, when frame 0 already has a conditioning frame, will have `is_init_cond_frame=False` and SAM2 will cross-attend to frame 0's memory. This is the desired behavior for interactive use: multiple conditioning frames should inform each other. This differs from `propagate_with_box()` in the detector pipeline, which always uses `is_init_cond_frame=True` to ignore memory (appropriate for automated detector-driven propagation where boxes come from a detector, not a human).

This is intentionally separate from `propagate_with_box()`, which was designed for the detector pipeline (takes a pre-read frame, has `add_as_conditioning`/`use_memory_with_prompt` params, no memory setup). The new method follows the interactive prompt pattern.

## What This Design Does NOT Include

- **Box refinement** — out of distribution for SAM2. Only points refine.
- **Box + points in a single request** — SAM2 supports this, but we keep the interaction simple: draw box first, then add correction points as separate requests.
- **Resizing/editing a drawn box** — draw a new box on the same frame (the handler clears old prompts from `localPrompts` and the segmentor pops old conditioning state before generating the new mask) or reset the frame.
