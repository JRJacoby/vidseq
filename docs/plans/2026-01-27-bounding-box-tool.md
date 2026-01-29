# Bounding Box Tool Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add an interactive bounding box drawing tool to the video detail screen, allowing users to prompt SAM3 with a bounding box for initial segmentation.

**Architecture:** The bounding box tool is a third option alongside positive/negative point tools. Users click-drag to draw a rectangle on the canvas overlay, which auto-submits to SAM3 on mouse release. After placement, corners can be dragged to resize and the interior can be dragged to move. Each edit re-submits after clearing only box prompts (point prompts are preserved). The backend reuses the existing `handle_add_prompt` TCP command handler which already supports boxes.

**Tech Stack:** Vue 3 (Composition API), TypeScript, FastAPI, Python, SAM3

---

## Task 1: Rename TCP command `add_point_prompt` → `add_prompt`

**Files:**
- Modify: `vidseq/services/sam3/server/tcp_server.py:308-309`
- Modify: `vidseq/services/sam3/server/commands.py:167`
- Modify: `vidseq/services/sam3/service.py:353-354`

**Steps:**

1. In `tcp_server.py`, rename the command dispatch key:

```python
# Line 308: change from
elif cmd_type == "add_point_prompt":
# to
elif cmd_type == "add_prompt":
```

2. In `commands.py`, rename the result type:

```python
# Line 167: change from
"type": "add_point_prompt_result",
# to
"type": "add_prompt_result",
```

3. In `service.py`, rename the command type in `add_point_prompt()`:

```python
# Line 353: change from
"type": "add_point_prompt",
# to
"type": "add_prompt",
```

**Commit:** `refactor: rename add_point_prompt TCP command to add_prompt`

---

## Task 2: Backend — extend SegmentRequest and SAM3Service for bounding box

**Files:**
- Modify: `vidseq/schemas/segmentation.py`
- Modify: `vidseq/services/sam3/service.py`
- Modify: `vidseq/api/routes/segmentation/inference.py`

**Steps:**

1. Update `SegmentRequest` in `schemas/segmentation.py` to accept bounding box:

```python
class SegmentRequest(BaseModel):
    """Request to run segmentation with a point or bounding box prompt."""
    frame_idx: int
    type: Literal["positive_point", "negative_point", "bounding_box"]
    details: dict[str, Any]

    @field_validator("details")
    @classmethod
    def validate_details(cls, v: dict, info) -> dict:
        prompt_type = info.data.get("type")
        if prompt_type in ("positive_point", "negative_point"):
            required = {"x", "y"}
            if not required.issubset(v.keys()):
                raise ValueError(f"point requires {required}")
        elif prompt_type == "bounding_box":
            required = {"x1", "y1", "x2", "y2"}
            if not required.issubset(v.keys()):
                raise ValueError(f"bounding_box requires {required}")
        return v
```

2. Add `add_box_prompt` method to `SAM3Service` (in `service.py`, after `add_point_prompt`):

```python
def add_box_prompt(
    self,
    project_id: int,
    video_id: int,
    video_path: Path,
    frame_idx: int,
    box: list[float],
) -> np.ndarray:
    """
    Add bounding box prompt and get the segmentation mask.

    Args:
        project_id: ID of the project
        video_id: ID of the video
        video_path: Path to video file
        frame_idx: Frame index to segment
        box: [x1, y1, x2, y2] in pixel coordinates

    Returns:
        Binary mask as numpy array (height, width), dtype=uint8, values 0 or 255
    """
    session = self.get_session(project_id, video_id)
    if session is None:
        session = self.init_session(project_id, video_id, video_path)

    result = self._send_and_wait({
        "type": "add_prompt",
        "video_id": video_id,
        "frame_idx": frame_idx,
        "box": box,
        "obj_id": OBJ_ID,
    }, timeout=120.0)

    if result.get("status") != "ok":
        raise RuntimeError(result.get("error", "Failed to add box prompt"))

    session.has_object = True

    mask_rle = result["mask_rle"]
    mask_shape = tuple(result["mask_shape"])
    mask_dtype = result.get("mask_dtype", "uint8")
    mask = _decode_mask_rle(mask_rle, mask_shape, mask_dtype)

    # Store box prompt info (clear previous box prompts first)
    self.delete_box_prompts_for_frame(frame_idx)
    prompt = {
        "type": "bounding_box",
        "x1": box[0],
        "y1": box[1],
        "x2": box[2],
        "y2": box[3],
        "frame_idx": frame_idx,
    }
    if frame_idx not in self._prompts:
        self._prompts[frame_idx] = []
    self._prompts[frame_idx].append(prompt)

    return mask
```

3. Add `delete_box_prompts_for_frame` method to `SAM3Service`:

```python
def delete_box_prompts_for_frame(self, frame_idx: int) -> None:
    """Delete only bounding box prompts for a specific frame (keeps point prompts)."""
    if frame_idx in self._prompts:
        self._prompts[frame_idx] = [
            p for p in self._prompts[frame_idx]
            if p["type"] != "bounding_box"
        ]
        if not self._prompts[frame_idx]:
            del self._prompts[frame_idx]
```

4. Add module-level wrapper function for `add_box_prompt` (at bottom of `service.py`, near the other wrappers):

```python
def add_box_prompt(
    project_id: int,
    video_id: int,
    video_path: Path,
    frame_idx: int,
    box: list[float],
) -> np.ndarray:
    """Add bounding box prompt and get the segmentation mask."""
    return SAM3Service.get_instance().add_box_prompt(
        project_id, video_id, video_path, frame_idx, box
    )
```

5. Update the inference route in `inference.py` to handle bounding box:

```python
@router.post("/projects/{project_id}/videos/{video_id}/segment")
async def run_segmentation(
    project_id: int,
    segment_request: SegmentRequest,
    video: Video = Depends(get_video),
    session: AsyncSession = Depends(get_project_session),
    project_path: Path = Depends(get_project_folder),
):
    """
    Run segmentation with a point or bounding box prompt.

    Point coords should be normalized [0,1].
    Bounding box coords are in pixel space [x1, y1, x2, y2].
    """
    video_path = Path(video.path)

    try:
        if segment_request.type == "bounding_box":
            mask = sam3_service.add_box_prompt(
                project_id=project_id,
                video_id=video.id,
                video_path=video_path,
                frame_idx=segment_request.frame_idx,
                box=[
                    segment_request.details["x1"],
                    segment_request.details["y1"],
                    segment_request.details["x2"],
                    segment_request.details["y2"],
                ],
            )
        else:
            label = 1 if segment_request.type == "positive_point" else 0
            mask = sam3_service.add_point_prompt(
                project_id=project_id,
                video_id=video.id,
                video_path=video_path,
                frame_idx=segment_request.frame_idx,
                points=[[segment_request.details["x"], segment_request.details["y"]]],
                labels=[label],
            )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    mask_storage.save_mask(
        project_path=project_path,
        video_id=video.id,
        frame_idx=segment_request.frame_idx,
        mask=mask,
        num_frames=video.num_frames,
        height=video.height,
        width=video.width,
    )

    has_content = bool(np.any(mask > 0))
    await frame_data_service.set_has_mask(
        session, video.id, segment_request.frame_idx, has_content
    )

    await conditioning_service.add_conditioning_frame(
        session=session,
        video_id=video.id,
        frame_idx=segment_request.frame_idx,
    )

    mask_png = segmentation_service.mask_to_png(mask)
    return Response(content=mask_png, media_type="image/png")
```

**Commit:** `feat: add bounding box prompt support to backend`

---

## Task 3: Frontend — extend types and add bbox API function

**Files:**
- Modify: `frontend/src/services/api.ts`
- Modify: `frontend/src/composables/useSegmentation.ts`

**Steps:**

1. Update `StoredPrompt` type and `runSegmentation` in `api.ts`:

```typescript
// Update StoredPrompt to support bounding box
export interface StoredPrompt {
    type: 'positive_point' | 'negative_point' | 'bounding_box'
    details: { x: number; y: number } | { x1: number; y1: number; x2: number; y2: number }
    createdAt: string
}

// Update runSegmentation to accept bounding_box type
export async function runSegmentation(
    projectId: number,
    videoId: number,
    frameIdx: number,
    type: 'positive_point' | 'negative_point' | 'bounding_box',
    details: { x: number; y: number } | { x1: number; y1: number; x2: number; y2: number }
): Promise<Blob> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/${videoId}/segment`,
        {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ frame_idx: frameIdx, type, details }),
        }
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to run segmentation'))
    }
    return response.blob()
}
```

2. Update `getPromptsForFrame` in `api.ts` to handle bbox prompts:

```typescript
export async function getPromptsForFrame(
    projectId: number,
    videoId: number,
    frameIdx: number
): Promise<StoredPrompt[]> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/${videoId}/prompts/${frameIdx}`
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to fetch prompts'))
    }
    const prompts = await response.json()
    return prompts.map((p: Record<string, any>) => {
        if (p.type === 'bounding_box') {
            return {
                type: 'bounding_box' as const,
                details: { x1: p.x1, y1: p.y1, x2: p.x2, y2: p.y2 },
                createdAt: new Date().toISOString(),
            }
        }
        return {
            type: p.type as 'positive_point' | 'negative_point',
            details: { x: p.x, y: p.y },
            createdAt: new Date().toISOString(),
        }
    })
}
```

3. Also update `getAllPrompts` in `api.ts` to use the same mapping logic.

4. Update `ToolType` in `useSegmentation.ts`:

```typescript
export type ToolType = 'none' | 'positive_point' | 'negative_point' | 'bounding_box'
```

5. Add `toggleBboxTool` and `handleBboxComplete` to `useSegmentation.ts`:

```typescript
const toggleBboxTool = () => {
    activeTool.value = activeTool.value === 'bounding_box' ? 'none' : 'bounding_box'
}

const handleBboxComplete = async (bbox: { x1: number; y1: number; x2: number; y2: number }) => {
    if (!projectId.value || !videoId.value) return

    isSegmenting.value = true

    try {
        const maskBlob = await runSegmentation(
            projectId.value,
            videoId.value,
            currentFrameIdx.value,
            'bounding_box',
            { x1: bbox.x1, y1: bbox.y1, x2: bbox.x2, y2: bbox.y2 }
        )

        const bitmap = await createImageBitmap(maskBlob)
        currentMask.value = bitmap

        const bboxResult = await getBbox(projectId.value, videoId.value, currentFrameIdx.value)
        currentBbox.value = bboxResult

        maskCache.set(currentFrameIdx.value, bitmap)
        bboxCache.set(currentFrameIdx.value, bboxResult)

        await fetchPromptsForFrame(currentFrameIdx.value)
    } catch (e) {
        console.error('Failed to add bbox:', e)
    } finally {
        isSegmenting.value = false
    }
}
```

6. Add `toggleBboxTool` and `handleBboxComplete` to the return object and `UseSegmentationReturn` interface.

**Commit:** `feat: add bounding box types, API, and composable logic`

---

## Task 4: Frontend — add bbox button and wire up VideoDetail

**Files:**
- Modify: `frontend/src/components/VideoDetail.vue`

**Steps:**

1. Import `handleBboxComplete` and `toggleBboxTool` from `useSegmentation` (add to the destructured return, around line 89-103).

2. Add bounding box button between negative-point and reset-frame buttons (after line 323):

```html
<button
  class="tool-button bounding-box"
  :class="{ active: activeTool === 'bounding_box' }"
  @click="toggleBboxTool"
  :disabled="isSegmenting || !segmentationIsReady"
>
  <span class="tool-icon">▢</span>
  <span class="tool-label">Bounding Box</span>
</button>
```

3. Add `@bbox-complete="handleBboxComplete"` to the `VideoOverlay` component props (around line 264).

4. Add CSS for active bounding box button:

```css
.tool-button.bounding-box.active {
  background-color: #dbeafe;
  border-color: #3b82f6;
  color: #1d4ed8;
}
```

**Commit:** `feat: add bounding box button to video detail toolbar`

---

## Task 5: Frontend — update VideoOverlay types and emit

**Files:**
- Modify: `frontend/src/components/VideoOverlay.vue`

**Steps:**

1. Update the `ToolType` export (line 5):

```typescript
export type ToolType = 'none' | 'positive_point' | 'negative_point' | 'bounding_box'
```

2. Add `bbox-complete` emit alongside `point-complete` (line 19-21):

```typescript
const emit = defineEmits<{
  (e: 'point-complete', point: { x: number; y: number; type: 'positive_point' | 'negative_point' }): void
  (e: 'bbox-complete', bbox: { x1: number; y1: number; x2: number; y2: number }): void
}>()
```

3. Add bounding box state refs (after `pendingPoint` on line 24):

```typescript
// Bounding box drawing state
const drawingBox = ref<{ startX: number; startY: number; endX: number; endY: number } | null>(null)
const placedBox = ref<{ x1: number; y1: number; x2: number; y2: number } | null>(null)
const editMode = ref<'none' | 'draw' | 'move' | 'resize-tl' | 'resize-tr' | 'resize-bl' | 'resize-br'>('none')
const editStart = ref<{ x: number; y: number; box: { x1: number; y1: number; x2: number; y2: number } } | null>(null)
```

**Commit:** `feat: add bbox types and state to VideoOverlay`

---

## Task 6: Frontend — implement bbox mouse event handling in VideoOverlay

**Files:**
- Modify: `frontend/src/components/VideoOverlay.vue`

**Steps:**

1. Add hit-test helper functions (before `onMouseDown`):

```typescript
const HANDLE_SIZE = 10 // pixels for corner hit detection

function getHitZone(coords: { x: number; y: number }, box: { x1: number; y1: number; x2: number; y2: number }): 'tl' | 'tr' | 'bl' | 'br' | 'inside' | 'outside' {
  const canvas = canvasRef.value
  if (!canvas) return 'outside'

  // Convert normalized coords to pixel for hit testing
  const px = coords.x * canvas.width
  const py = coords.y * canvas.height
  const bx1 = box.x1 * canvas.width
  const by1 = box.y1 * canvas.height
  const bx2 = box.x2 * canvas.width
  const by2 = box.y2 * canvas.height

  const threshold = HANDLE_SIZE

  // Check corners first
  if (Math.abs(px - bx1) < threshold && Math.abs(py - by1) < threshold) return 'tl'
  if (Math.abs(px - bx2) < threshold && Math.abs(py - by1) < threshold) return 'tr'
  if (Math.abs(px - bx1) < threshold && Math.abs(py - by2) < threshold) return 'bl'
  if (Math.abs(px - bx2) < threshold && Math.abs(py - by2) < threshold) return 'br'

  // Check inside
  if (px >= bx1 && px <= bx2 && py >= by1 && py <= by2) return 'inside'

  return 'outside'
}
```

2. Replace `onMouseDown` with expanded version:

```typescript
function onMouseDown(event: MouseEvent) {
  const coords = getNormalizedCoords(event)
  if (!coords) return

  if (props.activeTool === 'positive_point' || props.activeTool === 'negative_point') {
    pendingPoint.value = { x: coords.x, y: coords.y, type: props.activeTool }
    render()
    emit('point-complete', { x: coords.x, y: coords.y, type: props.activeTool })
  } else if (props.activeTool === 'bounding_box') {
    if (placedBox.value) {
      const zone = getHitZone(coords, placedBox.value)
      if (zone === 'tl' || zone === 'tr' || zone === 'bl' || zone === 'br') {
        editMode.value = `resize-${zone}` as any
        editStart.value = { x: coords.x, y: coords.y, box: { ...placedBox.value } }
        return
      } else if (zone === 'inside') {
        editMode.value = 'move'
        editStart.value = { x: coords.x, y: coords.y, box: { ...placedBox.value } }
        return
      }
    }
    // Start new box draw
    editMode.value = 'draw'
    drawingBox.value = { startX: coords.x, startY: coords.y, endX: coords.x, endY: coords.y }
    placedBox.value = null
  }
}
```

3. Add `onMouseMove` handler:

```typescript
function onMouseMove(event: MouseEvent) {
  const coords = getNormalizedCoords(event)
  if (!coords) return

  if (props.activeTool === 'bounding_box') {
    // Update cursor based on hover zone
    if (editMode.value === 'none' && placedBox.value) {
      const canvas = canvasRef.value
      if (canvas) {
        const zone = getHitZone(coords, placedBox.value)
        if (zone === 'tl' || zone === 'br') canvas.style.cursor = 'nwse-resize'
        else if (zone === 'tr' || zone === 'bl') canvas.style.cursor = 'nesw-resize'
        else if (zone === 'inside') canvas.style.cursor = 'move'
        else canvas.style.cursor = 'crosshair'
      }
    }

    if (editMode.value === 'draw' && drawingBox.value) {
      drawingBox.value.endX = coords.x
      drawingBox.value.endY = coords.y
      render()
    } else if (editMode.value === 'move' && editStart.value) {
      const dx = coords.x - editStart.value.x
      const dy = coords.y - editStart.value.y
      const orig = editStart.value.box
      placedBox.value = {
        x1: Math.max(0, Math.min(1, orig.x1 + dx)),
        y1: Math.max(0, Math.min(1, orig.y1 + dy)),
        x2: Math.max(0, Math.min(1, orig.x2 + dx)),
        y2: Math.max(0, Math.min(1, orig.y2 + dy)),
      }
      render()
    } else if (editMode.value.startsWith('resize-') && editStart.value) {
      const orig = editStart.value.box
      const corner = editMode.value.replace('resize-', '')
      const newBox = { ...orig }

      if (corner === 'tl') { newBox.x1 = coords.x; newBox.y1 = coords.y }
      else if (corner === 'tr') { newBox.x2 = coords.x; newBox.y1 = coords.y }
      else if (corner === 'bl') { newBox.x1 = coords.x; newBox.y2 = coords.y }
      else if (corner === 'br') { newBox.x2 = coords.x; newBox.y2 = coords.y }

      // Normalize so x1<x2, y1<y2
      placedBox.value = {
        x1: Math.max(0, Math.min(newBox.x1, newBox.x2)),
        y1: Math.max(0, Math.min(newBox.y1, newBox.y2)),
        x2: Math.min(1, Math.max(newBox.x1, newBox.x2)),
        y2: Math.min(1, Math.max(newBox.y1, newBox.y2)),
      }
      render()
    }
  }
}
```

4. Add `onMouseUp` handler:

```typescript
function onMouseUp(event: MouseEvent) {
  if (props.activeTool !== 'bounding_box') return

  if (editMode.value === 'draw' && drawingBox.value) {
    // Finalize drawn box
    const x1 = Math.max(0, Math.min(drawingBox.value.startX, drawingBox.value.endX))
    const y1 = Math.max(0, Math.min(drawingBox.value.startY, drawingBox.value.endY))
    const x2 = Math.min(1, Math.max(drawingBox.value.startX, drawingBox.value.endX))
    const y2 = Math.min(1, Math.max(drawingBox.value.startY, drawingBox.value.endY))

    // Ignore tiny boxes (accidental clicks)
    if (Math.abs(x2 - x1) > 0.01 && Math.abs(y2 - y1) > 0.01) {
      placedBox.value = { x1, y1, x2, y2 }
      submitBbox()
    }

    drawingBox.value = null
  } else if (editMode.value === 'move' || editMode.value.startsWith('resize-')) {
    // Finalize edit
    if (placedBox.value) {
      submitBbox()
    }
  }

  editMode.value = 'none'
  editStart.value = null
}
```

5. Add `submitBbox` helper:

```typescript
function submitBbox() {
  if (!placedBox.value) return
  const canvas = canvasRef.value
  if (!canvas) return

  // Convert normalized coords to pixel coords for the backend
  emit('bbox-complete', {
    x1: placedBox.value.x1 * canvas.width,
    y1: placedBox.value.y1 * canvas.height,
    x2: placedBox.value.x2 * canvas.width,
    y2: placedBox.value.y2 * canvas.height,
  })
}
```

6. Wire up mouse events in the template — add `@mousemove` and `@mouseup`:

```html
<canvas
  ref="canvasRef"
  class="video-overlay"
  :class="{ 'tool-active': activeTool !== 'none' }"
  @mousedown="onMouseDown"
  @mousemove="onMouseMove"
  @mouseup="onMouseUp"
/>
```

7. Clear `placedBox` when tool changes or frame changes (add a watcher):

```typescript
watch(() => props.activeTool, () => {
  placedBox.value = null
  drawingBox.value = null
  editMode.value = 'none'
  render()
})
```

**Commit:** `feat: implement bbox mouse interaction in VideoOverlay`

---

## Task 7: Frontend — implement bbox rendering in VideoOverlay

**Files:**
- Modify: `frontend/src/components/VideoOverlay.vue`

**Steps:**

1. Add bbox drawing to the `render()` function. After drawing stored prompts (after line 125), add:

```typescript
// Draw stored bounding box prompts
for (const prompt of props.prompts) {
  if (prompt.type === 'bounding_box') {
    const details = prompt.details as { x1: number; y1: number; x2: number; y2: number }
    const bx1 = details.x1 * canvas.width
    const by1 = details.y1 * canvas.height
    const bx2 = details.x2 * canvas.width
    const by2 = details.y2 * canvas.height

    // Semi-transparent fill
    ctx.fillStyle = 'rgba(59, 130, 246, 0.1)'
    ctx.fillRect(bx1, by1, bx2 - bx1, by2 - by1)

    // Blue stroke
    ctx.strokeStyle = '#3b82f6'
    ctx.lineWidth = 2
    ctx.setLineDash([])
    ctx.strokeRect(bx1, by1, bx2 - bx1, by2 - by1)
  }
}
```

2. Add drawing-in-progress box rendering (after pending point drawing, near end of `render()`):

```typescript
// Draw in-progress bounding box (while dragging)
if (drawingBox.value && props.activeTool === 'bounding_box') {
  const x1 = Math.min(drawingBox.value.startX, drawingBox.value.endX) * canvas.width
  const y1 = Math.min(drawingBox.value.startY, drawingBox.value.endY) * canvas.height
  const x2 = Math.max(drawingBox.value.startX, drawingBox.value.endX) * canvas.width
  const y2 = Math.max(drawingBox.value.startY, drawingBox.value.endY) * canvas.height

  ctx.strokeStyle = '#3b82f6'
  ctx.lineWidth = 2
  ctx.setLineDash([6, 4])
  ctx.strokeRect(x1, y1, x2 - x1, y2 - y1)
  ctx.fillStyle = 'rgba(59, 130, 246, 0.1)'
  ctx.fillRect(x1, y1, x2 - x1, y2 - y1)
}

// Draw placed bounding box (editable)
if (placedBox.value && props.activeTool === 'bounding_box') {
  const bx1 = placedBox.value.x1 * canvas.width
  const by1 = placedBox.value.y1 * canvas.height
  const bx2 = placedBox.value.x2 * canvas.width
  const by2 = placedBox.value.y2 * canvas.height

  // Fill
  ctx.fillStyle = 'rgba(59, 130, 246, 0.1)'
  ctx.fillRect(bx1, by1, bx2 - bx1, by2 - by1)

  // Stroke
  ctx.strokeStyle = '#3b82f6'
  ctx.lineWidth = 2
  ctx.setLineDash([])
  ctx.strokeRect(bx1, by1, bx2 - bx1, by2 - by1)

  // Corner handles (white squares with blue border)
  const handleSize = 6
  for (const [cx, cy] of [[bx1, by1], [bx2, by1], [bx1, by2], [bx2, by2]]) {
    ctx.fillStyle = '#fff'
    ctx.fillRect(cx - handleSize / 2, cy - handleSize / 2, handleSize, handleSize)
    ctx.strokeStyle = '#3b82f6'
    ctx.lineWidth = 1
    ctx.strokeRect(cx - handleSize / 2, cy - handleSize / 2, handleSize, handleSize)
  }
}
```

**Commit:** `feat: implement bbox rendering in VideoOverlay`

---

## Risk Notes

- **SAM3 box coord format:** The backend `handle_add_prompt` converts pixel XYXY to normalized XYWH before sending to SAM3. The frontend sends pixel coords (canvas dimensions). This matches the existing flow where canvas dimensions equal video dimensions.
- **State clearing:** `add_box_prompt` calls `delete_box_prompts_for_frame` to remove only box prompts from the in-memory store. SAM3's `add_prompt` with a box resets the detector's internal state automatically (by design). Point prompts that were added to the tracker remain in SAM3's inference state.
- **Coordinate space:** The canvas is sized to match `videoWidth`/`videoHeight` (actual video pixel dimensions), so normalized [0,1] canvas coords map directly to normalized [0,1] video coords. The `submitBbox` function converts back to pixel coords for the backend.
