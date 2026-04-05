# Box Prompting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add bounding box prompting to the VideoDetail screen as an alternative initial prompt type alongside points, threading through all five layers (overlay → composable → API → TCP → segmentor).

**Architecture:** New `POST /box-prompt/{frame_idx}` endpoint with its own schema, TCP command, and segmentor method. Frontend adds drag-to-draw box interaction on the canvas overlay. Box prompts create conditioning frames identical to point prompts; correction points after a box use the existing `refine_mask` flow.

**Tech Stack:** Vue 3 (Composition API), TypeScript, FastAPI, Pydantic, TCP JSON protocol, PyTorch/SAM2, HDF5

**Spec:** `docs/superpowers/specs/2026-04-05-box-prompting-design.md`

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| Modify | `vidseq/services/segmentation_model/streaming_segmentor.py` | New `add_box_prompt()` method |
| Modify | `vidseq/services/segmentation_commands.py` | New `handle_add_box_prompt()` handler |
| Modify | `vidseq/services/segmentation_tcp_server.py` | Register `add_box_prompt` command |
| Modify | `vidseq/services/segmentation_tcp_client.py` | New `add_box_prompt()` client method + module wrapper |
| Modify | `vidseq/schemas/segmentation.py` | New `BoxPromptRequest` schema |
| Modify | `vidseq/api/routes/segmentation/inference.py` | New `submit_box_prompt` route |
| Modify | `vidseq/services/segmentation_service.py` | New `submit_box_prompt()` service function |
| Modify | `frontend/src/services/api.ts` | New `submitBoxPrompt()` API function |
| Modify | `frontend/src/composables/useSegmentation.ts` | `ToolType` union, `LocalPrompt` union, `handleBoxComplete`, `toggleBoundingBoxTool` |
| Modify | `frontend/src/components/VideoOverlay.vue` | Box drag interaction, union prompt rendering |
| Modify | `frontend/src/components/VideoDetail.vue` | Toolbar button, wire up events |

---

### Task 1: Streaming Segmentor — add_box_prompt()

**Files:**
- Modify: `vidseq/services/segmentation_model/streaming_segmentor.py` (after `add_point_prompt` ending at line 667)

- [ ] **Step 1: Add `add_box_prompt` method**

Add this method after `add_point_prompt` (after line 667), before `refine_mask`:

```python
    def add_box_prompt(
        self,
        video_id: str,
        frame_idx: int,
        box: tuple[float, float, float, float],  # (x1, y1, x2, y2) in pixel coords
        frames,  # Indexable frame source: frames[idx] -> np.ndarray (H, W, 3)
        masks,   # Indexable mask source: masks[idx] -> np.ndarray (H, W)
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Add a bounding box prompt to a frame and generate initial mask.

        Creates a new mask from the box prompt. Any existing conditioning state
        for this frame is cleared first. For refining an existing mask with
        additional points, use refine_mask() instead.

        Args:
            video_id: The video identifier.
            frame_idx: Index of the frame to annotate.
            box: (x1, y1, x2, y2) bounding box in original frame pixel coords.
            frames: Indexable frame source returning BGR uint8 (H, W, 3).
            masks: Indexable mask source returning uint8 (H, W).

        Returns:
            Tuple of (mask, logits, score) where:
            - mask: Binary mask array (height, width) with dtype uint8, values 0 or 255.
            - logits: Low-res logits array (256, 256) for potential refinement.
            - score: Predicted IoU confidence in [0, 1].
        """
        # 1. Get session state
        session = self.sessions[video_id]
        frame_dims = session["frame_dims"]  # (height, width)
        output_dict = session["output_dict"]
        cond_frame_indices = session["cond_frame_indices"]

        # 2. Pop existing conditioning output for this frame to avoid self-bias
        output_dict["cond_frame_outputs"].pop(frame_idx, None)

        # 3. Prepare memory for arbitrary frame access
        self._set_memory_frame(video_id, frame_idx, frames, masks)

        # 4. Get frame from source
        frame = frames[frame_idx]

        # 5. Scale box coords to INPUT_SIZE (1024) space
        orig_h, orig_w = frame_dims
        x1, y1, x2, y2 = box
        scaled_x1 = x1 * self.INPUT_SIZE / orig_w
        scaled_y1 = y1 * self.INPUT_SIZE / orig_h
        scaled_x2 = x2 * self.INPUT_SIZE / orig_w
        scaled_y2 = y2 * self.INPUT_SIZE / orig_h

        # 6. Create point_inputs with SAM2 box convention (labels 2=TL, 3=BR)
        point_coords = torch.tensor(
            [[[scaled_x1, scaled_y1], [scaled_x2, scaled_y2]]],
            dtype=torch.float32,
            device=self.device,
        )
        point_labels = torch.tensor(
            [[2, 3]], dtype=torch.int32, device=self.device
        )
        point_inputs = {
            "point_coords": point_coords,
            "point_labels": point_labels,
        }

        # 7. Get image features and prepare backbone features
        _, backbone_out = self._get_image_features(video_id, frame_idx, frame)
        current_vision_feats, current_vision_pos_embeds, feat_sizes = (
            self._prepare_backbone_features(backbone_out)
        )

        # 8. Determine is_init_cond_frame (True if no conditioning frames exist)
        is_init_cond_frame = len(output_dict["cond_frame_outputs"]) == 0

        # 9. Call track_step with box as point_inputs
        with torch.inference_mode(), torch.autocast("cuda", torch.bfloat16):
            current_out = self.predictor.track_step(
                frame_idx=frame_idx,
                is_init_cond_frame=is_init_cond_frame,
                current_vision_feats=current_vision_feats,
                current_vision_pos_embeds=current_vision_pos_embeds,
                feat_sizes=feat_sizes,
                point_inputs=point_inputs,
                mask_inputs=None,
                output_dict=output_dict,
                num_frames=session["num_frames"],
            )

        # 10. Extract high-res mask, threshold, resize to original dims
        pred_mask_high_res = current_out["pred_masks_high_res"][0, 0]
        mask_binary = (pred_mask_high_res > 0).to(torch.uint8).mul(255).cpu().numpy()
        mask_resized = cv2.resize(
            mask_binary,
            (orig_w, orig_h),
            interpolation=cv2.INTER_NEAREST,
        )

        # 11. Get low-res logits for potential refinement
        pred_masks_low_res = current_out["pred_masks"][0, 0].cpu().numpy()

        # 12. Add frame_idx to cond_frame_indices
        cond_frame_indices.add(frame_idx)

        # 13. Store compact output in cond_frame_outputs
        output_dict["cond_frame_outputs"][frame_idx] = self._make_compact_output(current_out)

        # 14. Return mask, logits, and score
        score = self._extract_score(current_out)
        return mask_resized, pred_masks_low_res, score
```

- [ ] **Step 2: Verify imports exist**

The method uses `torch` and `cv2` which are already imported at the top of the file. No new imports needed.

- [ ] **Step 3: Commit**

```bash
git add vidseq/services/segmentation_model/streaming_segmentor.py
git commit -m "feat: add add_box_prompt() to streaming segmentor"
```

---

### Task 2: TCP Command Handler — handle_add_box_prompt()

**Files:**
- Modify: `vidseq/services/segmentation_commands.py` (after `handle_add_prompt` ending at line 282)

- [ ] **Step 1: Add `handle_add_box_prompt` function**

Add after `handle_add_prompt` (after line 282), before `handle_refine_mask`:

```python
def handle_add_box_prompt(
    params: dict,
    segmentor: StreamingSegmentor,
) -> dict:
    """Add bounding box prompt to a frame.

    Args:
        params: Command params with video_id, frame_idx, x1, y1, x2, y2
                (all coordinates normalized [0, 1])
        segmentor: StreamingSegmentor instance

    Returns:
        Response dict with mask_rle
    """
    video_id = params["video_id"]
    frame_idx = params["frame_idx"]
    x1 = params["x1"]  # normalized [0, 1]
    y1 = params["y1"]
    x2 = params["x2"]
    y2 = params["y2"]

    if segmentor is None:
        raise RuntimeError("Model not loaded")

    if video_id not in _video_resources:
        raise RuntimeError(f"No session for video {video_id}")

    resources = _video_resources[video_id]

    # Convert normalized coords to pixel coords
    px1 = x1 * resources.width
    py1 = y1 * resources.height
    px2 = x2 * resources.width
    py2 = y2 * resources.height

    with tracker_masks(resources.project_path, video_id, "a") as mask_data, \
         tracker_logits(resources.project_path, video_id, "a") as logits_data:
        mask_before = mask_data[frame_idx]
        before_sum = int(mask_before.sum())

        mask, logits, score = segmentor.add_box_prompt(
            video_id=str(video_id),
            frame_idx=frame_idx,
            box=(px1, py1, px2, py2),
            frames=resources.frame_source,
            masks=mask_data,
        )

        # Write both mask and logits — logits are required for point refinement
        mask_data[frame_idx] = mask
        logits_data[frame_idx] = logits

    after_sum = int(mask.sum())
    print(f"[Segmentation Worker] add_box_prompt frame={frame_idx} "
          f"box=({px1:.1f}, {py1:.1f}, {px2:.1f}, {py2:.1f}) "
          f"mask_sum: {before_sum} -> {after_sum}")

    return {
        "type": "add_box_prompt_result",
        "status": "ok",
        "mask_rle": encode_mask_rle(mask),
        "mask_shape": mask.shape,
        "mask_dtype": str(mask.dtype),
        "score": score,
    }
```

- [ ] **Step 2: Register in TCP server dispatch**

In `vidseq/services/segmentation_tcp_server.py`, add a new `elif` block after the `add_prompt` handler (after line 311). Find:

```python
            elif cmd_type == "refine_mask":
```

Add before it:

```python
            elif cmd_type == "add_box_prompt":
                if self._segmentor is None:
                    raise RuntimeError("Model not loaded")
                result = handle_add_box_prompt(cmd, self._segmentor)
```

Also add the import at the top of `segmentation_tcp_server.py` — find the existing import line for `handle_add_prompt` and add `handle_add_box_prompt` to it.

- [ ] **Step 3: Commit**

```bash
git add vidseq/services/segmentation_commands.py vidseq/services/segmentation_tcp_server.py
git commit -m "feat: add handle_add_box_prompt TCP command handler"
```

---

### Task 3: TCP Client — add_box_prompt()

**Files:**
- Modify: `vidseq/services/segmentation_tcp_client.py` (after `add_point_prompt` ending at line 562, and after module-level `add_point_prompt` at line 1032)

- [ ] **Step 1: Add class method**

Add after `add_point_prompt` method (after line 562), before `refine_mask`:

```python
    def add_box_prompt(
        self,
        project_id: int,
        video_id: int,
        frame_idx: int,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
    ) -> tuple[np.ndarray, float]:
        """
        Add a bounding box prompt and return the mask and confidence score.

        Requires an active session (call init_session first).

        Args:
            project_id: ID of the project
            video_id: ID of the video
            frame_idx: Frame index to segment
            x1, y1: Top-left corner in normalized [0, 1] coords
            x2, y2: Bottom-right corner in normalized [0, 1] coords

        Returns:
            Tuple of (mask, score) where mask is numpy array (height, width)
            and score is the predicted IoU confidence.
        """
        session = self.get_session(project_id, video_id)
        if session is None:
            raise RuntimeError("No session exists. Initialize session first.")

        result = self._send_and_wait({
            "type": "add_box_prompt",
            "video_id": video_id,
            "frame_idx": frame_idx,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
        }, timeout=120.0)

        if result.get("status") != "ok":
            raise RuntimeError(result.get("error", "Failed to add box prompt"))

        session.has_object = True

        mask_rle = result["mask_rle"]
        mask_shape = tuple(result["mask_shape"])
        mask_dtype = result.get("mask_dtype", "uint8")
        mask = _decode_mask_rle(mask_rle, mask_shape, mask_dtype)
        score = result.get("score", -1.0)

        return mask, score
```

- [ ] **Step 2: Add module-level wrapper**

Add after the module-level `add_point_prompt` function (after line 1032), before `refine_mask`:

```python
def add_box_prompt(
    project_id: int,
    video_id: int,
    frame_idx: int,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> tuple[np.ndarray, float]:
    """Add a box prompt and return the mask and confidence score."""
    return SegmentationService.get_instance().add_box_prompt(
        project_id, video_id, frame_idx, x1, y1, x2, y2
    )
```

- [ ] **Step 3: Commit**

```bash
git add vidseq/services/segmentation_tcp_client.py
git commit -m "feat: add add_box_prompt() to TCP client"
```

---

### Task 4: Schema + Route + Service

**Files:**
- Modify: `vidseq/schemas/segmentation.py` (after `PromptRequest` ending at line 37)
- Modify: `vidseq/api/routes/segmentation/inference.py` (after `submit_prompt` ending at line 60)
- Modify: `vidseq/services/segmentation_service.py` (after `submit_prompt` ending at line 310)

- [ ] **Step 1: Add BoxPromptRequest schema**

In `vidseq/schemas/segmentation.py`, add after `PromptRequest` class (after line 37), before `PropagateRequest`:

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

- [ ] **Step 2: Add service function**

In `vidseq/services/segmentation_service.py`, add after the `submit_prompt` function (after line 310), before the `propagate` function:

```python
async def submit_box_prompt(
    session: "AsyncSession",
    project_id: int,
    video_id: int,
    frame_idx: int,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> np.ndarray:
    """Submit a bounding box prompt for segmentation.

    Box is always an initial prompt — creates a new mask. Any existing
    conditioning state for this frame is cleared by the segmentor.

    Args:
        session: Async database session
        project_id: ID of the project
        video_id: ID of the video
        frame_idx: Frame index (0-based)
        x1, y1: Top-left corner in normalized [0, 1] coords
        x2, y2: Bottom-right corner in normalized [0, 1] coords

    Returns:
        Resulting mask as numpy array
    """
    mask, score = segmentation_tcp_client.add_box_prompt(
        project_id=project_id,
        video_id=video_id,
        frame_idx=frame_idx,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
    )

    # Ensure conditioning frame DB record exists (upsert pattern)
    from sqlalchemy import select
    from vidseq.models.conditioning_frame import ConditioningFrame

    existing = await session.execute(
        select(ConditioningFrame)
        .where(ConditioningFrame.video_id == video_id)
        .where(ConditioningFrame.frame_idx == frame_idx)
    )
    if existing.scalar_one_or_none() is None:
        session.add(ConditioningFrame(video_id=video_id, frame_idx=frame_idx))
        await session.commit()

    # Update mask presence index and save confidence score
    has_content = bool(np.any(mask > 0))
    await frame_data_service.set_has_tracker_mask(
        session, video_id, frame_idx, has_content
    )
    await frame_data_service.save_score(session, video_id, frame_idx, score)

    return mask
```

- [ ] **Step 3: Add route**

In `vidseq/api/routes/segmentation/inference.py`, add the `BoxPromptRequest` import. Find:

```python
from vidseq.schemas.segmentation import (
    PromptRequest,
    PropagateRequest,
    PropagateResponse,
)
```

Replace with:

```python
from vidseq.schemas.segmentation import (
    BoxPromptRequest,
    PromptRequest,
    PropagateRequest,
    PropagateResponse,
)
```

Then add the route after the `submit_prompt` route (after line 60), before the `propagate` route:

```python
@router.post("/projects/{project_id}/videos/{video_id}/box-prompt/{frame_idx}")
async def submit_box_prompt(
    project_id: int,
    frame_idx: int,
    request: BoxPromptRequest,
    video: Video = Depends(get_video),
    session: AsyncSession = Depends(get_project_session),
):
    """Submit a bounding box prompt to create an initial segmentation mask.

    Box coords should be normalized [0,1].
    """
    try:
        mask = await segmentation_service.submit_box_prompt(
            session=session,
            project_id=project_id,
            video_id=video.id,
            frame_idx=frame_idx,
            x1=request.x1,
            y1=request.y1,
            x2=request.x2,
            y2=request.y2,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    mask_png = segmentation_service.mask_to_png(mask)
    return Response(content=mask_png, media_type="image/png")
```

- [ ] **Step 4: Commit**

```bash
git add vidseq/schemas/segmentation.py vidseq/services/segmentation_service.py vidseq/api/routes/segmentation/inference.py
git commit -m "feat: add box prompt schema, service, and route"
```

---

### Task 5: Frontend API Function

**Files:**
- Modify: `frontend/src/services/api.ts` (after `submitPrompt` ending at line 253)

- [ ] **Step 1: Add submitBoxPrompt function**

Add after `submitPrompt` (after line 253), before `getTrackerMask`:

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
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to submit box prompt'))
    }
    return response.blob()
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/services/api.ts
git commit -m "feat: add submitBoxPrompt API function"
```

---

### Task 6: Frontend Composable — useSegmentation.ts

**Files:**
- Modify: `frontend/src/composables/useSegmentation.ts`

- [ ] **Step 1: Add import for submitBoxPrompt**

At line 6, add `submitBoxPrompt` to the import. Find:

```typescript
import {
    getTrackerMask,
    getTrackerMasks,
    getDetectorMasks,
    submitPrompt,
```

Replace with:

```typescript
import {
    getTrackerMask,
    getTrackerMasks,
    getDetectorMasks,
    submitPrompt,
    submitBoxPrompt,
```

- [ ] **Step 2: Update ToolType and add LocalPrompt union**

Replace line 21:

```typescript
export type ToolType = 'none' | 'positive_point' | 'negative_point'
```

With:

```typescript
export type ToolType = 'none' | 'positive_point' | 'negative_point' | 'bounding_box'

export type LocalPrompt =
    | { x: number; y: number; type: 'positive_point' | 'negative_point' }
    | { x1: number; y1: number; x2: number; y2: number; type: 'bounding_box' }
```

- [ ] **Step 3: Update UseSegmentationReturn interface**

Add to the interface (after line 33 `handlePointComplete`):

```typescript
    handleBoxComplete: (box: { x1: number; y1: number; x2: number; y2: number }) => Promise<void>
    toggleBoundingBoxTool: () => void
```

- [ ] **Step 4: Update localPrompts type**

Replace line 64:

```typescript
    const localPrompts = ref<Map<number, Array<{ x: number; y: number; type: 'positive_point' | 'negative_point' }>>>(new Map())
```

With:

```typescript
    const localPrompts = ref<Map<number, Array<LocalPrompt>>>(new Map())
```

- [ ] **Step 5: Update currentPrompts computed**

Replace lines 66-68:

```typescript
    const currentPrompts = computed(() => {
        return localPrompts.value.get(currentFrameIdx.value) || []
    })
```

With:

```typescript
    const currentPrompts = computed((): LocalPrompt[] => {
        return localPrompts.value.get(currentFrameIdx.value) || []
    })
```

- [ ] **Step 6: Add toggleBoundingBoxTool**

Add after `toggleNegativePointTool` (after line 243):

```typescript
    const toggleBoundingBoxTool = () => {
        activeTool.value = activeTool.value === 'bounding_box' ? 'none' : 'bounding_box'
    }
```

- [ ] **Step 7: Add handleBoxComplete handler**

Add after `handlePointComplete` (after line 281):

```typescript
    const handleBoxComplete = async (box: { x1: number; y1: number; x2: number; y2: number }) => {
        if (!projectId.value || !videoId.value) return

        isSegmenting.value = true

        // Clear existing prompts — box always starts fresh
        localPrompts.value.delete(currentFrameIdx.value)

        // Add box to local prompts for rendering
        const framePrompts: LocalPrompt[] = [{ ...box, type: 'bounding_box' as const }]
        localPrompts.value.set(currentFrameIdx.value, framePrompts)

        try {
            const maskBlob = await submitBoxPrompt(
                projectId.value,
                videoId.value,
                currentFrameIdx.value,
                box
            )

            const bitmap = await createImageBitmap(maskBlob)
            currentMask.value = bitmap
            maskCache.set(currentFrameIdx.value, bitmap)
        } catch (e) {
            console.error('Failed to submit box prompt:', e)
            // Rollback: remove the box we just added
            localPrompts.value.delete(currentFrameIdx.value)
        } finally {
            isSegmenting.value = false
        }
    }
```

- [ ] **Step 8: Update return object**

Add `toggleBoundingBoxTool` and `handleBoxComplete` to the return statement. Find:

```typescript
    return {
        activeTool,
        currentMask,
        detectorBbox,
        currentPrompts,
        isSegmenting,
        loadFrameData,
        seekToFrame,
        togglePositivePointTool,
        toggleNegativePointTool,
        handlePointComplete,
        handleResetFrame,
        handleResetVideo,
        clearMaskCache,
    }
```

Replace with:

```typescript
    return {
        activeTool,
        currentMask,
        detectorBbox,
        currentPrompts,
        isSegmenting,
        loadFrameData,
        seekToFrame,
        togglePositivePointTool,
        toggleNegativePointTool,
        toggleBoundingBoxTool,
        handlePointComplete,
        handleBoxComplete,
        handleResetFrame,
        handleResetVideo,
        clearMaskCache,
    }
```

- [ ] **Step 9: Commit**

```bash
git add frontend/src/composables/useSegmentation.ts
git commit -m "feat: add bounding box tool type and handleBoxComplete to composable"
```

---

### Task 7: Frontend VideoOverlay — Box Drawing and Rendering

**Files:**
- Modify: `frontend/src/components/VideoOverlay.vue`

- [ ] **Step 1: Update types and props**

Replace the `LocalPrompt` interface and props (lines 1-22):

```typescript
<script setup lang="ts">
import { ref, watch, onMounted } from 'vue'
import type { ToolType, LocalPrompt } from '@/composables/useSegmentation'

const props = defineProps<{
  videoWidth: number
  videoHeight: number
  activeTool: ToolType
  mask: ImageBitmap | null
  prompts: LocalPrompt[]
  showMask?: boolean
  showPrompts?: boolean
  detectorBbox?: { x1: number; y1: number; x2: number; y2: number } | null
  obbBbox?: { corners: [number, number][] } | null
}>()
```

- [ ] **Step 2: Update emits**

Replace the emits declaration (lines 24-26):

```typescript
const emit = defineEmits<{
  (e: 'point-complete', point: { x: number; y: number; type: 'positive_point' | 'negative_point' }): void
  (e: 'box-complete', box: { x1: number; y1: number; x2: number; y2: number }): void
}>()
```

- [ ] **Step 3: Add box drag state**

After `pendingPoint` ref (after line 29), add:

```typescript
const pendingBox = ref<{ x1: number; y1: number; x2: number; y2: number } | null>(null)
const isDragging = ref(false)
const dragStart = ref<{ x: number; y: number } | null>(null)
```

- [ ] **Step 4: Update mouse handlers**

Replace `onMouseDown` (lines 46-55) with three handlers:

```typescript
function onMouseDown(event: MouseEvent) {
  const coords = getNormalizedCoords(event)
  if (!coords) return

  if (props.activeTool === 'positive_point' || props.activeTool === 'negative_point') {
    pendingPoint.value = { x: coords.x, y: coords.y, type: props.activeTool }
    render()
    emit('point-complete', { x: coords.x, y: coords.y, type: props.activeTool })
  } else if (props.activeTool === 'bounding_box') {
    isDragging.value = true
    dragStart.value = coords
    pendingBox.value = { x1: coords.x, y1: coords.y, x2: coords.x, y2: coords.y }
    render()
  }
}

function onMouseMove(event: MouseEvent) {
  if (!isDragging.value || !dragStart.value) return

  const coords = getNormalizedCoords(event)
  if (!coords) return

  pendingBox.value = {
    x1: Math.min(dragStart.value.x, coords.x),
    y1: Math.min(dragStart.value.y, coords.y),
    x2: Math.max(dragStart.value.x, coords.x),
    y2: Math.max(dragStart.value.y, coords.y),
  }
  render()
}

function onMouseUp(_event: MouseEvent) {
  if (!isDragging.value || !pendingBox.value) return

  isDragging.value = false
  dragStart.value = null

  const box = pendingBox.value
  // Only emit if box has meaningful size (not a click)
  const minSize = 0.01
  if (Math.abs(box.x2 - box.x1) > minSize && Math.abs(box.y2 - box.y1) > minSize) {
    emit('box-complete', { x1: box.x1, y1: box.y1, x2: box.x2, y2: box.y2 })
  } else {
    pendingBox.value = null
    render()
  }
}
```

- [ ] **Step 5: Update render function — prompt drawing**

Replace the prompt drawing section (lines 103-163) with code that handles both point and box prompts:

```typescript
  // Draw prompts (points and boxes)
  if (props.showPrompts === false) return
  for (const prompt of props.prompts) {
    if (prompt.type === 'bounding_box') {
      // Draw completed box prompt
      const bx1 = prompt.x1 * canvas.width
      const by1 = prompt.y1 * canvas.height
      const bx2 = prompt.x2 * canvas.width
      const by2 = prompt.y2 * canvas.height
      ctx.strokeStyle = '#3b82f6'
      ctx.lineWidth = 3
      ctx.setLineDash([])
      ctx.strokeRect(bx1, by1, bx2 - bx1, by2 - by1)
    } else {
      // Draw point prompt (existing logic)
      const px = prompt.x * canvas.width
      const py = prompt.y * canvas.height
      const radius = 8

      ctx.beginPath()
      ctx.arc(px, py, radius, 0, Math.PI * 2)
      ctx.fillStyle = prompt.type === 'positive_point' ? '#22c55e' : '#ef4444'
      ctx.fill()
      ctx.strokeStyle = '#fff'
      ctx.lineWidth = 2
      ctx.setLineDash([])
      ctx.stroke()

      ctx.strokeStyle = '#fff'
      ctx.lineWidth = 2
      ctx.beginPath()
      ctx.moveTo(px - 4, py)
      ctx.lineTo(px + 4, py)
      if (prompt.type === 'positive_point') {
        ctx.moveTo(px, py - 4)
        ctx.lineTo(px, py + 4)
      }
      ctx.stroke()
    }
  }

  // Draw pending point
  if (pendingPoint.value) {
    const px = pendingPoint.value.x * canvas.width
    const py = pendingPoint.value.y * canvas.height
    const radius = 8

    ctx.beginPath()
    ctx.arc(px, py, radius + 4, 0, Math.PI * 2)
    ctx.strokeStyle = pendingPoint.value.type === 'positive_point' ? '#22c55e' : '#ef4444'
    ctx.lineWidth = 2
    ctx.setLineDash([4, 4])
    ctx.stroke()

    ctx.beginPath()
    ctx.arc(px, py, radius, 0, Math.PI * 2)
    ctx.fillStyle = pendingPoint.value.type === 'positive_point' ? 'rgba(34, 197, 94, 0.5)' : 'rgba(239, 68, 68, 0.5)'
    ctx.fill()
    ctx.strokeStyle = '#fff'
    ctx.lineWidth = 2
    ctx.setLineDash([])
    ctx.stroke()

    ctx.strokeStyle = '#fff'
    ctx.lineWidth = 2
    ctx.beginPath()
    ctx.moveTo(px - 4, py)
    ctx.lineTo(px + 4, py)
    if (pendingPoint.value.type === 'positive_point') {
      ctx.moveTo(px, py - 4)
      ctx.lineTo(px, py + 4)
    }
    ctx.stroke()
  }

  // Draw pending box (during drag)
  if (pendingBox.value) {
    const bx1 = pendingBox.value.x1 * canvas.width
    const by1 = pendingBox.value.y1 * canvas.height
    const bx2 = pendingBox.value.x2 * canvas.width
    const by2 = pendingBox.value.y2 * canvas.height
    ctx.strokeStyle = '#3b82f6'
    ctx.lineWidth = 2
    ctx.setLineDash([4, 4])
    ctx.strokeRect(bx1, by1, bx2 - bx1, by2 - by1)
    ctx.setLineDash([])
  }
```

- [ ] **Step 6: Update watchers**

Replace the watch on line 166 to include `pendingBox`:

```typescript
watch(() => [props.mask, props.prompts, props.detectorBbox, props.obbBbox, props.showMask, props.showPrompts], () => {
  pendingPoint.value = null
  pendingBox.value = null
  render()
}, { deep: true })
```

Also update the `activeTool` watcher (line 179) to clear box state:

```typescript
watch(() => props.activeTool, () => {
  pendingPoint.value = null
  pendingBox.value = null
  isDragging.value = false
  dragStart.value = null
  render()
})
```

- [ ] **Step 7: Update template**

Replace the template (lines 192-199):

```html
<template>
  <canvas
    ref="canvasRef"
    class="video-overlay"
    :class="{ 'tool-active': activeTool !== 'none' }"
    @mousedown="onMouseDown"
    @mousemove="onMouseMove"
    @mouseup="onMouseUp"
  />
</template>
```

- [ ] **Step 8: Commit**

```bash
git add frontend/src/components/VideoOverlay.vue
git commit -m "feat: add box drawing interaction and rendering to VideoOverlay"
```

---

### Task 8: Frontend VideoDetail.vue — Toolbar and Wiring

**Files:**
- Modify: `frontend/src/components/VideoDetail.vue`

- [ ] **Step 1: Destructure new composable exports**

Find the composable destructure (around line 131-143). Add `toggleBoundingBoxTool` and `handleBoxComplete`:

```typescript
const {
  activeTool,
  currentMask,
  detectorBbox,
  currentPrompts,
  isSegmenting,
  loadFrameData,
  seekToFrame,
  togglePositivePointTool,
  toggleNegativePointTool,
  toggleBoundingBoxTool,
  handlePointComplete,
  handleBoxComplete,
  handleResetFrame,
  handleResetVideo,
  clearMaskCache,
} = useSegmentation(projectId, videoId, currentFrameIdx, isPlaying, videoRef, fps, maskViewMode)
```

- [ ] **Step 2: Add handleBoxCompleteWithRefresh**

Find `handlePointCompleteWithRefresh` (it wraps `handlePointComplete` and refreshes the data track). Add a similar wrapper nearby:

```typescript
const handleBoxCompleteWithRefresh = async (box: { x1: number; y1: number; x2: number; y2: number }) => {
  await handleBoxComplete(box)
  await refreshDataTrack()
}
```

- [ ] **Step 3: Add toolbar button**

After the Negative Point button (after line 435), before the Reset Frame button (line 436), add:

```html
          <button
            class="tool-button bounding-box"
            :class="{ active: activeTool === 'bounding_box' }"
            @click="toggleBoundingBoxTool"
            :disabled="isSegmenting || !segmentationIsReady"
          >
            <span class="tool-icon">▢</span>
            <span class="tool-label">Bounding Box</span>
          </button>
```

- [ ] **Step 4: Wire up box-complete event on VideoOverlay**

Find the VideoOverlay component (around line 361-373). Add the `@box-complete` event:

```html
              <VideoOverlay
                v-if="videoWidth > 0 && videoHeight > 0"
                :video-width="videoWidth"
                :video-height="videoHeight"
                :active-tool="activeTool"
                :mask="currentMask"
                :prompts="currentPrompts"
                :show-mask="showMask"
                :show-prompts="showPrompts"
                :detector-bbox="detectorBbox"
                :obb-bbox="obbBbox"
                @point-complete="handlePointCompleteWithRefresh"
                @box-complete="handleBoxCompleteWithRefresh"
              />
```

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/VideoDetail.vue
git commit -m "feat: add bounding box tool button and wire up events"
```

---

### Task 9: Manual Integration Test

- [ ] **Step 1: Start the full stack**

```bash
# Terminal 1: Backend
vidseq

# Terminal 2: Frontend
cd frontend && npm run dev
```

- [ ] **Step 2: Test box prompt flow**

1. Open a project with a video
2. Initialize a SAM session
3. Click "Bounding Box" tool — verify crosshair cursor appears
4. Drag a rectangle on an object — verify dashed blue outline during drag
5. Release mouse — verify solid blue box rendered, mask appears on the object
6. Switch to "Positive Point" tool, click inside the mask — verify mask refines
7. Switch to "Negative Point" tool, click outside — verify mask refines

- [ ] **Step 3: Test multi-frame box prompts**

1. Draw a box on frame 0 — verify mask
2. Scrub to frame 50, draw another box — verify mask (memory from frame 0 should help)
3. Propagate from frame 0 — verify masks propagate and frame 50 is used as anchor

- [ ] **Step 4: Test re-boxing a frame**

1. Draw a box on frame 10 — verify mask
2. Draw a different box on the same frame — verify old prompts cleared, new mask appears

- [ ] **Step 5: Test reset**

1. Draw a box, then click "Reset Frame" — verify box and mask both cleared
