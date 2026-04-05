# Propagate Without Memory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a "Propagate Without Memory" button that propagates using only conditioning frame memories, preventing temporal drift.

**Architecture:** New `propagate_sequential_cond_only()` segmentor method, threaded through TCP command/client/service/route layers, with a new frontend button alongside existing propagation.

**Tech Stack:** Python (FastAPI, SAM2, HDF5), Vue 3/TypeScript

**Spec:** `docs/superpowers/specs/2026-04-05-propagate-without-memory-design.md`

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| Modify | `vidseq/services/segmentation_model/streaming_segmentor.py` | New `propagate_sequential_cond_only()` |
| Modify | `vidseq/services/segmentation_commands.py` | New `handle_propagate_without_memory()` |
| Modify | `vidseq/services/segmentation_tcp_server.py` | Register command |
| Modify | `vidseq/services/segmentation_tcp_client.py` | New client method + wrapper |
| Modify | `vidseq/services/segmentation_service.py` | New service function |
| Modify | `vidseq/api/routes/segmentation/inference.py` | New route |
| Modify | `frontend/src/services/api.ts` | New API function |
| Modify | `frontend/src/components/VideoDetail.vue` | New button + handler |

---

### Task 1: Streaming Segmentor — propagate_sequential_cond_only()

**Files:**
- Modify: `vidseq/services/segmentation_model/streaming_segmentor.py` (after `propagate_sequential` ending at line 1159)

- [ ] **Step 1: Add method**

Add after `propagate_sequential` (after line 1159), before `_compute_bbox_iou`:

```python
    def propagate_sequential_cond_only(
        self,
        video_id: str,
        start_frame: int,
        num_frames: int,
        frames,
        masks,
        on_result: Callable | None = None,
        progress_interval: int = 50,
    ) -> list[int]:
        """Propagate using only conditioning frame memories (no temporal window).

        Like propagate_sequential(), but clears non_cond_frame_outputs before
        each frame. This prevents temporal drift where propagated frames
        reinforce segmentation errors across the sliding window.

        Each frame is segmented using only the permanent conditioning frame
        memories from user prompts.

        Args:
            video_id: The video identifier.
            start_frame: Frame index to start propagation.
            num_frames: Maximum number of frames to propagate.
            frames: Indexable frame source: frames[idx] -> np.ndarray (H, W, 3).
            masks: Indexable mask source: masks[idx] -> np.ndarray (H, W).
            on_result: Callback called for each frame with (frame_idx, mask, logits, score).
            progress_interval: Print progress every N frames (0 to disable).

        Returns:
            List of frame indices that were propagated.

        Raises:
            KeyError: If video_id is not open.
            RuntimeError: If no conditioning frames exist.
        """
        session = self.sessions[video_id]
        cond_frame_indices = session["cond_frame_indices"]
        frame_dims = session["frame_dims"]
        output_dict = session["output_dict"]

        if len(output_dict["cond_frame_outputs"]) == 0:
            raise RuntimeError(
                "No conditioning frames exist. "
                "Use add_point_prompt() or add_box_prompt() first."
            )

        orig_h, orig_w = frame_dims
        propagated = []

        for i in range(num_frames):
            frame_idx = start_frame + i

            # Skip conditioning frames (don't overwrite user prompts)
            if frame_idx in cond_frame_indices:
                continue

            try:
                frame_bgr = frames[frame_idx]
            except (IndexError, KeyError):
                break

            # Clear non-cond memory so track_step only sees conditioning frames
            output_dict["non_cond_frame_outputs"].clear()

            # Get image features
            _, backbone_out = self._get_image_features(video_id, frame_idx, frame_bgr)
            current_vision_feats, current_vision_pos_embeds, feat_sizes = (
                self._prepare_backbone_features(backbone_out)
            )

            # track_step with only conditioning memory
            with torch.inference_mode(), torch.autocast("cuda", torch.bfloat16):
                current_out = self.predictor.track_step(
                    frame_idx=frame_idx,
                    is_init_cond_frame=False,
                    current_vision_feats=current_vision_feats,
                    current_vision_pos_embeds=current_vision_pos_embeds,
                    feat_sizes=feat_sizes,
                    point_inputs=None,
                    mask_inputs=None,
                    output_dict=output_dict,
                    num_frames=session["num_frames"],
                    run_mem_encoder=True,
                )

            # Extract mask
            pred_mask_high_res = current_out["pred_masks_high_res"][0, 0]
            mask_binary = (pred_mask_high_res > 0).to(torch.uint8).mul(255).cpu().numpy()
            mask_resized = cv2.resize(
                mask_binary,
                (orig_w, orig_h),
                interpolation=cv2.INTER_NEAREST,
            )

            pred_masks_low_res = current_out["pred_masks"][0, 0].cpu().numpy()
            score = self._extract_score(current_out)

            if on_result is not None:
                on_result(frame_idx, mask_resized, pred_masks_low_res, score)

            # Store in non_cond (will be cleared next iteration)
            output_dict["non_cond_frame_outputs"][frame_idx] = self._make_compact_output(
                current_out
            )

            propagated.append(frame_idx)

            if progress_interval > 0 and len(propagated) % progress_interval == 0:
                print(f"  Propagated {len(propagated)} frames (cond-only)...")

        return propagated
```

- [ ] **Step 2: Commit**

```bash
git add vidseq/services/segmentation_model/streaming_segmentor.py
git commit -m "feat: add propagate_sequential_cond_only() to streaming segmentor"
```

---

### Task 2: TCP Command Handler + Server Registration

**Files:**
- Modify: `vidseq/services/segmentation_commands.py` (after `handle_generate_training_masks` ending at line 541)
- Modify: `vidseq/services/segmentation_tcp_server.py`

- [ ] **Step 1: Add handler**

Add after `handle_generate_training_masks` (after line 541), before `handle_apply_detector`:

```python
def handle_propagate_without_memory(
    params: dict,
    segmentor: StreamingSegmentor,
    response_callback: Callable | None = None,
) -> dict:
    """Propagate using only conditioning frame memories (no temporal window).

    Args:
        params: Command params with video_id, start_frame_idx, max_frames
        segmentor: StreamingSegmentor instance
        response_callback: Optional callback to send progress messages

    Returns:
        Response dict with frames_processed, frame_indices
    """
    video_id = params["video_id"]
    start_frame_idx = params["start_frame_idx"]
    max_frames = params["max_frames"]

    if segmentor is None:
        raise RuntimeError("Model not loaded")

    if video_id not in _video_resources:
        raise RuntimeError(f"No session for video {video_id}")

    resources = _video_resources[video_id]

    scores: list[list] = []
    frames_propagated = 0

    with tracker_masks(resources.project_path, video_id, "a") as mask_data, \
         tracker_logits(resources.project_path, video_id, "a") as logits_data:
        def on_result(frame_idx: int, mask: np.ndarray, logits: np.ndarray, score: float) -> None:
            nonlocal frames_propagated
            mask_data[frame_idx] = mask
            logits_data[frame_idx] = logits
            scores.append([frame_idx, score])
            frames_propagated += 1
            if response_callback and frames_propagated % 50 == 0:
                response_callback({
                    "type": "progress",
                    "frame_idx": frames_propagated,
                    "total": max_frames,
                })

        frame_indices = segmentor.propagate_sequential_cond_only(
            video_id=str(video_id),
            start_frame=start_frame_idx,
            num_frames=max_frames,
            frames=resources.frame_source,
            masks=mask_data,
            on_result=on_result,
            progress_interval=50,
        )

    return {
        "type": "propagate_without_memory_result",
        "status": "ok",
        "frames_processed": len(frame_indices),
        "frame_indices": frame_indices,
        "scores": scores,
    }
```

- [ ] **Step 2: Register in TCP server**

In `vidseq/services/segmentation_tcp_server.py`, find the `generate_training_masks` elif block. Add after it:

```python
            elif cmd_type == "propagate_without_memory":
                if self._segmentor is None:
                    raise RuntimeError("Model not loaded")
                result = handle_propagate_without_memory(
                    cmd,
                    self._segmentor,
                    response_callback,
                )
```

Also add `handle_propagate_without_memory` to the import from `segmentation_commands`.

- [ ] **Step 3: Commit**

```bash
git add vidseq/services/segmentation_commands.py vidseq/services/segmentation_tcp_server.py
git commit -m "feat: add handle_propagate_without_memory TCP command"
```

---

### Task 3: TCP Client + Service + Route

**Files:**
- Modify: `vidseq/services/segmentation_tcp_client.py` (after `generate_training_masks`)
- Modify: `vidseq/services/segmentation_service.py` (after `propagate`)
- Modify: `vidseq/api/routes/segmentation/inference.py` (after propagate route)

- [ ] **Step 1: Add TCP client class method**

Add after `generate_training_masks` method (ends around line 826), before `apply_detector`:

```python
    def propagate_without_memory(
        self,
        project_id: int,
        video_id: int,
        start_frame_idx: int,
        max_frames: int,
        project_path: Path,
        num_frames: int,
        height: int,
        width: int,
    ) -> tuple[list[int], list[list]]:
        """
        Propagate using only conditioning frame memories (no temporal window).

        Same interface as generate_training_masks but uses cond-only propagation.
        """
        session = self.get_session(project_id, video_id)
        if session is None:
            raise RuntimeError("No session exists. Add a point prompt first.")

        if not session.has_object:
            raise RuntimeError("No object tracked. Add a point prompt first.")

        result = self._send_streaming({
            "type": "propagate_without_memory",
            "video_id": video_id,
            "start_frame_idx": start_frame_idx,
            "max_frames": max_frames,
            "project_path": str(project_path),
            "num_frames": num_frames,
            "height": height,
            "width": width,
        }, timeout=600.0)

        if result.get("status") != "ok":
            raise RuntimeError(result.get("error", "Failed to propagate without memory"))

        frame_indices = result.get("frame_indices", [])
        scores = result.get("scores", [])
        return frame_indices, scores
```

- [ ] **Step 2: Add module-level wrapper**

After the module-level `generate_training_masks` wrapper (find it by searching), add:

```python
def propagate_without_memory(
    project_id: int,
    video_id: int,
    start_frame_idx: int,
    max_frames: int,
    project_path: Path,
    num_frames: int,
    height: int,
    width: int,
) -> tuple[list[int], list[list]]:
    """Propagate using only conditioning frame memories."""
    return SegmentationService.get_instance().propagate_without_memory(
        project_id, video_id, start_frame_idx, max_frames,
        project_path, num_frames, height, width,
    )
```

- [ ] **Step 3: Add service function**

In `vidseq/services/segmentation_service.py`, add after `propagate` (ends at line 421), before `apply_detector`:

```python
async def propagate_without_memory(
    session: "AsyncSession",
    project_id: int,
    video_id: int,
    project_path: Path,
    start_frame_idx: int,
    max_frames: int,
    num_frames: int,
    height: int,
    width: int,
) -> int:
    """Propagate using only conditioning frame memories (no temporal window).

    Same as propagate() but uses cond-only mode to prevent drift.
    """
    frame_indices, scores = segmentation_tcp_client.propagate_without_memory(
        project_id=project_id,
        video_id=video_id,
        start_frame_idx=start_frame_idx,
        max_frames=max_frames,
        project_path=project_path,
        num_frames=num_frames,
        height=height,
        width=width,
    )

    await frame_data_service.set_has_tracker_mask(session, video_id, frame_indices, True)
    await frame_data_service.save_scores_batch(session, video_id, scores)

    return len(frame_indices)
```

- [ ] **Step 4: Add route**

In `vidseq/api/routes/segmentation/inference.py`, add after the existing `propagate` route (ends around line 100):

```python
@router.post(
    "/projects/{project_id}/videos/{video_id}/propagation-without-memory",
    response_model=PropagateResponse,
)
async def propagate_without_memory(
    project_id: int,
    request: PropagateRequest,
    video: Video = Depends(get_video),
    session: AsyncSession = Depends(get_project_session),
    project_path: Path = Depends(get_project_folder),
):
    """Propagate using only conditioning frame memories (no temporal window).

    Prevents drift by ignoring the sliding window of recent propagated frames.
    Each frame is segmented independently using only user-prompted conditioning frames.
    """
    try:
        frames_processed = await segmentation_service.propagate_without_memory(
            session=session,
            project_id=project_id,
            video_id=video.id,
            project_path=project_path,
            start_frame_idx=request.start_frame_idx,
            max_frames=request.max_frames,
            num_frames=video.num_frames,
            height=video.height,
            width=video.width,
        )
    except RuntimeError as e:
        logger.error(f"Propagation without memory failed: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in propagation without memory: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    return PropagateResponse(frames_processed=frames_processed)
```

- [ ] **Step 5: Commit**

```bash
git add vidseq/services/segmentation_tcp_client.py vidseq/services/segmentation_service.py vidseq/api/routes/segmentation/inference.py
git commit -m "feat: add propagate-without-memory TCP client, service, and route"
```

---

### Task 4: Frontend API + Button

**Files:**
- Modify: `frontend/src/services/api.ts` (after `createPropagation`)
- Modify: `frontend/src/components/VideoDetail.vue`

- [ ] **Step 1: Add API function**

In `frontend/src/services/api.ts`, add after `createPropagation` (find it — it POSTs to `/propagation`):

```typescript
export async function createPropagationWithoutMemory(
    projectId: number,
    videoId: number,
    startFrameIdx: number,
    maxFrames: number = 1000
): Promise<PropagateResponse> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/${videoId}/propagation-without-memory`,
        {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ start_frame_idx: startFrameIdx, max_frames: maxFrames }),
        }
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to propagate without memory'))
    }
    return response.json()
}
```

- [ ] **Step 2: Add import in VideoDetail.vue**

In VideoDetail.vue, find the import from `@/services/api` and add `createPropagationWithoutMemory` to it.

- [ ] **Step 3: Add handler in VideoDetail.vue**

After `handlePropagateMask` (find it — it calls `createPropagation`), add:

```typescript
const handlePropagateWithoutMemory = async () => {
  if (!projectId.value || !videoId.value) return

  isPropagating.value = true
  try {
    await createPropagationWithoutMemory(
      projectId.value,
      videoId.value,
      currentFrameIdx.value,
      maxFrames.value
    )
    clearMaskCache()
    await loadFrameData(currentFrameIdx.value)
    await refreshFrameRanges()
    await fetchDetectorScoresForView()
    await fetchObbScoresForView()
  } catch (e) {
    console.error('Failed to propagate without memory:', e)
    alert(e instanceof Error ? e.message : 'Failed to propagate without memory')
  } finally {
    isPropagating.value = false
  }
}
```

- [ ] **Step 4: Add button**

In the template, find the existing Propagate Mask button (around line 484-491). Add after it, within the same `propagate-section` div:

```html
          <button
            class="tool-button propagate-button"
            @click="handlePropagateWithoutMemory"
            :disabled="isSegmenting || isPropagating || !segmentationIsReady"
          >
            <span class="tool-icon">▶▷</span>
            <span class="tool-label">{{ isPropagating ? 'Propagating...' : 'Propagate Without Memory' }}</span>
          </button>
```

- [ ] **Step 5: Commit**

```bash
git add frontend/src/services/api.ts frontend/src/components/VideoDetail.vue
git commit -m "feat: add Propagate Without Memory button and API"
```
