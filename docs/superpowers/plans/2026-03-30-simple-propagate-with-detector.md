# Simple Propagate With Detector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a simplified batch segmentation mode that re-prompts SAM2 with a detector bbox every fps frames, without drift detection or backtracking.

**Architecture:** New `simple_propagate_with_detector()` method on StreamingSegmentor with a straightforward forward loop. New TCP command handler. Parameterize existing `segment_all_videos` with a `mode` parameter rather than duplicating the orchestration stack.

**Tech Stack:** FastAPI, SAM2, Ultralytics YOLO, TCP streaming, Vue 3

**Spec:** `docs/superpowers/specs/2026-03-30-simple-propagate-with-detector-design.md`

---

### Task 1: Streaming Segmentor Method

**Files:**
- Modify: `vidseq/services/segmentation_model/streaming_segmentor.py`

- [ ] **Step 1: Add `simple_propagate_with_detector` method**

Add after `propagate_with_detector` (after line ~1766). This is a simplified forward-only loop:

```python
    def simple_propagate_with_detector(
        self,
        video_id: str,
        num_frames: int,
        frames,
        get_detector_bbox,
        tracker_masks,
        final_masks,
        on_progress=None,
        reprompt_interval: int = 30,
        scores: dict | None = None,
        detector_bboxes: dict | None = None,
        detector_scores: dict | None = None,
    ) -> None:
        """Simple batch segmentation: propagate forward, re-prompt with detector every N frames.

        No drift detection, no search mode, no backtracking. Just forward propagation
        with periodic detector re-prompting to keep the tracker on target.

        Args:
            video_id: Video session ID (string).
            num_frames: Total frames in video.
            frames: Indexable frame source, frames[idx] -> np.ndarray BGR.
            get_detector_bbox: Callable(frame_idx, frame) -> (bbox|None, confidence).
            tracker_masks: MutableSequence to write tracker masks.
            final_masks: MutableSequence to write final masks.
            on_progress: Optional callback(frame_idx) for progress reporting.
            reprompt_interval: Re-prompt with detector every N frames.
            scores: Optional dict to collect {frame_idx: sam2_score}.
            detector_bboxes: Optional dict to collect {frame_idx: bbox}.
            detector_scores: Optional dict to collect {frame_idx: confidence}.
        """
        import cv2

        session = self.sessions.get(video_id)
        if session is None:
            raise RuntimeError(f"No session for video {video_id}")

        orig_h, orig_w = session["orig_h"], session["orig_w"]
        cond_frame_indices = set(session.get("cond_frame_indices", []))
        empty_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)

        if scores is None:
            scores = {}
        if detector_bboxes is None:
            detector_bboxes = {}
        if detector_scores is None:
            detector_scores = {}

        # Bootstrap: scan for first detector bbox
        start_frame = None
        for frame_idx in range(num_frames):
            if frame_idx in cond_frame_indices:
                continue
            frame = frames[frame_idx]
            det_bbox, det_conf = get_detector_bbox(frame_idx, frame)
            if det_bbox is not None:
                # Initialize with box prompt
                mask, logits, score = self._propagate_single_frame(
                    video_id, frame_idx, frame, box_prompt=det_bbox
                )
                mask_resized = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                tracker_masks[frame_idx] = mask_resized
                final_masks[frame_idx] = mask_resized
                scores[frame_idx] = score
                detector_bboxes[frame_idx] = det_bbox
                detector_scores[frame_idx] = det_conf
                start_frame = frame_idx
                # Write empty masks for skipped frames
                for i in range(frame_idx):
                    if i not in cond_frame_indices:
                        tracker_masks[i] = empty_mask
                        final_masks[i] = empty_mask
                break
            else:
                tracker_masks[frame_idx] = empty_mask
                final_masks[frame_idx] = empty_mask

        if start_frame is None:
            # No detection found in any frame
            print(f"[Simple Propagate] No object detected in video {video_id}")
            return

        if on_progress:
            on_progress(start_frame)

        # Track conditioning frames added by reprompting for eviction
        reprompt_cond_frames: list[int] = [start_frame]

        # Forward propagation
        for frame_idx in range(start_frame + 1, num_frames):
            if frame_idx in cond_frame_indices:
                # User conditioning frame, already in memory
                continue

            frame = frames[frame_idx]
            is_reprompt = (frame_idx - start_frame) % reprompt_interval == 0

            if is_reprompt:
                det_bbox, det_conf = get_detector_bbox(frame_idx, frame)
                if det_bbox is not None:
                    # Re-prompt with detector bbox
                    mask, logits, score = self._propagate_single_frame(
                        video_id, frame_idx, frame, box_prompt=det_bbox
                    )
                    detector_bboxes[frame_idx] = det_bbox
                    detector_scores[frame_idx] = det_conf

                    # Evict oldest reprompt conditioning frame if over limit
                    reprompt_cond_frames.append(frame_idx)
                    while len(reprompt_cond_frames) > self.MEM_WINDOW:
                        old_idx = reprompt_cond_frames.pop(0)
                        self.evict_conditioning_frame(video_id, old_idx)
                else:
                    # No detection, normal tracking
                    mask, logits, score = self._propagate_single_frame(
                        video_id, frame_idx, frame
                    )
            else:
                # Normal tracking
                mask, logits, score = self._propagate_single_frame(
                    video_id, frame_idx, frame
                )

            mask_resized = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            tracker_masks[frame_idx] = mask_resized
            final_masks[frame_idx] = mask_resized
            scores[frame_idx] = score

            if on_progress and (frame_idx % 50 == 0 or frame_idx == num_frames - 1):
                on_progress(frame_idx)
```

- [ ] **Step 2: Verify `evict_conditioning_frame` exists**

Search for `evict_conditioning_frame` in the file. If it doesn't exist, check for a similar method name. The existing `propagate_with_associated` uses it — find and verify the method signature.

- [ ] **Step 3: Check `_propagate_single_frame` return type**

Read `_propagate_single_frame` to confirm it returns `(mask, logits, score)` where mask is already in the right format or needs `cv2.resize`. The existing `propagate_with_detector` does resize — check if `_propagate_single_frame` returns masks in input space (1024x1024) or original resolution. Adjust the resize call accordingly.

- [ ] **Step 4: Commit**

```bash
git add vidseq/services/segmentation_model/streaming_segmentor.py
git commit -m "feat: add simple_propagate_with_detector method to StreamingSegmentor"
```

---

### Task 2: TCP Command Handler + Server Routing

**Files:**
- Modify: `vidseq/services/segmentation_commands.py`
- Modify: `vidseq/services/segmentation_tcp_server.py`

- [ ] **Step 1: Add `handle_simple_propagate_with_detector` to commands**

Add after `handle_propagate_with_detector`. Pattern matches the existing handler but simpler (no `iou_threshold`, no `training_frame_indices`):

```python
def handle_simple_propagate_with_detector(
    params: dict,
    segmentor: StreamingSegmentor,
    response_callback: Callable[[dict], None],
) -> dict:
    """Simple propagate: forward SAM2 with periodic detector re-prompting."""
    import torch
    from vidseq.services.detector_model import load_finetuned, detect, pick_best_detection

    video_id = params["video_id"]
    num_frames = params["num_frames"]
    project_path = Path(params["project_path"])
    reprompt_interval = params.get("reprompt_interval", 30)

    if segmentor is None:
        raise RuntimeError("Model not loaded")
    if video_id not in _video_resources:
        raise RuntimeError(f"No session for video {video_id}")

    resources = _video_resources[video_id]

    model_path = project_path / "models" / "detector.pt"
    if not model_path.exists():
        raise RuntimeError("No trained detector model found. Train first.")

    detector = None
    try:
        detector = load_finetuned(model_path, device="cuda")

        detector_scores: dict[int, float] = {}
        detector_bboxes: dict[int, tuple[float, float, float, float]] = {}

        def get_detector_bbox(
            frame_idx: int, frame: np.ndarray,
        ) -> tuple[tuple[float, float, float, float] | None, float]:
            detections = detect(detector, frame)
            bbox, conf = pick_best_detection(detections, None)
            return bbox, conf

        def on_progress(frame_idx: int) -> None:
            if frame_idx % 50 == 0 or frame_idx == num_frames - 1:
                response_callback({
                    "type": "progress",
                    "frame_idx": frame_idx,
                    "total": num_frames,
                })

        with tracker_masks(project_path, video_id, "a") as trk_mask_data, \
             final_masks(project_path, video_id, "a") as fin_mask_data:

            scores: dict[int, float] = {}
            segmentor.simple_propagate_with_detector(
                video_id=str(video_id),
                num_frames=num_frames,
                frames=resources.frame_source,
                get_detector_bbox=get_detector_bbox,
                tracker_masks=trk_mask_data,
                final_masks=fin_mask_data,
                on_progress=on_progress,
                reprompt_interval=reprompt_interval,
                scores=scores,
                detector_bboxes=detector_bboxes,
                detector_scores=detector_scores,
            )

        return {
            "type": "simple_propagate_with_detector_result",
            "status": "ok",
            "scores": [[idx, s] for idx, s in scores.items()],
            "detector_scores": [[idx, s] for idx, s in detector_scores.items()],
            "detector_bboxes": [
                [idx, *bbox] for idx, bbox in detector_bboxes.items()
            ],
        }

    finally:
        if detector is not None:
            del detector
            torch.cuda.empty_cache()
```

- [ ] **Step 2: Add TCP server routing**

In `segmentation_tcp_server.py`, add after the `propagate_with_detector` case in `_handle_command`:

```python
            elif cmd_type == "simple_propagate_with_detector":
                if self._segmentor is None:
                    raise RuntimeError("Model not loaded")
                result = handle_simple_propagate_with_detector(
                    cmd,
                    self._segmentor,
                    response_callback,
                )
```

Add `handle_simple_propagate_with_detector` to the import from `segmentation_commands`.

- [ ] **Step 3: Commit**

```bash
git add vidseq/services/segmentation_commands.py vidseq/services/segmentation_tcp_server.py
git commit -m "feat: add simple_propagate_with_detector TCP command handler and routing"
```

---

### Task 3: Parameterize segment_all_videos

**Files:**
- Modify: `vidseq/services/segmentation_tcp_client.py`
- Modify: `vidseq/services/segmentation_service.py`

- [ ] **Step 1: Add `mode` parameter to TCP client `segment_all_videos`**

Find the `segment_all_videos` method on `SegmentationService` class. Add `mode: str = "full"` parameter. In the per-video loop where `propagate_with_detector` command is sent, branch on mode:

```python
            if mode == "simple":
                result = self._send_streaming({
                    "type": "simple_propagate_with_detector",
                    "video_id": video.id,
                    "project_path": str(project_path),
                    "num_frames": video.num_frames,
                    "reprompt_interval": int(round(video.fps)),
                }, timeout=3600.0)
            else:
                result = self._send_streaming({
                    "type": "propagate_with_detector",
                    "video_id": video.id,
                    "project_path": str(project_path),
                    "num_frames": video.num_frames,
                    "iou_threshold": 0.7,
                    "training_frame_indices": (training_frames_by_video or {}).get(video.id, []),
                }, timeout=3600.0)
```

- [ ] **Step 2: Add `mode` parameter to module-level wrapper**

Find the module-level `segment_all_videos` wrapper function. Add `mode: str = "full"` and pass through.

- [ ] **Step 3: Add `mode` parameter to service layer**

Find `segment_all_videos` in `segmentation_service.py`. Add `mode: str = "full"` parameter. Pass through to the TCP client call. When `mode == "simple"`, skip the `training_frames_by_video` query (not needed).

- [ ] **Step 4: Commit**

```bash
git add vidseq/services/segmentation_tcp_client.py vidseq/services/segmentation_service.py
git commit -m "feat: parameterize segment_all_videos with mode for simple propagation"
```

---

### Task 4: API Endpoint + Frontend

**Files:**
- Modify: `vidseq/api/routes/segmentation/sessions.py`
- Modify: `frontend/src/services/api.ts`
- Modify: `frontend/src/components/VideoPipeline.vue`

- [ ] **Step 1: Add API endpoint**

In `sessions.py`, add after the existing `create_videos_segmentation`:

```python
@router.post("/projects/{project_id}/videos/segmentation/simple")
async def create_simple_segmentation(
    project_id: int,
    request: VideoSelectionRequest,
    session: AsyncSession = Depends(get_project_session),
    project_path: Path = Depends(get_project_folder),
):
    """Start simple batch segmentation (re-prompt every second, no drift detection)."""
    job_ids = await segmentation_service.segment_all_videos(
        session=session,
        project_id=project_id,
        project_path=project_path,
        video_ids=request.video_ids,
        mode="simple",
    )
    return {"job_ids": job_ids}
```

Make sure `VideoSelectionRequest` is imported (it should be from the existing endpoint).

- [ ] **Step 2: Add frontend API function**

In `api.ts`, add near `createVideosSegmentation`:

```typescript
export async function createSimpleSegmentation(projectId: number, videoIds: number[]): Promise<{ job_ids: number[] }> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/segmentation/simple`,
        {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ video_ids: videoIds }),
        }
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to create simple segmentation'))
    }
    return response.json()
}
```

- [ ] **Step 3: Add frontend button**

In `VideoPipeline.vue`:

1. Add `createSimpleSegmentation` to imports from `@/services/api`.

2. Add handler near `handleSegmentAll`:
```typescript
const handleSimpleSegmentAll = async () => {
  if (!projectId.value || isSegmenting.value) return
  isSegmenting.value = true
  try {
    await createSimpleSegmentation(projectId.value, selectedVideoIdsList.value)
    await loadVideos()
  } catch (e: any) {
    console.error('Failed to start simple segmentation:', e)
    alert(e.message || 'Failed to start simple segmentation')
  } finally {
    isSegmenting.value = false
  }
}
```

3. Add button after the existing "Segment All" button in the segmentation sidebar section:
```vue
          <button
            class="sidebar-button"
            @click="handleSimpleSegmentAll"
            :disabled="isSegmenting || selectedCount === 0"
          >
            <span class="button-label">{{ isSegmenting ? 'Segmenting...' : 'Simple Segment All' }}</span>
          </button>
```

- [ ] **Step 4: Commit**

```bash
git add vidseq/api/routes/segmentation/sessions.py frontend/src/services/api.ts frontend/src/components/VideoPipeline.vue
git commit -m "feat: add Simple Segment All endpoint and frontend button"
```
