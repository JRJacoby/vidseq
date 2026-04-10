# YOLO Segmentation Detector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a YOLO11x-seg detector that produces per-frame segmentation masks, with a full train/apply/view workflow parallel to the existing bbox detector.

**Architecture:** Extends existing detector infrastructure with a `"seg"` type — adds branches in training, label export, inference, and a new "Seg" view mode in the frontend. Masks are written to the existing (but unused) `detector_masks.h5` files. The GPU worker writes masks directly to H5 during batched inference.

**Tech Stack:** Ultralytics YOLO11x-seg, FastAPI, SQLAlchemy, H5PY, Vue 3

**Spec:** `docs/superpowers/specs/2026-03-31-yolo-seg-detector-design.md`

---

### Task 1: Detector Model — `detect_seg()` + PRETRAINED_MODELS

**Files:**
- Modify: `vidseq/services/detector_model.py`

- [ ] **Step 1: Add `"seg"` to PRETRAINED_MODELS**

Update the dict (line 16):

```python
PRETRAINED_MODELS = {
    "rtdetr": "rtdetr-x.pt",
    "yolo": "yolo11n.pt",
    "obb": "yolo11n-obb.pt",
    "seg": "yolo11x-seg.pt",
}
```

- [ ] **Step 2: Add `detect_seg()` function**

Add after `detect_obb()` (after line ~107):

```python
def detect_seg(
    model,
    frame: np.ndarray,
    conf: float = DETECTION_CONF_THRESHOLD,
) -> list[dict]:
    """Run segmentation detection on a single BGR frame.

    Args:
        model: Ultralytics YOLO-seg model instance.
        frame: BGR uint8 numpy array (H, W, 3).
        conf: Confidence threshold.

    Returns:
        List of detections sorted by confidence (descending).
        Each detection: {"bbox": (x1, y1, x2, y2), "mask": np.ndarray, "conf": float}
        bbox in pixel coords. mask is at YOLO's internal resolution, needs resize.
    """
    results = model(frame, conf=conf, verbose=False)
    detections = []
    if len(results) > 0 and results[0].boxes is not None and results[0].masks is not None:
        boxes = results[0].boxes
        masks = results[0].masks
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().tolist()
            mask = masks.data[i].cpu().numpy()
            detections.append({
                "bbox": (x1, y1, x2, y2),
                "mask": mask,
                "conf": boxes.conf[i].item(),
            })
    detections.sort(key=lambda d: d["conf"], reverse=True)
    return detections
```

- [ ] **Step 3: Commit**

```bash
git add vidseq/services/detector_model.py
git commit -m "feat: add YOLO-seg model variant and detect_seg function"
```

---

### Task 2: Frame Data Service — Async `set_has_detector_mask_batch`

**Files:**
- Modify: `vidseq/services/frame_data_service.py`

- [ ] **Step 1: Add async batch function**

Add after `set_has_detector_mask_batch_sync` (after line ~1230):

```python
async def set_has_detector_mask_batch(
    session: AsyncSession,
    video_id: int,
    frame_indices: list[int],
    has_mask: bool = True,
) -> None:
    """Batch update has_detector_mask flag for multiple frames (async version)."""
    if not frame_indices:
        return
    values = [
        {"video_id": video_id, "frame_idx": frame_idx, "has_detector_mask": 1 if has_mask else 0}
        for frame_idx in frame_indices
    ]
    await _chunked_upsert(session, values, ["video_id", "frame_idx"], ["has_detector_mask"])
```

- [ ] **Step 2: Commit**

```bash
git add vidseq/services/frame_data_service.py
git commit -m "feat: add async set_has_detector_mask_batch function"
```

---

### Task 3: Detector Service — Seg Training Flow

**Files:**
- Modify: `vidseq/services/detector_service.py`

This task adds all 7 branch points for `"seg"` plus the polygon label function and `seg_model_exists`.

- [ ] **Step 1: Add `_mask_to_polygon_label()` function**

Add after `_mask_to_obb_label` (after line ~98):

```python
def _mask_to_polygon_label(mask: np.ndarray, img_h: int, img_w: int) -> str | None:
    """Convert a binary mask to YOLO segmentation polygon label format.

    Args:
        mask: Binary mask (H, W) with values 0 or 255.
        img_h: Image height in pixels.
        img_w: Image width in pixels.

    Returns:
        YOLO seg format string "class x1 y1 x2 y2 ..." (normalized polygon points),
        or None if mask is empty or degenerate.
    """
    contours, _ = cv2.findContours(
        (mask > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None

    # Take the largest contour by area
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 1:
        return None

    # Simplify polygon
    perimeter = cv2.arcLength(largest, True)
    epsilon = 0.001 * perimeter
    simplified = cv2.approxPolyDP(largest, epsilon, True)

    # Need at least 3 points for a valid polygon
    if len(simplified) < 3:
        return None

    # Normalize to [0, 1]
    parts = []
    for point in simplified:
        px, py = point[0]
        parts.append(f"{px / img_w:.6f}")
        parts.append(f"{py / img_h:.6f}")

    return f"0 {' '.join(parts)}"
```

- [ ] **Step 2: Add `seg_model_exists()` method**

Add after `obb_model_exists` on the `DetectorService` class:

```python
    def seg_model_exists(self, project_path: Path) -> bool:
        """Check if trained seg detector model exists."""
        return (project_path / "models" / "seg_detector.pt").exists()
```

- [ ] **Step 3: Add all 7 branch points for `"seg"`**

Read the file and make these changes:

**Branch 1 — Detector type resolution** (line ~245): Change from:
```python
        if training_type == "obb":
            detector_type = "obb"
        else:
```
To:
```python
        if training_type == "obb":
            detector_type = "obb"
        elif training_type == "seg":
            detector_type = "seg"
        else:
```

**Branch 2 — Label dispatch in `_write_yolo_dataset`** (line ~452): Change from:
```python
            if detector_type == "obb":
                yolo_line = _mask_to_obb_label(mask, img_h, img_w)
            else:
                yolo_line = _mask_to_yolo_bbox(mask, img_h, img_w)
```
To:
```python
            if detector_type == "obb":
                yolo_line = _mask_to_obb_label(mask, img_h, img_w)
            elif detector_type == "seg":
                yolo_line = _mask_to_polygon_label(mask, img_h, img_w)
            else:
                yolo_line = _mask_to_yolo_bbox(mask, img_h, img_w)
```

**Branch 3 — Train workers/batch** (line ~352): Add seg case:
```python
        if detector_type == "obb":
            train_workers = 4
            train_batch = max(batch_size, 8)
            train_name = "obb_train"
        elif detector_type == "seg":
            train_workers = 4
            train_batch = max(batch_size, 8)
            train_name = "seg_train"
        elif detector_type == "yolo":
```

**Branch 4 — Task kwarg** (line ~382): Change from:
```python
        if detector_type == "obb":
            train_kwargs["task"] = "obb"
```
To:
```python
        if detector_type == "obb":
            train_kwargs["task"] = "obb"
        elif detector_type == "seg":
            train_kwargs["task"] = "segment"
```

**Branch 5 — Weights destination** (line ~388): Change from:
```python
        weights_dest = "obb_detector.pt" if training_type == "obb" else "detector.pt"
```
To:
```python
        weights_dest = {"obb": "obb_detector.pt", "seg": "seg_detector.pt"}.get(training_type, "detector.pt")
```

**Branch 6 — `_apply_to_training_data` weights path** (line ~523): Change from:
```python
        weights_name = "obb_detector.pt" if training_type == "obb" else "detector.pt"
```
To:
```python
        weights_name = {"obb": "obb_detector.pt", "seg": "seg_detector.pt"}.get(training_type, "detector.pt")
```

**Branch 7 — `_apply_to_training_data` detection + H5 write** (line ~554): Add seg branch before the else. This is the most complex branch — it must write masks to H5 AND set `has_detector_mask`:

```python
            if training_type == "seg":
                detections = detect_seg(detector, frame)
                if detections:
                    best = detections[0]
                    x1, y1, x2, y2 = best["bbox"]
                    conf = best["conf"]
                    mask = best["mask"]
                    mask_binary = (mask > 0.5).astype(np.uint8) * 255
                    mask_resized = cv2.resize(mask_binary, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
                    bboxes_to_save[video_id].append((frame_idx, x1, y1, x2, y2))
                    scores_to_save[video_id].append((frame_idx, conf))
                    masks_to_save.setdefault(video_id, []).append((frame_idx, mask_resized))
                else:
                    scores_to_save[video_id].append((frame_idx, 0.0))
            elif training_type == "obb":
```

Initialize `masks_to_save` dict before the loop:
```python
        masks_to_save: dict[int, list[tuple[int, np.ndarray]]] = {}
```

After the DB upsert block, add H5 writes and `has_detector_mask` updates for seg:
```python
            # Write seg masks to H5 and set has_detector_mask flags
            if training_type == "seg":
                for video_id, mask_list in masks_to_save.items():
                    if not mask_list:
                        continue
                    with detector_masks(project_path, video_id, "a") as mask_data:
                        for frame_idx, mask in mask_list:
                            mask_data[frame_idx] = mask
                    frame_indices = [fi for fi, _ in mask_list]
                    set_has_detector_mask_batch_sync(session, video_id, frame_indices, True)
```

Add the import for `detector_masks` from `array_storage` and `set_has_detector_mask_batch_sync` from `frame_data_service` at the top of `_apply_to_training_data` (inside the function, alongside existing local imports). Also add `detect_seg` to the import from `detector_model`.

- [ ] **Step 4: Commit**

```bash
git add vidseq/services/detector_service.py
git commit -m "feat: add YOLO-seg training flow with polygon labels and post-training mask apply"
```

---

### Task 4: GPU Worker — Extend `handle_apply_detector` for Seg

**Files:**
- Modify: `vidseq/services/segmentation_commands.py`

- [ ] **Step 1: Extend the per-video loop for seg mask writes**

In `handle_apply_detector`, the seg path needs to open `detector_masks.h5` per video and write masks during the batch loop. The key change is restructuring the per-video loop to optionally open the H5 file.

Find the per-video loop (line ~507). The current structure is:
```python
        for video_id, video_path in zip(video_ids, video_paths):
            vid_key = str(video_id)
            scores: list[list] = []
            bboxes: list[list] = []

            with VideoFrameSource(video_path) as frame_source:
                ...batch loop...

            all_scores[vid_key] = scores
            all_bboxes[vid_key] = bboxes
```

For seg, wrap the VideoFrameSource block with an optional `detector_masks` context:

```python
        for video_id, video_path in zip(video_ids, video_paths):
            vid_key = str(video_id)
            scores: list[list] = []
            bboxes: list[list] = []

            with VideoFrameSource(video_path) as frame_source:
                num_frames = frame_source.frame_count
                orig_w = frame_source.width
                orig_h = frame_source.height
                batch_size = 32
                frames_done = 0

                # Open H5 for seg mask writes
                mask_ctx = detector_masks(project_path, video_id, "a") if detector_type == "seg" else None
                mask_data = mask_ctx.__enter__() if mask_ctx else None
                try:
                    for batch_start in range(0, num_frames, batch_size):
                        ...existing batch code...

                        for idx, result in zip(batch_indices, results):
                            if detector_type == "seg":
                                if result.masks is not None and len(result.masks) > 0 and result.boxes is not None and len(result.boxes) > 0:
                                    best_i = result.boxes.conf.argmax()
                                    mask = result.masks.data[best_i].cpu().numpy()
                                    mask_binary = (mask > 0.5).astype(np.uint8) * 255
                                    mask_resized = cv2.resize(mask_binary, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                                    mask_data[idx] = mask_resized
                                    x1, y1, x2, y2 = result.boxes.xyxy[best_i].cpu().tolist()
                                    conf = result.boxes.conf[best_i].item()
                                    scores.append([idx, conf])
                                    bboxes.append([idx, x1, y1, x2, y2])
                                else:
                                    scores.append([idx, 0.0])
                            elif detector_type == "obb":
                                ...existing obb code...
                            else:
                                ...existing bbox code...

                        ...existing progress callback...
                finally:
                    if mask_ctx:
                        mask_ctx.__exit__(None, None, None)

            all_scores[vid_key] = scores
            all_bboxes[vid_key] = bboxes
```

Add `import cv2` at the top of the function (it may already be available from the module-level import) and add `detector_masks` to the imports from `array_storage` at the top of the file (if not already imported).

- [ ] **Step 2: Commit**

```bash
git add vidseq/services/segmentation_commands.py
git commit -m "feat: extend handle_apply_detector to write seg masks to H5 during inference"
```

---

### Task 5: TCP Client + Service Layer

**Files:**
- Modify: `vidseq/services/segmentation_tcp_client.py`
- Modify: `vidseq/services/segmentation_service.py`

- [ ] **Step 1: Add `apply_seg_detector` wrapper to TCP client**

Add at the end of the file near the other wrappers:

```python
def apply_seg_detector(
    project_path: Path,
    videos: list,
) -> tuple[dict[int, list[list]], dict[int, list[list]]]:
    """Run seg detector on all frames of given videos."""
    return SegmentationService.get_instance().apply_detector(project_path, videos, detector_type="seg")
```

- [ ] **Step 2: Add `apply_seg_detector` to service layer**

In `segmentation_service.py`, add after `apply_obb_detector`:

```python
async def apply_seg_detector(
    session: "AsyncSession",
    project_id: int,
    project_path: Path,
    video_ids: list[int],
) -> int:
    """Run seg detector on all frames of selected videos.

    Writes masks to detector_masks.h5 (via GPU worker) and scores/bboxes to DB.
    Returns the number of videos processed.
    """
    from vidseq.models.video import Video

    result = await session.execute(
        select(Video).where(Video.id.in_(video_ids))
    )
    videos = list(result.scalars().all())
    if not videos:
        raise RuntimeError("No videos found")

    model_path = project_path / "models" / "seg_detector.pt"
    if not model_path.exists():
        raise RuntimeError("No trained seg detector model found. Train first.")

    scores_by_video, bboxes_by_video = segmentation_tcp_client.apply_seg_detector(
        project_path=project_path,
        videos=videos,
    )

    for video in videos:
        vid_scores = scores_by_video.get(video.id, [])
        vid_bboxes = bboxes_by_video.get(video.id, [])
        if vid_scores:
            await frame_data_service.save_detector_scores_batch(
                session, video.id, vid_scores
            )
        if vid_bboxes:
            await frame_data_service.save_detector_bboxes_batch(
                session, video.id, vid_bboxes
            )
        # Set has_detector_mask flags for all frames with scores
        all_frame_indices = [int(s[0]) for s in vid_scores if len(s) >= 2 and s[1] > 0]
        if all_frame_indices:
            await frame_data_service.set_has_detector_mask_batch(
                session, video.id, all_frame_indices, True
            )

    await session.commit()
    return len(videos)
```

- [ ] **Step 3: Commit**

```bash
git add vidseq/services/segmentation_tcp_client.py vidseq/services/segmentation_service.py
git commit -m "feat: add apply_seg_detector TCP client wrapper and service orchestration"
```

---

### Task 6: API Endpoints

**Files:**
- Modify: `vidseq/api/routes/detector.py`

- [ ] **Step 1: Add seg status, training, apply, and mask-exists endpoints**

Add response model near the other models:
```python
class SegDetectorStatusResponse(BaseModel):
    """Response for seg detector status."""
    model_exists: bool
    is_training: bool
```

Add endpoints at the end of the file. These follow the exact pattern of the OBB endpoints:

```python
@router.get(
    "/projects/{project_id}/detection/seg/status",
    response_model=SegDetectorStatusResponse,
)
async def get_seg_detection_status(
    project_path: Path = Depends(get_project_folder),
):
    """Get seg detector model status."""
    service = DetectorService.get_instance()
    return SegDetectorStatusResponse(
        model_exists=service.seg_model_exists(project_path),
        is_training=service.is_training() and service._training_type == "seg",
    )


@router.post("/projects/{project_id}/detection/seg/training")
async def create_seg_training(
    request: TrainRequest,
    project_path: Path = Depends(get_project_folder),
):
    """Start seg detector training."""
    service = DetectorService.get_instance()
    if service.is_training():
        raise HTTPException(status_code=409, detail="Training already in progress")
    try:
        service.train(
            project_path=project_path,
            video_ids=request.video_ids,
            max_epochs=request.max_epochs,
            batch_size=request.batch_size,
            lr=request.lr,
            early_stop_patience=request.early_stop_patience,
            training_type="seg",
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "started"}


@router.delete("/projects/{project_id}/detection/seg/training", status_code=204)
async def delete_seg_training():
    """Stop seg detector training."""
    service = DetectorService.get_instance()
    if not service.is_training():
        raise HTTPException(status_code=400, detail="No training in progress")
    service.stop_training()
    return None


@router.get("/projects/{project_id}/detection/seg/training")
async def get_seg_training_progress():
    """Get seg training progress."""
    service = DetectorService.get_instance()
    return service.get_training_progress().to_dict()


@router.get("/projects/{project_id}/detection/seg/training/stream")
async def stream_seg_training():
    """SSE stream for seg training updates."""
    async def event_generator():
        service = DetectorService.get_instance()
        last_progress_str = None
        while True:
            progress = service.get_training_progress()
            progress_dict = progress.to_dict()
            progress_str = json.dumps(progress_dict)
            if progress_str != last_progress_str:
                yield f"data: {progress_str}\n\n"
                last_progress_str = progress_str
            if progress.status in ("completed", "failed", "stopped", "idle"):
                if not progress.is_training:
                    break
            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@router.post("/projects/{project_id}/videos/seg-detection")
async def apply_seg_detector(
    project_id: int,
    request: VideoSelectionRequest,
    project_path: Path = Depends(get_project_folder),
    session: AsyncSession = Depends(get_project_session),
):
    """Run seg detector on every frame of selected videos."""
    try:
        videos_processed = await segmentation_service.apply_seg_detector(
            session=session,
            project_id=project_id,
            project_path=project_path,
            video_ids=request.video_ids,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"videos_processed": videos_processed}


@router.get("/projects/{project_id}/videos/{video_id}/seg-detector-masks/exists")
async def seg_detector_masks_exist(
    video: Video = Depends(get_video),
    session: AsyncSession = Depends(get_project_session),
):
    """Check if seg detector masks exist for a video (checks DB flags, not H5)."""
    result = await session.execute(
        select(FrameData.id)
        .where(
            FrameData.video_id == video.id,
            FrameData.has_detector_mask == 1,
        )
        .limit(1)
    )
    return {"exists": result.first() is not None}
```

Make sure `FrameData` is imported (add `from vidseq.models.frame_data import FrameData` if needed) and `select` from sqlalchemy.

- [ ] **Step 2: Commit**

```bash
git add vidseq/api/routes/detector.py
git commit -m "feat: add seg detector API endpoints for status, training, apply, and mask existence"
```

---

### Task 7: Frontend API + Composable

**Files:**
- Modify: `frontend/src/services/api.ts`
- Create: `frontend/src/composables/useSegDetector.ts`

- [ ] **Step 1: Add seg detector API functions to api.ts**

Add near the other detector functions:

```typescript
// ----- Seg Detector -----

export interface SegDetectorStatus {
    model_exists: boolean
    is_training: boolean
}

export async function getSegDetectionStatus(projectId: number): Promise<SegDetectorStatus> {
    const response = await fetch(`${API_BASE}/projects/${projectId}/detection/seg/status`)
    if (!response.ok) throw new Error(await getErrorMessage(response, 'Failed to get seg status'))
    return response.json()
}

export async function createSegTraining(
    projectId: number,
    maxEpochs: number,
    videoIds: number[],
    earlyStopPatience: number = 20,
): Promise<void> {
    const response = await fetch(`${API_BASE}/projects/${projectId}/detection/seg/training`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            video_ids: videoIds,
            max_epochs: maxEpochs,
            early_stop_patience: earlyStopPatience,
        }),
    })
    if (!response.ok) throw new Error(await getErrorMessage(response, 'Failed to start seg training'))
}

export async function deleteSegTraining(projectId: number): Promise<void> {
    const response = await fetch(`${API_BASE}/projects/${projectId}/detection/seg/training`, {
        method: 'DELETE',
    })
    if (!response.ok) throw new Error(await getErrorMessage(response, 'Failed to stop seg training'))
}

export async function applySegDetector(projectId: number, videoIds: number[]): Promise<{ videos_processed: number }> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/seg-detection`,
        {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ video_ids: videoIds }),
        }
    )
    if (!response.ok) throw new Error(await getErrorMessage(response, 'Failed to apply seg detector'))
    return response.json()
}

export async function segDetectorMasksExist(projectId: number, videoId: number): Promise<{ exists: boolean }> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/${videoId}/seg-detector-masks/exists`
    )
    if (!response.ok) throw new Error(await getErrorMessage(response, 'Failed to check seg masks'))
    return response.json()
}
```

- [ ] **Step 2: Create `useSegDetector.ts` composable**

Create `frontend/src/composables/useSegDetector.ts`:

```typescript
import { ref, watch, onUnmounted, type Ref } from 'vue'
import {
    getSegDetectionStatus,
    createSegTraining,
    deleteSegTraining,
} from '@/services/api'

export function useSegDetector(projectId: Ref<number | null>) {
    const isTraining = ref(false)
    const modelExists = ref(false)

    const checkStatus = async () => {
        if (!projectId.value) return
        try {
            const status = await getSegDetectionStatus(projectId.value)
            modelExists.value = status.model_exists
            isTraining.value = status.is_training
        } catch (e) {
            console.error('Failed to check seg detector status:', e)
        }
    }

    const startTraining = async (maxEpochs: number = 1000, videoIds: number[] = []) => {
        if (!projectId.value || isTraining.value) return
        isTraining.value = true
        try {
            await createSegTraining(projectId.value, maxEpochs, videoIds)
        } catch (e) {
            console.error('Failed to start seg training:', e)
            isTraining.value = false
            throw e
        }
    }

    const stopTraining = async () => {
        if (!projectId.value) return
        try {
            await deleteSegTraining(projectId.value)
        } catch (e) {
            console.error('Failed to stop seg training:', e)
            throw e
        }
    }

    watch(projectId, async (newId) => {
        if (newId !== null) {
            await checkStatus()
        }
    }, { immediate: true })

    let pollInterval: number | null = null

    watch(isTraining, (training) => {
        if (training) {
            if (pollInterval === null && projectId.value !== null) {
                pollInterval = window.setInterval(async () => {
                    await checkStatus()
                    if (!isTraining.value && pollInterval !== null) {
                        clearInterval(pollInterval)
                        pollInterval = null
                    }
                }, 2000)
            }
        } else {
            if (pollInterval !== null) {
                clearInterval(pollInterval)
                pollInterval = null
            }
        }
    })

    onUnmounted(() => {
        if (pollInterval !== null) {
            clearInterval(pollInterval)
            pollInterval = null
        }
    })

    return {
        isTraining,
        modelExists,
        startTraining,
        stopTraining,
        checkStatus,
    }
}
```

- [ ] **Step 3: Commit**

```bash
git add frontend/src/services/api.ts frontend/src/composables/useSegDetector.ts
git commit -m "feat: add seg detector API functions and useSegDetector composable"
```

---

### Task 8: Frontend VideoPipeline Sidebar

**Files:**
- Modify: `frontend/src/components/VideoPipeline.vue`

- [ ] **Step 1: Add imports, composable, state, handlers, and button**

Add `applySegDetector` to imports from `@/services/api`. Add:
```typescript
import { useSegDetector } from '@/composables/useSegDetector'
```

Initialize composable:
```typescript
const {
  isTraining: isSegTraining,
  modelExists: segModelExists,
  startTraining: startSegTraining,
} = useSegDetector(projectId)
```

Add state and handlers:
```typescript
const isApplyingSeg = ref(false)

const handleTrainSeg = async () => {
  if (!projectId.value || isSegTraining.value) return
  try {
    await startSegTraining(1000, selectedVideoIdsList.value)
    router.push(`/project/${projectId.value}/detector`)
  } catch (e: any) {
    alert(e.message || 'Failed to start seg training')
  }
}

const handleApplySeg = async () => {
  if (!projectId.value || isApplyingSeg.value) return
  isApplyingSeg.value = true
  try {
    await applySegDetector(projectId.value, selectedVideoIdsList.value)
    await loadVideos()
  } catch (e: any) {
    console.error('Failed to apply seg detector:', e)
    alert(e.message || 'Failed to apply seg detector')
  } finally {
    isApplyingSeg.value = false
  }
}
```

Add sidebar section after OBB Detector, before Associated Videos:
```vue
          <h4 class="sidebar-section-title">Seg Detector</h4>
          <button
            class="sidebar-button"
            @click="handleTrainSeg"
            :disabled="isSegTraining || isDetectorTraining || isObbTraining || selectedCount === 0"
          >
            <span class="button-label">{{ isSegTraining ? 'Training...' : 'Train Seg Detector' }}</span>
          </button>
          <button
            v-if="segModelExists"
            class="sidebar-button"
            @click="handleApplySeg"
            :disabled="isApplyingSeg || isSegTraining || selectedCount === 0"
          >
            <span class="button-label">{{ isApplyingSeg ? 'Applying...' : 'Apply Seg Detector' }}</span>
          </button>
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/components/VideoPipeline.vue
git commit -m "feat: add Seg Detector section to VideoPipeline sidebar"
```

---

### Task 9: Frontend VideoDetail + useSegmentation — "Seg" View Mode

**Files:**
- Modify: `frontend/src/components/VideoDetail.vue`
- Modify: `frontend/src/composables/useSegmentation.ts`

- [ ] **Step 1: Update `useSegmentation.ts` for `'seg'` mode**

Update the type signature (line ~50) to include `'seg'`:
```typescript
maskViewMode: Ref<'tracker' | 'detector' | 'final' | 'obb' | 'seg'> = ref('tracker'),
```

Update `batchFn` selection (line ~102):
```typescript
const batchFn = maskViewMode.value === 'detector' || maskViewMode.value === 'seg'
    ? getDetectorMasks
    : maskViewMode.value === 'final'
        ? getFinalMasks
        : getTrackerMasks
```

Update `fetchMaskForFrame` (line ~163) — add `'seg'` branch that fetches actual mask (not bbox):
```typescript
        if (maskViewMode.value === 'detector') {
            // Fetch bbox instead of mask
            const result = await getDetectorBbox(projectId.value, videoId.value, frameIdx)
            detectorBbox.value = result.bbox
            return null  // No mask to render
        } else if (maskViewMode.value === 'seg') {
            // Fetch actual mask from detector_masks.h5
            return await getDetectorMask(projectId.value, videoId.value, frameIdx)
        } else if (maskViewMode.value === 'final') {
```

Make sure `getDetectorMask` (single-frame PNG endpoint) is imported in `useSegmentation.ts`. It should already be available — check the imports.

Update `loadFrameData` (line ~182) — skip bbox fetch for `'seg'` mode:
```typescript
    if (maskViewMode.value === 'detector') {
        // ...existing bbox fetch...
    }
```
This already only runs for `'detector'` mode, so `'seg'` correctly falls through to the mask cache path. No change needed here.

- [ ] **Step 2: Update `VideoDetail.vue` for Seg view mode**

Add `segDetectorMasksExist` to imports from `@/services/api`.

Update `MaskViewMode` type:
```typescript
type MaskViewMode = 'tracker' | 'detector' | 'final' | 'obb' | 'seg'
```

Add state:
```typescript
const hasSegMasks = ref(false)
```

Add availability check:
```typescript
const checkSegMasks = async () => {
  if (!projectId.value || !videoId.value) return
  try {
    const result = await segDetectorMasksExist(projectId.value, videoId.value)
    hasSegMasks.value = result.exists
  } catch {
    hasSegMasks.value = false
  }
}
```

Call in `onMounted`:
```typescript
  await checkSegMasks()
```

Add dropdown option after the OBB option:
```vue
    <option value="seg" :disabled="!hasSegMasks">
      Seg {{ hasSegMasks ? '' : '(not available)' }}
    </option>
```

- [ ] **Step 3: Commit**

```bash
git add frontend/src/composables/useSegmentation.ts frontend/src/components/VideoDetail.vue
git commit -m "feat: add Seg view mode for YOLO segmentation masks in VideoDetail"
```
