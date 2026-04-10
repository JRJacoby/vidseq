# Standalone Detector Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an "Apply Detector" operation that runs the trained detector on every frame of selected videos, producing bboxes and scores without SAM2.

**Architecture:** New TCP command `apply_detector` on the GPU worker loads the detector, iterates videos frame-by-frame in batches of 32, and streams progress. The service layer saves results to existing `FrameData` columns. Frontend adds an "Apply Detector" button in the VideoPipeline detector sidebar.

**Tech Stack:** FastAPI, Ultralytics YOLO/RT-DETR, TCP streaming protocol, Vue 3

**Spec:** `docs/superpowers/specs/2026-03-27-standalone-detector-inference-design.md`

---

### Task 1: GPU Worker Command Handler

**Files:**
- Modify: `vidseq/services/segmentation_commands.py` (add handler after line 463)

- [ ] **Step 1: Add `handle_apply_detector` function**

Add this function after `handle_generate_training_masks` (after line 463):

```python
def handle_apply_detector(
    params: dict,
    segmentor: StreamingSegmentor,
    response_callback: Callable | None = None,
) -> dict:
    """Run trained detector on every frame of given videos (no SAM2).

    Loads the detector on GPU, iterates videos in batches of 32 frames,
    and returns bboxes + scores grouped by video ID.
    """
    import torch
    from vidseq.services.detector_model import load_finetuned, DETECTION_CONF_THRESHOLD

    video_ids = params["video_ids"]
    project_path = Path(params["project_path"])
    video_paths = params["video_paths"]

    model_path = project_path / "models" / "detector.pt"
    if not model_path.exists():
        raise RuntimeError("No trained detector model found. Train first.")

    detector = None
    try:
        detector = load_finetuned(model_path, device="cuda")

        all_scores: dict[str, list[list]] = {}
        all_bboxes: dict[str, list[list]] = {}

        for video_id, video_path in zip(video_ids, video_paths):
            vid_key = str(video_id)
            scores: list[list] = []
            bboxes: list[list] = []

            frame_source = VideoFrameSource(video_path)
            num_frames = frame_source.frame_count
            batch_size = 32
            frames_done = 0

            for batch_start in range(0, num_frames, batch_size):
                batch_end = min(batch_start + batch_size, num_frames)
                batch_frames = [frame_source[i] for i in range(batch_start, batch_end)]
                batch_indices = list(range(batch_start, batch_end))

                results = detector(batch_frames, conf=DETECTION_CONF_THRESHOLD, verbose=False)

                for idx, result in zip(batch_indices, results):
                    if result.boxes is not None and len(result.boxes) > 0:
                        # Take highest confidence detection
                        best_i = result.boxes.conf.argmax()
                        x1, y1, x2, y2 = result.boxes.xyxy[best_i].cpu().tolist()
                        conf = result.boxes.conf[best_i].item()
                        scores.append([idx, conf])
                        bboxes.append([idx, x1, y1, x2, y2])
                    else:
                        scores.append([idx, 0.0])

                frames_done += len(batch_frames)
                if response_callback and frames_done % 50 < batch_size:
                    response_callback({
                        "type": "progress",
                        "frame_idx": frames_done,
                        "total": num_frames,
                        "video_id": video_id,
                    })

            frame_source.close()
            all_scores[vid_key] = scores
            all_bboxes[vid_key] = bboxes

        return {
            "type": "apply_detector_result",
            "status": "ok",
            "detector_scores": all_scores,
            "detector_bboxes": all_bboxes,
        }

    finally:
        if detector is not None:
            del detector
            torch.cuda.empty_cache()
```

- [ ] **Step 2: Check `VideoFrameSource` has a `close()` method**

Read `VideoFrameSource` in the same file (around line 31). If it doesn't have a `close()` method, add one:

```python
def close(self):
    """Release the video capture."""
    if self._cap is not None:
        self._cap.release()
        self._cap = None
```

- [ ] **Step 3: Commit**

```bash
git add vidseq/services/segmentation_commands.py
git commit -m "feat: add handle_apply_detector command handler for standalone detector inference"
```

---

### Task 2: TCP Server Routing

**Files:**
- Modify: `vidseq/services/segmentation_tcp_server.py` (in `_handle_command`, after the `generate_training_masks` block around line 328)

- [ ] **Step 1: Add command routing for `apply_detector`**

In `_handle_command()`, add a new `elif` block after the `generate_training_masks` case (after line 328). The handler does not require a loaded segmentor (detector is independent), but we pass it for the standard signature:

```python
            elif cmd_type == "apply_detector":
                result = handle_apply_detector(
                    cmd,
                    self._segmentor,
                    response_callback,
                )
```

- [ ] **Step 2: Add import**

Add `handle_apply_detector` to the imports from `segmentation_commands` at the top of the file. Find the existing import line and add it:

```python
from vidseq.services.segmentation_commands import (
    # ... existing imports ...
    handle_apply_detector,
)
```

- [ ] **Step 3: Commit**

```bash
git add vidseq/services/segmentation_tcp_server.py
git commit -m "feat: route apply_detector command to handler in TCP server"
```

---

### Task 3: TCP Client Method

**Files:**
- Modify: `vidseq/services/segmentation_tcp_client.py`

- [ ] **Step 1: Add `apply_detector` method to `SegmentationService` class**

Add this method after `generate_training_masks` (after line 773). Follow the pattern of other streaming methods:

```python
    def apply_detector(
        self,
        project_path: Path,
        videos: list,
    ) -> tuple[dict[int, list[list]], dict[int, list[list]]]:
        """Run trained detector on all frames of given videos.

        Args:
            project_path: Path to the project folder
            videos: List of Video objects with .id, .path, .num_frames

        Returns:
            Tuple of (scores_by_video, bboxes_by_video) where each is
            {video_id: [[frame_idx, ...], ...]}
        """
        result = self._send_streaming({
            "type": "apply_detector",
            "video_ids": [v.id for v in videos],
            "video_paths": [v.path for v in videos],
            "project_path": str(project_path),
        }, timeout=3600.0)

        if result.get("status") != "ok":
            raise RuntimeError(result.get("error", "Failed to apply detector"))

        raw_scores = result.get("detector_scores", {})
        raw_bboxes = result.get("detector_bboxes", {})

        scores_by_video = {int(k): v for k, v in raw_scores.items()}
        bboxes_by_video = {int(k): v for k, v in raw_bboxes.items()}
        return scores_by_video, bboxes_by_video
```

- [ ] **Step 2: Add module-level wrapper function**

Add at the end of the file, near the other module-level wrappers (around line 1090):

```python
def apply_detector(
    project_path: Path,
    videos: list,
) -> tuple[dict[int, list[list]], dict[int, list[list]]]:
    """Run trained detector on all frames of given videos."""
    return SegmentationService.get_instance().apply_detector(project_path, videos)
```

- [ ] **Step 3: Commit**

```bash
git add vidseq/services/segmentation_tcp_client.py
git commit -m "feat: add apply_detector TCP client method with streaming"
```

---

### Task 4: Service Layer Orchestration

**Files:**
- Modify: `vidseq/services/segmentation_service.py`

- [ ] **Step 1: Add `apply_detector` function**

Add this function after the `propagate` function. Follow the pattern from `segment_all_videos` for DB saves:

```python
async def apply_detector(
    session: "AsyncSession",
    project_id: int,
    project_path: Path,
    video_ids: list[int],
) -> int:
    """Run trained detector on all frames of selected videos.

    Returns the number of videos processed.
    """
    from vidseq.models.video import Video

    # Fetch video objects
    result = await session.execute(
        select(Video).where(Video.id.in_(video_ids))
    )
    videos = list(result.scalars().all())
    if not videos:
        raise RuntimeError("No videos found")

    # Validate detector model exists
    model_path = project_path / "models" / "detector.pt"
    if not model_path.exists():
        raise RuntimeError("No trained detector model found. Train first.")

    # Run detector via TCP (blocking call to GPU worker)
    scores_by_video, bboxes_by_video = segmentation_tcp_client.apply_detector(
        project_path=project_path,
        videos=videos,
    )

    # Save results to database
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

    await session.commit()
    return len(videos)
```

- [ ] **Step 2: Verify imports**

Check that `segmentation_tcp_client`, `frame_data_service`, and `select` are already imported at the top of the file. They should be, since `segment_all_videos` uses them. If `select` is missing, add:

```python
from sqlalchemy import select
```

- [ ] **Step 3: Commit**

```bash
git add vidseq/services/segmentation_service.py
git commit -m "feat: add apply_detector service function with DB orchestration"
```

---

### Task 5: API Endpoint

**Files:**
- Modify: `vidseq/api/routes/detector.py`

- [ ] **Step 1: Add the endpoint**

Add at the end of the file (after line 276). Reuse the existing `VideoSelectionRequest` schema:

```python
@router.post("/projects/{project_id}/videos/detection")
async def apply_detector(
    project_id: int,
    request: VideoSelectionRequest,
    project_path: Path = Depends(get_project_folder),
    session: AsyncSession = Depends(get_project_session),
):
    """Run trained detector on every frame of selected videos."""
    try:
        videos_processed = await segmentation_service.apply_detector(
            session=session,
            project_id=project_id,
            project_path=project_path,
            video_ids=request.video_ids,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"videos_processed": videos_processed}
```

- [ ] **Step 2: Verify imports**

Check that `VideoSelectionRequest` and `segmentation_service` are already imported. They should be — `segmentation_service` is imported at line 16 and `VideoSelectionRequest` would need to be added if not present:

```python
from vidseq.api.schemas import VideoSelectionRequest
```

- [ ] **Step 3: Commit**

```bash
git add vidseq/api/routes/detector.py
git commit -m "feat: add POST /videos/detection endpoint for standalone detector inference"
```

---

### Task 6: Frontend API Function

**Files:**
- Modify: `frontend/src/services/api.ts`

- [ ] **Step 1: Add `applyDetector` function**

Add near the other detector functions (around line 1343, after `createDetectionTraining`):

```typescript
export async function applyDetector(projectId: number, videoIds: number[]): Promise<{ videos_processed: number }> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/detection`,
        {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ video_ids: videoIds }),
        }
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to apply detector'))
    }
    return response.json()
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/services/api.ts
git commit -m "feat: add applyDetector API function"
```

---

### Task 7: Frontend Button

**Files:**
- Modify: `frontend/src/components/VideoPipeline.vue`

- [ ] **Step 1: Add import**

Add `applyDetector` to the import list from `@/services/api` at the top of the `<script setup>` block (around line 4):

```typescript
import {
  // ... existing imports ...
  applyDetector,
  // ...
} from '@/services/api'
```

- [ ] **Step 2: Add loading state ref**

Add near the other loading refs (around line 43):

```typescript
const isApplyingDetector = ref(false)
```

- [ ] **Step 3: Add handler function**

Add near `handleTrainDetector` (find it in the script section):

```typescript
const handleApplyDetector = async () => {
  if (!projectId.value || isApplyingDetector.value) return
  isApplyingDetector.value = true
  try {
    await applyDetector(projectId.value, selectedVideoIdsList.value)
    await loadVideos()
  } catch (e: any) {
    console.error('Failed to apply detector:', e)
    alert(e.message || 'Failed to apply detector')
  } finally {
    isApplyingDetector.value = false
  }
}
```

- [ ] **Step 4: Add button to template**

Add the "Apply Detector" button after the "View Detector Training" button (after line 755), before the "Associated Videos" section:

```vue
          <button
            v-if="detectorModelExists"
            class="sidebar-button"
            @click="handleApplyDetector"
            :disabled="isApplyingDetector || isDetectorTraining || selectedCount === 0"
          >
            <span class="button-label">{{ isApplyingDetector ? 'Applying...' : 'Apply Detector' }}</span>
          </button>
```

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/VideoPipeline.vue
git commit -m "feat: add Apply Detector button to VideoPipeline sidebar"
```
