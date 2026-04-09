# Keypoint Pose Estimation Design

**Date:** 2026-04-09
**Branch:** feat/associated-videos
**Replaces:** Alignment training (DINOv2+U-Net in alignment_service.py) and alignment API routes. The alignment-apply step (frame rotation) is retained as a separate function.

## Overview

Replace the existing alignment training pipeline (DINOv2 encoder + U-Net decoder, custom training loop) with an Ultralytics YOLO pose model for predicting two fixed keypoints — **front** and **rear** — on each frame. This simplifies the architecture by reusing the same Ultralytics training/inference patterns already established by the detector pipeline.

Labeling happens on **original (uncropped) videos** in VideoDetail. The pose model trains and predicts in original-frame coordinates.

## Data Model & Storage

### Labels (training data)

Rename the existing `alignment_labels` table to `pose_labels`. The schema is unchanged:

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER | Primary key |
| `video_id` | INTEGER | FK to videos |
| `frame_idx` | INTEGER | Frame index |
| `front_x` | FLOAT | Front keypoint x (normalized 0-1) |
| `front_y` | FLOAT | Front keypoint y (normalized 0-1) |
| `rear_x` | FLOAT | Rear keypoint x (normalized 0-1) |
| `rear_y` | FLOAT | Rear keypoint y (normalized 0-1) |

Unique constraint on `(video_id, frame_idx)`.

Rename the SQLAlchemy model from `AlignmentLabel` to `PoseLabel`.

### Predictions (inference output)

Add columns to `FrameData`:

| Column | Type | Description |
|--------|------|-------------|
| `pose_front_x` | FLOAT | Predicted front x (normalized 0-1) |
| `pose_front_y` | FLOAT | Predicted front y (normalized 0-1) |
| `pose_rear_x` | FLOAT | Predicted rear x (normalized 0-1) |
| `pose_rear_y` | FLOAT | Predicted rear y (normalized 0-1) |
| `pose_score` | FLOAT | Confidence score (default -1.0, matching existing convention) |
| `has_pose` | INTEGER | Whether predictions exist for this frame (0/1/NULL, Integer for SQLite compatibility) |

### Array storage

Change `alignment_keypoints.h5` schema from `(num_frames, 2)` heading vectors to `(num_frames, 4)` raw keypoint coordinates: `[front_x, front_y, rear_x, rear_y]`. Update `create_alignment_keypoints_array()` accordingly.

The alignment-apply step (retained, see below) reads these raw coordinates and computes heading vectors (`atan2(front - rear)`) at rotation time.

### Trained model

Saved to `<project_folder>/models/pose.pt`.

## Backend Service

### `pose_service.py`

Singleton class. Handles training and inference for the YOLO pose model.

**Training:**
1. Query all frames with `PoseLabel` entries across selected videos
2. Read frames from original (uncropped) videos
3. Convert to YOLO pose format: 1 class ("animal"), 2 keypoints per instance, bounding box derived from the keypoint positions (padded)
4. Write temporary YOLO dataset directory (`images/train`, `images/val`, `labels/train`, `labels/val`, `dataset.yaml`), cleaned up in a `finally` block
5. Train with Ultralytics pose model (e.g., `yolo11n-pose.pt` base)
6. Save best weights to `models/pose.pt`
7. Stream progress via SSE (epoch, train_loss, val_loss, lr)

**Applying:**
1. Load `models/pose.pt`
2. Run inference per frame on selected videos (GPU-bound, runs in FastAPI background thread like detector `_apply_to_training_data`)
3. Write `pose_front_x/y`, `pose_rear_x/y`, `pose_score`, `has_pose` to `FrameData`
4. Write `[front_x, front_y, rear_x, rear_y]` to `alignment_keypoints.h5`
5. Stream per-video progress via SSE (video_idx, total_videos, frame_idx, total_frames)

**YOLO pose format notes:**
- Each label line: `class_id x_center y_center width height kp1_x kp1_y kp1_visible kp2_x kp2_y kp2_visible`
- Visibility flag: 2 = visible (always, since user clicked it)
- `dataset.yaml` specifies `kpt_shape: [2, 3]` (2 keypoints, 3 values each: x, y, visibility)

### Training data bounding box derivation

The bounding box for YOLO pose format is derived from the two keypoint positions:
1. Compute the bounding box that encloses both keypoints
2. Pad by 20% of frame dimensions (not box diagonal — ensures a reasonable box even when keypoints are close together)
3. Clamp to frame bounds (0-1)
4. Convert to YOLO normalized center format (x_center, y_center, width, height)

### Alignment-apply (retained from alignment_service.py)

Extract the frame-rotation logic from `alignment_service.py` into a standalone function (or small module). This step:
1. Reads raw keypoint coordinates from `alignment_keypoints.h5` (shape `(N, 4)`)
2. Computes heading angle: `atan2(front_y - rear_y, front_x - rear_x)`
3. Rotates frames to canonical heading, producing aligned videos and `aligned_masks.h5`

The rest of `alignment_service.py` (DINOv2 model, training loop, feature averaging, Savitzky-Golay smoothing) is deleted.

## API Routes

New file: `api/routes/pose.py`, mounted under `/api/projects/{project_id}/`.

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/videos/{video_id}/pose/labels` | List labeled frames for a video |
| `POST` | `/videos/{video_id}/pose/labels/{frame_idx}` | Save front/rear label for a frame |
| `DELETE` | `/videos/{video_id}/pose/labels/{frame_idx}` | Remove label for a frame |
| `POST` | `/pose/training` | Start training (body: `video_ids`, `max_epochs`) |
| `DELETE` | `/pose/training` | Stop training |
| `GET` | `/pose/training/stream` | SSE progress stream (training) |
| `GET` | `/pose/status` | Model existence + training state |
| `POST` | `/videos/pose` | Apply pose model to selected videos |
| `GET` | `/videos/pose/apply/stream` | SSE progress stream (applying) |
| `GET` | `/videos/{id}/pose-scores-downsampled` | Confidence scores for DataTrack |

### Schemas

```python
class PoseLabelRequest(BaseModel):
    front_x: float = Field(ge=0.0, le=1.0)
    front_y: float = Field(ge=0.0, le=1.0)
    rear_x: float = Field(ge=0.0, le=1.0)
    rear_y: float = Field(ge=0.0, le=1.0)

class PoseLabelResponse(BaseModel):
    frame_idx: int
    front_x: float
    front_y: float
    rear_x: float
    rear_y: float

class PoseTrainingRequest(BaseModel):
    video_ids: list[int]
    max_epochs: int = 300

class PoseStatusResponse(BaseModel):
    model_exists: bool
    is_training: bool
```

## Frontend — Annotation (VideoDetail)

### Keypoint labeling mode

New button in the VideoDetail action bar: **"Label Keypoints"**. Toggles `isLabelingKeypoints` state.

Labeling occurs on the **original (uncropped) video** — the same video shown in VideoDetail.

**State machine** when active:

```
awaiting_front → (click) → awaiting_rear → (click) → saving → (saved) → advance frame → awaiting_front
```

- **awaiting_front**: Hint text "Click front". Crosshair cursor. First click places a green dot and records `front_x, front_y`.
- **awaiting_rear**: Hint text "Click rear". Second click places a red dot, records `rear_x, rear_y`, and saves the label via `POST /videos/{video_id}/pose/labels/{frame_idx}`.
- **saving**: Brief state while API call completes. On success, auto-advance to next frame (seek video element to next frame's timestamp), reset to `awaiting_front`.

**Editing existing labels:**
- When navigating to a frame that has a label, display the front (green) and rear (red) dots.
- Entering labeling mode on a labeled frame allows re-clicking to overwrite.
- Delete key removes the label for the current frame.

### Composable: `usePoseLabels(projectId, videoId)`

**State:**
- `currentLabel: Ref<{ front_x, front_y, rear_x, rear_y } | null>` — label for current frame
- `labeledFrameCount: Ref<number>` — total labeled frames for this video
- `labelingState: Ref<'awaiting_front' | 'awaiting_rear' | 'idle'>` — annotation state machine

**Methods:**
- `saveLabel(frameIdx, front_x, front_y, rear_x, rear_y)` — POST to API
- `deleteLabel(frameIdx)` — DELETE from API
- `loadLabel(frameIdx)` — fetch label for a frame (or pull from a local cache)
- `refresh()` — reload label count

### VideoOverlay changes

New rendering layer for keypoint dots:
- **Labeled keypoints**: Green filled circle (front), red filled circle (rear) — 8px radius
- **Predicted keypoints** (after applying): Same colors but outline/ring style to distinguish from hand labels
- Rendered regardless of `maskViewMode` — keypoints overlay on top of whatever mask is showing
- Controlled by a `showPoseKeypoints` toggle in the action bar

## Frontend — Training & Applying (VideoPipeline)

### VideoPipeline.vue additions

New section in the pipeline controls (alongside detector buttons):

- **"Train Pose"** button — calls `startPoseTraining(maxEpochs, selectedVideoIds)`, navigates to pose training progress view
- **"Apply Pose"** button — calls `applyPose(projectId, selectedVideoIds)`, shows progress via SSE
- Video list shows `pose_label_count` per video (computed from a count query on `pose_labels` grouped by `video_id`)

### Training progress

Reuse `DetectorTraining.vue` with a mode/type prop, or create a minimal `PoseTraining.vue` that follows the same SSE pattern (epoch counter, loss chart, stop button). The SSE stream shape is identical to detector training.

### Composable: `usePoseDetector(projectId)`

Same shape as `useDetector` (simplified — no detector type switching):
- `isTraining`, `modelExists` — reactive state
- `startTraining(maxEpochs, videoIds)`, `stopTraining()` — actions
- Status polling while training is active

## Frontend — Visualization

### VideoOverlay

After applying the pose model, predicted keypoints render as two dots per frame:
- Front: green outline circle (distinguishable from filled label dots)
- Rear: red outline circle
- Visible during playback and scrubbing

### DataTrack

New confidence line: **pose score** in magenta. Toggled via "Pose Confidence" button in the action bar, alongside existing detector/OBB confidence toggles.

Data fetched from `GET /videos/{id}/pose-scores-downsampled` with the same LTTB downsampling pattern.

## Cleanup

| File/Component | Action |
|----------------|--------|
| `vidseq/services/alignment_service.py` | Delete training code (DINOv2 model, training loop, feature averaging, Savitzky-Golay smoothing). Extract frame-rotation logic into a standalone function. |
| `vidseq/api/routes/alignment.py` | Delete entirely (replaced by pose routes) |
| Frontend alignment composables/API calls | Remove |
| DINOv2 model dependencies | Remove if not used elsewhere |
| `AlignmentLabel` model | Rename to `PoseLabel`, keep schema |
| `alignment_keypoints.h5` context manager | Keep, update schema from `(N, 2)` to `(N, 4)` |
| `video_service.py` cascade delete | Update `AlignmentLabel` import to `PoseLabel` |
| Downstream pipeline (PCA, aligned videos) | Update alignment-apply to read `(N, 4)` keypoints and compute heading |

## Migration

1. Rename table: `ALTER TABLE alignment_labels RENAME TO pose_labels` (project uses `create_all()`, no Alembic — apply via manual migration or startup check)
2. Add columns to `frame_data`: `pose_front_x`, `pose_front_y`, `pose_rear_x`, `pose_rear_y` (FLOAT), `pose_score` (FLOAT, default -1.0), `has_pose` (INTEGER)
3. Existing labeled data (if any) is preserved through the table rename
