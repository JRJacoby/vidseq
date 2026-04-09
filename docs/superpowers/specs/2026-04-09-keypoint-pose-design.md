# Keypoint Pose Estimation Design

**Date:** 2026-04-09
**Branch:** feat/associated-videos
**Replaces:** Alignment system (DINOv2+U-Net alignment_service.py, alignment API routes)

## Overview

Replace the existing alignment pipeline (DINOv2 encoder + U-Net decoder, custom training loop) with an Ultralytics YOLO pose model for predicting two fixed keypoints — **front** and **back** — on each frame. This simplifies the architecture by reusing the same Ultralytics training/inference patterns already established by the detector pipeline.

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
| `rear_x` | FLOAT | Back keypoint x (normalized 0-1) |
| `rear_y` | FLOAT | Back keypoint y (normalized 0-1) |

Unique constraint on `(video_id, frame_idx)`.

Rename the SQLAlchemy model from `AlignmentLabel` to `PoseLabel`. Alembic migration renames the table.

### Predictions (inference output)

Add columns to `FrameData`:

| Column | Type | Description |
|--------|------|-------------|
| `pose_front_x` | FLOAT | Predicted front x (normalized 0-1) |
| `pose_front_y` | FLOAT | Predicted front y (normalized 0-1) |
| `pose_back_x` | FLOAT | Predicted back x (normalized 0-1) |
| `pose_back_y` | FLOAT | Predicted back y (normalized 0-1) |
| `pose_score` | FLOAT | Confidence score |
| `has_pose` | BOOLEAN | Whether predictions exist for this frame |

### Array storage

Write predictions to `alignment_keypoints.h5` — the same file the downstream cropping/alignment pipeline already consumes. This ensures zero changes to downstream stages.

### Trained model

Saved to `<project_folder>/models/pose.pt`.

## Backend Service

### `pose_service.py`

Follows `detector_service.py` patterns. Singleton class.

**Training:**
1. Query all frames with `PoseLabel` entries across selected videos
2. Convert to YOLO pose format: 1 class ("animal"), 2 keypoints per instance, bounding box derived from the keypoint positions (padded)
3. Write temporary YOLO dataset directory (`images/train`, `images/val`, `labels/train`, `labels/val`, `dataset.yaml`)
4. Train with Ultralytics pose model (e.g., `yolo11n-pose.pt` base)
5. Save best weights to `models/pose.pt`
6. Stream progress via SSE (epoch, train_loss, val_loss, lr)

**Applying:**
1. Load `models/pose.pt`
2. Run inference per frame on selected videos
3. Write `pose_front_x/y`, `pose_back_x/y`, `pose_score`, `has_pose` to `FrameData`
4. Write coordinates to `alignment_keypoints.h5` for downstream pipeline compatibility

**YOLO pose format notes:**
- Each label line: `class_id x_center y_center width height kp1_x kp1_y kp1_visible kp2_x kp2_y kp2_visible`
- Bounding box can be derived from keypoints with padding (e.g., expand by 20% of frame around the two points)
- Visibility flag: 2 = visible (always, since user clicked it)
- `dataset.yaml` specifies `kpt_shape: [2, 3]` (2 keypoints, 3 values each: x, y, visibility)

### Training data format

The bounding box for YOLO pose format is derived from the two keypoint positions:
1. Compute the bounding box that encloses both keypoints
2. Pad by a fixed margin (e.g., 20% of the box diagonal or a percentage of frame dimensions)
3. Clamp to frame bounds
4. Convert to YOLO normalized center format (x_center, y_center, width, height)

## API Routes

New file: `api/routes/pose.py`, mounted under `/api/projects/{project_id}/`.

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/pose/labels` | List labeled frames for a video (query param: `video_id`) |
| `POST` | `/pose/labels/{frame_idx}` | Save front/back label for a frame |
| `DELETE` | `/pose/labels/{frame_idx}` | Remove label for a frame |
| `POST` | `/pose/training` | Start training (body: `video_ids`, `max_epochs`) |
| `DELETE` | `/pose/training` | Stop training |
| `GET` | `/pose/training/stream` | SSE progress stream |
| `GET` | `/pose/status` | Model existence + training state |
| `POST` | `/videos/pose` | Apply pose model to selected videos |
| `GET` | `/videos/{id}/pose-scores-downsampled` | Confidence scores for DataTrack |

### Schemas

```python
class PoseLabelRequest(BaseModel):
    front_x: float  # normalized 0-1
    front_y: float
    back_x: float
    back_y: float

class PoseLabelResponse(BaseModel):
    frame_idx: int
    front_x: float
    front_y: float
    back_x: float
    back_y: float

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

**State machine** when active:

```
awaiting_front → (click) → awaiting_back → (click) → saving → (saved) → advance frame → awaiting_front
```

- **awaiting_front**: Hint text "Click front". Crosshair cursor. First click places a green dot and records `front_x, front_y`.
- **awaiting_back**: Hint text "Click back". Second click places a red dot, records `back_x, back_y`, and saves the label via `POST /pose/labels/{frame_idx}`.
- **saving**: Brief state while API call completes. On success, auto-advance `currentFrameIdx` by 1, reset to `awaiting_front`.

**Editing existing labels:**
- When navigating to a frame that has a label, display the front (green) and back (red) dots.
- Entering labeling mode on a labeled frame allows re-clicking to overwrite.
- Delete key removes the label for the current frame (`DELETE /pose/labels/{frame_idx}`).

### Composable: `usePoseLabels(projectId, videoId)`

**State:**
- `currentLabel: Ref<{ front_x, front_y, back_x, back_y } | null>` — label for current frame
- `labeledFrameCount: Ref<number>` — total labeled frames for this video
- `labelingState: Ref<'awaiting_front' | 'awaiting_back' | 'idle'>` — annotation state machine

**Methods:**
- `saveLabel(frameIdx, front_x, front_y, back_x, back_y)` — POST to API
- `deleteLabel(frameIdx)` — DELETE from API
- `loadLabel(frameIdx)` — fetch label for a frame (or pull from a local cache)
- `refresh()` — reload label count

### VideoOverlay changes

New rendering layer for keypoint dots:
- **Labeled keypoints**: Green circle (front), red circle (back) — filled, 8px radius
- **Predicted keypoints** (after applying): Same colors but slightly different style (e.g., ring/outline only, or smaller) to distinguish from hand labels
- Rendered regardless of `maskViewMode` — keypoints overlay on top of whatever mask is showing
- Controlled by a `showPoseKeypoints` toggle in the action bar

## Frontend — Training & Applying (VideoPipeline)

### VideoPipeline.vue additions

New section in the pipeline controls (alongside detector buttons):

- **"Train Pose"** button — calls `startPoseTraining(maxEpochs, selectedVideoIds)`, navigates to pose training progress view
- **"Apply Pose"** button — calls `applyPose(projectId, selectedVideoIds)`
- Video list shows `pose_label_count` per video (number of labeled frames)

### Training progress

Reuse `DetectorTraining.vue` with a mode/type prop, or create a minimal `PoseTraining.vue` that follows the same SSE pattern (epoch counter, loss chart, stop button). The SSE stream shape is identical to detector training.

### Composable: `usePoseDetector(projectId)`

Same shape as `useDetector`:
- `isTraining`, `modelExists` — reactive state
- `startTraining(maxEpochs, videoIds)`, `stopTraining()` — actions
- Status polling while training is active

## Frontend — Visualization

### VideoOverlay

After applying the pose model, predicted keypoints render as two dots per frame:
- Front: green outline circle (distinguishable from filled label dots)
- Back: red outline circle
- Visible during playback and scrubbing

### DataTrack

New confidence line: **pose score** in magenta. Toggled via "Pose Confidence" button in the action bar, alongside existing detector/OBB confidence toggles.

Data fetched from `GET /videos/{id}/pose-scores-downsampled` with the same LTTB downsampling pattern.

## Cleanup — Code to Remove

| File/Component | Action |
|----------------|--------|
| `vidseq/services/alignment_service.py` | Delete entirely |
| `vidseq/api/routes/alignment.py` | Delete entirely |
| Frontend alignment composables/API calls | Remove |
| DINOv2 model dependencies | Remove if not used elsewhere |
| `AlignmentLabel` model | Rename to `PoseLabel`, keep schema |
| `alignment_keypoints.h5` context manager | Keep (predictions write here) |
| Downstream pipeline consumers | No changes needed |

## Migration

1. Alembic migration: rename `alignment_labels` → `pose_labels`
2. Alembic migration: add `pose_front_x`, `pose_front_y`, `pose_back_x`, `pose_back_y`, `pose_score`, `has_pose` columns to `frame_data`
3. Existing labeled data (if any) is preserved through the table rename
