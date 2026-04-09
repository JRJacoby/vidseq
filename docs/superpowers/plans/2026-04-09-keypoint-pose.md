# Keypoint Pose Estimation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the DINOv2+U-Net alignment pipeline with a YOLO pose model for front/rear keypoint prediction, with labeling in VideoDetail and training/applying from VideoPipeline.

**Architecture:** New `pose_service.py` (Ultralytics YOLO pose training/inference) parallels `detector_service.py`. Frontend adds a keypoint labeling mode to VideoDetail with auto-advance, and train/apply controls to VideoPipeline. The alignment-apply step (frame rotation) is retained as a standalone module consuming keypoint predictions.

**Tech Stack:** Python (FastAPI, Ultralytics, SQLAlchemy, H5PY), TypeScript (Vue 3, Pinia), SSE streaming

**Spec:** `docs/superpowers/specs/2026-04-09-keypoint-pose-design.md`

---

### Task 1: Rename AlignmentLabel → PoseLabel

**Files:**
- Modify: `vidseq/models/alignment_label.py`
- Modify: `vidseq/services/video_service.py:549,584`

- [ ] **Step 1: Rename the model class and table**

In `vidseq/models/alignment_label.py`, rename the class and table:

```python
class PoseLabel(Base):
    """Per-frame pose labels with front and rear keypoint coordinates.

    Used for training the YOLO pose model. Coordinates are
    normalized (0-1) relative to original video frame dimensions.
    """
    __tablename__ = "pose_labels"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    video_id: Mapped[int] = mapped_column(Integer, ForeignKey("videos.id"), nullable=False)
    frame_idx: Mapped[int] = mapped_column(Integer, nullable=False)

    # Front point (nose) - normalized coordinates (0-1)
    front_x: Mapped[float] = mapped_column(Float, nullable=False)
    front_y: Mapped[float] = mapped_column(Float, nullable=False)

    # Rear point (tail) - normalized coordinates (0-1)
    rear_x: Mapped[float] = mapped_column(Float, nullable=False)
    rear_y: Mapped[float] = mapped_column(Float, nullable=False)

    __table_args__ = (
        Index("ix_pose_label_video_frame", "video_id", "frame_idx", unique=True),
    )
```

Also rename the file itself from `alignment_label.py` to `pose_label.py`.

- [ ] **Step 2: Update video_service.py cascade delete**

In `vidseq/services/video_service.py`, update the import (line 549) and delete (line 584):

```python
# Line 549: change import
from vidseq.models.pose_label import PoseLabel

# Line 584: change delete
await session.execute(
    delete(PoseLabel).where(PoseLabel.video_id == video.id)
)
```

- [ ] **Step 3: Verify no other imports of AlignmentLabel remain**

Search for `AlignmentLabel` and `alignment_label` across the codebase. Update any remaining references. Key locations to check:
- `vidseq/services/alignment_service.py:43` — will be cleaned up in Task 5
- `vidseq/api/routes/alignment.py` — will be deleted in Task 5

For now, only fix `video_service.py`. The alignment service references will be handled when we delete it.

- [ ] **Step 4: Commit**

```bash
git add vidseq/models/pose_label.py vidseq/services/video_service.py
git rm vidseq/models/alignment_label.py
git commit -m "refactor: rename AlignmentLabel to PoseLabel"
```

---

### Task 2: Add Pose Columns to FrameData

**Files:**
- Modify: `vidseq/models/frame_data.py`

- [ ] **Step 1: Add pose columns**

Add after the `obb_score` line (line 59) in `vidseq/models/frame_data.py`:

```python
    # Pose keypoint coordinates (NULL = no prediction)
    pose_front_x: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
    pose_front_y: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
    pose_rear_x: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
    pose_rear_y: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)

    # Pose confidence score (-1.0 = not computed)
    pose_score: Mapped[float] = mapped_column(Float, nullable=False, default=-1.0)

    # Pose presence flag: 1 = has pose, 0 = no pose, NULL = unknown
    # Using Integer for SQLite compatibility (no native boolean type)
    has_pose: Mapped[int | None] = mapped_column(Integer, nullable=True, default=None)
```

Add an index inside `__table_args__`:

```python
        Index("ix_frame_data_video_has_pose", "video_id", "has_pose"),
```

- [ ] **Step 2: Commit**

```bash
git add vidseq/models/frame_data.py
git commit -m "feat: add pose keypoint columns to FrameData"
```

---

### Task 3: Update H5 Schema for Keypoint Coordinates

**Files:**
- Modify: `vidseq/services/array_storage.py:454-475`

- [ ] **Step 1: Change shape from (N, 2) to (N, 4)**

In `vidseq/services/array_storage.py`, update `create_alignment_keypoints_array` (line 454):

```python
def create_alignment_keypoints_array(
    project_path: Path,
    video_id: int,
    num_frames: int,
) -> None:
    """Create alignment keypoints array for a video.

    Stores raw keypoint coordinates: [front_x, front_y, rear_x, rear_y]
    per frame, normalized 0-1.

    Args:
        project_path: Path to the project folder
        video_id: ID of the video
        num_frames: Total number of frames in the video
    """
    h5_path = _construct_h5_path(project_path, video_id, "alignment_keypoints.h5")
    with open_h5_with_lock(h5_path, mode="w") as f:
        f.create_dataset(
            "data",
            shape=(num_frames, 4),
            dtype=np.float32,
            fillvalue=0.0,
            chunks=(1, 4),
            compression=None,
        )
```

- [ ] **Step 2: Commit**

```bash
git add vidseq/services/array_storage.py
git commit -m "feat: change alignment_keypoints H5 schema from (N,2) heading vectors to (N,4) raw coordinates"
```

---

### Task 4: Create Pose Service

**Files:**
- Create: `vidseq/services/pose_service.py`
- Create: `vidseq/schemas/pose.py`

- [ ] **Step 1: Create pose progress schema**

Create `vidseq/schemas/pose.py`:

```python
"""Schemas for pose model training and inference."""
from dataclasses import dataclass, field
from typing import Optional
import time


@dataclass
class PoseTrainingProgress:
    """Tracks training progress for SSE streaming."""
    is_training: bool = False
    current_epoch: int = 0
    max_epochs: int = 0
    current_train_loss: float = 0.0
    current_val_loss: float = 0.0
    train_loss_history: list[float] = field(default_factory=list)
    val_loss_history: list[float] = field(default_factory=list)
    best_val_loss: Optional[float] = None
    best_epoch: int = 0
    current_lr: float = 0.0
    epochs_without_improvement: int = 0
    lr_patience: int = 10
    early_stop_patience: int = 20
    status: str = "idle"
    started_at: float = 0.0
    error_message: Optional[str] = None
    num_train_frames: int = 0
    num_val_frames: int = 0
    current_batch: int = 0
    total_batches: int = 0
    batch_loss: float = 0.0
    # Apply progress
    apply_current: int = 0
    apply_total: int = 0

    def to_dict(self) -> dict:
        return {
            "is_training": self.is_training,
            "current_epoch": self.current_epoch,
            "max_epochs": self.max_epochs,
            "current_train_loss": self.current_train_loss,
            "current_val_loss": self.current_val_loss,
            "train_loss_history": self.train_loss_history,
            "val_loss_history": self.val_loss_history,
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
            "current_lr": self.current_lr,
            "epochs_without_improvement": self.epochs_without_improvement,
            "lr_patience": self.lr_patience,
            "early_stop_patience": self.early_stop_patience,
            "status": self.status,
            "started_at": self.started_at,
            "error_message": self.error_message,
            "num_train_frames": self.num_train_frames,
            "num_val_frames": self.num_val_frames,
            "current_batch": self.current_batch,
            "total_batches": self.total_batches,
            "batch_loss": self.batch_loss,
            "apply_current": self.apply_current,
            "apply_total": self.apply_total,
        }
```

- [ ] **Step 2: Create pose_service.py with training**

Create `vidseq/services/pose_service.py`. This follows the `detector_service.py` singleton pattern. Key differences: training data comes from `PoseLabel` (not masks), and YOLO dataset format includes keypoints.

```python
"""Pose estimation service for YOLO pose model training and inference."""
import logging
import os
import random
import shutil
import tempfile
import threading
import time
from pathlib import Path

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import cv2
import numpy as np
import yaml
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from vidseq.models.frame_data import FrameData
from vidseq.models.pose_label import PoseLabel
from vidseq.models.video import Video
from vidseq.schemas.pose import PoseTrainingProgress
from vidseq.services.array_storage import (
    alignment_keypoints,
    create_alignment_keypoints_array,
)
from vidseq.services.database_manager import DatabaseManager

logger = logging.getLogger(__name__)

MIN_FRAMES_FOR_VALIDATION = 10
BBOX_PAD_FRACTION = 0.20  # Pad bounding box by 20% of frame dimensions


class PoseService:
    """Singleton service for YOLO pose model training and inference."""

    _instance = None

    def __init__(self):
        self._is_training = False
        self._stop_requested = False
        self._training_thread: threading.Thread | None = None
        self._training_progress = PoseTrainingProgress()

    @classmethod
    def get_instance(cls) -> "PoseService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def is_training(self) -> bool:
        return self._is_training

    def get_training_progress(self) -> PoseTrainingProgress:
        return self._training_progress

    def stop_training(self) -> None:
        self._stop_requested = True

    def model_exists(self, project_path: Path) -> bool:
        return (project_path / "models" / "pose.pt").exists()

    def train(
        self,
        project_path: Path,
        video_ids: list[int],
        max_epochs: int = 300,
    ) -> bool:
        """Start pose training in background thread."""
        if self._is_training:
            raise RuntimeError("Training already in progress")

        self._is_training = True
        self._stop_requested = False

        def _train_thread():
            try:
                self._train_sync(project_path, video_ids, max_epochs)
            except Exception as e:
                logger.exception("Pose training failed")
                self._training_progress.status = "failed"
                self._training_progress.error_message = str(e)
            finally:
                self._is_training = False

        self._training_thread = threading.Thread(target=_train_thread, daemon=True)
        self._training_thread.start()
        return True

    def _gather_training_frames(
        self, project_path: Path, video_ids: list[int]
    ) -> list[tuple[Path, int, int, float, float, float, float]]:
        """Gather frames with pose labels.

        Returns list of (video_path, video_id, frame_idx, front_x, front_y, rear_x, rear_y).
        """
        db_manager = DatabaseManager.get_instance()
        engine = db_manager.get_project_engine(project_path)

        frames = []
        with Session(engine) as session:
            for video_id in video_ids:
                video = session.execute(
                    select(Video).where(Video.id == video_id)
                ).scalar_one_or_none()
                if video is None:
                    continue

                video_path = Path(video.filepath)
                if not video_path.exists():
                    logger.warning(f"Video file not found: {video_path}")
                    continue

                labels = session.execute(
                    select(PoseLabel).where(PoseLabel.video_id == video_id)
                ).scalars().all()

                for label in labels:
                    frames.append((
                        video_path, video_id, label.frame_idx,
                        label.front_x, label.front_y,
                        label.rear_x, label.rear_y,
                    ))

        return frames

    def _write_yolo_pose_dataset(
        self,
        frames: list[tuple[Path, int, int, float, float, float, float]],
        split: str,
        tmp_path: Path,
    ) -> None:
        """Write frames and YOLO pose labels to dataset directory."""
        images_dir = tmp_path / "images" / split
        labels_dir = tmp_path / "labels" / split

        for i, (video_path, video_id, frame_idx, fx, fy, rx, ry) in enumerate(frames):
            cap = cv2.VideoCapture(str(video_path))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            cap.release()

            if not ret:
                logger.warning(f"Failed to read frame {frame_idx} from {video_path}")
                continue

            # Save image
            img_name = f"v{video_id}_f{frame_idx}_{i}.jpg"
            cv2.imwrite(str(images_dir / img_name), frame)

            # Derive bounding box from keypoints with padding
            img_h, img_w = frame.shape[:2]
            min_x = min(fx, rx)
            max_x = max(fx, rx)
            min_y = min(fy, ry)
            max_y = max(fy, ry)

            pad_x = BBOX_PAD_FRACTION
            pad_y = BBOX_PAD_FRACTION
            x1 = max(0.0, min_x - pad_x)
            y1 = max(0.0, min_y - pad_y)
            x2 = min(1.0, max_x + pad_x)
            y2 = min(1.0, max_y + pad_y)

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1

            # YOLO pose format: class x_center y_center width height kp1_x kp1_y kp1_vis kp2_x kp2_y kp2_vis
            label_name = img_name.replace(".jpg", ".txt")
            with open(labels_dir / label_name, "w") as f:
                f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {fx:.6f} {fy:.6f} 2 {rx:.6f} {ry:.6f} 2\n")

    def _train_sync(
        self,
        project_path: Path,
        video_ids: list[int],
        max_epochs: int,
    ) -> None:
        """Synchronous training implementation."""
        from ultralytics import YOLO
        from vidseq.services import segmentation_service

        # Free GPU memory
        segmentation_service.shutdown()

        # Gather training data
        all_frames = self._gather_training_frames(project_path, video_ids)
        if len(all_frames) == 0:
            raise RuntimeError("No pose labels found. Label keypoints first.")

        # Train/val split
        random.shuffle(all_frames)
        use_validation = len(all_frames) >= MIN_FRAMES_FOR_VALIDATION

        if use_validation:
            split_idx = int(len(all_frames) * 0.8)
            train_frames = all_frames[:split_idx]
            val_frames = all_frames[split_idx:]
        else:
            train_frames = all_frames
            val_frames = []

        # Initialize progress
        self._training_progress = PoseTrainingProgress(
            is_training=True,
            current_epoch=0,
            max_epochs=max_epochs,
            status="training",
            started_at=time.time(),
            num_train_frames=len(train_frames),
            num_val_frames=len(val_frames),
        )

        tmp_dir = tempfile.mkdtemp(prefix="vidseq_pose_")
        tmp_path = Path(tmp_dir)

        try:
            # Create directory structure
            for split in ("train", "val"):
                (tmp_path / "images" / split).mkdir(parents=True, exist_ok=True)
                (tmp_path / "labels" / split).mkdir(parents=True, exist_ok=True)

            # Write dataset
            self._write_yolo_pose_dataset(train_frames, "train", tmp_path)
            if val_frames:
                self._write_yolo_pose_dataset(val_frames, "val", tmp_path)

            # Write dataset.yaml
            yaml_path = tmp_path / "dataset.yaml"
            dataset_config = {
                "path": str(tmp_path),
                "train": "images/train",
                "val": "images/val" if val_frames else "images/train",
                "nc": 1,
                "names": ["animal"],
                "kpt_shape": [2, 3],  # 2 keypoints, 3 values each (x, y, visibility)
            }
            with open(yaml_path, "w") as f:
                yaml.dump(dataset_config, f, default_flow_style=False)

            # Callbacks
            def check_stop(trainer):
                if self._stop_requested:
                    raise KeyboardInterrupt("Training stopped by user")

            def on_epoch_end(trainer):
                epoch = trainer.epoch
                total_loss = (
                    float(trainer.tloss.sum())
                    if trainer.tloss is not None
                    else 0.0
                )
                metrics = trainer.metrics or {}
                self._training_progress.current_epoch = epoch
                self._training_progress.current_train_loss = total_loss
                val_loss = metrics.get("fitness", 0.0)
                self._training_progress.current_val_loss = val_loss
                self._training_progress.train_loss_history.append(total_loss)
                self._training_progress.val_loss_history.append(val_loss)

            model_save_dir = project_path / "models"
            model_save_dir.mkdir(exist_ok=True)

            # Load pretrained YOLO pose model
            model = YOLO("yolo11n-pose.pt")
            model.add_callback("on_fit_epoch_end", on_epoch_end)
            model.add_callback("on_train_epoch_start", check_stop)

            model.train(
                data=str(yaml_path),
                epochs=max_epochs,
                imgsz=640,
                batch=8,
                lr0=1e-4,
                optimizer="AdamW",
                patience=20,
                single_cls=True,
                device=0,
                workers=4,
                plots=False,
                project=str(model_save_dir),
                name="pose_train",
                exist_ok=True,
                verbose=False,
            )

            # Copy best weights
            best_pt = model_save_dir / "pose_train" / "weights" / "best.pt"
            if best_pt.exists():
                shutil.copy2(best_pt, model_save_dir / "pose.pt")
                logger.info(f"Best pose weights saved to {model_save_dir / 'pose.pt'}")
            else:
                logger.warning("best.pt not found after pose training")

            if self._stop_requested:
                self._training_progress.status = "stopped"
            else:
                self._training_progress.status = "completed"

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        self._training_progress.is_training = False

    def apply(
        self,
        project_path: Path,
        video_ids: list[int],
    ) -> bool:
        """Start pose inference in background thread."""
        if self._is_training:
            raise RuntimeError("Training or apply already in progress")

        self._is_training = True
        self._stop_requested = False

        def _apply_thread():
            try:
                self._apply_sync(project_path, video_ids)
            except Exception as e:
                logger.exception("Pose apply failed")
                self._training_progress.status = "failed"
                self._training_progress.error_message = str(e)
            finally:
                self._is_training = False

        self._training_thread = threading.Thread(target=_apply_thread, daemon=True)
        self._training_thread.start()
        return True

    def _apply_sync(
        self,
        project_path: Path,
        video_ids: list[int],
    ) -> None:
        """Apply pose model to videos."""
        from ultralytics import YOLO
        from vidseq.services import segmentation_service

        segmentation_service.shutdown()

        model_path = project_path / "models" / "pose.pt"
        if not model_path.exists():
            raise RuntimeError("No trained pose model found")

        model = YOLO(str(model_path))

        db_manager = DatabaseManager.get_instance()
        engine = db_manager.get_project_engine(project_path)

        with Session(engine) as session:
            videos = session.execute(
                select(Video).where(Video.id.in_(video_ids))
            ).scalars().all()

            # Count total frames for progress
            total_frames = sum(v.num_frames for v in videos)
            self._training_progress = PoseTrainingProgress(
                is_training=True,
                status="applying",
                started_at=time.time(),
                apply_total=total_frames,
                apply_current=0,
            )

            frames_done = 0
            for video in videos:
                video_path = Path(video.filepath)
                if not video_path.exists():
                    logger.warning(f"Video not found: {video_path}")
                    frames_done += video.num_frames
                    continue

                # Create/recreate alignment_keypoints.h5 with new shape
                create_alignment_keypoints_array(project_path, video.id, video.num_frames)

                cap = cv2.VideoCapture(str(video_path))
                batch_updates: list[dict] = []

                with alignment_keypoints(project_path, video.id, "a") as kp_data:
                    for frame_idx in range(video.num_frames):
                        if self._stop_requested:
                            break

                        ret, frame = cap.read()
                        if not ret:
                            break

                        # Run inference
                        results = model(frame, verbose=False)
                        result = results[0]

                        if result.keypoints is not None and len(result.keypoints.xy) > 0:
                            # Take best detection (highest confidence)
                            kps = result.keypoints.xy[0].cpu().numpy()  # (2, 2)
                            confs = result.keypoints.conf[0].cpu().numpy() if result.keypoints.conf is not None else np.ones(2)
                            box_conf = float(result.boxes.conf[0].cpu()) if result.boxes is not None and len(result.boxes) > 0 else 0.0

                            img_h, img_w = frame.shape[:2]
                            fx = float(kps[0, 0]) / img_w
                            fy = float(kps[0, 1]) / img_h
                            rx = float(kps[1, 0]) / img_w
                            ry = float(kps[1, 1]) / img_h

                            # Write to H5
                            kp_data[frame_idx] = [fx, fy, rx, ry]

                            batch_updates.append({
                                "video_id": video.id,
                                "frame_idx": frame_idx,
                                "pose_front_x": fx,
                                "pose_front_y": fy,
                                "pose_rear_x": rx,
                                "pose_rear_y": ry,
                                "pose_score": box_conf,
                                "has_pose": 1,
                            })
                        else:
                            batch_updates.append({
                                "video_id": video.id,
                                "frame_idx": frame_idx,
                                "pose_score": -1.0,
                                "has_pose": 0,
                            })

                        frames_done += 1
                        self._training_progress.apply_current = frames_done

                cap.release()

                # Batch upsert to DB
                self._batch_upsert_pose(session, batch_updates)

            if self._stop_requested:
                self._training_progress.status = "stopped"
            else:
                self._training_progress.status = "completed"

        self._training_progress.is_training = False

    def _batch_upsert_pose(self, session: Session, updates: list[dict]) -> None:
        """Batch upsert pose predictions to FrameData."""
        from sqlalchemy.dialects.sqlite import insert

        CHUNK_SIZE = 500
        for i in range(0, len(updates), CHUNK_SIZE):
            chunk = updates[i:i + CHUNK_SIZE]
            stmt = insert(FrameData).values(chunk)
            stmt = stmt.on_conflict_do_update(
                index_elements=["video_id", "frame_idx"],
                set_={
                    "pose_front_x": stmt.excluded.pose_front_x,
                    "pose_front_y": stmt.excluded.pose_front_y,
                    "pose_rear_x": stmt.excluded.pose_rear_x,
                    "pose_rear_y": stmt.excluded.pose_rear_y,
                    "pose_score": stmt.excluded.pose_score,
                    "has_pose": stmt.excluded.has_pose,
                },
            )
            session.execute(stmt)
        session.commit()
```

- [ ] **Step 3: Commit**

```bash
git add vidseq/schemas/pose.py vidseq/services/pose_service.py
git commit -m "feat: add pose service with YOLO pose training and inference"
```

---

### Task 5: Create Pose API Routes

**Files:**
- Create: `vidseq/api/routes/pose.py`
- Modify: `vidseq/server.py:54`

- [ ] **Step 1: Create pose routes**

Create `vidseq/api/routes/pose.py`:

```python
"""API routes for keypoint pose estimation."""
import asyncio
import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from vidseq.api.dependencies import get_project_db, get_project_folder, get_video
from vidseq.models.pose_label import PoseLabel
from vidseq.models.video import Video
from vidseq.services.pose_service import PoseService

router = APIRouter()


# --- Schemas ---

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


class VideoSelectionRequest(BaseModel):
    video_ids: list[int]


# --- Label Routes ---

@router.get("/projects/{project_id}/videos/{video_id}/pose/labels")
async def get_pose_labels(
    video_id: int,
    session: AsyncSession = Depends(get_project_db),
):
    """List all pose labels for a video."""
    result = await session.execute(
        select(PoseLabel)
        .where(PoseLabel.video_id == video_id)
        .order_by(PoseLabel.frame_idx)
    )
    labels = result.scalars().all()
    return [
        PoseLabelResponse(
            frame_idx=l.frame_idx,
            front_x=l.front_x,
            front_y=l.front_y,
            rear_x=l.rear_x,
            rear_y=l.rear_y,
        )
        for l in labels
    ]


@router.post("/projects/{project_id}/videos/{video_id}/pose/labels/{frame_idx}")
async def save_pose_label(
    video_id: int,
    frame_idx: int,
    body: PoseLabelRequest,
    session: AsyncSession = Depends(get_project_db),
):
    """Save or update a pose label for a frame."""
    # Upsert: delete existing then insert
    await session.execute(
        delete(PoseLabel)
        .where(PoseLabel.video_id == video_id)
        .where(PoseLabel.frame_idx == frame_idx)
    )
    label = PoseLabel(
        video_id=video_id,
        frame_idx=frame_idx,
        front_x=body.front_x,
        front_y=body.front_y,
        rear_x=body.rear_x,
        rear_y=body.rear_y,
    )
    session.add(label)
    await session.commit()
    return {"status": "ok"}


@router.delete(
    "/projects/{project_id}/videos/{video_id}/pose/labels/{frame_idx}",
    status_code=204,
)
async def delete_pose_label(
    video_id: int,
    frame_idx: int,
    session: AsyncSession = Depends(get_project_db),
):
    """Delete a pose label for a frame."""
    await session.execute(
        delete(PoseLabel)
        .where(PoseLabel.video_id == video_id)
        .where(PoseLabel.frame_idx == frame_idx)
    )
    await session.commit()
    return None


@router.get("/projects/{project_id}/videos/{video_id}/pose/label-count")
async def get_pose_label_count(
    video_id: int,
    session: AsyncSession = Depends(get_project_db),
):
    """Get count of labeled frames for a video."""
    result = await session.execute(
        select(func.count())
        .select_from(PoseLabel)
        .where(PoseLabel.video_id == video_id)
    )
    count = result.scalar() or 0
    return {"count": count}


# --- Training Routes ---

@router.get("/projects/{project_id}/pose/status", response_model=PoseStatusResponse)
async def get_pose_status(
    project_path=Depends(get_project_folder),
):
    """Get pose model status."""
    service = PoseService.get_instance()
    return PoseStatusResponse(
        model_exists=service.model_exists(project_path),
        is_training=service.is_training(),
    )


@router.post("/projects/{project_id}/pose/training")
async def create_pose_training(
    request: PoseTrainingRequest,
    project_path=Depends(get_project_folder),
):
    """Start pose model training."""
    service = PoseService.get_instance()
    if service.is_training():
        raise HTTPException(status_code=409, detail="Training already in progress")

    try:
        service.train(
            project_path=project_path,
            video_ids=request.video_ids,
            max_epochs=request.max_epochs,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"status": "started"}


@router.delete("/projects/{project_id}/pose/training", status_code=204)
async def delete_pose_training():
    """Stop pose training or apply."""
    service = PoseService.get_instance()
    if not service.is_training():
        raise HTTPException(status_code=400, detail="No training or apply in progress")
    service.stop_training()
    return None


@router.get("/projects/{project_id}/pose/training")
async def get_pose_training():
    """Get current training progress."""
    service = PoseService.get_instance()
    progress = service.get_training_progress()
    return progress.to_dict()


@router.get("/projects/{project_id}/pose/training/stream")
async def stream_pose_training():
    """SSE stream for real-time training/apply updates."""

    async def event_generator():
        service = PoseService.get_instance()
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
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


# --- Apply Routes ---

@router.post("/projects/{project_id}/videos/pose")
async def apply_pose(
    request: VideoSelectionRequest,
    project_path=Depends(get_project_folder),
):
    """Apply pose model to selected videos."""
    service = PoseService.get_instance()
    if service.is_training():
        raise HTTPException(status_code=409, detail="Training or apply already in progress")

    try:
        service.apply(
            project_path=project_path,
            video_ids=request.video_ids,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"status": "started"}


# --- Prediction Routes ---

@router.get("/projects/{project_id}/videos/{video_id}/pose/prediction/{frame_idx}")
async def get_pose_prediction(
    video_id: int,
    frame_idx: int,
    session: AsyncSession = Depends(get_project_db),
):
    """Get pose prediction for a single frame."""
    from vidseq.models.frame_data import FrameData

    result = await session.execute(
        select(FrameData)
        .where(FrameData.video_id == video_id)
        .where(FrameData.frame_idx == frame_idx)
    )
    fd = result.scalar_one_or_none()
    if fd is None or fd.has_pose != 1:
        return {"prediction": None}

    return {
        "prediction": {
            "front_x": fd.pose_front_x,
            "front_y": fd.pose_front_y,
            "rear_x": fd.pose_rear_x,
            "rear_y": fd.pose_rear_y,
            "score": fd.pose_score,
        }
    }


# --- Score Routes ---

@router.get("/projects/{project_id}/videos/{video_id}/pose-scores-downsampled")
async def get_pose_scores_downsampled(
    video_id: int,
    max_samples: int = 800,
    start_frame: int | None = None,
    end_frame: int | None = None,
    session: AsyncSession = Depends(get_project_db),
):
    """Get downsampled pose confidence scores for DataTrack visualization."""
    from vidseq.services.frame_data_service import get_scores_downsampled

    scores = await get_scores_downsampled(
        session, video_id, "pose_score", max_samples, start_frame, end_frame
    )
    return {"scores": scores}
```

**Note:** The `get_scores_downsampled` function in `frame_data_service.py` may need to be made generic enough to accept a column name parameter. If it currently only works with specific score columns, add a simple function that queries `FrameData.pose_score` with the same LTTB downsampling pattern used by detector scores. Check the existing implementation and adapt.

- [ ] **Step 2: Register the router in server.py**

In `vidseq/server.py`, add after the alignment import and router registration:

```python
from vidseq.api.routes import pose
# ...
app.include_router(pose.router, prefix="/api", tags=["pose"])
```

Replace the alignment router line (line 54):
```python
# Remove: app.include_router(alignment.router, prefix="/api", tags=["alignment"])
app.include_router(pose.router, prefix="/api", tags=["pose"])
```

- [ ] **Step 3: Commit**

```bash
git add vidseq/api/routes/pose.py vidseq/server.py
git commit -m "feat: add pose API routes for labels, training, and inference"
```

---

### Task 6: Clean Up Alignment System

**Files:**
- Modify: `vidseq/services/alignment_service.py` (extract rotation, delete rest)
- Delete: `vidseq/api/routes/alignment.py`
- Modify: `vidseq/server.py`
- Modify: `frontend/src/router/index.ts`

- [ ] **Step 1: Extract rotation functions into standalone module**

Create `vidseq/services/frame_rotation.py` with the rotation functions extracted from `alignment_service.py`:

```python
"""Frame rotation utilities for egocentric alignment.

Reads keypoint coordinates from alignment_keypoints.h5 (shape: num_frames x 4),
computes heading angles, and rotates frames/masks to canonical orientation.
"""
import cv2
import numpy as np


def rotate_frame(frame: np.ndarray, angle_degrees: float) -> np.ndarray:
    """Rotate frame around center by given angle.

    Args:
        frame: (H, W, 3) BGR image
        angle_degrees: Rotation angle (positive = counterclockwise)

    Returns:
        Rotated frame (same dimensions)
    """
    h, w = frame.shape[:2]
    center = (w / 2, h / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle_degrees, scale=1.0)
    return cv2.warpAffine(frame, rotation_matrix, (w, h))


def rotate_mask(mask: np.ndarray, angle_degrees: float, threshold: int = 127) -> np.ndarray:
    """Rotate a binary mask around center and re-threshold to binary.

    Args:
        mask: (H, W) uint8 mask
        angle_degrees: Rotation angle (positive = counterclockwise)
        threshold: Threshold for re-binarization (default 127)

    Returns:
        Rotated binary mask (same dimensions)
    """
    h, w = mask.shape[:2]
    center = (w / 2, h / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle_degrees, scale=1.0)
    rotated = cv2.warpAffine(mask, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR)
    _, binary = cv2.threshold(rotated, threshold, 255, cv2.THRESH_BINARY)
    return binary


def compute_heading_angles(keypoints: np.ndarray) -> np.ndarray:
    """Compute heading angles from front/rear keypoint coordinates.

    Args:
        keypoints: (N, 4) array of [front_x, front_y, rear_x, rear_y] per frame

    Returns:
        (N,) array of angles in degrees
    """
    from scipy.signal import savgol_filter

    front_x = keypoints[:, 0]
    front_y = keypoints[:, 1]
    rear_x = keypoints[:, 2]
    rear_y = keypoints[:, 3]

    dx = front_x - rear_x
    dy = front_y - rear_y

    # Smooth with Savitzky-Golay filter
    window = min(11, len(dx) if len(dx) % 2 == 1 else len(dx) - 1)
    if window >= 5:
        dx = savgol_filter(dx, window, 3)
        dy = savgol_filter(dy, window, 3)

    return np.degrees(np.arctan2(dy, dx))
```

- [ ] **Step 2: Update alignment_service.py apply_alignment_sync to use new H5 format**

The `apply_alignment_sync` function in `alignment_service.py` needs to be updated to:
1. Read `alignment_keypoints.h5` with new `(N, 4)` shape (raw coordinates)
2. Use `compute_heading_angles()` from `frame_rotation.py` instead of the DINOv2 pass 1
3. Remove the DINOv2/decoder inference code from `apply_alignment_sync`

This is a significant refactor of `apply_alignment_sync`. The new flow becomes:
- Read precomputed keypoints from H5 (written by pose_service apply)
- Compute heading angles from keypoints
- Pass 2 only: rotate frames using heading angles

Delete everything in `alignment_service.py` except:
- `apply_alignment_sync` (refactored to use precomputed keypoints)
- `AlignmentProgress` dataclass
- Singleton pattern for managing the apply operation
- The `_reencode_to_h264` helper

Delete from `alignment_service.py`:
- All DINOv2 / decoder loading functions
- `train_model_sync` and all training code
- `AlignmentDataset` class
- `HeadingVectorDecoder` class
- All augmentation functions
- `_extract_features`, `_decode_features`, `predict_sync`

- [ ] **Step 3: Delete alignment routes**

Delete `vidseq/api/routes/alignment.py`.

Remove the alignment import and router registration from `vidseq/server.py` (the pose router was already added in Task 5).

- [ ] **Step 4: Clean up frontend router**

In `frontend/src/router/index.ts`, remove the Alignment import and route:

```typescript
// Remove: import Alignment from '../components/Alignment.vue'
// Remove the alignment route object from children array
```

Keep the `DetectorTraining` route — we'll add a pose training route in Task 9.

- [ ] **Step 5: Remove alignment API functions from api.ts**

Remove all the alignment-related functions from `frontend/src/services/api.ts` (lines ~770-1040 covering `getAlignmentStatus`, `getAlignmentRandomFrame`, `saveAlignmentLabel`, etc.).

- [ ] **Step 6: Commit**

```bash
git add vidseq/services/frame_rotation.py
git add vidseq/services/alignment_service.py
git rm vidseq/api/routes/alignment.py
git add vidseq/server.py
git add frontend/src/router/index.ts
git add frontend/src/services/api.ts
git commit -m "refactor: extract frame rotation, delete alignment training and routes"
```

---

### Task 7: Frontend API Functions

**Files:**
- Modify: `frontend/src/services/api.ts`

- [ ] **Step 1: Add pose label API functions**

Add to `frontend/src/services/api.ts`:

```typescript
// --- Pose Labels ---

export interface PoseLabel {
    frame_idx: number
    front_x: number
    front_y: number
    rear_x: number
    rear_y: number
}

export async function getPoseLabels(projectId: number, videoId: number): Promise<PoseLabel[]> {
    const response = await fetch(`${API_BASE}/projects/${projectId}/videos/${videoId}/pose/labels`)
    if (!response.ok) throw new Error(await getErrorMessage(response, 'Failed to fetch pose labels'))
    return response.json()
}

export async function savePoseLabel(
    projectId: number,
    videoId: number,
    frameIdx: number,
    frontX: number,
    frontY: number,
    rearX: number,
    rearY: number,
): Promise<void> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/${videoId}/pose/labels/${frameIdx}`,
        {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ front_x: frontX, front_y: frontY, rear_x: rearX, rear_y: rearY }),
        },
    )
    if (!response.ok) throw new Error(await getErrorMessage(response, 'Failed to save pose label'))
}

export async function deletePoseLabel(projectId: number, videoId: number, frameIdx: number): Promise<void> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/${videoId}/pose/labels/${frameIdx}`,
        { method: 'DELETE' },
    )
    if (!response.ok) throw new Error(await getErrorMessage(response, 'Failed to delete pose label'))
}

export async function getPoseLabelCount(projectId: number, videoId: number): Promise<number> {
    const response = await fetch(`${API_BASE}/projects/${projectId}/videos/${videoId}/pose/label-count`)
    if (!response.ok) throw new Error(await getErrorMessage(response, 'Failed to fetch pose label count'))
    const data = await response.json()
    return data.count
}

// --- Pose Training ---

export interface PoseStatus {
    model_exists: boolean
    is_training: boolean
}

export async function getPoseStatus(projectId: number): Promise<PoseStatus> {
    const response = await fetch(`${API_BASE}/projects/${projectId}/pose/status`)
    if (!response.ok) throw new Error(await getErrorMessage(response, 'Failed to fetch pose status'))
    return response.json()
}

export async function createPoseTraining(
    projectId: number,
    maxEpochs: number,
    videoIds: number[],
): Promise<void> {
    const response = await fetch(`${API_BASE}/projects/${projectId}/pose/training`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ video_ids: videoIds, max_epochs: maxEpochs }),
    })
    if (!response.ok) throw new Error(await getErrorMessage(response, 'Failed to start pose training'))
}

export async function deletePoseTraining(projectId: number): Promise<void> {
    const response = await fetch(`${API_BASE}/projects/${projectId}/pose/training`, { method: 'DELETE' })
    if (!response.ok) throw new Error(await getErrorMessage(response, 'Failed to stop pose training'))
}

export async function getPoseTraining(projectId: number): Promise<DetectorTrainingProgress> {
    const response = await fetch(`${API_BASE}/projects/${projectId}/pose/training`)
    if (!response.ok) throw new Error(await getErrorMessage(response, 'Failed to get pose training status'))
    return response.json()
}

export function connectPoseTrainingStream(projectId: number): EventSource {
    return new EventSource(`${API_BASE}/projects/${projectId}/pose/training/stream`)
}

export async function applyPose(projectId: number, videoIds: number[]): Promise<void> {
    const response = await fetch(`${API_BASE}/projects/${projectId}/videos/pose`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ video_ids: videoIds }),
    })
    if (!response.ok) throw new Error(await getErrorMessage(response, 'Failed to apply pose model'))
}

export interface PosePrediction {
    front_x: number
    front_y: number
    rear_x: number
    rear_y: number
    score: number
}

export async function getPosePrediction(
    projectId: number,
    videoId: number,
    frameIdx: number,
): Promise<PosePrediction | null> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/${videoId}/pose/prediction/${frameIdx}`,
    )
    if (!response.ok) return null
    const data = await response.json()
    return data.prediction
}

export async function getPoseScoresDownsampled(
    projectId: number,
    videoId: number,
    maxSamples: number = 800,
    startFrame?: number,
    endFrame?: number,
): Promise<{ scores: MaskScore[] }> {
    const params = new URLSearchParams({ max_samples: maxSamples.toString() })
    if (startFrame !== undefined) params.set('start_frame', startFrame.toString())
    if (endFrame !== undefined) params.set('end_frame', endFrame.toString())
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/${videoId}/pose-scores-downsampled?${params}`,
    )
    if (!response.ok) throw new Error(await getErrorMessage(response, 'Failed to fetch pose scores'))
    return response.json()
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/services/api.ts
git commit -m "feat: add pose API functions to frontend"
```

---

### Task 8: Frontend Composables

**Files:**
- Create: `frontend/src/composables/usePoseLabels.ts`
- Create: `frontend/src/composables/usePoseDetector.ts`

- [ ] **Step 1: Create usePoseLabels composable**

Create `frontend/src/composables/usePoseLabels.ts`:

```typescript
import { ref, type Ref } from 'vue'
import {
    getPoseLabels,
    savePoseLabel,
    deletePoseLabel,
    getPoseLabelCount,
    type PoseLabel,
} from '@/services/api'

export type LabelingState = 'idle' | 'awaiting_front' | 'awaiting_rear'

export interface UsePoseLabelsReturn {
    currentLabel: Ref<PoseLabel | null>
    labeledFrameCount: Ref<number>
    labelingState: Ref<LabelingState>
    pendingFront: Ref<{ x: number; y: number } | null>
    loadLabel: (frameIdx: number) => Promise<void>
    saveLabel: (frameIdx: number, frontX: number, frontY: number, rearX: number, rearY: number) => Promise<void>
    deleteLabel: (frameIdx: number) => Promise<void>
    refresh: () => Promise<void>
    startLabeling: () => void
    stopLabeling: () => void
    handleClick: (x: number, y: number, frameIdx: number, advanceFrame: () => void) => Promise<void>
}

export function usePoseLabels(
    projectId: Ref<number>,
    videoId: Ref<number>,
): UsePoseLabelsReturn {
    const currentLabel = ref<PoseLabel | null>(null)
    const labeledFrameCount = ref(0)
    const labelingState = ref<LabelingState>('idle')
    const pendingFront = ref<{ x: number; y: number } | null>(null)

    // Cache: frame_idx -> PoseLabel
    const labelCache = new Map<number, PoseLabel | null>()

    const loadLabel = async (frameIdx: number) => {
        if (labelCache.has(frameIdx)) {
            currentLabel.value = labelCache.get(frameIdx) ?? null
            return
        }
        // Labels are fetched in bulk, so check if cache is populated
        currentLabel.value = null
    }

    const refresh = async () => {
        if (!projectId.value || !videoId.value) return
        try {
            // Fetch all labels for this video and cache them
            const labels = await getPoseLabels(projectId.value, videoId.value)
            labelCache.clear()
            for (const label of labels) {
                labelCache.set(label.frame_idx, label)
            }
            labeledFrameCount.value = labels.length
        } catch (e) {
            console.error('Failed to refresh pose labels:', e)
        }
    }

    const saveLabel = async (
        frameIdx: number,
        frontX: number,
        frontY: number,
        rearX: number,
        rearY: number,
    ) => {
        await savePoseLabel(projectId.value, videoId.value, frameIdx, frontX, frontY, rearX, rearY)
        const label: PoseLabel = { frame_idx: frameIdx, front_x: frontX, front_y: frontY, rear_x: rearX, rear_y: rearY }
        labelCache.set(frameIdx, label)
        currentLabel.value = label
        labeledFrameCount.value = labelCache.size
    }

    const deleteLabel = async (frameIdx: number) => {
        await deletePoseLabel(projectId.value, videoId.value, frameIdx)
        labelCache.delete(frameIdx)
        currentLabel.value = null
        labeledFrameCount.value = labelCache.size
    }

    const startLabeling = () => {
        labelingState.value = 'awaiting_front'
        pendingFront.value = null
    }

    const stopLabeling = () => {
        labelingState.value = 'idle'
        pendingFront.value = null
    }

    const handleClick = async (x: number, y: number, frameIdx: number, advanceFrame: () => void) => {
        if (labelingState.value === 'awaiting_front') {
            pendingFront.value = { x, y }
            labelingState.value = 'awaiting_rear'
        } else if (labelingState.value === 'awaiting_rear' && pendingFront.value) {
            const front = pendingFront.value
            await saveLabel(frameIdx, front.x, front.y, x, y)
            pendingFront.value = null
            labelingState.value = 'awaiting_front'
            advanceFrame()
        }
    }

    return {
        currentLabel,
        labeledFrameCount,
        labelingState,
        pendingFront,
        loadLabel,
        saveLabel,
        deleteLabel,
        refresh,
        startLabeling,
        stopLabeling,
        handleClick,
    }
}
```

- [ ] **Step 2: Create usePoseDetector composable**

Create `frontend/src/composables/usePoseDetector.ts`:

```typescript
import { ref, watch, onUnmounted, type Ref } from 'vue'
import {
    getPoseStatus,
    createPoseTraining,
    deletePoseTraining,
} from '@/services/api'

export interface UsePoseDetectorReturn {
    isTraining: Ref<boolean>
    modelExists: Ref<boolean>
    startTraining: (maxEpochs?: number, videoIds?: number[]) => Promise<void>
    stopTraining: () => Promise<void>
    checkStatus: () => Promise<void>
}

export function usePoseDetector(projectId: Ref<number | null>): UsePoseDetectorReturn {
    const isTraining = ref(false)
    const modelExists = ref(false)

    const checkStatus = async () => {
        if (!projectId.value) return
        try {
            const status = await getPoseStatus(projectId.value)
            modelExists.value = status.model_exists
            isTraining.value = status.is_training
        } catch (e) {
            console.error('Failed to check pose status:', e)
        }
    }

    const startTraining = async (maxEpochs: number = 300, videoIds: number[] = []) => {
        if (!projectId.value || isTraining.value) return
        isTraining.value = true
        try {
            await createPoseTraining(projectId.value, maxEpochs, videoIds)
        } catch (e) {
            console.error('Failed to start pose training:', e)
            isTraining.value = false
            throw e
        }
    }

    const stopTraining = async () => {
        if (!projectId.value) return
        try {
            await deletePoseTraining(projectId.value)
        } catch (e) {
            console.error('Failed to stop pose training:', e)
            throw e
        }
    }

    watch(projectId, async (newId) => {
        if (newId !== null) {
            await checkStatus()
        }
    }, { immediate: true })

    // Poll while training
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
git add frontend/src/composables/usePoseLabels.ts frontend/src/composables/usePoseDetector.ts
git commit -m "feat: add usePoseLabels and usePoseDetector composables"
```

---

### Task 9: Frontend Annotation UI — VideoOverlay

**Files:**
- Modify: `frontend/src/components/VideoOverlay.vue`

- [ ] **Step 1: Add keypoint props and emit**

In `VideoOverlay.vue`, extend the props and emits:

```typescript
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
  // Pose keypoints
  poseLabel?: { front_x: number; front_y: number; rear_x: number; rear_y: number } | null
  posePrediction?: { front_x: number; front_y: number; rear_x: number; rear_y: number } | null
  pendingFront?: { x: number; y: number } | null
  showPoseKeypoints?: boolean
  isLabelingKeypoints?: boolean
}>()

const emit = defineEmits<{
  (e: 'point-complete', point: { x: number; y: number; type: 'positive_point' | 'negative_point' }): void
  (e: 'box-complete', box: { x1: number; y1: number; x2: number; y2: number }): void
  (e: 'keypoint-click', point: { x: number; y: number }): void
}>()
```

- [ ] **Step 2: Handle keypoint clicks in onMouseDown**

Add keypoint click handling at the beginning of `onMouseDown`, before the existing tool checks:

```typescript
function onMouseDown(event: MouseEvent) {
  const coords = getNormalizedCoords(event)
  if (!coords) return

  if (props.isLabelingKeypoints) {
    emit('keypoint-click', { x: coords.x, y: coords.y })
    return
  }

  // ... existing tool handling
}
```

- [ ] **Step 3: Add keypoint rendering to render()**

Add after the OBB drawing (after line 135) but before the prompts loop (line 137):

```typescript
  // Draw pose keypoints
  if (props.showPoseKeypoints !== false) {
    const drawKeypoint = (x: number, y: number, color: string, filled: boolean) => {
      const px = x * canvas.width
      const py = y * canvas.height
      const radius = 8
      ctx.beginPath()
      ctx.arc(px, py, radius, 0, Math.PI * 2)
      if (filled) {
        ctx.fillStyle = color
        ctx.fill()
      }
      ctx.strokeStyle = color
      ctx.lineWidth = 2
      ctx.setLineDash([])
      ctx.stroke()
    }

    // Draw hand-labeled keypoints (filled circles)
    if (props.poseLabel) {
      drawKeypoint(props.poseLabel.front_x, props.poseLabel.front_y, '#22c55e', true)
      drawKeypoint(props.poseLabel.rear_x, props.poseLabel.rear_y, '#ef4444', true)
    }

    // Draw predicted keypoints (outline circles, only if no label exists)
    if (props.posePrediction && !props.poseLabel) {
      drawKeypoint(props.posePrediction.front_x, props.posePrediction.front_y, '#22c55e', false)
      drawKeypoint(props.posePrediction.rear_x, props.posePrediction.rear_y, '#ef4444', false)
    }

    // Draw pending front point during labeling
    if (props.pendingFront) {
      drawKeypoint(props.pendingFront.x, props.pendingFront.y, '#22c55e', true)
    }
  }
```

- [ ] **Step 4: Update the tool-active class to include keypoint mode**

Update the template class binding:

```vue
:class="{ 'tool-active': activeTool !== 'none' || isLabelingKeypoints }"
```

- [ ] **Step 5: Update the watch to include new props**

Update the watch (line 226):

```typescript
watch(() => [props.mask, props.prompts, props.detectorBbox, props.obbBbox, props.showMask, props.showPrompts, props.poseLabel, props.posePrediction, props.pendingFront, props.showPoseKeypoints], () => {
  pendingPoint.value = null
  pendingBox.value = null
  render()
}, { deep: true })
```

- [ ] **Step 6: Commit**

```bash
git add frontend/src/components/VideoOverlay.vue
git commit -m "feat: add keypoint rendering and click handling to VideoOverlay"
```

---

### Task 10: Frontend Annotation UI — VideoDetail

**Files:**
- Modify: `frontend/src/components/VideoDetail.vue`

- [ ] **Step 1: Import and initialize usePoseLabels**

Add to the imports at the top of VideoDetail.vue:

```typescript
import { usePoseLabels } from '@/composables/usePoseLabels'
```

Add after the existing composable initializations:

```typescript
const {
    currentLabel: poseLabel,
    labeledFrameCount: poseLabelCount,
    labelingState,
    pendingFront,
    loadLabel: loadPoseLabel,
    deleteLabel: deletePoseLabel,
    refresh: refreshPoseLabels,
    startLabeling,
    stopLabeling,
    handleClick: handleKeypointClick,
} = usePoseLabels(projectId, videoId)

const isLabelingKeypoints = ref(false)

const toggleLabelingMode = () => {
    isLabelingKeypoints.value = !isLabelingKeypoints.value
    if (isLabelingKeypoints.value) {
        startLabeling()
    } else {
        stopLabeling()
    }
}

const showPoseKeypoints = ref(true)
const posePrediction = ref<PosePrediction | null>(null)
```

Import `getPosePrediction` and `type PosePrediction` from api.ts.

- [ ] **Step 2: Add frame advance and keypoint event handler**

Add an `advanceFrame` function and connect the keypoint click handler:

```typescript
const advanceFrame = () => {
    if (!video.value) return
    const nextFrame = currentFrameIdx.value + 1
    if (nextFrame < video.value.num_frames) {
        seekToFrame(nextFrame)
    }
}

const onKeypointClick = async (point: { x: number; y: number }) => {
    await handleKeypointClick(point.x, point.y, currentFrameIdx.value, advanceFrame)
}
```

- [ ] **Step 3: Load pose label when frame changes**

Add a watcher that loads the pose label when the frame changes. Find the existing frame-change watcher pattern and add:

```typescript
watch(currentFrameIdx, async (frameIdx) => {
    if (frameIdx !== undefined) {
        await loadPoseLabel(frameIdx)
        // Load prediction for this frame
        try {
            posePrediction.value = await getPosePrediction(projectId.value, videoId.value, frameIdx)
        } catch {
            posePrediction.value = null
        }
    }
})
```

Also call `refreshPoseLabels()` on mount.

- [ ] **Step 4: Handle Delete key for pose labels**

Add a keyboard handler for Delete key when in labeling mode. Add to existing keyboard handling or create one:

```typescript
const handleKeyDown = (e: KeyboardEvent) => {
    if ((e.key === 'Delete' || e.key === 'Backspace') && isLabelingKeypoints.value && poseLabel.value) {
        deletePoseLabel(currentFrameIdx.value)
    }
}
```

Register/unregister in onMounted/onUnmounted.

- [ ] **Step 5: Pass props to VideoOverlay**

Update the `<VideoOverlay>` component in the template to pass the new props:

```vue
<VideoOverlay
  :video-width="videoWidth"
  :video-height="videoHeight"
  :active-tool="activeTool"
  :mask="currentMask"
  :prompts="currentPrompts"
  :show-mask="showMask"
  :show-prompts="showPrompts"
  :detector-bbox="detectorBbox"
  :obb-bbox="obbBbox"
  :pose-label="poseLabel"
  :pose-prediction="posePrediction"
  :pending-front="pendingFront"
  :show-pose-keypoints="showPoseKeypoints"
  :is-labeling-keypoints="isLabelingKeypoints"
  @point-complete="handlePointComplete"
  @box-complete="handleBoxComplete"
  @keypoint-click="onKeypointClick"
/>
```

- [ ] **Step 6: Add UI controls to action bar**

After the "Training Data" section (line 548), add a new section:

```vue
<h4 class="action-bar-title">Pose Keypoints</h4>
<div class="tool-buttons">
  <button
    class="tool-button"
    :class="{ active: isLabelingKeypoints }"
    @click="toggleLabelingMode"
    :disabled="isSegmenting || isPropagating"
  >
    <span class="tool-icon">+</span>
    <span class="tool-label">{{ isLabelingKeypoints ? 'Exit Labeling' : 'Label Keypoints' }}</span>
  </button>
</div>
<p v-if="isLabelingKeypoints" class="marking-hint">
  {{ labelingState === 'awaiting_front' ? 'Click front (nose)' : 'Click rear (tail)' }}
</p>
<p v-if="poseLabelCount > 0" class="marking-hint">
  {{ poseLabelCount }} frame(s) labeled
</p>
```

Also add the show/hide toggle in the "Data Track" section (after OBB confidence toggle):

```vue
<button
  class="tool-button toggle-button pose-confidence-toggle"
  :class="{ active: showPoseKeypoints }"
  @click="showPoseKeypoints = !showPoseKeypoints"
>
  <span class="tool-icon">*</span>
  <span class="tool-label">{{ showPoseKeypoints ? 'Pose Keypoints' : 'Pose Keypoints Off' }}</span>
</button>
```

- [ ] **Step 7: Commit**

```bash
git add frontend/src/components/VideoDetail.vue
git commit -m "feat: add keypoint labeling mode to VideoDetail"
```

---

### Task 11: Frontend Pipeline & Training Progress

**Files:**
- Modify: `frontend/src/components/VideoPipeline.vue`
- Create: `frontend/src/components/PoseTraining.vue`
- Modify: `frontend/src/router/index.ts`

- [ ] **Step 1: Add pose section to VideoPipeline**

In `VideoPipeline.vue`, import and use the pose composable:

```typescript
import { usePoseDetector } from '@/composables/usePoseDetector'

const {
    isTraining: isPoseTraining,
    modelExists: poseModelExists,
    startTraining: startPoseTraining,
    stopTraining: stopPoseTraining,
    checkStatus: checkPoseStatus,
} = usePoseDetector(projectId)
```

Add handler functions:

```typescript
const handleTrainPose = async () => {
    if (!projectId.value || isPoseTraining.value) return
    try {
        await startPoseTraining(300, selectedVideoIdsList.value)
        router.push(`/project/${projectId.value}/pose`)
    } catch (e: any) {
        alert(e.message || 'Failed to start pose training')
    }
}

const handleApplyPose = async () => {
    if (!projectId.value) return
    try {
        await applyPose(projectId.value, selectedVideoIdsList.value)
        router.push(`/project/${projectId.value}/pose`)
    } catch (e: any) {
        alert(e.message || 'Failed to apply pose model')
    }
}
```

Add UI section in the template (after the detector/OBB/seg buttons):

```vue
<h4>Pose Model</h4>
<div class="pipeline-buttons">
  <button
    class="pipeline-button"
    @click="handleTrainPose"
    :disabled="isPoseTraining || selectedVideoIds.size === 0"
  >
    Train Pose
  </button>
  <button
    class="pipeline-button"
    @click="handleApplyPose"
    :disabled="isPoseTraining || !poseModelExists || selectedVideoIds.size === 0"
  >
    Apply Pose
  </button>
</div>
```

Import `applyPose` from api.ts.

- [ ] **Step 2: Create PoseTraining.vue**

Create `frontend/src/components/PoseTraining.vue`. This mirrors `DetectorTraining.vue` exactly but connects to the pose SSE stream:

```typescript
// Same structure as DetectorTraining.vue, but:
// - Uses connectPoseTrainingStream instead of connectDetectorTrainingStream
// - Uses deletePoseTraining instead of deleteDetectionTraining
// - Title says "Pose Training" instead of "Detector Training"
```

Copy `DetectorTraining.vue` and change:
1. Import `connectPoseTrainingStream` and `deletePoseTraining` from api
2. Replace the stream connection call
3. Replace the stop training call
4. Update the title text

- [ ] **Step 3: Add route**

In `frontend/src/router/index.ts`, add the pose training route:

```typescript
import PoseTraining from '../components/PoseTraining.vue'

// Add to children array:
{
    path: 'pose',
    name: 'pose',
    component: PoseTraining
}
```

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/VideoPipeline.vue
git add frontend/src/components/PoseTraining.vue
git add frontend/src/router/index.ts
git commit -m "feat: add pose training/apply to VideoPipeline with training progress page"
```

---

### Task 12: Frontend DataTrack — Pose Confidence Line

**Files:**
- Modify: `frontend/src/components/DataTrack.vue`
- Modify: `frontend/src/components/VideoDetail.vue`

- [ ] **Step 1: Add pose confidence props to DataTrack**

In `DataTrack.vue`, add to the props interface (around line 20):

```typescript
  showPoseConfidence?: boolean
  poseScores?: { frame_idx: number; score: number }[]
```

Add defaults (around line 34):

```typescript
  showPoseConfidence: false,
  poseScores: () => [],
```

- [ ] **Step 2: Draw pose confidence line in drawPlot**

In the `drawPlot` function (around line 448, after OBB drawing):

```typescript
  // Draw pose confidence scores (magenta line)
  if (props.showPoseConfidence && props.poseScores && props.poseScores.length > 0) {
    const validScores = props.poseScores.filter(s => s.score >= 0)
    if (validScores.length > 0) {
      const poseRange = drawScoreLine(ctx, validScores, 'rgba(192, 38, 211, 0.8)', width, height)
      if (poseRange && !activeRange) activeRange = poseRange
    }
  }
```

Add a watch trigger for the new prop:

```typescript
watch(() => props.showPoseConfidence, drawPlot)
watch(() => props.poseScores, drawPlot, { deep: true })
```

- [ ] **Step 3: Wire up in VideoDetail**

In `VideoDetail.vue`, add state and data fetching for pose scores:

```typescript
const showPoseConfidence = ref(true)
const poseScores = ref<MaskScore[]>([])

const fetchPoseScores = useDebounceFn(async () => {
    if (!projectId.value || !videoId.value) return
    try {
        const response = await getPoseScoresDownsampled(
            projectId.value, videoId.value, 800,
            Math.floor(viewStart.value * video.value!.fps),
            Math.floor(viewEnd.value * video.value!.fps),
        )
        poseScores.value = response.scores
    } catch {
        poseScores.value = []
    }
}, 150)
```

Import `getPoseScoresDownsampled` from api.ts.

Pass to `DataTrack`:

```vue
:show-pose-confidence="showPoseConfidence"
:pose-scores="poseScores"
```

Add a toggle button in the Data Track section of the action bar:

```vue
<button
  class="tool-button toggle-button"
  :class="{ active: showPoseConfidence }"
  @click="showPoseConfidence = !showPoseConfidence"
>
  <span class="tool-icon">*</span>
  <span class="tool-label">{{ showPoseConfidence ? 'Pose Confidence' : 'Pose Confidence Off' }}</span>
</button>
```

Call `fetchPoseScores()` when the view changes (add to existing view-change handler).

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/DataTrack.vue frontend/src/components/VideoDetail.vue
git commit -m "feat: add pose confidence line to DataTrack"
```

---

### Task 13: Verify End-to-End

- [ ] **Step 1: Start backend**

```bash
vidseq
```

Verify it starts without import errors.

- [ ] **Step 2: Start frontend**

```bash
cd frontend && npm run dev
```

Verify it compiles without TypeScript errors.

- [ ] **Step 3: Test labeling flow**

Open a video in VideoDetail, click "Label Keypoints", click front then rear on a frame. Verify:
- Green dot appears on first click
- Red dot appears on second click
- Frame auto-advances
- Label persists when navigating back

- [ ] **Step 4: Test training flow**

Go to VideoPipeline, select videos with labels, click "Train Pose". Verify:
- Navigates to pose training page
- SSE stream shows epoch/loss progress
- Training completes and model is saved

- [ ] **Step 5: Test apply flow**

Click "Apply Pose" on videos. Verify:
- Progress updates via SSE
- Keypoint dots appear on frames in VideoDetail
- Pose confidence line appears in DataTrack

- [ ] **Step 6: Final commit**

```bash
git add -A
git commit -m "feat: keypoint pose estimation — complete implementation"
```
