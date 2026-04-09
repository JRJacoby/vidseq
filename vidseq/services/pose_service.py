"""Pose Service for YOLO pose training and inference."""

import logging
import random
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import yaml
from sqlalchemy import select
from sqlalchemy.orm import Session

from vidseq.models.frame_data import FrameData
from vidseq.models.pose_label import PoseLabel
from vidseq.models.video import Video
from vidseq.schemas.pose import PoseTrainingProgress
from vidseq.services.array_storage import alignment_keypoints, create_alignment_keypoints_array
from vidseq.services.database_manager import DatabaseManager

logger = logging.getLogger("vidseq.pose")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "[%(asctime)s] [Pose] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

MIN_FRAMES_FOR_VALIDATION = 10
UPSERT_CHUNK_SIZE = 500


class PoseService:
    """Singleton service for YOLO pose training and inference."""

    _instance: Optional["PoseService"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "PoseService":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._is_training = False
        self._stop_requested = False
        self._training_thread: Optional[threading.Thread] = None
        self._training_progress = PoseTrainingProgress()

        self._initialized = True

    @classmethod
    def get_instance(cls) -> "PoseService":
        """Get the singleton instance."""
        return cls()

    def is_training(self) -> bool:
        """Check if training or applying is in progress."""
        return self._is_training

    def get_training_progress(self) -> PoseTrainingProgress:
        """Get current training progress."""
        return self._training_progress

    def stop_training(self) -> bool:
        """Request training to stop."""
        if self._is_training:
            self._stop_requested = True
            return True
        return False

    def model_exists(self, project_path: Path) -> bool:
        """Check if trained pose model exists."""
        return (project_path / "models" / "pose.pt").exists()

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------

    def train(
        self,
        project_path: Path,
        video_ids: list[int],
        max_epochs: int = 300,
    ) -> bool:
        """Start pose training in background thread.

        Returns immediately. Use get_training_progress() to monitor.
        """
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

    def _train_sync(
        self,
        project_path: Path,
        video_ids: list[int],
        max_epochs: int,
    ) -> None:
        """Synchronous pose training implementation using Ultralytics YOLO."""
        from ultralytics import YOLO
        from vidseq.services import segmentation_service

        # Free GPU memory
        segmentation_service.shutdown()

        logger.info(f"Starting YOLO pose training: max_epochs={max_epochs}")

        # Gather training data from PoseLabel table
        all_frames = self._gather_training_frames(project_path, video_ids)
        if len(all_frames) == 0:
            raise RuntimeError("No pose labels found. Annotate frames with keypoints first.")

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

            # Write frames and labels
            self._write_yolo_pose_dataset(train_frames, "train", tmp_path)
            if val_frames:
                self._write_yolo_pose_dataset(val_frames, "val", tmp_path)

            # Write dataset.yaml with kpt_shape for pose
            yaml_path = tmp_path / "dataset.yaml"
            dataset_config = {
                "path": str(tmp_path),
                "train": "images/train",
                "val": "images/val" if val_frames else "images/train",
                "nc": 1,
                "names": ["animal"],
                "kpt_shape": [2, 3],
            }
            with open(yaml_path, "w") as f:
                yaml.dump(dataset_config, f, default_flow_style=False)

            logger.info(
                f"YOLO pose dataset written: {len(train_frames)} train, "
                f"{len(val_frames)} val frames in {tmp_dir}"
            )

            # Callbacks
            def check_stop(trainer):
                if self._stop_requested:
                    raise KeyboardInterrupt("Pose training stopped by user")

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

            model = YOLO("yolo11n-pose.pt")
            model.add_callback("on_fit_epoch_end", on_epoch_end)
            model.add_callback("on_train_epoch_start", check_stop)

            model.train(
                data=str(yaml_path),
                epochs=max_epochs,
                imgsz=640,
                batch=8,
                optimizer="AdamW",
                patience=50,
                single_cls=True,
                device=0,
                workers=4,
                plots=False,
                project=str(model_save_dir),
                name="pose_train",
                exist_ok=True,
                verbose=False,
            )

            # Copy best weights to standard location
            best_pt = model_save_dir / "pose_train" / "weights" / "best.pt"
            dest_pt = model_save_dir / "pose.pt"
            if best_pt.exists():
                shutil.copy2(best_pt, dest_pt)
                logger.info(f"Best weights saved to {dest_pt}")
            else:
                logger.warning("best.pt not found after pose training")

            if self._stop_requested:
                self._training_progress.status = "stopped"
                logger.info("Pose training stopped by user")
            else:
                self._training_progress.status = "completed"

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        self._training_progress.is_training = False
        logger.info("Pose training completed")

    def _gather_training_frames(
        self,
        project_path: Path,
        video_ids: list[int],
    ) -> list[tuple[Path, int, int, float, float, float, float]]:
        """Gather all frames with PoseLabel entries.

        Returns:
            List of (video_path, video_id, frame_idx, front_x, front_y, rear_x, rear_y).
        """
        db_manager = DatabaseManager.get_instance()
        engine = db_manager.get_project_engine(project_path)

        frames = []
        with Session(engine) as session:
            videos = session.execute(
                select(Video).where(Video.id.in_(video_ids))
            ).scalars().all()
            video_map = {v.id: v for v in videos}

            pose_labels = session.execute(
                select(PoseLabel).where(PoseLabel.video_id.in_(video_ids))
            ).scalars().all()

            for label in pose_labels:
                video = video_map.get(label.video_id)
                if video is None:
                    continue
                frames.append((
                    Path(video.path),
                    video.id,
                    label.frame_idx,
                    label.front_x,
                    label.front_y,
                    label.rear_x,
                    label.rear_y,
                ))

        logger.info(f"Gathered {len(frames)} pose-labeled frames from {project_path}")
        return frames

    def _write_yolo_pose_dataset(
        self,
        frames: list[tuple[Path, int, int, float, float, float, float]],
        split: str,
        tmp_path: Path,
    ) -> None:
        """Write frames and YOLO pose labels to the dataset directory.

        YOLO pose label format per line:
            0 cx cy w h fx fy 2 rx ry 2
        where (cx, cy, w, h) is the normalized bounding box derived from keypoints,
        (fx, fy) is the front keypoint (normalized), (rx, ry) is the rear keypoint,
        and 2 = visible confidence for each keypoint.

        Args:
            frames: List of (video_path, video_id, frame_idx, front_x, front_y, rear_x, rear_y).
            split: "train" or "val".
            tmp_path: Root of the temporary dataset directory.
        """
        images_dir = tmp_path / "images" / split
        labels_dir = tmp_path / "labels" / split

        for i, (video_path, video_id, frame_idx, front_x, front_y, rear_x, rear_y) in enumerate(frames):
            cap = cv2.VideoCapture(str(video_path))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            cap.release()

            if not ret:
                logger.warning(
                    f"Failed to read frame {frame_idx} from {video_path}, skipping"
                )
                continue

            img_h, img_w = frame.shape[:2]

            # Derive bounding box from keypoints with 20% of frame dimensions padding
            pad_x = 0.20
            pad_y = 0.20

            kp_x_min = min(front_x, rear_x)
            kp_x_max = max(front_x, rear_x)
            kp_y_min = min(front_y, rear_y)
            kp_y_max = max(front_y, rear_y)

            x1 = max(0.0, kp_x_min - pad_x)
            y1 = max(0.0, kp_y_min - pad_y)
            x2 = min(1.0, kp_x_max + pad_x)
            y2 = min(1.0, kp_y_max + pad_y)

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            w = x2 - x1
            h = y2 - y1

            # YOLO pose label: class cx cy w h kp1_x kp1_y kp1_vis kp2_x kp2_y kp2_vis
            yolo_line = (
                f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} "
                f"{front_x:.6f} {front_y:.6f} 2 "
                f"{rear_x:.6f} {rear_y:.6f} 2"
            )

            name = f"v{video_id}_f{frame_idx}_{i:06d}"
            cv2.imwrite(str(images_dir / f"{name}.jpg"), frame)
            with open(labels_dir / f"{name}.txt", "w") as f:
                f.write(yolo_line + "\n")

    # -------------------------------------------------------------------------
    # Applying
    # -------------------------------------------------------------------------

    def apply(
        self,
        project_path: Path,
        video_ids: list[int],
    ) -> bool:
        """Start applying pose model to videos in background thread.

        Returns immediately. Use get_training_progress() to monitor.
        """
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
        """Synchronous implementation of applying pose model to videos."""
        from ultralytics import YOLO
        from vidseq.services import segmentation_service

        # Free GPU memory
        segmentation_service.shutdown()

        weights_path = project_path / "models" / "pose.pt"
        if not weights_path.exists():
            raise RuntimeError(f"Pose model not found at {weights_path}. Train first.")

        db_manager = DatabaseManager.get_instance()
        engine = db_manager.get_project_engine(project_path)

        # Get video metadata
        with Session(engine) as session:
            videos = session.execute(
                select(Video).where(Video.id.in_(video_ids))
            ).scalars().all()
            video_list = list(videos)

        total_frames = sum(v.num_frames for v in video_list)
        self._training_progress.apply_total = total_frames
        self._training_progress.apply_current = 0
        self._training_progress.status = "applying"

        model = YOLO(str(weights_path))

        try:
            for video in video_list:
                video_path = Path(video.path)
                video_id = video.id
                num_frames = video.num_frames

                # Create alignment_keypoints.h5 for this video
                create_alignment_keypoints_array(project_path, video_id, num_frames)

                cap = cv2.VideoCapture(str(video_path))
                pose_updates = []  # (frame_idx, front_x, front_y, rear_x, rear_y, score)

                try:
                    with alignment_keypoints(project_path, video_id, mode="a") as kp_data:
                        for frame_idx in range(num_frames):
                            if self._stop_requested:
                                self._training_progress.status = "stopped"
                                logger.info("Pose apply stopped by user")
                                return

                            ret, frame = cap.read()
                            if not ret:
                                self._training_progress.apply_current += 1
                                continue

                            results = model(frame, verbose=False)
                            front_x = front_y = rear_x = rear_y = 0.0
                            score = 0.0

                            if results and len(results) > 0:
                                result = results[0]
                                if (
                                    result.keypoints is not None
                                    and result.keypoints.xy is not None
                                    and len(result.keypoints.xy) > 0
                                ):
                                    img_h, img_w = frame.shape[:2]
                                    kps = result.keypoints.xy[0].cpu().numpy()  # (2, 2)
                                    confs = result.keypoints.conf
                                    if confs is not None:
                                        kp_conf = confs[0].cpu().numpy()  # (2,)
                                        score = float(np.mean(kp_conf))
                                    else:
                                        score = 1.0

                                    if len(kps) >= 2:
                                        front_x = float(kps[0, 0]) / img_w
                                        front_y = float(kps[0, 1]) / img_h
                                        rear_x = float(kps[1, 0]) / img_w
                                        rear_y = float(kps[1, 1]) / img_h

                            pose_updates.append((frame_idx, front_x, front_y, rear_x, rear_y, score))

                            # Write to H5
                            kp_data[frame_idx] = [front_x, front_y, rear_x, rear_y]

                            self._training_progress.apply_current += 1
                finally:
                    cap.release()

                # Batch upsert pose data to FrameData
                with Session(engine) as session:
                    self._batch_upsert_pose(session, video_id, pose_updates)

                logger.info(f"Applied pose model to video {video_id}: {num_frames} frames")

        finally:
            del model
            torch.cuda.empty_cache()

        self._training_progress.status = "completed"
        self._training_progress.is_training = False
        logger.info("Pose apply completed")

    def _batch_upsert_pose(
        self,
        session: Session,
        video_id: int,
        updates: list[tuple[int, float, float, float, float, float]],
    ) -> None:
        """Batch upsert pose keypoints and score into FrameData.

        Args:
            session: Sync SQLAlchemy session.
            video_id: Video ID.
            updates: List of (frame_idx, front_x, front_y, rear_x, rear_y, score).
        """
        from sqlalchemy.dialects.sqlite import insert as sqlite_insert

        rows = [
            {
                "video_id": video_id,
                "frame_idx": int(frame_idx),
                "pose_front_x": float(front_x),
                "pose_front_y": float(front_y),
                "pose_rear_x": float(rear_x),
                "pose_rear_y": float(rear_y),
                "pose_score": float(score),
                "has_pose": 1 if score > 0.0 else 0,
            }
            for frame_idx, front_x, front_y, rear_x, rear_y, score in updates
        ]

        for i in range(0, len(rows), UPSERT_CHUNK_SIZE):
            chunk = rows[i : i + UPSERT_CHUNK_SIZE]
            stmt = sqlite_insert(FrameData).values(chunk)
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
