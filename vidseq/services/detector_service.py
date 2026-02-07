"""Detector Service for RT-DETR-based object detection training and inference."""

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
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from vidseq.models.frame_data import FrameData
from vidseq.models.video import Video
from vidseq.schemas.detector import DetectorTrainingProgress
from vidseq.services.array_storage import tracker_masks
from vidseq.services.database_manager import DatabaseManager

logger = logging.getLogger("vidseq.detector")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "[%(asctime)s] [Detector] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

MIN_FRAMES_FOR_VALIDATION = 10


def _mask_to_yolo_bbox(mask: np.ndarray, img_h: int, img_w: int) -> str | None:
    """Convert a binary mask to YOLO bbox format.

    Args:
        mask: Binary mask (H, W) with values 0 or 255.
        img_h: Image height in pixels.
        img_w: Image width in pixels.

    Returns:
        YOLO format string "class cx cy w h" (normalized), or None if mask is empty.
    """
    rows = np.any(mask > 127, axis=1)
    cols = np.any(mask > 127, axis=0)
    if not rows.any() or not cols.any():
        return None
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    cx = (x1 + x2 + 1) / 2 / img_w
    cy = (y1 + y2 + 1) / 2 / img_h
    w = (x2 - x1 + 1) / img_w
    h = (y2 - y1 + 1) / img_h
    return f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


class DetectorService:
    """Singleton service for RT-DETR detector training and inference."""

    _instance: Optional["DetectorService"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "DetectorService":
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
        self._training_progress = DetectorTrainingProgress()

        self._initialized = True

    @classmethod
    def get_instance(cls) -> "DetectorService":
        """Get the singleton instance."""
        return cls()

    def is_training(self) -> bool:
        """Check if training is in progress."""
        return self._is_training

    def get_training_progress(self) -> DetectorTrainingProgress:
        """Get current training progress."""
        return self._training_progress

    def stop_training(self) -> bool:
        """Request training to stop."""
        if self._is_training:
            self._stop_requested = True
            return True
        return False

    def model_exists(self, project_path: Path) -> bool:
        """Check if trained model exists."""
        return (project_path / "models" / "detector.pt").exists()

    def train(
        self,
        project_path: Path,
        max_epochs: int = 100,
        batch_size: int = 2,
        lr: float = 1e-4,
        early_stop_patience: int = 20,
    ) -> bool:
        """Start training in background thread.

        Returns immediately. Use get_training_progress() to monitor.
        """
        if self._is_training:
            raise RuntimeError("Training already in progress")

        self._is_training = True
        self._stop_requested = False

        def _train_thread():
            try:
                self._train_sync(
                    project_path,
                    max_epochs,
                    batch_size,
                    lr,
                    early_stop_patience,
                )
            except Exception as e:
                logger.exception("Training failed")
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
        max_epochs: int,
        batch_size: int,
        lr: float,
        early_stop_patience: int,
    ) -> None:
        """Synchronous training implementation using Ultralytics RT-DETR."""
        from ultralytics import RTDETR
        from vidseq.services import segmentation_service

        # Free GPU memory by shutting down SAM2 worker
        # Raises RuntimeError if SAM2 has active sessions
        segmentation_service.shutdown()

        logger.info(
            f"Starting RT-DETR training: max_epochs={max_epochs}, "
            f"batch_size={batch_size}, lr={lr}"
        )

        # Gather training data
        all_frames = self._gather_training_frames(project_path)
        if len(all_frames) == 0:
            raise RuntimeError("No training frames found. Mark training ranges first.")

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
        self._training_progress = DetectorTrainingProgress(
            is_training=True,
            current_epoch=0,
            max_epochs=max_epochs,
            current_lr=lr,
            early_stop_patience=early_stop_patience,
            status="training",
            started_at=time.time(),
            num_train_frames=len(train_frames),
            num_val_frames=len(val_frames),
        )

        # Write YOLO dataset to temp directory
        tmp_dir = tempfile.mkdtemp(prefix="vidseq_detector_")
        tmp_path = Path(tmp_dir)

        try:
            # Create directory structure
            for split in ("train", "val"):
                (tmp_path / "images" / split).mkdir(parents=True, exist_ok=True)
                (tmp_path / "labels" / split).mkdir(parents=True, exist_ok=True)

            # Write frames and labels
            self._write_yolo_dataset(
                train_frames, "train", tmp_path, project_path
            )
            if val_frames:
                self._write_yolo_dataset(
                    val_frames, "val", tmp_path, project_path
                )

            # Write dataset.yaml
            yaml_path = tmp_path / "dataset.yaml"
            dataset_config = {
                "path": str(tmp_path),
                "train": "images/train",
                "val": "images/val" if val_frames else "images/train",
                "nc": 1,
                "names": ["animal"],
            }
            with open(yaml_path, "w") as f:
                yaml.dump(dataset_config, f, default_flow_style=False)

            logger.info(
                f"YOLO dataset written: {len(train_frames)} train, "
                f"{len(val_frames)} val frames in {tmp_dir}"
            )

            # Create and train model
            model = RTDETR("rtdetr-x.pt")

            # Graceful stop callback
            def check_stop(trainer):
                if self._stop_requested:
                    raise KeyboardInterrupt("Training stopped by user")

            # Progress callback
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

            model.add_callback("on_fit_epoch_end", on_epoch_end)
            model.add_callback("on_train_epoch_start", check_stop)

            model_save_dir = project_path / "models"
            model_save_dir.mkdir(exist_ok=True)

            model.train(
                data=str(yaml_path),
                epochs=max_epochs,
                imgsz=640,
                batch=batch_size,
                lr0=lr,
                optimizer="AdamW",
                patience=early_stop_patience,
                single_cls=True,
                device=0,
                workers=0,  # RT-DETR transforms contain unpicklable lambdas
                plots=False,  # avoid accumulating plot data across epochs
                project=str(model_save_dir),
                name="rtdetr_train",
                exist_ok=True,
                verbose=False,
            )

            # Copy best weights to standard location
            best_pt = model_save_dir / "rtdetr_train" / "weights" / "best.pt"
            if best_pt.exists():
                shutil.copy2(best_pt, model_save_dir / "detector.pt")
                logger.info(f"Best weights saved to {model_save_dir / 'detector.pt'}")
            else:
                logger.warning("best.pt not found after training")

            # Mark completed and apply to training data
            if self._stop_requested:
                self._training_progress.status = "stopped"
                logger.info("Training stopped by user")
            else:
                self._training_progress.status = "applying"
                self._apply_to_training_data(project_path)
                self._training_progress.status = "completed"

        finally:
            # Clean up temp directory
            shutil.rmtree(tmp_dir, ignore_errors=True)

        self._training_progress.is_training = False
        logger.info("Training completed")

    def _write_yolo_dataset(
        self,
        frames: list[tuple[Path, int, int]],
        split: str,
        tmp_path: Path,
        project_path: Path,
    ) -> None:
        """Write frames and YOLO labels to the dataset directory.

        Args:
            frames: List of (video_path, video_id, frame_idx) tuples.
            split: "train" or "val".
            tmp_path: Root of the temporary dataset directory.
            project_path: Project folder path for accessing mask files.
        """
        images_dir = tmp_path / "images" / split
        labels_dir = tmp_path / "labels" / split

        for i, (video_path, video_id, frame_idx) in enumerate(frames):
            # Read frame from video
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

            # Read mask from tracker masks
            with tracker_masks(project_path, video_id) as masks:
                mask = np.asarray(masks[frame_idx])

            # Convert mask to YOLO bbox
            yolo_line = _mask_to_yolo_bbox(mask, img_h, img_w)
            if yolo_line is None:
                continue

            # Write image and label with unique name
            name = f"v{video_id}_f{frame_idx}_{i:06d}"
            cv2.imwrite(str(images_dir / f"{name}.jpg"), frame)
            with open(labels_dir / f"{name}.txt", "w") as f:
                f.write(yolo_line + "\n")

    def _gather_training_frames(
        self,
        project_path: Path,
    ) -> list[tuple[Path, int, int]]:
        """Gather all training frames from all videos in project.

        Training frames are identified by FrameData.frame_type == 'train'.

        Returns:
            List of (video_path, video_id, frame_idx) tuples.
        """
        db_manager = DatabaseManager.get_instance()
        engine = db_manager.get_project_engine(project_path)

        frames = []
        with Session(engine) as session:
            # Get all videos
            videos = session.execute(select(Video)).scalars().all()
            video_map = {v.id: v for v in videos}

            # Get all training frames
            training_frames = session.execute(
                select(FrameData).where(FrameData.frame_type == "train")
            ).scalars().all()

            for frame_data in training_frames:
                video = video_map.get(frame_data.video_id)
                if video is None:
                    continue

                video_path = Path(video.path)
                frames.append((video_path, video.id, frame_data.frame_idx))

        logger.info(f"Gathered {len(frames)} training frames from {project_path}")
        return frames

    def _apply_to_training_data(
        self,
        project_path: Path,
    ) -> None:
        """Apply trained detector to all training frames and save bboxes/scores to DB."""
        from vidseq.services.detector_model import detect, load_finetuned

        logger.info("Applying detector to training data...")

        all_frames = self._gather_training_frames(project_path)
        self._training_progress.apply_total = len(all_frames)
        self._training_progress.apply_current = 0

        weights_path = project_path / "models" / "detector.pt"
        detector = load_finetuned(weights_path)

        # Group frames by video for efficient batch processing
        frames_by_video: dict[int, list[tuple[Path, int]]] = {}
        for video_path, video_id, frame_idx in all_frames:
            if video_id not in frames_by_video:
                frames_by_video[video_id] = []
            frames_by_video[video_id].append((video_path, frame_idx))

        # Collect bboxes and scores to batch-insert
        bboxes_to_save: dict[int, list[tuple[int, float, float, float, float]]] = {}
        scores_to_save: dict[int, list[tuple[int, float]]] = {}

        for video_id, frame_list in frames_by_video.items():
            bboxes_to_save[video_id] = []
            scores_to_save[video_id] = []

            for video_path, frame_idx in frame_list:
                # Load frame
                cap = cv2.VideoCapture(str(video_path))
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                cap.release()

                if not ret:
                    self._training_progress.apply_current += 1
                    continue

                # Run detection
                detections = detect(detector, frame)

                if detections:
                    best = detections[0]  # Highest confidence
                    x1, y1, x2, y2 = best["bbox"]
                    conf = best["conf"]
                    bboxes_to_save[video_id].append(
                        (frame_idx, x1, y1, x2, y2)
                    )
                    scores_to_save[video_id].append((frame_idx, conf))
                else:
                    scores_to_save[video_id].append((frame_idx, 0.0))

                self._training_progress.apply_current += 1

        # Batch insert to database
        db_manager = DatabaseManager.get_instance()
        engine = db_manager.get_project_engine(project_path)

        with Session(engine) as session:
            for video_id, bbox_list in bboxes_to_save.items():
                if not bbox_list:
                    continue
                rows = [
                    {
                        "video_id": video_id,
                        "frame_idx": int(fi),
                        "detector_bbox_x1": float(x1),
                        "detector_bbox_y1": float(y1),
                        "detector_bbox_x2": float(x2),
                        "detector_bbox_y2": float(y2),
                    }
                    for fi, x1, y1, x2, y2 in bbox_list
                ]
                stmt = sqlite_insert(FrameData).values(rows)
                stmt = stmt.on_conflict_do_update(
                    index_elements=["video_id", "frame_idx"],
                    set_={
                        "detector_bbox_x1": stmt.excluded.detector_bbox_x1,
                        "detector_bbox_y1": stmt.excluded.detector_bbox_y1,
                        "detector_bbox_x2": stmt.excluded.detector_bbox_x2,
                        "detector_bbox_y2": stmt.excluded.detector_bbox_y2,
                    },
                )
                session.execute(stmt)

            for video_id, score_list in scores_to_save.items():
                if not score_list:
                    continue
                rows = [
                    {
                        "video_id": video_id,
                        "frame_idx": int(fi),
                        "detector_score": float(score),
                    }
                    for fi, score in score_list
                ]
                stmt = sqlite_insert(FrameData).values(rows)
                stmt = stmt.on_conflict_do_update(
                    index_elements=["video_id", "frame_idx"],
                    set_={"detector_score": stmt.excluded.detector_score},
                )
                session.execute(stmt)

            session.commit()

        # Cleanup
        del detector
        torch.cuda.empty_cache()

        logger.info(f"Applied detector to {len(all_frames)} training frames")
