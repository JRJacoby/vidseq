"""Detector Service for DINOv2-based segmentation training and inference."""

import logging
import os
import threading
from pathlib import Path
from typing import Optional

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from sqlalchemy import select
from sqlalchemy.orm import Session

from vidseq.models.video import Video
from vidseq.models.frame_data import FrameData
from vidseq.schemas.detector import DetectorTrainingProgress
from vidseq.services.database_manager import DatabaseManager

# Disable HDF5's internal file locking
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

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


class DetectorDataset(Dataset):
    """Dataset for detector training from video frames and tracker masks."""

    def __init__(
        self,
        frames: list[tuple[Path, int, int]],  # (video_path, video_id, frame_idx)
        project_path: Path,
        target_size: int = 518,
    ):
        """Initialize dataset.

        Args:
            frames: List of (video_path, video_id, frame_idx) tuples.
            project_path: Path to project folder (for HDF5 mask files).
            target_size: Target size for longest edge.
        """
        self.frames = frames
        self.project_path = project_path
        self.target_size = target_size
        self._h5_cache: dict[int, h5py.File] = {}

    def __len__(self) -> int:
        return len(self.frames)

    def _get_h5_file(self, video_id: int) -> h5py.File:
        """Get or open HDF5 file for video."""
        if video_id not in self._h5_cache:
            h5_path = self.project_path / "masks" / f"{video_id}.h5"
            self._h5_cache[video_id] = h5py.File(h5_path, "r")
        return self._h5_cache[video_id]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        video_path, video_id, frame_idx = self.frames[idx]

        # Load frame from video
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise RuntimeError(f"Failed to read frame {frame_idx} from {video_path}")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]

        # Load mask from HDF5
        h5_file = self._get_h5_file(video_id)
        mask = h5_file["masks"][frame_idx]
        mask = np.array(mask, dtype=np.float32) / 255.0

        # Resize preserving aspect ratio, then trim to multiple of 14
        scale = self.target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        # Trim to multiple of 14
        new_h = (frame.shape[0] // 14) * 14
        new_w = (frame.shape[1] // 14) * 14
        frame = frame[:new_h, :new_w]
        mask = mask[:new_h, :new_w]

        # Convert to tensors
        frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask).unsqueeze(0).float()

        return frame, mask

    def close(self):
        """Close all HDF5 files."""
        for f in self._h5_cache.values():
            f.close()
        self._h5_cache.clear()


class DetectorService:
    """Singleton service for detector training and inference."""

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
                h5_path = project_path / "masks" / f"{video.id}.h5"

                if not h5_path.exists():
                    continue

                frames.append((video_path, video.id, frame_data.frame_idx))

        logger.info(f"Gathered {len(frames)} training frames from {project_path}")
        return frames
