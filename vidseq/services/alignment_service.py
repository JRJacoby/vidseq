"""Alignment Service for egocentric alignment training and inference.

Provides mock training and inference for keypoint detection (front/rear of animal).
"""

import io
import json
import logging
import random
import threading
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from vidseq.models.alignment_label import AlignmentLabel
from vidseq.models.video import Video

# Configure logger for alignment service
logger = logging.getLogger("vidseq.alignment")
logger.setLevel(logging.DEBUG)

# Add console handler if not already present
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "[%(asctime)s] [Alignment] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def generate_gaussian_heatmap(
    x: float,
    y: float,
    height: int,
    width: int,
    sigma: float = 0.05,
) -> np.ndarray:
    """Generate a gaussian heatmap centered at (x, y).

    Args:
        x: Normalized x coordinate (0-1)
        y: Normalized y coordinate (0-1)
        height: Height of output heatmap
        width: Width of output heatmap
        sigma: Standard deviation of gaussian (normalized, default 0.05 = 5% of image)

    Returns:
        Heatmap array of shape (height, width) with values in [0, 1]
    """
    logger.debug(f"generate_gaussian_heatmap: center=({x:.3f}, {y:.3f}), size={width}x{height}, sigma={sigma}")

    # Create coordinate grids (normalized 0-1)
    yy, xx = np.mgrid[0:height, 0:width]
    xx = xx / width
    yy = yy / height

    # Compute gaussian
    dist_sq = (xx - x) ** 2 + (yy - y) ** 2
    heatmap = np.exp(-dist_sq / (2 * sigma ** 2))

    max_val = float(np.max(heatmap))
    logger.debug(f"generate_gaussian_heatmap: max_value={max_val:.4f}")

    return heatmap.astype(np.float32)


def heatmap_to_png(heatmap: np.ndarray) -> bytes:
    """Convert (H, W, 2) heatmap to PNG bytes.

    Encodes front probability as R channel, rear probability as G channel.

    Args:
        heatmap: Array of shape (H, W, 2) with values in [0, 1]

    Returns:
        PNG image bytes
    """
    h, w, c = heatmap.shape
    logger.debug(f"heatmap_to_png: input_shape=({h}, {w}, {c})")

    # Scale to 0-255 range
    r_channel = (heatmap[:, :, 0] * 255).astype(np.uint8)  # Front
    g_channel = (heatmap[:, :, 1] * 255).astype(np.uint8)  # Rear
    b_channel = np.zeros((h, w), dtype=np.uint8)

    # Stack into RGB image
    rgb = np.stack([r_channel, g_channel, b_channel], axis=2)

    # Convert to PNG
    img = Image.fromarray(rgb, mode="RGB")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    png_bytes = buffer.getvalue()

    logger.debug(f"heatmap_to_png: output_size={len(png_bytes)} bytes, R_max={r_channel.max()}, G_max={g_channel.max()}")

    return png_bytes


class AlignmentService:
    """Singleton service for alignment model training and inference."""

    _instance: Optional["AlignmentService"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "AlignmentService":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
                    logger.info("AlignmentService singleton instance created")
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._is_training = False
        self._is_applying = False

        self._initialized = True
        logger.info("AlignmentService initialized: is_training=False, is_applying=False")

    @classmethod
    def get_instance(cls) -> "AlignmentService":
        """Get the singleton instance."""
        return cls()

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton (useful for testing)."""
        with cls._lock:
            if cls._instance is not None:
                logger.info("AlignmentService singleton instance reset")
                cls._instance = None

    def is_training(self) -> bool:
        """Check if training is in progress."""
        logger.debug(f"is_training() -> {self._is_training}")
        return self._is_training

    def is_applying(self) -> bool:
        """Check if applying alignment is in progress."""
        logger.debug(f"is_applying() -> {self._is_applying}")
        return self._is_applying

    def get_model_path(self, project_path: Path) -> Path:
        """Get path to the alignment model state file."""
        path = project_path / "alignment_model.json"
        logger.debug(f"get_model_path: {path}")
        return path

    def is_model_trained(self, project_path: Path) -> bool:
        """Check if alignment model has been trained."""
        exists = self.get_model_path(project_path).exists()
        logger.debug(f"is_model_trained: project={project_path.name}, exists={exists}")
        return exists

    def delete_model(self, project_path: Path) -> bool:
        """Delete the alignment model file.

        Returns:
            True if model was deleted, False if it didn't exist
        """
        model_path = self.get_model_path(project_path)
        logger.info(f"delete_model: attempting to delete {model_path}")
        if model_path.exists():
            model_path.unlink()
            logger.info(f"delete_model: deleted model file")
            return True
        else:
            logger.info(f"delete_model: model file did not exist")
            return False

    async def get_label_count(self, session: AsyncSession) -> int:
        """Get count of alignment labels."""
        result = await session.execute(select(func.count(AlignmentLabel.id)))
        count = result.scalar() or 0
        logger.info(f"get_label_count: count={count}")
        return count

    async def get_all_labels(self, session: AsyncSession) -> list[AlignmentLabel]:
        """Get all alignment labels."""
        result = await session.execute(
            select(AlignmentLabel).order_by(AlignmentLabel.id)
        )
        labels = list(result.scalars().all())
        logger.info(f"get_all_labels: returning {len(labels)} labels")
        for label in labels:
            logger.debug(
                f"  Label id={label.id}: video={label.video_id}, frame={label.frame_idx}, "
                f"front=({label.front_x:.3f}, {label.front_y:.3f}), "
                f"rear=({label.rear_x:.3f}, {label.rear_y:.3f})"
            )
        return labels

    async def save_label(
        self,
        session: AsyncSession,
        video_id: int,
        frame_idx: int,
        front_x: float,
        front_y: float,
        rear_x: float,
        rear_y: float,
    ) -> AlignmentLabel:
        """Save or update an alignment label.

        Args:
            session: Database session
            video_id: Video ID
            frame_idx: Frame index
            front_x, front_y: Front (nose) coordinates (normalized 0-1)
            rear_x, rear_y: Rear (tail) coordinates (normalized 0-1)

        Returns:
            The saved AlignmentLabel
        """
        logger.info(
            f"save_label: video_id={video_id}, frame_idx={frame_idx}, "
            f"front=({front_x:.4f}, {front_y:.4f}), rear=({rear_x:.4f}, {rear_y:.4f})"
        )

        # Validate coordinates are in range
        for name, val in [("front_x", front_x), ("front_y", front_y), ("rear_x", rear_x), ("rear_y", rear_y)]:
            if not (0.0 <= val <= 1.0):
                logger.warning(f"save_label: {name}={val} is outside [0, 1] range!")

        # Check for existing label
        result = await session.execute(
            select(AlignmentLabel).where(
                AlignmentLabel.video_id == video_id,
                AlignmentLabel.frame_idx == frame_idx,
            )
        )
        existing = result.scalar_one_or_none()

        if existing:
            # Update existing
            logger.info(f"save_label: updating existing label id={existing.id}")
            existing.front_x = front_x
            existing.front_y = front_y
            existing.rear_x = rear_x
            existing.rear_y = rear_y
            await session.commit()
            logger.info(f"save_label: updated label id={existing.id}")
            return existing
        else:
            # Create new
            label = AlignmentLabel(
                video_id=video_id,
                frame_idx=frame_idx,
                front_x=front_x,
                front_y=front_y,
                rear_x=rear_x,
                rear_y=rear_y,
            )
            session.add(label)
            await session.commit()
            await session.refresh(label)
            logger.info(f"save_label: created new label id={label.id}")
            return label

    async def delete_label(
        self,
        session: AsyncSession,
        video_id: int,
        frame_idx: int,
    ) -> bool:
        """Delete an alignment label.

        Returns:
            True if label was deleted, False if not found
        """
        logger.info(f"delete_label: video_id={video_id}, frame_idx={frame_idx}")
        result = await session.execute(
            delete(AlignmentLabel).where(
                AlignmentLabel.video_id == video_id,
                AlignmentLabel.frame_idx == frame_idx,
            )
        )
        await session.commit()
        deleted = result.rowcount > 0
        logger.info(f"delete_label: deleted={deleted}, rowcount={result.rowcount}")
        return deleted

    async def delete_all_labels(self, session: AsyncSession) -> int:
        """Delete all alignment labels.

        Returns:
            Number of labels deleted
        """
        logger.info("delete_all_labels: starting")
        result = await session.execute(delete(AlignmentLabel))
        await session.commit()
        deleted_count = result.rowcount
        logger.info(f"delete_all_labels: deleted {deleted_count} labels")
        return deleted_count

    async def get_random_unlabeled_frame(
        self,
        session: AsyncSession,
    ) -> Optional[tuple[int, int]]:
        """Get a random unlabeled frame from videos with cropping completed.

        Returns:
            (video_id, frame_idx) tuple or None if no frames available
        """
        logger.info("get_random_unlabeled_frame: starting")

        # Get videos with cropping completed
        videos_result = await session.execute(
            select(Video).where(Video.cropping_status == "completed")
        )
        videos = list(videos_result.scalars().all())
        logger.info(f"get_random_unlabeled_frame: found {len(videos)} videos with cropping completed")

        if not videos:
            logger.warning("get_random_unlabeled_frame: no videos with cropping completed")
            return None

        for v in videos:
            logger.debug(f"  Video id={v.id}, name={v.name}, num_frames={v.num_frames}")

        # Get all existing labels
        labels_result = await session.execute(select(AlignmentLabel))
        labels = list(labels_result.scalars().all())
        labeled_set = {(l.video_id, l.frame_idx) for l in labels}
        logger.info(f"get_random_unlabeled_frame: {len(labels)} existing labels")

        # Build list of all possible frames
        total_frames = sum(v.num_frames for v in videos)
        unlabeled_count = total_frames - len(labeled_set)
        logger.info(f"get_random_unlabeled_frame: total_frames={total_frames}, unlabeled={unlabeled_count}")

        all_frames = []
        for video in videos:
            for frame_idx in range(video.num_frames):
                if (video.id, frame_idx) not in labeled_set:
                    all_frames.append((video.id, frame_idx))

        if not all_frames:
            logger.warning("get_random_unlabeled_frame: all frames are labeled!")
            return None

        choice = random.choice(all_frames)
        logger.info(f"get_random_unlabeled_frame: selected video_id={choice[0]}, frame_idx={choice[1]}")
        return choice

    def train_model_sync(
        self,
        project_path: Path,
        epochs: int = 10,
    ) -> bool:
        """Mock training: just saves a model state file.

        In the real implementation, this would train a neural network.
        For now, it just marks the model as trained.

        Args:
            project_path: Path to project folder
            epochs: Number of training epochs (for future use)

        Returns:
            True if successful
        """
        logger.info(f"train_model_sync: starting training with epochs={epochs}, project={project_path.name}")
        self._is_training = True
        try:
            # Mock training - just save a state file
            model_state = {
                "trained": True,
                "epochs": epochs,
                "version": "mock_v1",
            }

            model_path = self.get_model_path(project_path)
            logger.info(f"train_model_sync: saving model state to {model_path}")

            with open(model_path, "w") as f:
                json.dump(model_state, f, indent=2)

            logger.info(f"train_model_sync: mock training complete, model_state={model_state}")
            return True
        except Exception as e:
            logger.error(f"train_model_sync: training failed with error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
        finally:
            self._is_training = False
            logger.debug("train_model_sync: is_training set to False")

    def predict_sync(
        self,
        project_path: Path,
        height: int,
        width: int,
    ) -> np.ndarray:
        """Mock prediction: returns random gaussian heatmaps.

        In the real implementation, this would run the trained model.
        For now, returns random predictions.

        Args:
            project_path: Path to project folder (unused for mock)
            height: Output height
            width: Output width

        Returns:
            Heatmap array of shape (height, width, 2)
            Channel 0 = front probability, Channel 1 = rear probability
        """
        logger.info(f"predict_sync: generating prediction, size={width}x{height}")

        # Generate random positions
        front_x = random.uniform(0.2, 0.8)
        front_y = random.uniform(0.2, 0.8)
        rear_x = random.uniform(0.2, 0.8)
        rear_y = random.uniform(0.2, 0.8)

        logger.info(
            f"predict_sync: mock prediction - front=({front_x:.3f}, {front_y:.3f}), "
            f"rear=({rear_x:.3f}, {rear_y:.3f})"
        )

        # Generate gaussian heatmaps
        front_heatmap = generate_gaussian_heatmap(front_x, front_y, height, width)
        rear_heatmap = generate_gaussian_heatmap(rear_x, rear_y, height, width)

        # Stack into (H, W, 2) array
        heatmap = np.stack([front_heatmap, rear_heatmap], axis=2)

        logger.debug(f"predict_sync: output_shape={heatmap.shape}, dtype={heatmap.dtype}")

        return heatmap

    def predict_to_png(
        self,
        project_path: Path,
        height: int,
        width: int,
    ) -> bytes:
        """Get mock prediction as PNG bytes.

        Args:
            project_path: Path to project folder
            height: Output height
            width: Output width

        Returns:
            PNG image bytes (R = front, G = rear)
        """
        logger.info(f"predict_to_png: project={project_path.name}, size={width}x{height}")
        heatmap = self.predict_sync(project_path, height, width)
        png_bytes = heatmap_to_png(heatmap)
        logger.info(f"predict_to_png: returning {len(png_bytes)} bytes")
        return png_bytes

    def apply_alignment_sync(
        self,
        project_path: Path,
        project_engine,
    ) -> bool:
        """Apply alignment to all cropped videos.

        Reads cropped videos, rotates frames so animal faces right,
        saves to aligned_videos folder.

        This is a mock implementation that just copies videos without rotation.

        Args:
            project_path: Path to project folder
            project_engine: SQLAlchemy engine for project DB

        Returns:
            True if successful
        """
        logger.info(f"apply_alignment_sync: starting, project={project_path.name}")
        self._is_applying = True
        try:
            from vidseq.services.cropped_video_service import get_cropped_video_path

            # Get videos with cropping completed
            with Session(project_engine) as session:
                result = session.execute(
                    select(Video).where(Video.cropping_status == "completed")
                )
                videos = list(result.scalars().all())

            logger.info(f"apply_alignment_sync: found {len(videos)} videos with cropping completed")

            if not videos:
                logger.warning("apply_alignment_sync: no cropped videos to align")
                return False

            # Create output directory
            output_dir = project_path / "aligned_videos"
            output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"apply_alignment_sync: output_dir={output_dir}")

            for video in videos:
                logger.info(f"apply_alignment_sync: processing video id={video.id}, name={video.name}")
                cropped_path = get_cropped_video_path(project_path, video.name)

                if not cropped_path.exists():
                    logger.warning(f"apply_alignment_sync: cropped video not found: {cropped_path}")
                    continue

                output_path = output_dir / f"{cropped_path.stem}_aligned.mp4"
                logger.info(f"apply_alignment_sync: copying {cropped_path} -> {output_path}")

                # Mock: just copy the video without rotation
                # In real implementation, would rotate each frame based on predictions
                import shutil
                shutil.copy2(cropped_path, output_path)

                file_size = output_path.stat().st_size
                logger.info(f"apply_alignment_sync: aligned video saved, size={file_size} bytes")

            logger.info(f"apply_alignment_sync: completed successfully for {len(videos)} videos")
            return True

        except Exception as e:
            logger.error(f"apply_alignment_sync: failed with error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
        finally:
            self._is_applying = False
            logger.debug("apply_alignment_sync: is_applying set to False")


# Module-level singleton instance
alignment_service = AlignmentService.get_instance()
