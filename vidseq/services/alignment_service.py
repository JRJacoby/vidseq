"""Alignment Service for egocentric alignment training and inference.

Provides training and inference for keypoint detection (front/rear of animal)
using a U-Net model with ResNet18 encoder.
"""

import io
import logging
import random
import threading
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from PIL import Image
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session
from torch.utils.data import Dataset, DataLoader

from vidseq.models.alignment_label import AlignmentLabel
from vidseq.models.video import Video
from vidseq.services.cropped_video_service import get_cropped_video_path

# Constants
ALIGNMENT_INPUT_SIZE = 128  # Fixed input size for model

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


class AlignmentDataset(Dataset):
    """PyTorch Dataset for alignment model training.

    Loads frames from cropped videos and generates gaussian heatmap targets.
    Applies augmentations (rotations, flips) to expand training data 6x.
    """

    # Augmentation types applied to every sample
    AUG_TYPES = ["none", "rot90", "rot180", "rot270", "flip_h", "flip_v"]

    def __init__(
        self,
        labels: list,
        project_path: Path,
        video_name_map: dict[int, str],
    ):
        """Initialize the dataset.

        Args:
            labels: List of AlignmentLabel objects (or dicts with same fields)
            project_path: Path to project folder
            video_name_map: Dict mapping video_id -> video.name for file lookup
        """
        self.project_path = project_path
        self.video_name_map = video_name_map
        self.size = ALIGNMENT_INPUT_SIZE

        # Expand labels with all augmentation types (6x expansion)
        self.samples: list[tuple] = []
        for label in labels:
            for aug_type in self.AUG_TYPES:
                self.samples.append((label, aug_type))

        logger.info(
            f"AlignmentDataset: {len(labels)} labels expanded to {len(self.samples)} samples "
            f"(6x augmentation: {self.AUG_TYPES})"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _apply_augmentation(
        self,
        frame: np.ndarray,
        front_x: float,
        front_y: float,
        rear_x: float,
        rear_y: float,
        aug_type: str,
    ) -> tuple[np.ndarray, float, float, float, float]:
        """Apply augmentation to frame and keypoint coordinates.

        Args:
            frame: Image array (H, W, C)
            front_x, front_y: Front keypoint normalized coords (0-1)
            rear_x, rear_y: Rear keypoint normalized coords (0-1)
            aug_type: Augmentation type

        Returns:
            (augmented_frame, new_front_x, new_front_y, new_rear_x, new_rear_y)
        """
        if aug_type == "none":
            return frame, front_x, front_y, rear_x, rear_y

        elif aug_type == "rot90":
            # Rotate 90° clockwise
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            # (x, y) → (1-y, x)
            new_front_x, new_front_y = 1 - front_y, front_x
            new_rear_x, new_rear_y = 1 - rear_y, rear_x

        elif aug_type == "rot180":
            # Rotate 180°
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            # (x, y) → (1-x, 1-y)
            new_front_x, new_front_y = 1 - front_x, 1 - front_y
            new_rear_x, new_rear_y = 1 - rear_x, 1 - rear_y

        elif aug_type == "rot270":
            # Rotate 270° clockwise (= 90° counter-clockwise)
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            # (x, y) → (y, 1-x)
            new_front_x, new_front_y = front_y, 1 - front_x
            new_rear_x, new_rear_y = rear_y, 1 - rear_x

        elif aug_type == "flip_h":
            # Flip horizontal
            frame = cv2.flip(frame, 1)
            # (x, y) → (1-x, y)
            new_front_x, new_front_y = 1 - front_x, front_y
            new_rear_x, new_rear_y = 1 - rear_x, rear_y

        elif aug_type == "flip_v":
            # Flip vertical
            frame = cv2.flip(frame, 0)
            # (x, y) → (x, 1-y)
            new_front_x, new_front_y = front_x, 1 - front_y
            new_rear_x, new_rear_y = rear_x, 1 - rear_y

        else:
            raise ValueError(f"Unknown augmentation type: {aug_type}")

        return frame, new_front_x, new_front_y, new_rear_x, new_rear_y

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        label, aug_type = self.samples[idx]

        # Get video name for file lookup
        video_name = self.video_name_map[label.video_id]
        cropped_path = get_cropped_video_path(self.project_path, video_name)

        # Load frame from video
        cap = cv2.VideoCapture(str(cropped_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, label.frame_idx)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise RuntimeError(f"Failed to read frame {label.frame_idx} from {cropped_path}")

        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize to fixed size (before augmentation for consistency)
        frame = cv2.resize(frame, (self.size, self.size), interpolation=cv2.INTER_LINEAR)

        # Apply augmentation to frame and coordinates
        frame, front_x, front_y, rear_x, rear_y = self._apply_augmentation(
            frame,
            label.front_x,
            label.front_y,
            label.rear_x,
            label.rear_y,
            aug_type,
        )

        # Normalize to [0, 1] and convert to (C, H, W) tensor
        frame_tensor = torch.from_numpy(frame).float() / 255.0
        frame_tensor = frame_tensor.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)

        # Generate target heatmaps at the fixed size using augmented coordinates
        front_heatmap = generate_gaussian_heatmap(
            front_x, front_y, self.size, self.size
        )
        rear_heatmap = generate_gaussian_heatmap(
            rear_x, rear_y, self.size, self.size
        )

        # Stack into (2, H, W) tensor
        target = torch.from_numpy(
            np.stack([front_heatmap, rear_heatmap], axis=0)
        ).float()

        return frame_tensor, target


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

        # Model caching
        self._model: Optional[nn.Module] = None
        self._model_path: Optional[Path] = None
        self._device: Optional[torch.device] = None

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
        """Get path to the alignment model weights file."""
        path = project_path / "alignment_model.pt"
        logger.debug(f"get_model_path: {path}")
        return path

    def is_model_trained(self, project_path: Path) -> bool:
        """Check if alignment model has been trained."""
        exists = self.get_model_path(project_path).exists()
        logger.debug(f"is_model_trained: project={project_path.name}, exists={exists}")
        return exists

    def delete_model(self, project_path: Path) -> bool:
        """Delete the alignment model file and clear cache.

        Returns:
            True if model was deleted, False if it didn't exist
        """
        # Clear cached model
        self._model = None
        self._model_path = None
        self._device = None
        logger.info("delete_model: cleared model cache")

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
        labels: list,
        video_name_map: dict[int, str],
        epochs: int = 10,
        batch_size: int = 8,
        lr: float = 1e-4,
    ) -> bool:
        """Train the alignment model using labeled data.

        Args:
            project_path: Path to project folder
            labels: List of AlignmentLabel objects
            video_name_map: Dict mapping video_id -> video.name
            epochs: Number of training epochs
            batch_size: Training batch size
            lr: Learning rate

        Returns:
            True if successful
        """
        logger.info(
            f"train_model_sync: starting training with epochs={epochs}, "
            f"batch_size={batch_size}, lr={lr}, num_labels={len(labels)}"
        )
        self._is_training = True

        try:
            # Setup device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"train_model_sync: using device={device}")

            # Create model
            model = smp.Unet(
                encoder_name="resnet18",
                encoder_weights="imagenet",
                in_channels=3,
                classes=2,
            )
            model = model.to(device)
            logger.info("train_model_sync: created U-Net model with ResNet18 encoder")

            # Create dataset and dataloader
            dataset = AlignmentDataset(labels, project_path, video_name_map)
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,  # Keep simple for now
            )
            logger.info(f"train_model_sync: created dataloader with {len(dataset)} samples")

            # Setup optimizer and loss
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            criterion = nn.MSELoss()

            # Training loop
            model.train()
            for epoch in range(epochs):
                epoch_loss = 0.0
                num_batches = 0

                for batch_idx, (frames, targets) in enumerate(dataloader):
                    frames = frames.to(device)
                    targets = targets.to(device)

                    # Forward pass
                    optimizer.zero_grad()
                    outputs = model(frames)

                    # Apply sigmoid to get probabilities
                    outputs = torch.sigmoid(outputs)

                    # Compute loss
                    loss = criterion(outputs, targets)

                    # Backward pass
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    num_batches += 1

                avg_loss = epoch_loss / max(num_batches, 1)
                logger.info(f"train_model_sync: epoch {epoch + 1}/{epochs}, loss={avg_loss:.6f}")

            # Save model weights
            model_path = self.get_model_path(project_path)
            torch.save(model.state_dict(), model_path)
            logger.info(f"train_model_sync: saved model weights to {model_path}")

            # Clear cached model so next predict loads the new weights
            self._model = None
            self._model_path = None

            logger.info("train_model_sync: training complete")
            return True

        except Exception as e:
            logger.error(f"train_model_sync: training failed with error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

        finally:
            self._is_training = False
            logger.debug("train_model_sync: is_training set to False")

    def _load_model(self, project_path: Path) -> nn.Module:
        """Load the model from disk, using cache if available.

        Args:
            project_path: Path to project folder

        Returns:
            Loaded model in eval mode

        Raises:
            FileNotFoundError: If model weights file doesn't exist
        """
        model_path = self.get_model_path(project_path)

        # Check if we need to reload
        if self._model is not None and self._model_path == model_path:
            logger.debug(f"_load_model: using cached model from {model_path}")
            return self._model

        if not model_path.exists():
            raise FileNotFoundError(f"Model weights not found: {model_path}")

        logger.info(f"_load_model: loading model from {model_path}")

        # Setup device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"_load_model: using device={device}")

        # Create model architecture
        model = smp.Unet(
            encoder_name="resnet18",
            encoder_weights=None,  # We'll load our own weights
            in_channels=3,
            classes=2,
        )

        # Load weights
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()

        # Cache for future use
        self._model = model
        self._model_path = model_path
        self._device = device

        logger.info(f"_load_model: model loaded and cached")
        return model

    def predict_sync(
        self,
        project_path: Path,
        frame: np.ndarray,
    ) -> np.ndarray:
        """Run inference on a frame to predict front/rear keypoint heatmaps.

        Args:
            project_path: Path to project folder
            frame: Input frame as (H, W, 3) uint8 BGR array

        Returns:
            Heatmap array of shape (H, W, 2) with values in [0, 1]
            Channel 0 = front probability, Channel 1 = rear probability
        """
        orig_h, orig_w = frame.shape[:2]
        logger.info(f"predict_sync: input frame size={orig_w}x{orig_h}")

        # Load model (uses cache)
        model = self._load_model(project_path)
        device = self._device

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize to model input size
        frame_resized = cv2.resize(
            frame_rgb, (ALIGNMENT_INPUT_SIZE, ALIGNMENT_INPUT_SIZE),
            interpolation=cv2.INTER_LINEAR
        )

        # Normalize to [0, 1] and convert to (C, H, W) tensor
        frame_tensor = torch.from_numpy(frame_resized).float() / 255.0
        frame_tensor = frame_tensor.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
        frame_tensor = frame_tensor.unsqueeze(0)  # Add batch dimension
        frame_tensor = frame_tensor.to(device)

        # Run inference
        with torch.no_grad():
            output = model(frame_tensor)
            output = torch.sigmoid(output)  # Ensure [0, 1] range

        # Convert to numpy: (1, 2, H, W) -> (H, W, 2)
        heatmap = output[0].cpu().numpy()  # (2, H, W)
        heatmap = np.transpose(heatmap, (1, 2, 0))  # (H, W, 2)

        logger.debug(f"predict_sync: model output shape={heatmap.shape}, range=[{heatmap.min():.3f}, {heatmap.max():.3f}]")

        # Resize back to original dimensions
        heatmap_resized = cv2.resize(
            heatmap, (orig_w, orig_h),
            interpolation=cv2.INTER_LINEAR
        )

        logger.info(f"predict_sync: output heatmap size={orig_w}x{orig_h}")

        return heatmap_resized

    def predict_to_png(
        self,
        project_path: Path,
        frame: np.ndarray,
    ) -> bytes:
        """Run inference and return prediction as PNG bytes.

        Args:
            project_path: Path to project folder
            frame: Input frame as (H, W, 3) uint8 BGR array

        Returns:
            PNG image bytes (R = front, G = rear)
        """
        height, width = frame.shape[:2]
        logger.info(f"predict_to_png: project={project_path.name}, frame size={width}x{height}")
        heatmap = self.predict_sync(project_path, frame)
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
