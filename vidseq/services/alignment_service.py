"""Alignment Service for egocentric alignment training and inference.

Provides training and inference for keypoint detection (front/rear of animal)
using a U-Net model with ResNet18 encoder.
"""

import asyncio
import collections
import copy
import logging
import os
import random
import subprocess
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

# Disable HDF5's internal file locking (we use our own approach)
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import imageio_ffmpeg

from vidseq.services.array_storage import (
    cropped_masks,
    aligned_masks,
    alignment_keypoints,
    create_aligned_masks_array,
    create_alignment_keypoints_array,
)

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from scipy.signal import savgol_filter
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session
from torch.utils.data import Dataset, DataLoader

from vidseq.models.alignment_label import AlignmentLabel
from vidseq.models.video import Video
from vidseq.services.cropped_video_service import cropped_video_exists, get_cropped_video_path
from vidseq.services.database_manager import DatabaseManager
from vidseq.services.exceptions import AlignmentTrainingError

# Constants
DINOV2_INPUT_SIZE = 224  # Input size for DINOv2 (must be divisible by 14)

# ImageNet normalization for DINOv2
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# For backward compatibility
ALIGNMENT_INPUT_SIZE = DINOV2_INPUT_SIZE

# Savitzky-Golay smoothing for keypoint coordinates (bidirectional polynomial filter)
SAVGOL_WINDOW_LENGTH = 11  # ~367ms at 30fps — fits cubic polynomial over this window
SAVGOL_POLYORDER = 3       # Cubic polynomial — preserves acceleration in real movements

# DINOv2 feature averaging window (centered sliding window before decoder)
FEATURE_AVG_WINDOW = 5     # ~167ms at 30fps — average 5 DINOv2 feature tensors before decoding


@dataclass
class TrainingProgress:
    """Real-time training progress state for SSE streaming."""

    is_training: bool = False
    current_epoch: int = 0
    max_epochs: int = 100

    # Training loss
    current_train_loss: float = 0.0
    train_loss_history: list[float] = field(default_factory=list)

    # Validation loss
    current_val_loss: float = 0.0
    val_loss_history: list[float] = field(default_factory=list)

    # Best model tracking (based on validation loss)
    best_val_loss: float = float("inf")
    best_epoch: int = 0

    # Learning rate
    current_lr: float = 1e-4

    # Patience counters (based on val loss)
    epochs_without_improvement: int = 0
    lr_patience: int = 3
    early_stop_patience: int = 5
    lr_reduced_this_plateau: bool = False

    # Status
    status: str = "idle"  # idle, training, completed, stopped, failed
    started_at: Optional[float] = None

    # Dataset info
    num_train_labels: int = 0
    num_val_labels: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        # Convert inf to None for JSON compatibility
        if d["best_val_loss"] == float("inf"):
            d["best_val_loss"] = None
        # Add backward-compatible field aliases
        d["current_loss"] = d["current_train_loss"]
        d["best_loss"] = d["best_val_loss"]
        d["loss_history"] = d["train_loss_history"]
        return d


@dataclass
class AlignmentProgress:
    """Real-time alignment (apply) progress state for SSE streaming."""

    is_aligning: bool = False
    status: str = "idle"  # idle, aligning, completed, failed

    # Video-level progress
    current_video_index: int = 0
    total_videos: int = 0
    current_video_name: str = ""

    # Frame-level progress
    current_frame: int = 0
    total_frames: int = 0

    # Performance metrics
    fps: float = 0.0  # Rolling average
    eta_seconds: float = 0.0

    # Timestamps
    started_at: Optional[float] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)



# Configure logger for alignment service
logger = logging.getLogger("vidseq.alignment")
logger.setLevel(logging.DEBUG)


# --- DINOv2 Model Loading ---
_dinov2_model: Optional[nn.Module] = None
_dinov2_device: Optional[torch.device] = None

# Enable cuDNN benchmark for fixed input sizes (224x224)
torch.backends.cudnn.benchmark = True


def _load_dinov2(device: torch.device) -> nn.Module:
    """Load frozen DINOv2 ViT-g model (cached singleton).

    Applies torch.compile() for optimized inference on first load.

    Args:
        device: Device to load model on

    Returns:
        Frozen DINOv2 model in eval mode
    """
    global _dinov2_model, _dinov2_device

    if _dinov2_model is not None and _dinov2_device == device:
        return _dinov2_model

    logger.info("_load_dinov2: loading dinov2_vitg14_reg (this may take a while on first run)")
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_reg')
    model = model.to(device)
    model.eval()

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Compile for faster inference (one-time cost on first forward pass)
    if device.type == 'cuda':
        logger.info("_load_dinov2: compiling model with torch.compile() (first inference will be slow)")
        model = torch.compile(model)

    _dinov2_model = model
    _dinov2_device = device
    logger.info("_load_dinov2: model loaded and frozen")

    return model


class HeadingVectorDecoder(nn.Module):
    """Predicts heading direction (cos θ, sin θ) from DINOv2 features.

    Uses global average pooling to collapse spatial dimensions, then an MLP
    to predict a 2D unit vector representing the animal's heading direction.
    """

    def __init__(self, in_channels: int = 1536):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: DINOv2 patch tokens reshaped to (B, 1536, 16, 16)

        Returns:
            Unit vectors of shape (B, 2) representing (cos θ, sin θ)
        """
        x = self.pool(x)          # (B, 1536, 1, 1)
        x = x.flatten(1)          # (B, 1536)
        x = self.mlp(x)           # (B, 2)
        x = nn.functional.normalize(x, dim=-1)  # Unit vector
        return x


def preprocess_for_dinov2(
    frame: np.ndarray,
) -> tuple[torch.Tensor, float, int, int, int, int]:
    """Preprocess frame for DINOv2 input.

    Rescales so longest side = 224, pads to 224×224 square with gray,
    applies ImageNet normalization.

    Args:
        frame: Input frame (H, W, 3) RGB uint8

    Returns:
        (tensor, scale, pad_left, pad_top, orig_w, orig_h)
        - tensor: (3, 224, 224) normalized tensor
        - scale: Scale factor applied
        - pad_left, pad_top: Padding offsets
        - orig_w, orig_h: Original dimensions
    """
    orig_h, orig_w = frame.shape[:2]

    # Calculate scale to fit longest side to 224
    scale = DINOV2_INPUT_SIZE / max(orig_w, orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)

    # Resize
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Pad to 224×224 with gray (centered)
    pad_left = (DINOV2_INPUT_SIZE - new_w) // 2
    pad_top = (DINOV2_INPUT_SIZE - new_h) // 2
    pad_right = DINOV2_INPUT_SIZE - new_w - pad_left
    pad_bottom = DINOV2_INPUT_SIZE - new_h - pad_top

    padded = cv2.copyMakeBorder(
        resized,
        pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT,
        value=(128, 128, 128)  # Gray
    )

    # Convert to float and normalize
    tensor = torch.from_numpy(padded).float() / 255.0
    tensor = tensor.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)

    # Apply ImageNet normalization
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    tensor = (tensor - mean) / std

    return tensor, scale, pad_left, pad_top, orig_w, orig_h


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


# --- Augmentation Configuration ---
AUGMENT_VERSIONS_PER_SAMPLE = 10  # Number of augmented versions per training sample
AUGMENT_CATEGORY_PROB = 0.5  # 50% chance each category is activated

# --- Train/Validation Split Configuration ---
TRAIN_VAL_SPLIT_SEED = 42  # Fixed seed for reproducible splits
TRAIN_VAL_SPLIT_RATIO = 0.8  # 80% train, 20% val
MIN_LABELS_FOR_VALIDATION = 5  # Minimum labels needed for meaningful validation


def split_labels_train_val(
    labels: list,
    train_ratio: float = TRAIN_VAL_SPLIT_RATIO,
    seed: int = TRAIN_VAL_SPLIT_SEED,
) -> tuple[list, list]:
    """Split labels into train and validation sets.

    Args:
        labels: List of AlignmentLabel objects
        train_ratio: Fraction of labels for training (default 0.8)
        seed: Random seed for reproducible splits

    Returns:
        (train_labels, val_labels) tuple
    """
    # Copy list to avoid modifying original
    labels_copy = list(labels)

    # Shuffle with fixed seed
    rng = random.Random(seed)
    rng.shuffle(labels_copy)

    # Split
    split_idx = int(len(labels_copy) * train_ratio)
    train_labels = labels_copy[:split_idx]
    val_labels = labels_copy[split_idx:]

    logger.info(
        f"split_labels_train_val: {len(labels)} total -> "
        f"{len(train_labels)} train, {len(val_labels)} val "
        f"(ratio={train_ratio}, seed={seed})"
    )

    return train_labels, val_labels


def _apply_geometric_augmentation(
    frame: np.ndarray,
    front_x: float,
    front_y: float,
    rear_x: float,
    rear_y: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float, float, float, float]:
    """Apply random rotation and optional flips.

    Args:
        frame: Input frame (H, W, 3)
        front_x, front_y: Front keypoint normalized coords (0-1)
        rear_x, rear_y: Rear keypoint normalized coords (0-1)
        rng: NumPy random generator for reproducibility

    Returns:
        (augmented_frame, new_front_x, new_front_y, new_rear_x, new_rear_y)
    """
    h, w = frame.shape[:2]

    # Random rotation angle
    angle = rng.uniform(-180, 180)
    center = (w / 2, h / 2)
    rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    frame = cv2.warpAffine(frame, rot_matrix, (w, h))

    # Transform coordinates (rotate around center)
    def rotate_point(x: float, y: float) -> tuple[float, float]:
        px, py = x * w, y * h
        cos_a, sin_a = np.cos(np.radians(angle)), np.sin(np.radians(angle))
        cx, cy = w / 2, h / 2
        px_new = cos_a * (px - cx) - sin_a * (py - cy) + cx
        py_new = sin_a * (px - cx) + cos_a * (py - cy) + cy
        return px_new / w, py_new / h

    front_x, front_y = rotate_point(front_x, front_y)
    rear_x, rear_y = rotate_point(rear_x, rear_y)

    # Optional horizontal flip (50% chance)
    if rng.random() < 0.5:
        frame = cv2.flip(frame, 1)
        front_x, rear_x = 1 - front_x, 1 - rear_x

    # Optional vertical flip (50% chance)
    if rng.random() < 0.5:
        frame = cv2.flip(frame, 0)
        front_y, rear_y = 1 - front_y, 1 - rear_y

    return frame, front_x, front_y, rear_x, rear_y


def _apply_color_augmentation(
    frame: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply brightness, contrast, saturation adjustments.

    Args:
        frame: Input frame (H, W, 3) uint8
        rng: NumPy random generator for reproducibility

    Returns:
        Augmented frame (H, W, 3) uint8
    """
    frame = frame.astype(np.float32)

    # Brightness: multiply by factor
    brightness = rng.uniform(0.7, 1.3)
    frame = frame * brightness

    # Contrast: scale around mean
    contrast = rng.uniform(0.7, 1.3)
    mean = frame.mean()
    frame = (frame - mean) * contrast + mean

    # Saturation: blend with grayscale
    saturation = rng.uniform(0.8, 1.2)
    gray = cv2.cvtColor(frame.clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB).astype(np.float32)
    frame = gray_rgb + saturation * (frame - gray_rgb)

    # Optional Gaussian blur (30% chance)
    if rng.random() < 0.3:
        kernel_size = int(rng.choice([3, 5]))
        frame = cv2.GaussianBlur(
            frame.clip(0, 255).astype(np.uint8),
            (kernel_size, kernel_size),
            0,
        ).astype(np.float32)

    return frame.clip(0, 255).astype(np.uint8)


def _apply_erasing_augmentation(
    frame: np.ndarray,
    front_x: float,
    front_y: float,
    rear_x: float,
    rear_y: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply random erasing (cutout or coarse dropout).

    Avoids erasing regions near keypoints.

    Args:
        frame: Input frame (H, W, 3)
        front_x, front_y: Front keypoint normalized coords (0-1)
        rear_x, rear_y: Rear keypoint normalized coords (0-1)
        rng: NumPy random generator for reproducibility

    Returns:
        Augmented frame with erased regions
    """
    h, w = frame.shape[:2]
    frame = frame.copy()

    # Define keypoint exclusion zones (don't erase keypoints)
    exclusion_radius = 0.1  # 10% of image around each keypoint

    def is_safe_region(cx: float, cy: float, size: float) -> bool:
        """Check if erasing region doesn't overlap keypoints."""
        for kx, ky in [(front_x, front_y), (rear_x, rear_y)]:
            if abs(cx - kx) < (size / 2 + exclusion_radius) and abs(cy - ky) < (
                size / 2 + exclusion_radius
            ):
                return False
        return True

    if rng.random() < 0.7:
        # Single rectangular cutout (70% chance)
        for _ in range(10):  # Try up to 10 times to find safe region
            size = rng.uniform(0.1, 0.2)
            cx = rng.uniform(size / 2, 1 - size / 2)
            cy = rng.uniform(size / 2, 1 - size / 2)
            if is_safe_region(cx, cy, size):
                x1 = int((cx - size / 2) * w)
                y1 = int((cy - size / 2) * h)
                x2 = int((cx + size / 2) * w)
                y2 = int((cy + size / 2) * h)
                frame[y1:y2, x1:x2] = rng.integers(
                    0, 255, size=(y2 - y1, x2 - x1, 3), dtype=np.uint8
                )
                break
    else:
        # Coarse dropout (30% chance) - multiple small squares
        n_squares = rng.integers(5, 11)
        for _ in range(n_squares):
            size = rng.uniform(0.02, 0.05)
            cx = rng.uniform(size / 2, 1 - size / 2)
            cy = rng.uniform(size / 2, 1 - size / 2)
            if is_safe_region(cx, cy, size):
                x1 = int((cx - size / 2) * w)
                y1 = int((cy - size / 2) * h)
                x2 = int((cx + size / 2) * w)
                y2 = int((cy + size / 2) * h)
                frame[y1:y2, x1:x2] = rng.integers(
                    0, 255, size=(y2 - y1, x2 - x1, 3), dtype=np.uint8
                )

    return frame


def _apply_scaling_augmentation(
    frame: np.ndarray,
    front_x: float,
    front_y: float,
    rear_x: float,
    rear_y: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float, float, float, float]:
    """Apply random scaling (zoom in/out).

    Args:
        frame: Input frame (H, W, 3)
        front_x, front_y: Front keypoint normalized coords (0-1)
        rear_x, rear_y: Rear keypoint normalized coords (0-1)
        rng: NumPy random generator for reproducibility

    Returns:
        (augmented_frame, new_front_x, new_front_y, new_rear_x, new_rear_y)
    """
    h, w = frame.shape[:2]
    scale = rng.uniform(0.85, 1.15)

    # Calculate new dimensions
    new_h, new_w = int(h * scale), int(w * scale)

    if scale > 1:
        # Zoom in: resize larger, then center crop
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        start_x = (new_w - w) // 2
        start_y = (new_h - h) // 2
        frame = resized[start_y : start_y + h, start_x : start_x + w]

        # Adjust coordinates: shift toward center
        front_x = (front_x - 0.5) * scale + 0.5
        front_y = (front_y - 0.5) * scale + 0.5
        rear_x = (rear_x - 0.5) * scale + 0.5
        rear_y = (rear_y - 0.5) * scale + 0.5
    else:
        # Zoom out: resize smaller, then pad
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        pad_x = (w - new_w) // 2
        pad_y = (h - new_h) // 2
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

        # Adjust coordinates: shift away from center
        front_x = (front_x - 0.5) * scale + 0.5
        front_y = (front_y - 0.5) * scale + 0.5
        rear_x = (rear_x - 0.5) * scale + 0.5
        rear_y = (rear_y - 0.5) * scale + 0.5

    # Clamp coordinates to [0, 1]
    front_x = float(np.clip(front_x, 0, 1))
    front_y = float(np.clip(front_y, 0, 1))
    rear_x = float(np.clip(rear_x, 0, 1))
    rear_y = float(np.clip(rear_y, 0, 1))

    return frame, front_x, front_y, rear_x, rear_y



def rotate_frame(frame: np.ndarray, angle_degrees: float) -> np.ndarray:
    """Rotate frame around center by given angle.

    Args:
        frame: (H, W, 3) BGR image
        angle_degrees: Rotation angle (positive = counterclockwise)

    Returns:
        Rotated frame (same dimensions)
    """
    h, w = frame.shape[:2]
    center = (w / 2, h / 2)  # Frame center

    rotation_matrix = cv2.getRotationMatrix2D(center, angle_degrees, scale=1.0)
    rotated = cv2.warpAffine(frame, rotation_matrix, (w, h))

    return rotated


def rotate_mask(mask: np.ndarray, angle_degrees: float, threshold: int = 127) -> np.ndarray:
    """Rotate a binary mask around center and re-threshold to binary.

    Args:
        mask: (H, W) uint8 mask
        angle_degrees: Rotation angle (positive = counterclockwise)
        threshold: Threshold value for re-binarization after rotation (default 127)

    Returns:
        Rotated binary mask (same dimensions)
    """
    h, w = mask.shape[:2]
    center = (w / 2, h / 2)

    rotation_matrix = cv2.getRotationMatrix2D(center, angle_degrees, scale=1.0)
    # Use INTER_LINEAR for smoother rotation, then threshold back to binary
    rotated = cv2.warpAffine(mask, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR)

    # Re-threshold to binary (values > threshold become 255, else 0)
    _, binary = cv2.threshold(rotated, threshold, 255, cv2.THRESH_BINARY)

    return binary


class AlignmentDataset(Dataset):
    """PyTorch Dataset for alignment model training.

    Loads frames from cropped videos and generates gaussian heatmap targets.
    Optionally applies probabilistic augmentations (geometric, color, erasing, scaling)
    with 50% chance per category, expanding training data 10x.
    """

    def __init__(
        self,
        labels: list,
        project_path: Path,
        video_name_map: dict[int, str],
        versions_per_sample: int = AUGMENT_VERSIONS_PER_SAMPLE,
        augment: bool = True,
    ):
        """Initialize the dataset.

        Args:
            labels: List of AlignmentLabel objects (or dicts with same fields)
            project_path: Path to project folder
            video_name_map: Dict mapping video_id -> video.name for file lookup
            versions_per_sample: Number of augmented versions per label (default 10)
            augment: Whether to apply data augmentation (default True)
        """
        self.project_path = project_path
        self.video_name_map = video_name_map
        self.size = ALIGNMENT_INPUT_SIZE
        self.augment = augment
        self.labels = labels

        if augment:
            # Create sample indices: (label_idx, version_idx)
            # Each version gets a unique seed for reproducibility
            self.versions_per_sample = versions_per_sample
            self.samples: list[tuple[int, int]] = []
            for label_idx in range(len(labels)):
                for version_idx in range(versions_per_sample):
                    self.samples.append((label_idx, version_idx))

            logger.info(
                f"AlignmentDataset: {len(labels)} labels expanded to {len(self.samples)} samples "
                f"({versions_per_sample}x augmentation, augment=True)"
            )
        else:
            # No augmentation: one sample per label
            self.versions_per_sample = 1
            self.samples = [(label_idx, 0) for label_idx in range(len(labels))]

            logger.info(
                f"AlignmentDataset: {len(labels)} labels = {len(self.samples)} samples "
                f"(no augmentation, augment=False)"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        label_idx, version_idx = self.samples[idx]
        label = self.labels[label_idx]

        # Fresh RNG each time - different augmentation every epoch (standard practice)
        rng = np.random.default_rng()

        # Get video name for file lookup
        video_name = self.video_name_map[label.video_id]
        cropped_path = get_cropped_video_path(self.project_path, video_name)

        # Load frame from video
        cap = cv2.VideoCapture(str(cropped_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, label.frame_idx)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise RuntimeError(
                f"Failed to read frame {label.frame_idx} from {cropped_path}"
            )

        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = frame.shape[:2]

        # Get keypoint coordinates (normalized 0-1)
        front_x, front_y = label.front_x, label.front_y
        rear_x, rear_y = label.rear_x, label.rear_y

        # Apply augmentations only for training set (not validation)
        if self.augment:
            # Apply augmentations probabilistically (50% chance each category)

            # 1. Geometric (rotation + flips) - affects coordinates
            if rng.random() < AUGMENT_CATEGORY_PROB:
                frame, front_x, front_y, rear_x, rear_y = _apply_geometric_augmentation(
                    frame, front_x, front_y, rear_x, rear_y, rng
                )

            # 2. Color (brightness, contrast, saturation, blur) - no coord change
            if rng.random() < AUGMENT_CATEGORY_PROB:
                frame = _apply_color_augmentation(frame, rng)

            # 3. Random erasing (cutout or dropout) - no coord change
            if rng.random() < AUGMENT_CATEGORY_PROB:
                frame = _apply_erasing_augmentation(
                    frame, front_x, front_y, rear_x, rear_y, rng
                )

            # 4. Scaling (zoom in/out) - affects coordinates
            if rng.random() < AUGMENT_CATEGORY_PROB:
                frame, front_x, front_y, rear_x, rear_y = _apply_scaling_augmentation(
                    frame, front_x, front_y, rear_x, rear_y, rng
                )

        # Preprocess for DINOv2 (rescale longest side to 224, pad to square, normalize)
        frame_tensor, scale, pad_left, pad_top, _, _ = preprocess_for_dinov2(frame)

        # Compute heading angle from keypoints
        dx = (front_x - rear_x) * orig_w
        dy = (front_y - rear_y) * orig_h
        angle = np.arctan2(dy, dx)

        # Target is (cos θ, sin θ) unit vector
        target = torch.tensor([np.cos(angle), np.sin(angle)], dtype=torch.float32)

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

        # Training progress tracking for real-time updates
        self._training_progress = TrainingProgress()

        # Alignment (apply) progress tracking for real-time updates
        self._alignment_progress = AlignmentProgress()
        self._fps_timestamps: collections.deque = collections.deque(maxlen=30)

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

    def get_training_progress(self) -> TrainingProgress:
        """Get current training progress for SSE streaming."""
        return self._training_progress

    def get_alignment_progress(self) -> AlignmentProgress:
        """Get current alignment (apply) progress for SSE streaming."""
        return self._alignment_progress

    def _calculate_rolling_fps(self) -> float:
        """Calculate rolling average FPS from recent frame timestamps."""
        if len(self._fps_timestamps) < 2:
            return 0.0
        # Time span across all tracked frames
        time_span = self._fps_timestamps[-1] - self._fps_timestamps[0]
        if time_span <= 0:
            return 0.0
        # FPS = (num_frames - 1) / time_span
        return (len(self._fps_timestamps) - 1) / time_span

    def get_model_path(self, project_path: Path) -> Path:
        """Get path to the alignment model weights file."""
        return project_path / "alignment_model.pt"

    def is_model_trained(self, project_path: Path) -> bool:
        """Check if alignment model has been trained."""
        return self.get_model_path(project_path).exists()

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
        project_path: Path,
    ) -> Optional[tuple[int, int]]:
        """Get a random unlabeled frame from videos with cropping completed.

        Returns:
            (video_id, frame_idx) tuple or None if no frames available
        """
        logger.info("get_random_unlabeled_frame: starting")

        # Get videos with cropping completed (check filesystem)
        videos_result = await session.execute(select(Video))
        all_videos = list(videos_result.scalars().all())
        videos = await asyncio.to_thread(
            lambda: [v for v in all_videos if cropped_video_exists(project_path, v.name)]
        )
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

    async def get_alignment_status(
        self, session: AsyncSession, project_path: Path
    ) -> dict:
        """Get current alignment status including label count and cropped video status.

        Args:
            session: Async database session
            project_path: Path to the project folder

        Returns:
            Dict with keys: label_count, model_trained, is_training, is_applying, all_videos_cropped
        """
        label_count = await self.get_label_count(session)
        model_trained = self.is_model_trained(project_path)
        is_training = self.is_training()
        is_applying = self.is_applying()

        # Check if all videos have cropping completed
        result = await session.execute(select(Video))
        videos = list(result.scalars().all())
        video_count = len(videos)

        def _count_cropped() -> int:
            return sum(
                1 for v in videos
                if cropped_video_exists(project_path, v.name)
            )

        cropped_count = await asyncio.to_thread(_count_cropped)
        all_videos_cropped = video_count > 0 and cropped_count == video_count

        logger.info(
            f"get_alignment_status: label_count={label_count}, model_trained={model_trained}, "
            f"is_training={is_training}, is_applying={is_applying}, "
            f"videos={video_count}, cropped={cropped_count}, all_videos_cropped={all_videos_cropped}"
        )

        return {
            "label_count": label_count,
            "model_trained": model_trained,
            "is_training": is_training,
            "is_applying": is_applying,
            "all_videos_cropped": all_videos_cropped,
        }

    async def create_alignment_training(
        self,
        session: AsyncSession,
        project_path: Path,
        video_ids: list[int],
        epochs: int = 100,
        augment: bool = True,
        early_stop_patience: int = 5,
        lr_patience: int = 3,
    ) -> dict:
        """Start alignment model training.

        Fetches labels for selected videos, builds video name map,
        validates, and starts training in a background thread.

        Args:
            session: Async database session
            project_path: Path to the project folder
            video_ids: List of video IDs to use for training
            epochs: Maximum training epochs
            augment: Whether to use data augmentation
            early_stop_patience: Epochs without improvement before stopping
            lr_patience: Epochs without improvement before reducing LR

        Returns:
            Dict with status and training parameters

        Raises:
            AlignmentTrainingError: If training already in progress or no labels available
        """
        if self.is_training():
            raise AlignmentTrainingError("Training already in progress")

        # Fetch labels only for selected videos
        result = await session.execute(
            select(AlignmentLabel)
            .where(AlignmentLabel.video_id.in_(video_ids))
            .order_by(AlignmentLabel.id)
        )
        labels = list(result.scalars().all())

        if len(labels) == 0:
            raise AlignmentTrainingError("No labels available for training in selected videos")

        # Build video_name_map: video_id -> video.name
        label_video_ids = list({label.video_id for label in labels})
        result = await session.execute(select(Video).where(Video.id.in_(label_video_ids)))
        videos = list(result.scalars().all())
        video_name_map = {v.id: v.name for v in videos}

        logger.info(
            f"create_alignment_training: starting training with {len(labels)} labels, "
            f"{len(video_name_map)} videos, max_epochs={epochs}, augment={augment}"
        )

        # Start training in background thread
        asyncio.create_task(
            asyncio.to_thread(
                self.train_model_sync,
                project_path,
                labels,
                video_name_map,
                max_epochs=epochs,
                augment=augment,
                early_stop_patience=early_stop_patience,
                lr_patience=lr_patience,
            )
        )

        return {"status": "started", "max_epochs": epochs, "augment": augment}

    async def create_videos_alignment(self, project_path: Path, video_ids: list[int]) -> dict:
        """Apply alignment to selected cropped videos.

        Creates aligned videos in <project>/aligned_videos/ folder.
        Starts processing in a background thread.

        Args:
            project_path: Path to the project folder
            video_ids: List of video IDs to apply alignment to

        Returns:
            Dict with status

        Raises:
            AlignmentTrainingError: If alignment already in progress or model not trained
        """
        if self.is_applying():
            raise AlignmentTrainingError("Alignment already in progress")

        if not self.is_model_trained(project_path):
            raise AlignmentTrainingError("Model not trained yet")

        # Get project engine for sync operations
        db_manager = DatabaseManager.get_instance()
        project_engine = db_manager.get_project_engine(project_path)

        logger.info(
            f"create_videos_alignment: starting alignment in background for project at {project_path}"
        )

        # Start alignment in background thread
        asyncio.create_task(
            asyncio.to_thread(
                self.apply_alignment_sync,
                project_path,
                project_engine,
                video_ids,
            )
        )

        return {"status": "started"}

    def train_model_sync(
        self,
        project_path: Path,
        labels: list,
        video_name_map: dict[int, str],
        max_epochs: int = 100,
        batch_size: int = 8,
        lr: float = 1e-4,
        lr_patience: int = 3,
        lr_factor: float = 0.25,
        early_stop_patience: int = 5,
        augment: bool = True,
    ) -> bool:
        """Train the alignment model using labeled data with train/val split.

        Uses early stopping and learning rate reduction on plateau based on
        validation loss (or training loss if insufficient labels for validation):
        - If no improvement for `lr_patience` epochs, reduce LR by `lr_factor`
        - If no improvement for `early_stop_patience` epochs, stop training
        - Saves the best model (lowest validation loss), not the final epoch

        Args:
            project_path: Path to project folder
            labels: List of AlignmentLabel objects
            video_name_map: Dict mapping video_id -> video.name
            max_epochs: Maximum number of training epochs (default 100)
            batch_size: Training batch size
            lr: Initial learning rate
            lr_patience: Epochs without improvement before reducing LR (default 3)
            lr_factor: Factor to multiply LR by when reducing (default 0.25)
            early_stop_patience: Epochs without improvement before stopping (default 5)

        Returns:
            True if successful
        """
        logger.info(
            f"train_model_sync: starting training with max_epochs={max_epochs}, "
            f"batch_size={batch_size}, lr={lr}, num_labels={len(labels)}"
        )
        logger.info(
            f"train_model_sync: lr_patience={lr_patience}, lr_factor={lr_factor}, "
            f"early_stop_patience={early_stop_patience}"
        )
        self._is_training = True

        # Split labels into train/val sets
        use_validation = len(labels) >= MIN_LABELS_FOR_VALIDATION
        if use_validation:
            train_labels, val_labels = split_labels_train_val(labels)
        else:
            train_labels = labels
            val_labels = []
            logger.warning(
                f"train_model_sync: only {len(labels)} labels, need >= {MIN_LABELS_FOR_VALIDATION} "
                f"for validation split. Using training loss for early stopping."
            )

        # Initialize training progress
        self._training_progress = TrainingProgress(
            is_training=True,
            current_epoch=0,
            max_epochs=max_epochs,
            current_train_loss=0.0,
            train_loss_history=[],
            current_val_loss=0.0,
            val_loss_history=[],
            best_val_loss=float("inf"),
            best_epoch=0,
            current_lr=lr,
            epochs_without_improvement=0,
            lr_patience=lr_patience,
            early_stop_patience=early_stop_patience,
            lr_reduced_this_plateau=False,
            status="training",
            started_at=time.time(),
            num_train_labels=len(train_labels),
            num_val_labels=len(val_labels),
        )

        try:
            # Setup device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"train_model_sync: using device={device}")

            # Load frozen DINOv2 feature extractor
            dinov2 = _load_dinov2(device)
            logger.info("train_model_sync: loaded frozen DINOv2 ViT-g")

            # Create trainable decoder
            decoder = HeadingVectorDecoder(in_channels=1536)
            decoder = decoder.to(device)

            # Count trainable parameters
            num_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
            logger.info(f"train_model_sync: created HeadingVectorDecoder with {num_params:,} trainable parameters")

            # Create train dataset with augmentation
            train_dataset = AlignmentDataset(
                train_labels, project_path, video_name_map, augment=augment
            )
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
            )
            logger.info(
                f"train_model_sync: train dataset - {len(train_labels)} labels -> "
                f"{len(train_dataset)} samples (with augmentation)"
            )

            # Create val dataset without augmentation (if we have validation labels)
            val_dataloader = None
            if use_validation:
                val_dataset = AlignmentDataset(
                    val_labels, project_path, video_name_map, augment=False
                )
                val_dataloader = DataLoader(
                    val_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=0,
                )
                logger.info(
                    f"train_model_sync: val dataset - {len(val_labels)} labels -> "
                    f"{len(val_dataset)} samples (no augmentation)"
                )

            # Setup optimizer (only decoder params) and loss
            optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
            criterion = nn.MSELoss()

            # Early stopping and LR reduction tracking (based on val loss)
            best_val_loss = float("inf")
            best_model_state = None
            best_epoch = 0
            epochs_without_improvement = 0
            current_lr = lr
            lr_reduced_this_plateau = False
            min_lr = 1e-7

            # Use bfloat16 autocast for faster training on Ampere+ GPUs
            use_amp = device.type == 'cuda'
            amp_dtype = torch.bfloat16

            # Training loop
            for epoch in range(max_epochs):
                # --- Training phase ---
                decoder.train()
                train_loss = 0.0
                train_batches = 0

                for frames, targets in train_dataloader:
                    frames = frames.to(device)
                    targets = targets.to(device)

                    optimizer.zero_grad()

                    with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                        # Extract DINOv2 features (no grad needed for frozen encoder)
                        with torch.no_grad():
                            # DINOv2 forward returns class token + patch tokens
                            # We use forward_features to get intermediate features
                            features = dinov2.forward_features(frames)
                            # features dict contains 'x_norm_patchtokens' of shape (B, 256, 1536)
                            # Reshape to (B, 1536, 16, 16) for the decoder
                            patch_tokens = features["x_norm_patchtokens"]  # (B, 256, 1536)
                            B = patch_tokens.shape[0]
                            patch_tokens = patch_tokens.permute(0, 2, 1)  # (B, 1536, 256)
                            patch_tokens = patch_tokens.reshape(B, 1536, 16, 16)  # (B, 1536, 16, 16)

                        # Pass through trainable decoder
                        outputs = decoder(patch_tokens)
                        loss = criterion(outputs, targets)

                    # Backward pass outside autocast (bfloat16 doesn't need GradScaler)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()
                    train_batches += 1

                avg_train_loss = train_loss / max(train_batches, 1)

                # --- Validation phase ---
                if use_validation and val_dataloader is not None:
                    decoder.eval()
                    val_loss = 0.0
                    val_batches = 0

                    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                        for frames, targets in val_dataloader:
                            frames = frames.to(device)
                            targets = targets.to(device)

                            # Extract DINOv2 features
                            features = dinov2.forward_features(frames)
                            patch_tokens = features["x_norm_patchtokens"]
                            B = patch_tokens.shape[0]
                            patch_tokens = patch_tokens.permute(0, 2, 1)
                            patch_tokens = patch_tokens.reshape(B, 1536, 16, 16)

                            outputs = decoder(patch_tokens)
                            loss = criterion(outputs, targets)

                            val_loss += loss.item()
                            val_batches += 1

                    avg_val_loss = val_loss / max(val_batches, 1)
                else:
                    # No validation set - use training loss for early stopping
                    avg_val_loss = avg_train_loss

                # Use validation loss for improvement tracking
                loss_for_comparison = avg_val_loss

                # Check for improvement
                if loss_for_comparison < best_val_loss:
                    # Improvement - save best decoder state
                    best_val_loss = loss_for_comparison
                    best_model_state = copy.deepcopy(decoder.state_dict())
                    best_epoch = epoch + 1
                    epochs_without_improvement = 0
                    lr_reduced_this_plateau = False
                    if use_validation:
                        logger.info(
                            f"train_model_sync: epoch {epoch + 1}/{max_epochs}, "
                            f"train={avg_train_loss:.6f}, val={avg_val_loss:.6f} (new best), "
                            f"lr={current_lr:.2e}"
                        )
                    else:
                        logger.info(
                            f"train_model_sync: epoch {epoch + 1}/{max_epochs}, "
                            f"loss={avg_train_loss:.6f} (new best), lr={current_lr:.2e}"
                        )
                else:
                    # No improvement
                    epochs_without_improvement += 1
                    if use_validation:
                        logger.info(
                            f"train_model_sync: epoch {epoch + 1}/{max_epochs}, "
                            f"train={avg_train_loss:.6f}, val={avg_val_loss:.6f}, "
                            f"no improvement x{epochs_without_improvement}, lr={current_lr:.2e}"
                        )
                    else:
                        logger.info(
                            f"train_model_sync: epoch {epoch + 1}/{max_epochs}, "
                            f"loss={avg_train_loss:.6f}, no improvement x{epochs_without_improvement}, "
                            f"lr={current_lr:.2e}"
                        )

                    # Check for early stopping
                    if epochs_without_improvement >= early_stop_patience:
                        logger.info(
                            f"train_model_sync: early stopping after {epoch + 1} epochs "
                            f"(no improvement for {early_stop_patience} epochs)"
                        )
                        # Update progress for early stop
                        self._training_progress.current_epoch = epoch + 1
                        self._training_progress.current_train_loss = avg_train_loss
                        self._training_progress.train_loss_history.append(avg_train_loss)
                        self._training_progress.current_val_loss = avg_val_loss
                        self._training_progress.val_loss_history.append(avg_val_loss)
                        self._training_progress.best_val_loss = best_val_loss
                        self._training_progress.best_epoch = best_epoch
                        self._training_progress.epochs_without_improvement = epochs_without_improvement
                        self._training_progress.is_training = False
                        self._training_progress.status = "stopped"
                        break

                    # Check for LR reduction (only reduce once per plateau)
                    if (
                        epochs_without_improvement >= lr_patience
                        and not lr_reduced_this_plateau
                        and current_lr > min_lr
                    ):
                        current_lr *= lr_factor
                        for param_group in optimizer.param_groups:
                            param_group["lr"] = current_lr
                        lr_reduced_this_plateau = True
                        logger.info(
                            f"train_model_sync: reducing learning rate to {current_lr:.2e}"
                        )

                # Update training progress after each epoch
                self._training_progress.current_epoch = epoch + 1
                self._training_progress.current_train_loss = avg_train_loss
                self._training_progress.train_loss_history.append(avg_train_loss)
                self._training_progress.current_val_loss = avg_val_loss
                self._training_progress.val_loss_history.append(avg_val_loss)
                self._training_progress.best_val_loss = best_val_loss
                self._training_progress.best_epoch = best_epoch
                self._training_progress.current_lr = current_lr
                self._training_progress.epochs_without_improvement = epochs_without_improvement
                self._training_progress.lr_reduced_this_plateau = lr_reduced_this_plateau

            # Save best decoder weights (not final)
            model_path = self.get_model_path(project_path)
            if best_model_state is not None:
                decoder.load_state_dict(best_model_state)
                logger.info(
                    f"train_model_sync: restoring best decoder from epoch {best_epoch} "
                    f"(val_loss={best_val_loss:.6f})"
                )
            torch.save(decoder.state_dict(), model_path)
            logger.info(f"train_model_sync: saved decoder weights to {model_path}")

            # Clear cached model so next predict loads the new weights
            self._model = None
            self._model_path = None

            logger.info(
                f"train_model_sync: training complete (best val_loss={best_val_loss:.6f} "
                f"at epoch {best_epoch})"
            )

            # Mark training as completed (if not already stopped by early stopping)
            if self._training_progress.status == "training":
                self._training_progress.is_training = False
                self._training_progress.status = "completed"

            return True

        except Exception as e:
            logger.error(f"train_model_sync: training failed with error: {e}")
            import traceback
            logger.error(traceback.format_exc())

            # Mark training as failed
            self._training_progress.is_training = False
            self._training_progress.status = "failed"

            return False

        finally:
            self._is_training = False
            logger.debug("train_model_sync: is_training set to False")

    def _load_model(self, project_path: Path) -> nn.Module:
        """Load the decoder from disk, using cache if available.

        Args:
            project_path: Path to project folder

        Returns:
            Loaded decoder in eval mode

        Raises:
            FileNotFoundError: If model weights file doesn't exist
        """
        model_path = self.get_model_path(project_path)

        # Check if we need to reload
        if self._model is not None and self._model_path == model_path:
            return self._model

        if not model_path.exists():
            raise FileNotFoundError(f"Model weights not found: {model_path}")

        logger.info(f"_load_model: loading decoder from {model_path}")

        # Setup device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"_load_model: using device={device}")

        # Create decoder architecture
        decoder = HeadingVectorDecoder(in_channels=1536)

        # Load weights
        state_dict = torch.load(model_path, map_location=device)
        decoder.load_state_dict(state_dict)
        decoder = decoder.to(device)
        decoder.eval()

        # Cache for future use
        self._model = decoder
        self._model_path = model_path
        self._device = device

        logger.info(f"_load_model: decoder loaded and cached")
        return decoder

    def _extract_features(
        self,
        frame: np.ndarray,
        dinov2: nn.Module,
        device: torch.device,
    ) -> torch.Tensor:
        """Extract DINOv2 patch token features from a frame.

        Args:
            frame: Input frame as (H, W, 3) uint8 BGR array
            dinov2: Loaded DINOv2 model
            device: Torch device

        Returns:
            Patch tokens tensor of shape (1, 1536, 16, 16)
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor, scale, pad_left, pad_top, _, _ = preprocess_for_dinov2(frame_rgb)
        frame_tensor = frame_tensor.unsqueeze(0).to(device)

        use_amp = device.type == 'cuda'
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
            features = dinov2.forward_features(frame_tensor)
            patch_tokens = features["x_norm_patchtokens"]  # (1, 256, 1536)
            patch_tokens = patch_tokens.permute(0, 2, 1)  # (1, 1536, 256)
            patch_tokens = patch_tokens.reshape(1, 1536, 16, 16)

        return patch_tokens

    def _decode_features(
        self,
        patch_tokens: torch.Tensor,
        decoder: nn.Module,
    ) -> tuple[float, float]:
        """Decode averaged DINOv2 features into a heading vector.

        Args:
            patch_tokens: Features of shape (1, 1536, 16, 16)
            decoder: Loaded HeadingVectorDecoder

        Returns:
            (cos_theta, sin_theta) tuple
        """
        device = patch_tokens.device
        use_amp = device.type == 'cuda'

        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
            output = decoder(patch_tokens)  # (1, 2)

        vec = output[0].float().cpu().numpy()  # (2,)
        return float(vec[0]), float(vec[1])

    def predict_sync(
        self,
        project_path: Path,
        frame: np.ndarray,
    ) -> tuple[float, float]:
        """Run inference on a frame to predict heading direction.

        Args:
            project_path: Path to project folder
            frame: Input frame as (H, W, 3) uint8 BGR array

        Returns:
            (cos_theta, sin_theta) unit vector
        """
        decoder = self._load_model(project_path)
        device = self._device
        dinov2 = _load_dinov2(device)

        patch_tokens = self._extract_features(frame, dinov2, device)
        return self._decode_features(patch_tokens, decoder)

    def _reencode_to_h264(self, input_path: Path, output_path: Path) -> bool:
        """Re-encode video to H.264 for browser compatibility.

        Args:
            input_path: Path to input video (any codec)
            output_path: Path to output video (H.264/MP4)

        Returns:
            True if successful, False otherwise
        """
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

        cmd = [
            ffmpeg_path,
            "-y",  # Overwrite output
            "-i", str(input_path),
            "-c:v", "libx264",  # H.264 codec
            "-preset", "fast",  # Encoding speed/quality tradeoff
            "-crf", "23",  # Quality (lower = better, 18-28 is typical)
            "-pix_fmt", "yuv420p",  # Pixel format for browser compatibility
            "-movflags", "+faststart",  # Move moov atom to start for streaming
            str(output_path),
        ]

        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"_reencode_to_h264: FFmpeg error: {e.stderr}")
            return False

    def apply_alignment_sync(
        self,
        project_path: Path,
        project_engine,
        video_ids: list[int],
    ) -> bool:
        """Apply alignment to selected cropped videos.

        Reads cropped videos, rotates frames so animal faces right,
        saves to aligned_videos folder. Also rotates cropped masks
        and saves to array_data/{video_id}/aligned_masks.h5.

        Args:
            project_path: Path to project folder
            project_engine: SQLAlchemy engine for project DB
            video_ids: List of video IDs to apply alignment to

        Returns:
            True if successful
        """
        logger.info(f"apply_alignment_sync: starting, project={project_path.name}")
        self._is_applying = True

        # Initialize alignment progress
        self._alignment_progress = AlignmentProgress(
            is_aligning=True,
            status="aligning",
            started_at=time.time(),
        )
        self._fps_timestamps.clear()
        last_progress_update = time.time()

        try:
            # Get selected videos with cropping completed (check filesystem)
            with Session(project_engine) as session:
                result = session.execute(
                    select(Video).where(Video.id.in_(video_ids))
                )
                all_videos = list(result.scalars().all())

            videos = [v for v in all_videos if cropped_video_exists(project_path, v.name)]
            logger.info(f"apply_alignment_sync: found {len(videos)} cropped videos")

            if not videos:
                logger.warning("apply_alignment_sync: no cropped videos to align")
                return False

            # Create output directory for aligned videos
            output_dir = project_path / "aligned_videos"
            output_dir.mkdir(parents=True, exist_ok=True)

            # Filter out videos that already have aligned files on disk
            videos = [
                v for v in videos
                if not (output_dir / f"{get_cropped_video_path(project_path, v.name).stem}_aligned.mp4").exists()
            ]
            logger.info(f"apply_alignment_sync: {len(videos)} videos need alignment")

            if not videos:
                logger.info("apply_alignment_sync: all videos already aligned")
                self._alignment_progress.status = "completed"
                self._alignment_progress.is_aligning = False
                return True

            self._alignment_progress.total_videos = len(videos)
            logger.info(f"apply_alignment_sync: output_dir={output_dir}")

            for video_idx, video in enumerate(videos):
                logger.info(f"apply_alignment_sync: processing video {video_idx + 1}/{len(videos)}: {video.name}")
                cropped_path = get_cropped_video_path(project_path, video.name)

                if not cropped_path.exists():
                    logger.warning(f"apply_alignment_sync: cropped video not found: {cropped_path}")
                    continue

                # Open input video
                cap = cv2.VideoCapture(str(cropped_path))
                if not cap.isOpened():
                    logger.error(f"apply_alignment_sync: failed to open video: {cropped_path}")
                    continue

                # Get video properties
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                # Update progress for this video
                self._alignment_progress.current_video_index = video_idx + 1
                self._alignment_progress.current_video_name = video.name
                self._alignment_progress.total_frames = frame_count
                self._alignment_progress.current_frame = 0
                self._fps_timestamps.clear()

                # --- Pass 1: Extract features with averaging, decode, collect heading vectors ---
                logger.info(f"apply_alignment_sync: pass 1 - extracting heading vectors ({frame_count} frames)")

                # Load models once for the video
                decoder = self._load_model(project_path)
                device = self._device
                dinov2 = _load_dinov2(device)

                # Sliding window buffer for DINOv2 feature averaging
                half_win = FEATURE_AVG_WINDOW // 2  # e.g., 2 for window=5
                feature_buffer: collections.deque[torch.Tensor] = collections.deque(maxlen=FEATURE_AVG_WINDOW)

                raw_cos = np.zeros(frame_count, dtype=np.float64)
                raw_sin = np.zeros(frame_count, dtype=np.float64)

                # Create temp output file (mp4v codec, then re-encode to H.264)
                temp_path = output_dir / f"{cropped_path.stem}_aligned.temp.mp4"
                output_path = output_dir / f"{cropped_path.stem}_aligned.mp4"

                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(str(temp_path), fourcc, fps, (width, height))

                if not writer.isOpened():
                    logger.error(f"apply_alignment_sync: failed to create video writer: {temp_path}")
                    cap.release()
                    continue

                # Get mask dimensions from cropped masks (may differ from video dimensions)
                with cropped_masks(project_path, video.id) as masks:
                    mask_shape = masks.shape
                    mask_height, mask_width = mask_shape[1], mask_shape[2]

                # Create H5 files upfront using array_storage create functions
                create_aligned_masks_array(project_path, video.id, frame_count, mask_height)
                create_alignment_keypoints_array(project_path, video.id, frame_count, height)

                # Use context managers for H5 files with proper locking (append mode)
                with cropped_masks(project_path, video.id) as cropped_mask_data, \
                     aligned_masks(project_path, video.id, "a") as aligned_mask_data, \
                     alignment_keypoints(project_path, video.id, "a") as keypoints_data:

                    for frame_idx in range(frame_count):
                        ret, frame = cap.read()
                        if not ret:
                            logger.warning(f"apply_alignment_sync: pass 1 failed to read frame {frame_idx}")
                            break

                        # Extract DINOv2 features for this frame
                        patch_tokens = self._extract_features(frame, dinov2, device)
                        feature_buffer.append(patch_tokens)

                        # Decode the frame at the center of the window
                        # decode_idx is the frame whose features are now centered in the buffer
                        decode_idx = frame_idx - half_win

                        if decode_idx >= 0:
                            # Average all features in the buffer
                            avg_features = torch.stack(list(feature_buffer)).mean(dim=0)
                            cos_val, sin_val = self._decode_features(avg_features, decoder)
                            keypoints_data[decode_idx] = [cos_val, sin_val]
                            raw_cos[decode_idx] = cos_val
                            raw_sin[decode_idx] = sin_val

                        # Update progress (pass 1)
                        now = time.time()
                        if now - last_progress_update >= 1.0:
                            self._alignment_progress.current_frame = frame_idx + 1
                            self._alignment_progress.total_frames = frame_count * 2
                            self._fps_timestamps.append(now)
                            self._alignment_progress.fps = self._calculate_rolling_fps()
                            if self._alignment_progress.fps > 0:
                                remaining = (frame_count - frame_idx - 1) + frame_count
                                self._alignment_progress.eta_seconds = remaining / self._alignment_progress.fps
                            last_progress_update = now

                    # Drain remaining frames from buffer (last half_win frames)
                    for i in range(1, half_win + 1):
                        decode_idx = frame_count - half_win + i - 1
                        if decode_idx >= frame_count:
                            break
                        feature_buffer.popleft()
                        if len(feature_buffer) > 0:
                            avg_features = torch.stack(list(feature_buffer)).mean(dim=0)
                            cos_val, sin_val = self._decode_features(avg_features, decoder)
                            keypoints_data[decode_idx] = [cos_val, sin_val]
                            raw_cos[decode_idx] = cos_val
                            raw_sin[decode_idx] = sin_val

                    logger.info(f"apply_alignment_sync: smoothing heading vectors (window={SAVGOL_WINDOW_LENGTH}, polyorder={SAVGOL_POLYORDER})")

                    win = min(SAVGOL_WINDOW_LENGTH, frame_count)
                    if win % 2 == 0:
                        win -= 1
                    if win < SAVGOL_POLYORDER + 2:
                        logger.warning(f"apply_alignment_sync: too few frames ({frame_count}) for SavGol, using raw heading")
                        smooth_cos = raw_cos
                        smooth_sin = raw_sin
                    else:
                        smooth_cos = savgol_filter(raw_cos, win, SAVGOL_POLYORDER)
                        smooth_sin = savgol_filter(raw_sin, win, SAVGOL_POLYORDER)

                    # Re-normalize smoothed vectors back to unit circle
                    norms = np.sqrt(smooth_cos**2 + smooth_sin**2)
                    norms = np.maximum(norms, 1e-8)
                    smooth_cos = smooth_cos / norms
                    smooth_sin = smooth_sin / norms

                    # Convert to angles (degrees)
                    angles = np.degrees(np.arctan2(smooth_sin, smooth_cos))

                    # --- Pass 2: Re-read frames, rotate using smoothed angles, write output ---
                    logger.info(f"apply_alignment_sync: pass 2 - rotating frames ({frame_count} frames)")

                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self._fps_timestamps.clear()

                    for frame_idx in range(frame_count):
                        ret, frame = cap.read()
                        if not ret:
                            logger.warning(f"apply_alignment_sync: pass 2 failed to read frame {frame_idx}")
                            break

                        angle = float(angles[frame_idx])

                        # Rotate frame
                        rotated = rotate_frame(frame, angle)
                        writer.write(rotated)

                        # Load cropped mask, rotate, threshold, and save
                        cropped_mask = cropped_mask_data[frame_idx]
                        aligned_mask = rotate_mask(cropped_mask, angle)
                        aligned_mask_data[frame_idx] = aligned_mask

                        # Track frame timestamp for FPS calculation
                        self._fps_timestamps.append(time.time())

                        # Update progress (pass 2)
                        now = time.time()
                        if now - last_progress_update >= 1.0:
                            self._alignment_progress.current_frame = frame_count + frame_idx + 1
                            self._alignment_progress.total_frames = frame_count * 2
                            self._alignment_progress.fps = self._calculate_rolling_fps()
                            if self._alignment_progress.fps > 0:
                                remaining = frame_count - frame_idx - 1
                                self._alignment_progress.eta_seconds = remaining / self._alignment_progress.fps
                            last_progress_update = now

                    # Final progress update for this video
                    self._alignment_progress.current_frame = frame_count * 2

                # Release video resources outside the H5 context managers
                cap.release()
                writer.release()

                # Re-encode to H.264 for browser compatibility
                # Write to a _tmp file first, then atomically rename so that
                # existence checks only see fully-written files.
                tmp_output_path = output_path.with_name(output_path.stem + "_tmp.mp4")
                if not self._reencode_to_h264(temp_path, tmp_output_path):
                    logger.error(f"apply_alignment_sync: failed to re-encode video {video.id}")
                    temp_path.unlink(missing_ok=True)
                    tmp_output_path.unlink(missing_ok=True)
                    continue

                # Atomic rename to final path
                tmp_output_path.rename(output_path)

                # Clean up mp4v temp file
                temp_path.unlink(missing_ok=True)

            # Mark completion
            self._alignment_progress.status = "completed"
            self._alignment_progress.is_aligning = False
            self._alignment_progress.eta_seconds = 0.0
            logger.info(f"apply_alignment_sync: completed successfully for {len(videos)} videos")
            return True

        except Exception as e:
            logger.error(f"apply_alignment_sync: failed with error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self._alignment_progress.status = "failed"
            self._alignment_progress.is_aligning = False
            return False
        finally:
            self._is_applying = False


# Module-level singleton instance
alignment_service = AlignmentService.get_instance()


# ----- Alignment Label CRUD -----


async def get_video_alignment_label_frames(
    session: AsyncSession,
    video_id: int,
) -> list[int]:
    """Get frame indices that have alignment labels for a video.

    Args:
        session: Project database session
        video_id: ID of the video

    Returns:
        List of frame indices with alignment labels, ordered ascending
    """
    result = await session.execute(
        select(AlignmentLabel.frame_idx)
        .where(AlignmentLabel.video_id == video_id)
        .order_by(AlignmentLabel.frame_idx)
    )
    return list(result.scalars().all())


async def delete_video_alignment_label(
    session: AsyncSession,
    video_id: int,
    frame_idx: int,
) -> None:
    """Delete an alignment label for a specific frame.

    Args:
        session: Project database session
        video_id: ID of the video
        frame_idx: Frame index
    """
    await session.execute(
        delete(AlignmentLabel)
        .where(AlignmentLabel.video_id == video_id)
        .where(AlignmentLabel.frame_idx == frame_idx)
    )
    await session.commit()


async def delete_video_alignment_labels(
    session: AsyncSession,
    video_id: int,
) -> None:
    """Delete all alignment labels for a video.

    Args:
        session: Project database session
        video_id: ID of the video
    """
    await session.execute(
        delete(AlignmentLabel).where(AlignmentLabel.video_id == video_id)
    )
    await session.commit()
