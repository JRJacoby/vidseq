"""Alignment Service for egocentric alignment training and inference.

Provides training and inference for keypoint detection (front/rear of animal)
using a U-Net model with ResNet18 encoder.
"""

import copy
import io
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

import h5py
import imageio_ffmpeg

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from PIL import Image
from scipy.optimize import curve_fit
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session
from torch.utils.data import Dataset, DataLoader

from vidseq.models.alignment_label import AlignmentLabel
from vidseq.models.video import Video
from vidseq.services.cropped_video_service import get_cropped_video_path

# Constants
ALIGNMENT_INPUT_SIZE = 128  # Fixed input size for model

# OneEuro filter defaults for temporal smoothing
ONE_EURO_MIN_CUTOFF = 1.0  # Minimum cutoff frequency (Hz) - lower = more smoothing
ONE_EURO_BETA = 0.0        # Speed coefficient - 0 = simple low-pass filter (no adaptive behavior)
ONE_EURO_D_CUTOFF = 1.0    # Derivative cutoff frequency (Hz)


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


class OneEuroFilter:
    """OneEuro filter for temporal smoothing of noisy signals.

    Attempt to smooth out jitter. The closer to zero min_cutoff is,
    the more smoothing there is. It's recommended to leave min_cutoff
    alone and tweak beta to find the optimal smoothing for your signal.

    Reference: https://cristal.univ-lille.fr/~casiez/1euro/
    """

    def __init__(
        self,
        freq: float,
        min_cutoff: float = ONE_EURO_MIN_CUTOFF,
        beta: float = ONE_EURO_BETA,
        d_cutoff: float = ONE_EURO_D_CUTOFF,
    ):
        """Initialize the filter.

        Args:
            freq: Sampling frequency in Hz (e.g., video FPS)
            min_cutoff: Minimum cutoff frequency
            beta: Speed coefficient
            d_cutoff: Derivative cutoff frequency
        """
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

        # State
        self.x_prev: Optional[float] = None
        self.dx_prev: float = 0.0

    def _smoothing_factor(self, cutoff: float) -> float:
        """Compute the smoothing factor (alpha) for a given cutoff frequency."""
        tau = 1.0 / (2.0 * np.pi * cutoff)
        te = 1.0 / self.freq
        return 1.0 / (1.0 + tau / te)

    def _exponential_smoothing(self, alpha: float, x: float, x_prev: float) -> float:
        """Apply exponential smoothing."""
        return alpha * x + (1.0 - alpha) * x_prev

    def reset(self) -> None:
        """Reset filter state."""
        self.x_prev = None
        self.dx_prev = 0.0

    def __call__(self, x: float) -> float:
        """Filter a single value.

        Args:
            x: Input value

        Returns:
            Filtered value
        """
        if self.x_prev is None:
            # First sample - initialize state
            self.x_prev = x
            self.dx_prev = 0.0
            return x

        # Estimate derivative
        dx = (x - self.x_prev) * self.freq

        # Smooth the derivative
        alpha_d = self._smoothing_factor(self.d_cutoff)
        dx_smooth = self._exponential_smoothing(alpha_d, dx, self.dx_prev)
        self.dx_prev = dx_smooth

        # Compute adaptive cutoff based on speed
        cutoff = self.min_cutoff + self.beta * abs(dx_smooth)

        # Smooth the signal
        alpha = self._smoothing_factor(cutoff)
        x_smooth = self._exponential_smoothing(alpha, x, self.x_prev)
        self.x_prev = x_smooth

        return x_smooth

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
    sigma: float = 0.10,
) -> np.ndarray:
    """Generate a gaussian heatmap centered at (x, y).

    Args:
        x: Normalized x coordinate (0-1)
        y: Normalized y coordinate (0-1)
        height: Height of output heatmap
        width: Width of output heatmap
        sigma: Standard deviation of gaussian (normalized, default 0.10 = 10% of image)

    Returns:
        Heatmap array of shape (height, width) with values in [0, 1]
    """
    # Create coordinate grids (normalized 0-1)
    yy, xx = np.mgrid[0:height, 0:width]
    xx = xx / width
    yy = yy / height

    # Compute gaussian
    dist_sq = (xx - x) ** 2 + (yy - y) ** 2
    heatmap = np.exp(-dist_sq / (2 * sigma ** 2))

    return heatmap.astype(np.float32)


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


# --- Prediction Storage (HDF5) ---


def _get_predictions_dir(project_path: Path) -> Path:
    """Get the alignment predictions directory for a project."""
    return project_path / "alignment_predictions"


def _get_predictions_h5_path(project_path: Path, video_id: int) -> Path:
    """Get path to the HDF5 file for a video's alignment predictions."""
    return _get_predictions_dir(project_path) / f"{video_id}.h5"


def save_prediction(
    project_path: Path,
    video_id: int,
    frame_idx: int,
    heatmap: np.ndarray,
) -> None:
    """Save prediction heatmap to HDF5.

    Args:
        project_path: Path to project folder
        video_id: Video ID
        frame_idx: Frame index
        heatmap: (H, W, 2) float32 array with front/rear probabilities
    """
    predictions_dir = _get_predictions_dir(project_path)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    h5_path = _get_predictions_h5_path(project_path, video_id)

    with h5py.File(h5_path, "a") as f:
        # Create predictions group if it doesn't exist
        if "predictions" not in f:
            f.create_group("predictions")

        predictions_group = f["predictions"]
        dataset_name = str(frame_idx)

        # Delete existing dataset if present (overwrite behavior)
        if dataset_name in predictions_group:
            del predictions_group[dataset_name]

        # Save with gzip compression
        predictions_group.create_dataset(
            dataset_name,
            data=heatmap.astype(np.float32),
            compression="gzip",
            compression_opts=4,
        )


def load_prediction(
    project_path: Path,
    video_id: int,
    frame_idx: int,
) -> Optional[np.ndarray]:
    """Load prediction heatmap from HDF5.

    Args:
        project_path: Path to project folder
        video_id: Video ID
        frame_idx: Frame index

    Returns:
        (H, W, 2) float32 array or None if not found
    """
    h5_path = _get_predictions_h5_path(project_path, video_id)

    if not h5_path.exists():
        return None

    try:
        with h5py.File(h5_path, "r") as f:
            if "predictions" not in f:
                return None

            predictions_group = f["predictions"]
            dataset_name = str(frame_idx)

            if dataset_name not in predictions_group:
                return None

            return predictions_group[dataset_name][:]
    except Exception as e:
        logger.warning(f"load_prediction: failed to load frame {frame_idx} for video {video_id}: {e}")
        return None


def predictions_exist(project_path: Path, video_id: int) -> bool:
    """Check if predictions exist for a video.

    Args:
        project_path: Path to project folder
        video_id: Video ID

    Returns:
        True if predictions HDF5 file exists and has data
    """
    h5_path = _get_predictions_h5_path(project_path, video_id)

    if not h5_path.exists():
        return False

    try:
        with h5py.File(h5_path, "r") as f:
            if "predictions" not in f:
                return False
            return len(f["predictions"]) > 0
    except Exception:
        return False


def _gaussian_2d(coords: tuple, amplitude: float, x0: float, y0: float, sigma: float) -> np.ndarray:
    """2D Gaussian function for curve fitting.

    Args:
        coords: Tuple of (x, y) coordinate arrays
        amplitude: Peak amplitude
        x0, y0: Center coordinates (normalized 0-1)
        sigma: Standard deviation (normalized)

    Returns:
        Flattened gaussian values
    """
    x, y = coords
    return amplitude * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))


def fit_gaussian_to_heatmap(heatmap: np.ndarray) -> tuple[float, float]:
    """Fit 2D Gaussian to heatmap and return (x, y) of peak.

    Args:
        heatmap: (H, W) array with values in [0, 1]

    Returns:
        (x, y) normalized coordinates (0-1) of gaussian center
    """
    h, w = heatmap.shape

    # Create normalized coordinate grids
    yy, xx = np.mgrid[0:h, 0:w]
    xx_norm = xx / w
    yy_norm = yy / h

    # Initial guess from argmax
    max_idx = np.argmax(heatmap)
    max_y, max_x = np.unravel_index(max_idx, heatmap.shape)
    x0_init = max_x / w
    y0_init = max_y / h
    amp_init = heatmap[max_y, max_x]

    # Flatten arrays for curve_fit
    x_flat = xx_norm.ravel()
    y_flat = yy_norm.ravel()
    z_flat = heatmap.ravel()

    try:
        # Fit gaussian with bounds
        popt, _ = curve_fit(
            _gaussian_2d,
            (x_flat, y_flat),
            z_flat,
            p0=[amp_init, x0_init, y0_init, 0.05],  # Initial: amp, x0, y0, sigma
            bounds=(
                [0, 0, 0, 0.01],      # Lower bounds
                [2, 1, 1, 0.5]        # Upper bounds
            ),
            maxfev=1000,
        )
        _, x0, y0, _ = popt
        logger.debug(f"fit_gaussian_to_heatmap: fitted center=({x0:.3f}, {y0:.3f})")
        return x0, y0

    except (RuntimeError, ValueError) as e:
        # Fall back to argmax if fitting fails
        logger.warning(f"fit_gaussian_to_heatmap: fitting failed ({e}), falling back to argmax")
        return x0_init, y0_init


def calculate_rotation_angle(
    front_x: float,
    front_y: float,
    rear_x: float,
    rear_y: float,
    width: int = 1,
    height: int = 1,
) -> float:
    """Calculate rotation angle to make animal face right.

    Args:
        front_x, front_y: Front (nose) keypoint coordinates (normalized 0-1)
        rear_x, rear_y: Rear (tail) keypoint coordinates (normalized 0-1)
        width, height: Frame dimensions for aspect ratio correction

    Returns:
        Angle in degrees for cv2.getRotationMatrix2D
    """
    # Vector from rear to front, scaled by dimensions for correct aspect ratio
    dx = (front_x - rear_x) * width
    dy = (front_y - rear_y) * height

    # Current angle (radians) - note: y increases downward in image coords
    # arctan2(dy, dx) gives angle from positive x-axis
    angle_rad = np.arctan2(dy, dx)

    # Convert to degrees
    # We want animal facing right (angle = 0)
    # In OpenCV with image coords (y-down), positive angle = visually clockwise
    # To rotate the animal TO 0°, we need to rotate BY the negative of its current angle
    # But since positive rotation is CW (decreases angle), we use the angle directly
    rotation_degrees = np.degrees(angle_rad)

    logger.debug(
        f"calculate_rotation_angle: front=({front_x:.3f}, {front_y:.3f}), "
        f"rear=({rear_x:.3f}, {rear_y:.3f}), dims={width}x{height}, angle={rotation_degrees:.1f}°"
    )

    return rotation_degrees


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

        # Resize to fixed size
        frame = cv2.resize(
            frame, (self.size, self.size), interpolation=cv2.INTER_LINEAR
        )

        # Get keypoint coordinates
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

        # Normalize to [0, 1] and convert to (C, H, W) tensor
        frame_tensor = torch.from_numpy(frame).float() / 255.0
        frame_tensor = frame_tensor.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)

        # Generate target heatmaps at the fixed size using augmented coordinates
        front_heatmap = generate_gaussian_heatmap(
            front_x, front_y, self.size, self.size
        )
        rear_heatmap = generate_gaussian_heatmap(rear_x, rear_y, self.size, self.size)

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

        # Training progress tracking for real-time updates
        self._training_progress = TrainingProgress()

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
        max_epochs: int = 100,
        batch_size: int = 8,
        lr: float = 1e-4,
        lr_patience: int = 3,
        lr_factor: float = 0.25,
        early_stop_patience: int = 5,
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

            # Create model
            model = smp.Unet(
                encoder_name="resnet152",
                encoder_weights="imagenet",
                in_channels=3,
                classes=2,
            )
            model = model.to(device)
            logger.info("train_model_sync: created U-Net model with ResNet18 encoder")

            # Create train dataset with augmentation
            train_dataset = AlignmentDataset(
                train_labels, project_path, video_name_map, augment=True
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

            # Setup optimizer and loss
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            criterion = nn.MSELoss()

            # Early stopping and LR reduction tracking (based on val loss)
            best_val_loss = float("inf")
            best_model_state = None
            best_epoch = 0
            epochs_without_improvement = 0
            current_lr = lr
            lr_reduced_this_plateau = False
            min_lr = 1e-7

            # Training loop
            for epoch in range(max_epochs):
                # --- Training phase ---
                model.train()
                train_loss = 0.0
                train_batches = 0

                for frames, targets in train_dataloader:
                    frames = frames.to(device)
                    targets = targets.to(device)

                    optimizer.zero_grad()
                    outputs = model(frames)
                    outputs = torch.sigmoid(outputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()
                    train_batches += 1

                avg_train_loss = train_loss / max(train_batches, 1)

                # --- Validation phase ---
                if use_validation and val_dataloader is not None:
                    model.eval()
                    val_loss = 0.0
                    val_batches = 0

                    with torch.no_grad():
                        for frames, targets in val_dataloader:
                            frames = frames.to(device)
                            targets = targets.to(device)

                            outputs = model(frames)
                            outputs = torch.sigmoid(outputs)
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
                    # Improvement - save best model state
                    best_val_loss = loss_for_comparison
                    best_model_state = copy.deepcopy(model.state_dict())
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

            # Save best model weights (not final)
            model_path = self.get_model_path(project_path)
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
                logger.info(
                    f"train_model_sync: restoring best model from epoch {best_epoch} "
                    f"(val_loss={best_val_loss:.6f})"
                )
            torch.save(model.state_dict(), model_path)
            logger.info(f"train_model_sync: saved model weights to {model_path}")

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
            encoder_name="resnet152",
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
    ) -> bool:
        """Apply alignment to all cropped videos.

        Reads cropped videos, rotates frames so animal faces right,
        saves to aligned_videos folder. Also rotates cropped masks
        and saves to aligned_masks folder.

        Args:
            project_path: Path to project folder
            project_engine: SQLAlchemy engine for project DB

        Returns:
            True if successful
        """
        logger.info(f"apply_alignment_sync: starting, project={project_path.name}")
        self._is_applying = True
        try:
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

            # Check that all videos have cropped masks
            cropped_masks_dir = project_path / "cropped_masks"
            missing_masks = []
            for video in videos:
                cropped_mask_path = cropped_masks_dir / f"{video.id}.h5"
                if not cropped_mask_path.exists():
                    missing_masks.append(video.id)

            if missing_masks:
                logger.error(
                    f"apply_alignment_sync: cropped masks missing for videos: {missing_masks}. "
                    f"Please re-run cropped video extraction."
                )
                return False

            logger.info("apply_alignment_sync: all cropped masks verified")

            # Create output directories
            output_dir = project_path / "aligned_videos"
            output_dir.mkdir(parents=True, exist_ok=True)
            aligned_masks_dir = project_path / "aligned_masks"
            aligned_masks_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"apply_alignment_sync: output_dir={output_dir}, aligned_masks_dir={aligned_masks_dir}")

            for video in videos:
                logger.info(f"apply_alignment_sync: processing video id={video.id}, name={video.name}")
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

                logger.info(
                    f"apply_alignment_sync: video properties - "
                    f"{width}x{height}, {fps:.2f} fps, {frame_count} frames"
                )

                # Create OneEuro filter for temporal smoothing of angle directly
                angle_filter = OneEuroFilter(freq=fps)
                # For angle unwrapping (handle -180/180 discontinuity)
                prev_raw_angle: Optional[float] = None
                unwrapped_angle: float = 0.0
                logger.info(
                    f"apply_alignment_sync: OneEuro angle filter initialized "
                    f"(min_cutoff={ONE_EURO_MIN_CUTOFF}, beta={ONE_EURO_BETA})"
                )

                # Create temp output file (mp4v codec, then re-encode to H.264)
                temp_path = output_dir / f"{cropped_path.stem}_aligned.temp.mp4"
                output_path = output_dir / f"{cropped_path.stem}_aligned.mp4"

                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(str(temp_path), fourcc, fps, (width, height))

                if not writer.isOpened():
                    logger.error(f"apply_alignment_sync: failed to create video writer: {temp_path}")
                    cap.release()
                    continue

                # Open cropped masks HDF5 for reading
                cropped_mask_h5_path = cropped_masks_dir / f"{video.id}.h5"
                cropped_mask_h5 = h5py.File(cropped_mask_h5_path, "r")

                # Get mask dimensions from cropped masks (may differ from video dimensions)
                mask_shape = cropped_mask_h5["masks"].shape
                mask_height, mask_width = mask_shape[1], mask_shape[2]

                # Create aligned masks HDF5 for writing (use mask dimensions, not video dimensions)
                aligned_mask_h5_path = aligned_masks_dir / f"{video.id}.h5"
                aligned_mask_h5 = h5py.File(aligned_mask_h5_path, "w")
                aligned_mask_h5.create_dataset(
                    "masks",
                    shape=(frame_count, mask_height, mask_width),
                    dtype=np.uint8,
                    fillvalue=0,
                    chunks=(1, mask_height, mask_width),
                    compression=None,
                )
                logger.info(
                    f"apply_alignment_sync: opened cropped masks from {cropped_mask_h5_path.name} "
                    f"(shape {mask_shape}), creating aligned masks at {aligned_mask_h5_path.name}"
                )

                # Process each frame
                log_interval = max(1, frame_count // 10)  # Log every 10%
                for frame_idx in range(frame_count):
                    ret, frame = cap.read()
                    if not ret:
                        logger.warning(f"apply_alignment_sync: failed to read frame {frame_idx}")
                        break

                    # Run prediction to get heatmap
                    heatmap = self.predict_sync(project_path, frame)

                    # Save prediction for debugging
                    save_prediction(project_path, video.id, frame_idx, heatmap)

                    # Fit gaussians to find keypoints
                    front_x, front_y = fit_gaussian_to_heatmap(heatmap[:, :, 0])
                    rear_x, rear_y = fit_gaussian_to_heatmap(heatmap[:, :, 1])

                    # Calculate raw rotation angle (pass dimensions for aspect ratio correction)
                    raw_angle = calculate_rotation_angle(front_x, front_y, rear_x, rear_y, width, height)

                    # Unwrap angle to handle -180/180 discontinuity
                    if prev_raw_angle is None:
                        unwrapped_angle = raw_angle
                    else:
                        delta = raw_angle - prev_raw_angle
                        # Take the shortest path across the boundary
                        if delta > 180:
                            delta -= 360
                        elif delta < -180:
                            delta += 360
                        unwrapped_angle += delta
                    prev_raw_angle = raw_angle

                    # Apply temporal smoothing to the unwrapped angle
                    angle = angle_filter(unwrapped_angle)

                    # Rotate frame
                    rotated = rotate_frame(frame, angle)

                    # Write rotated frame
                    writer.write(rotated)

                    # Load cropped mask, rotate, threshold, and save
                    cropped_mask = np.array(cropped_mask_h5["masks"][frame_idx])
                    aligned_mask = rotate_mask(cropped_mask, angle)
                    aligned_mask_h5["masks"][frame_idx] = aligned_mask

                    # Log progress
                    if frame_idx % log_interval == 0 or frame_idx == frame_count - 1:
                        progress = (frame_idx + 1) / frame_count * 100
                        logger.info(
                            f"apply_alignment_sync: video {video.id} - "
                            f"frame {frame_idx + 1}/{frame_count} ({progress:.0f}%)"
                        )

                # Release resources
                cap.release()
                writer.release()
                cropped_mask_h5.close()
                aligned_mask_h5.close()
                logger.info(f"apply_alignment_sync: video {video.id} - aligned masks saved")

                # Re-encode to H.264 for browser compatibility
                logger.info(f"apply_alignment_sync: re-encoding to H.264: {output_path.name}")
                if not self._reencode_to_h264(temp_path, output_path):
                    logger.error(f"apply_alignment_sync: failed to re-encode video {video.id}")
                    temp_path.unlink(missing_ok=True)
                    continue

                # Clean up temp file
                temp_path.unlink(missing_ok=True)

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
