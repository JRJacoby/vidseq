"""Mask storage - per-video HDF5 files for segmentation masks.

This module handles storage of binary segmentation masks in per-video HDF5 files.
Each video's masks are stored in a directory at `array_data/{video_id}/` with separate
files for tracker, detector, and final masks, allowing concurrent writes to different
videos without locking conflicts.
"""

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

# Disable HDF5's internal file locking (we use our own lock files)
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import h5py
import numpy as np


# Module-level cache for H5 file handles
_cache: dict[Path, h5py.File] = {}


def close_all_h5() -> None:
    """Close all cached H5 file handles.

    Call this during cleanup/shutdown or when you need to ensure
    all files are flushed and closed.
    """
    for f in _cache.values():
        try:
            f.close()
        except Exception:
            pass
    _cache.clear()


def _get_lock_path(h5_path: Path) -> Path:
    """Get the lock file path for an HDF5 file."""
    return h5_path.with_suffix(".h5.lock")


@contextmanager
def open_h5_with_lock(h5_path: Path, mode: str):
    """Context manager for HDF5 file access with caching and locking.

    Caching strategy:
    - mode='r': Cached at module level for fast repeated reads
    - mode='a'/'w': NOT cached - opens fresh with lock, closes on exit

    If 'r' is cached and 'a'/'w' is requested, the cached handle is evicted first.

    Args:
        h5_path: Path to the HDF5 file
        mode: File mode - 'r' for read, 'a' for append/write, 'w' for truncate

    Yields:
        h5py.File: The HDF5 file handle

    Raises:
        ValueError: If mode is not 'r', 'a', or 'w'
        RuntimeError: If file is locked by another process (write modes only)
        FileNotFoundError: If file doesn't exist (read mode only)
    """
    if mode not in ("r", "a", "w"):
        raise ValueError(f"Invalid mode '{mode}'. Must be 'r', 'a', or 'w'")

    h5_path = h5_path.resolve()  # Normalize for cache key
    lock_path = _get_lock_path(h5_path)

    # mode='w' truncates - evict cache, lock, create fresh, close on exit
    if mode == "w":
        h5_path.parent.mkdir(parents=True, exist_ok=True)
        if h5_path in _cache:
            try:
                _cache[h5_path].close()
            except Exception:
                pass
            del _cache[h5_path]
        if lock_path.exists():
            raise RuntimeError(f"HDF5 file is locked by another process: {lock_path}")
        lock_path.touch()
        try:
            with h5py.File(h5_path, "w") as f:
                yield f
        finally:
            lock_path.unlink(missing_ok=True)
        return

    # mode='a' - evict cache if exists, lock, open fresh, close on exit (not cached)
    if mode == "a":
        h5_path.parent.mkdir(parents=True, exist_ok=True)
        if h5_path in _cache:
            try:
                _cache[h5_path].close()
            except Exception:
                pass
            del _cache[h5_path]
        if lock_path.exists():
            raise RuntimeError(f"HDF5 file is locked by another process: {lock_path}")
        lock_path.touch()
        h5_file = h5py.File(h5_path, "a")
        try:
            yield h5_file
        finally:
            h5_file.close()
            lock_path.unlink(missing_ok=True)
        return

    # mode='r' - use cache
    if h5_path not in _cache:
        if not h5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {h5_path}")
        _cache[h5_path] = h5py.File(h5_path, "r")
    yield _cache[h5_path]


@contextmanager
def tracker_h5(project_path: Path, video_id: int, mode: str = "r"):
    """Context manager for tracker masks H5 file: array_data/{video_id}/tracker_masks.h5

    Usage:
        with tracker_h5(project_path, video_id) as f:
            mask = f["masks"][frame_idx]

        with tracker_h5(project_path, video_id, mode="a") as f:
            f["masks"][frame_idx] = mask
    """
    h5_path = project_path / "array_data" / str(video_id) / "tracker_masks.h5"
    with open_h5_with_lock(h5_path, mode) as f:
        yield f


@contextmanager
def detector_h5(project_path: Path, video_id: int, mode: str = "r"):
    """Context manager for detector masks H5 file: array_data/{video_id}/detector_masks.h5"""
    h5_path = project_path / "array_data" / str(video_id) / "detector_masks.h5"
    with open_h5_with_lock(h5_path, mode) as f:
        yield f


@contextmanager
def final_h5(project_path: Path, video_id: int, mode: str = "r"):
    """Context manager for final masks H5 file: array_data/{video_id}/final_masks.h5"""
    h5_path = project_path / "array_data" / str(video_id) / "final_masks.h5"
    with open_h5_with_lock(h5_path, mode) as f:
        yield f


@contextmanager
def cropped_h5(project_path: Path, video_id: int, mode: str = "r"):
    """Context manager for cropped masks H5 file: array_data/{video_id}/cropped_masks.h5"""
    h5_path = project_path / "array_data" / str(video_id) / "cropped_masks.h5"
    with open_h5_with_lock(h5_path, mode) as f:
        yield f


@contextmanager
def aligned_h5(project_path: Path, video_id: int, mode: str = "r"):
    """Context manager for aligned masks H5 file: array_data/{video_id}/aligned_masks.h5"""
    h5_path = project_path / "array_data" / str(video_id) / "aligned_masks.h5"
    with open_h5_with_lock(h5_path, mode) as f:
        yield f


@contextmanager
def predictions_h5(project_path: Path, video_id: int, mode: str = "r"):
    """Context manager for alignment keypoints H5 file: array_data/{video_id}/alignment_keypoints.h5"""
    h5_path = project_path / "array_data" / str(video_id) / "alignment_keypoints.h5"
    with open_h5_with_lock(h5_path, mode) as f:
        yield f


@contextmanager
def pca_scores_h5(project_path: Path, video_id: int, mode: str = "r"):
    """Context manager for PCA scores H5 file: array_data/{video_id}/pca_scores.h5"""
    h5_path = project_path / "array_data" / str(video_id) / "pca_scores.h5"
    with open_h5_with_lock(h5_path, mode) as f:
        yield f


def save_mask(
    project_path: Path,
    video_id: int,
    frame_idx: int,
    mask: np.ndarray,
    h5_file: Optional[h5py.File] = None,
) -> None:
    """Save a tracker mask to the HDF5 file (array_data/{video_id}/tracker_masks.h5).

    The H5 file and masks dataset must already exist (created by create_video_segmentation_files).

    Args:
        project_path: Path to the project folder
        video_id: ID of the video
        frame_idx: Frame index (0-based)
        mask: Binary mask array (height, width) with dtype uint8
        h5_file: Optional pre-opened h5py.File in write mode
    """
    if h5_file is not None:
        h5_file["masks"][frame_idx] = mask
    else:
        with tracker_h5(project_path, video_id, "a") as f:
            f["masks"][frame_idx] = mask
            f.flush()


def load_mask(
    project_path: Path,
    video_id: int,
    frame_idx: int,
    h5_file: Optional[h5py.File] = None,
) -> np.ndarray:
    """Load a tracker mask from the HDF5 file (array_data/{video_id}/tracker_masks.h5).

    The H5 file must exist (created by create_video_segmentation_files).

    Args:
        project_path: Path to the project folder
        video_id: ID of the video
        frame_idx: Frame index (0-based)
        h5_file: Optional pre-opened h5py.File

    Returns:
        Mask array (height, width) with dtype uint8

    Raises:
        FileNotFoundError: If the H5 file doesn't exist
        KeyError: If the masks dataset doesn't exist
    """
    if h5_file is not None:
        return np.array(h5_file["masks"][frame_idx])

    with tracker_h5(project_path, video_id, "r") as f:
        return np.array(f["masks"][frame_idx])


def load_masks_batch(
    project_path: Path,
    video_id: int,
    start_frame: int,
    count: int,
    num_frames: int,
    h5_file: Optional[h5py.File] = None,
) -> np.ndarray:
    """Load multiple tracker masks efficiently using HDF5 slice indexing.

    The H5 file must exist (created by create_video_segmentation_files).

    Args:
        project_path: Path to the project folder
        video_id: ID of the video
        start_frame: Starting frame index (0-based)
        count: Number of frames to load
        num_frames: Total number of frames in the video
        h5_file: Optional pre-opened h5py.File

    Returns:
        Array of shape (actual_count, height, width) where actual_count
        may be less than count if start_frame + count exceeds num_frames.

    Raises:
        FileNotFoundError: If the H5 file doesn't exist
        KeyError: If the masks dataset doesn't exist
    """
    end_frame = min(start_frame + count, num_frames)

    if h5_file is not None:
        return np.array(h5_file["masks"][start_frame:end_frame])

    with tracker_h5(project_path, video_id, "r") as f:
        return np.array(f["masks"][start_frame:end_frame])


def clear_mask(
    project_path: Path,
    video_id: int,
    frame_idx: int,
    h5_file: Optional[h5py.File] = None,
) -> None:
    """Clear (zero out) a tracker mask for a specific frame.

    The H5 file must exist (created by create_video_segmentation_files).

    Args:
        project_path: Path to the project folder
        video_id: ID of the video
        frame_idx: Frame index (0-based)
        h5_file: Optional pre-opened h5py.File in write mode

    Raises:
        FileNotFoundError: If the H5 file doesn't exist
    """
    if h5_file is not None:
        ds = h5_file["masks"]
        ds[frame_idx] = np.zeros((ds.shape[1], ds.shape[2]), dtype=np.uint8)
        h5_file.flush()
        return

    with tracker_h5(project_path, video_id, "a") as f:
        ds = f["masks"]
        ds[frame_idx] = np.zeros((ds.shape[1], ds.shape[2]), dtype=np.uint8)
        f.flush()


def compute_bbox_from_mask(mask: np.ndarray) -> Optional[np.ndarray]:
    """Compute bounding box [x1, y1, x2, y2] from a binary mask.

    Args:
        mask: Binary mask array (height, width), values 0 or non-zero

    Returns:
        Bounding box as numpy array [x1, y1, x2, y2] or None if mask is empty
    """
    mask_binary = mask > 0
    if not np.any(mask_binary):
        return None

    rows = np.any(mask_binary, axis=1)
    cols = np.any(mask_binary, axis=0)

    if not np.any(rows) or not np.any(cols):
        return None

    y_indices = np.where(rows)[0]
    x_indices = np.where(cols)[0]
    y1, y2 = y_indices[0], y_indices[-1]
    x1, x2 = x_indices[0], x_indices[-1]

    return np.array([x1, y1, x2, y2], dtype=np.float32)


# ----- Final masks support -----
# Final masks are stored in array_data/{video_id}/final_masks.h5 and contain the
# tracker-detector fusion result (tracker if IoU >= 0.5, corrected otherwise)


def load_final_mask(
    project_path: Path,
    video_id: int,
    frame_idx: int,
) -> np.ndarray:
    """Load a final (corrected) mask from the HDF5 file.

    The H5 file must exist (created by create_video_segmentation_files).

    Args:
        project_path: Path to the project folder
        video_id: ID of the video
        frame_idx: Frame index (0-based)

    Returns:
        Mask array (height, width) with dtype uint8

    Raises:
        FileNotFoundError: If the H5 file doesn't exist
        KeyError: If the masks dataset doesn't exist
    """
    with final_h5(project_path, video_id, mode="r") as f:
        return np.array(f["masks"][frame_idx])


def load_final_masks_batch(
    project_path: Path,
    video_id: int,
    start_frame: int,
    count: int,
    num_frames: int,
) -> np.ndarray:
    """Load multiple final (corrected) masks efficiently.

    The H5 file must exist (created by create_video_segmentation_files).

    Args:
        project_path: Path to the project folder
        video_id: ID of the video
        start_frame: Starting frame index (0-based)
        count: Number of frames to load
        num_frames: Total number of frames in the video

    Returns:
        Array of shape (actual_count, height, width)

    Raises:
        FileNotFoundError: If the H5 file doesn't exist
        KeyError: If the masks dataset doesn't exist
    """
    end_frame = min(start_frame + count, num_frames)

    with final_h5(project_path, video_id, mode="r") as f:
        return np.array(f["masks"][start_frame:end_frame])


# ----- Delete functions for clearing individual frame data -----


def delete_tracker_mask(project_path: Path, video_id: int, frame_idx: int) -> None:
    """Zero out tracker mask for a frame.

    The H5 file must exist (created by create_video_segmentation_files).

    Raises:
        FileNotFoundError: If the H5 file doesn't exist
    """
    with tracker_h5(project_path, video_id, mode="a") as f:
        f["masks"][frame_idx] = 0
        f.flush()


def delete_tracker_logits(project_path: Path, video_id: int, frame_idx: int) -> None:
    """Zero out tracker logits for a frame.

    The H5 file must exist (created by create_video_segmentation_files).

    Raises:
        FileNotFoundError: If the H5 file doesn't exist
    """
    with tracker_h5(project_path, video_id, mode="a") as f:
        f["logits"][frame_idx] = 0.0
        f.flush()


def delete_detector_mask(project_path: Path, video_id: int, frame_idx: int) -> None:
    """Zero out detector mask for a frame.

    The H5 file must exist (created by create_video_segmentation_files).

    Raises:
        FileNotFoundError: If the H5 file doesn't exist
    """
    with detector_h5(project_path, video_id, mode="a") as f:
        f["masks"][frame_idx] = 0
        f.flush()


def delete_final_mask(project_path: Path, video_id: int, frame_idx: int) -> None:
    """Zero out final (corrected) mask for a frame.

    The H5 file must exist (created by create_video_segmentation_files).

    Raises:
        FileNotFoundError: If the H5 file doesn't exist
    """
    with final_h5(project_path, video_id, mode="a") as f:
        f["masks"][frame_idx] = 0
        f.flush()


# ----- Video-level H5 file management -----


def create_video_segmentation_files(
    project_path: Path,
    video_id: int,
    num_frames: int,
    height: int,
    width: int,
    logits_size: int = 256,
) -> None:
    """Create all segmentation H5 files for a video upfront.

    Creates tracker, detector, and final mask files with pre-allocated datasets.
    Files are stored in array_data/{video_id}/ directory:
      - tracker_masks.h5: masks and logits
      - detector_masks.h5: compressed masks
      - final_masks.h5: masks

    Args:
        project_path: Path to the project folder
        video_id: ID of the video
        num_frames: Total number of frames in the video
        height: Video height in pixels
        width: Video width in pixels
        logits_size: Size of low-res logits (SAM2=256)
    """
    # Create video directory under array_data
    video_dir = project_path / "array_data" / str(video_id)
    video_dir.mkdir(parents=True, exist_ok=True)

    # 1. Tracker: tracker_masks.h5 with 'masks' and 'logits'
    with tracker_h5(project_path, video_id, mode="w") as f:
        f.create_dataset(
            "masks",
            shape=(num_frames, height, width),
            dtype=np.uint8,
            chunks=(1, height, width),
            fillvalue=0,
        )
        f.create_dataset(
            "logits",
            shape=(num_frames, logits_size, logits_size),
            dtype=np.float32,
            chunks=(1, logits_size, logits_size),
            fillvalue=0.0,
        )

    # 2. Detector: detector_masks.h5 (compressed)
    with detector_h5(project_path, video_id, mode="w") as f:
        f.create_dataset(
            "masks",
            shape=(num_frames, height, width),
            dtype=np.uint8,
            chunks=(1, height, width),
            compression="gzip",
            fillvalue=0,
        )

    # 3. Final: final_masks.h5
    with final_h5(project_path, video_id, mode="w") as f:
        f.create_dataset(
            "masks",
            shape=(num_frames, height, width),
            dtype=np.uint8,
            chunks=(1, height, width),
            fillvalue=0,
        )


def delete_video_segmentation_files(project_path: Path, video_id: int) -> None:
    """Delete all segmentation H5 files for a video.

    Deletes tracker, detector, and final mask files from array_data/{video_id}/.
    Does NOT touch cropped/aligned masks.

    Also evicts any cached read handles for these files.

    Args:
        project_path: Path to the project folder
        video_id: ID of the video
    """
    video_dir = project_path / "array_data" / str(video_id)
    file_names = ["tracker_masks.h5", "detector_masks.h5", "final_masks.h5"]

    for file_name in file_names:
        h5_path = video_dir / file_name
        lock_path = _get_lock_path(h5_path)

        # Check for lock before deleting
        if lock_path.exists():
            raise RuntimeError(
                f"Cannot delete - file is locked by another process: {lock_path}"
            )

        # Evict from cache if present
        resolved = h5_path.resolve()
        if resolved in _cache:
            try:
                _cache[resolved].close()
            except Exception:
                pass
            del _cache[resolved]

        if h5_path.exists():
            h5_path.unlink()


def reset_video_segmentation_files(
    project_path: Path,
    video_id: int,
    num_frames: int,
    height: int,
    width: int,
    logits_size: int = 256,
) -> None:
    """Reset all segmentation H5 files by deleting and recreating with zeros.

    Resets tracker, detector, and final masks in array_data/{video_id}/.
    Does NOT touch cropped/aligned masks.

    Args:
        project_path: Path to the project folder
        video_id: ID of the video
        num_frames: Total number of frames in the video
        height: Video height in pixels
        width: Video width in pixels
        logits_size: Size of low-res logits (SAM2=256)
    """
    delete_video_segmentation_files(project_path, video_id)
    create_video_segmentation_files(project_path, video_id, num_frames, height, width, logits_size)


# ----- Pipeline H5 file creation -----
# These files are created lazily when their respective pipeline runs


def create_cropped_h5(
    project_path: Path,
    video_id: int,
    num_frames: int,
    crop_size: int,
) -> None:
    """Create cropped masks H5 file for a video.

    Args:
        project_path: Path to the project folder
        video_id: ID of the video
        num_frames: Total number of frames in the video
        crop_size: Square crop dimension
    """
    with cropped_h5(project_path, video_id, mode="w") as f:
        f.create_dataset(
            "masks",
            shape=(num_frames, crop_size, crop_size),
            dtype=np.uint8,
            fillvalue=0,
            chunks=(1, crop_size, crop_size),
            compression=None,
        )


def create_aligned_h5(
    project_path: Path,
    video_id: int,
    num_frames: int,
    crop_size: int,
) -> None:
    """Create aligned masks H5 file for a video.

    Args:
        project_path: Path to the project folder
        video_id: ID of the video
        num_frames: Total number of frames in the video
        crop_size: Square crop dimension (same as cropped masks)
    """
    with aligned_h5(project_path, video_id, mode="w") as f:
        f.create_dataset(
            "masks",
            shape=(num_frames, crop_size, crop_size),
            dtype=np.uint8,
            fillvalue=0,
            chunks=(1, crop_size, crop_size),
            compression=None,
        )


def create_alignment_predictions_h5(
    project_path: Path,
    video_id: int,
    num_frames: int,
    crop_size: int,
) -> None:
    """Create alignment predictions (heatmaps) H5 file for a video.

    Args:
        project_path: Path to the project folder
        video_id: ID of the video
        num_frames: Total number of frames in the video
        crop_size: Heatmap dimensions (same as cropped masks)
    """
    with predictions_h5(project_path, video_id, mode="w") as f:
        f.create_dataset(
            "heatmaps",
            shape=(num_frames, crop_size, crop_size, 2),
            dtype=np.float32,
            fillvalue=0.0,
            chunks=(1, crop_size, crop_size, 2),
            compression=None,
        )


def create_pca_scores_h5(
    project_path: Path,
    video_id: int,
    num_frames: int,
    n_components: int,
) -> None:
    """Create PCA scores H5 file for a video.

    Args:
        project_path: Path to the project folder
        video_id: ID of the video
        num_frames: Total number of frames in the video
        n_components: Number of PCA components
    """
    with pca_scores_h5(project_path, video_id, mode="w") as f:
        f.create_dataset(
            "scores",
            shape=(num_frames, n_components),
            dtype=np.float32,
            fillvalue=0.0,
        )


# =============================================================================
# Existence Checks
# =============================================================================


def cropped_h5_exists(project_path: Path, video_id: int) -> bool:
    """Check if cropped masks H5 file exists for a video.

    Args:
        project_path: Path to the project folder
        video_id: ID of the video

    Returns:
        True if the cropped masks H5 file exists
    """
    h5_path = project_path / "array_data" / str(video_id) / "cropped_masks.h5"
    return h5_path.exists()
