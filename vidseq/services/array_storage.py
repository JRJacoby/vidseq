"""Array storage - per-video array data with HDF5 backend.

This module provides array-like access to per-video data. Callers receive
dataset handles directly and can index them without knowing the storage format.
Each video's data is stored in `array_data/{video_id}/` with separate files,
allowing concurrent writes to different videos without locking conflicts.
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


def _construct_h5_path(project_path: Path, video_id: int, filename: str) -> Path:
    """Construct path to an H5 file in the video's array_data directory."""
    return project_path / "array_data" / str(video_id) / filename


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
def tracker_masks(project_path: Path, video_id: int, mode: str = "r"):
    """Tracker masks array: array_data/{video_id}/tracker_masks.h5

    Usage:
        with tracker_masks(project_path, video_id) as masks:
            mask = masks[frame_idx]

        with tracker_masks(project_path, video_id, mode="a") as masks:
            masks[frame_idx] = mask
    """
    h5_path = _construct_h5_path(project_path, video_id, "tracker_masks.h5")
    with open_h5_with_lock(h5_path, mode) as f:
        yield f["data"]


@contextmanager
def tracker_logits(project_path: Path, video_id: int, mode: str = "r"):
    """Tracker logits array: array_data/{video_id}/tracker_logits.h5

    Usage:
        with tracker_logits(project_path, video_id) as logits:
            prev_logits = logits[frame_idx]
    """
    h5_path = _construct_h5_path(project_path, video_id, "tracker_logits.h5")
    with open_h5_with_lock(h5_path, mode) as f:
        yield f["data"]


@contextmanager
def detector_masks(project_path: Path, video_id: int, mode: str = "r"):
    """Detector masks array: array_data/{video_id}/detector_masks.h5"""
    h5_path = _construct_h5_path(project_path, video_id, "detector_masks.h5")
    with open_h5_with_lock(h5_path, mode) as f:
        yield f["data"]


@contextmanager
def final_masks(project_path: Path, video_id: int, mode: str = "r"):
    """Final masks array: array_data/{video_id}/final_masks.h5"""
    h5_path = _construct_h5_path(project_path, video_id, "final_masks.h5")
    with open_h5_with_lock(h5_path, mode) as f:
        yield f["data"]


@contextmanager
def cropped_masks(project_path: Path, video_id: int, mode: str = "r"):
    """Cropped masks array: array_data/{video_id}/cropped_masks.h5"""
    h5_path = _construct_h5_path(project_path, video_id, "cropped_masks.h5")
    with open_h5_with_lock(h5_path, mode) as f:
        yield f["data"]


@contextmanager
def aligned_masks(project_path: Path, video_id: int, mode: str = "r"):
    """Aligned masks array: array_data/{video_id}/aligned_masks.h5"""
    h5_path = _construct_h5_path(project_path, video_id, "aligned_masks.h5")
    with open_h5_with_lock(h5_path, mode) as f:
        yield f["data"]


@contextmanager
def alignment_keypoints(project_path: Path, video_id: int, mode: str = "r"):
    """Alignment keypoints array: array_data/{video_id}/alignment_keypoints.h5"""
    h5_path = _construct_h5_path(project_path, video_id, "alignment_keypoints.h5")
    with open_h5_with_lock(h5_path, mode) as f:
        yield f["data"]


@contextmanager
def pca_scores(project_path: Path, video_id: int, mode: str = "r"):
    """PCA scores array: array_data/{video_id}/pca_scores.h5"""
    h5_path = _construct_h5_path(project_path, video_id, "pca_scores.h5")
    with open_h5_with_lock(h5_path, mode) as f:
        yield f["data"]


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


# ----- Video-level H5 file management -----


def create_video_segmentation_arrays(
    project_path: Path,
    video_id: int,
    num_frames: int,
    height: int,
    width: int,
    logits_size: int = 256,
) -> None:
    """Create all segmentation arrays for a video upfront.

    Creates tracker, detector, and final mask arrays with pre-allocated datasets.
    Arrays are stored in array_data/{video_id}/ directory:
      - tracker_masks.h5
      - tracker_logits.h5
      - detector_masks.h5
      - final_masks.h5

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

    # 1. Tracker masks: tracker_masks.h5
    tracker_masks_path = _construct_h5_path(project_path, video_id, "tracker_masks.h5")
    with open_h5_with_lock(tracker_masks_path, mode="w") as f:
        f.create_dataset(
            "data",
            shape=(num_frames, height, width),
            dtype=np.uint8,
            chunks=(1, height, width),
            fillvalue=0,
        )

    # 2. Tracker logits: tracker_logits.h5
    tracker_logits_path = _construct_h5_path(project_path, video_id, "tracker_logits.h5")
    with open_h5_with_lock(tracker_logits_path, mode="w") as f:
        f.create_dataset(
            "data",
            shape=(num_frames, logits_size, logits_size),
            dtype=np.float32,
            chunks=(1, logits_size, logits_size),
            fillvalue=0.0,
        )

    # 3. Detector: detector_masks.h5 (compressed)
    detector_masks_path = _construct_h5_path(project_path, video_id, "detector_masks.h5")
    with open_h5_with_lock(detector_masks_path, mode="w") as f:
        f.create_dataset(
            "data",
            shape=(num_frames, height, width),
            dtype=np.uint8,
            chunks=(1, height, width),
            compression="gzip",
            fillvalue=0,
        )

    # 4. Final: final_masks.h5
    final_masks_path = _construct_h5_path(project_path, video_id, "final_masks.h5")
    with open_h5_with_lock(final_masks_path, mode="w") as f:
        f.create_dataset(
            "data",
            shape=(num_frames, height, width),
            dtype=np.uint8,
            chunks=(1, height, width),
            fillvalue=0,
        )


def delete_video_segmentation_arrays(project_path: Path, video_id: int) -> None:
    """Delete all segmentation arrays for a video.

    Deletes tracker_masks.h5, tracker_logits.h5, detector_masks.h5, and final_masks.h5
    from array_data/{video_id}/. Does NOT touch cropped/aligned masks.

    Also evicts any cached read handles for these files.

    Args:
        project_path: Path to the project folder
        video_id: ID of the video
    """
    video_dir = project_path / "array_data" / str(video_id)
    file_names = ["tracker_masks.h5", "tracker_logits.h5", "detector_masks.h5", "final_masks.h5"]

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


def reset_video_segmentation_arrays(
    project_path: Path,
    video_id: int,
    num_frames: int,
    height: int,
    width: int,
    logits_size: int = 256,
) -> None:
    """Reset all segmentation arrays by deleting and recreating with zeros.

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
    delete_video_segmentation_arrays(project_path, video_id)
    create_video_segmentation_arrays(project_path, video_id, num_frames, height, width, logits_size)


# ----- Pipeline H5 file creation -----
# These files are created lazily when their respective pipeline runs


def create_cropped_masks_array(
    project_path: Path,
    video_id: int,
    num_frames: int,
    crop_size: int,
) -> None:
    """Create cropped masks array for a video.

    Args:
        project_path: Path to the project folder
        video_id: ID of the video
        num_frames: Total number of frames in the video
        crop_size: Square crop dimension
    """
    h5_path = _construct_h5_path(project_path, video_id, "cropped_masks.h5")
    with open_h5_with_lock(h5_path, mode="w") as f:
        f.create_dataset(
            "data",
            shape=(num_frames, crop_size, crop_size),
            dtype=np.uint8,
            fillvalue=0,
            chunks=(1, crop_size, crop_size),
            compression=None,
        )


def create_aligned_masks_array(
    project_path: Path,
    video_id: int,
    num_frames: int,
    crop_size: int,
) -> None:
    """Create aligned masks array for a video.

    Args:
        project_path: Path to the project folder
        video_id: ID of the video
        num_frames: Total number of frames in the video
        crop_size: Square crop dimension (same as cropped masks)
    """
    h5_path = _construct_h5_path(project_path, video_id, "aligned_masks.h5")
    with open_h5_with_lock(h5_path, mode="w") as f:
        f.create_dataset(
            "data",
            shape=(num_frames, crop_size, crop_size),
            dtype=np.uint8,
            fillvalue=0,
            chunks=(1, crop_size, crop_size),
            compression=None,
        )


def create_alignment_keypoints_array(
    project_path: Path,
    video_id: int,
    num_frames: int,
    crop_size: int,
) -> None:
    """Create alignment keypoints (heatmaps) array for a video.

    Args:
        project_path: Path to the project folder
        video_id: ID of the video
        num_frames: Total number of frames in the video
        crop_size: Heatmap dimensions (same as cropped masks)
    """
    h5_path = _construct_h5_path(project_path, video_id, "alignment_keypoints.h5")
    with open_h5_with_lock(h5_path, mode="w") as f:
        f.create_dataset(
            "data",
            shape=(num_frames, crop_size, crop_size, 2),
            dtype=np.float32,
            fillvalue=0.0,
            chunks=(1, crop_size, crop_size, 2),
            compression=None,
        )


def create_pca_scores_array(
    project_path: Path,
    video_id: int,
    num_frames: int,
    n_components: int,
) -> None:
    """Create PCA scores array for a video.

    Args:
        project_path: Path to the project folder
        video_id: ID of the video
        num_frames: Total number of frames in the video
        n_components: Number of PCA components
    """
    h5_path = _construct_h5_path(project_path, video_id, "pca_scores.h5")
    with open_h5_with_lock(h5_path, mode="w") as f:
        f.create_dataset(
            "data",
            shape=(num_frames, n_components),
            dtype=np.float32,
            fillvalue=0.0,
        )

