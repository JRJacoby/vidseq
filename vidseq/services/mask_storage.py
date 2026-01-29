"""Mask storage - per-video HDF5 files for segmentation masks.

This module handles storage of binary segmentation masks in per-video HDF5 files.
Each video's masks are stored in a separate file at `masks/{video_id}.h5`, which
allows concurrent writes to different videos without locking conflicts.
"""

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

# Disable HDF5's internal file locking (we use our own lock files)
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import h5py
import numpy as np


def _get_masks_dir(project_path: Path, mask_subdir: str = "masks") -> Path:
    """Get the masks directory for a project.

    Args:
        project_path: Path to the project folder
        mask_subdir: Subdirectory name ("masks", "cropped_masks", or "aligned_masks")
    """
    return project_path / mask_subdir


def _get_video_h5_path(project_path: Path, video_id: int, mask_subdir: str = "masks") -> Path:
    """Get path to the HDF5 file for a specific video's masks."""
    return _get_masks_dir(project_path, mask_subdir) / f"{video_id}.h5"


def _get_lock_path(h5_path: Path) -> Path:
    """Get the lock file path for an HDF5 file."""
    return h5_path.with_suffix(".h5.lock")


@contextmanager
def open_video_h5(project_path: Path, video_id: int, mode: str, mask_subdir: str = "masks"):
    """Context manager for per-video HDF5 mask file access with file locking.

    Args:
        project_path: Path to the project folder
        video_id: ID of the video
        mode: File mode - 'r' for read-only, 'a' for append/write
        mask_subdir: Subdirectory name ("masks", "cropped_masks", or "aligned_masks")

    Yields:
        h5py.File: The opened HDF5 file handle

    Raises:
        ValueError: If mode is not 'r' or 'a'
        FileNotFoundError: If project_path doesn't exist
        RuntimeError: If file is locked by another process (write mode only)
    """
    if mode not in ("r", "a"):
        raise ValueError(f"Invalid mode '{mode}'. Must be 'r' or 'a'")

    if not project_path.exists():
        raise FileNotFoundError(f"Project path does not exist: {project_path}")

    masks_dir = _get_masks_dir(project_path, mask_subdir)
    if mode == "a":
        masks_dir.mkdir(parents=True, exist_ok=True)

    h5_path = _get_video_h5_path(project_path, video_id, mask_subdir)
    lock_path = _get_lock_path(h5_path)

    lock_created = False
    if mode == "a":
        if lock_path.exists():
            raise RuntimeError(
                f"HDF5 file is locked by another process. Lock file: {lock_path}"
            )
        try:
            lock_path.touch(exist_ok=False)
            lock_created = True
        except FileExistsError:
            raise RuntimeError(
                f"HDF5 file is locked by another process. Lock file: {lock_path}"
            )

    h5_file = None
    try:
        h5_file = h5py.File(h5_path, mode)
        yield h5_file
    finally:
        if h5_file is not None:
            h5_file.close()
        if lock_created and lock_path.exists():
            lock_path.unlink()


def _ensure_dataset(
    h5_file: h5py.File,
    num_frames: int,
    height: int,
    width: int,
) -> None:
    """Ensure the masks dataset exists in the file."""
    if "masks" not in h5_file:
        h5_file.create_dataset(
            "masks",
            shape=(num_frames, height, width),
            dtype=np.uint8,
            fillvalue=0,
            chunks=(1, height, width),
            compression=None,
        )
        h5_file.flush()


def save_mask(
    project_path: Path,
    video_id: int,
    frame_idx: int,
    mask: np.ndarray,
    num_frames: int,
    height: int,
    width: int,
    h5_file: Optional[h5py.File] = None,
    mask_subdir: str = "masks",
) -> None:
    """Save a mask to the per-video HDF5 file.

    Args:
        project_path: Path to the project folder
        video_id: ID of the video
        frame_idx: Frame index (0-based)
        mask: Binary mask array (height, width) with dtype uint8
        num_frames: Total number of frames in the video
        height: Video height in pixels
        width: Video width in pixels
        h5_file: Optional pre-opened h5py.File in write mode
        mask_subdir: Subdirectory name ("masks", "cropped_masks", or "aligned_masks")
    """
    if h5_file is not None:
        _ensure_dataset(h5_file, num_frames, height, width)
        h5_file["masks"][frame_idx] = mask
    else:
        with open_video_h5(project_path, video_id, "a", mask_subdir) as f:
            _ensure_dataset(f, num_frames, height, width)
            f["masks"][frame_idx] = mask
            f.flush()


def load_mask(
    project_path: Path,
    video_id: int,
    frame_idx: int,
    num_frames: int,
    height: int,
    width: int,
    h5_file: Optional[h5py.File] = None,
    mask_subdir: str = "masks",
) -> np.ndarray:
    """Load a mask from the per-video HDF5 file.

    Returns zeros if the file or dataset doesn't exist.

    Args:
        project_path: Path to the project folder
        video_id: ID of the video
        frame_idx: Frame index (0-based)
        num_frames: Total number of frames in the video
        height: Video height in pixels
        width: Video width in pixels
        h5_file: Optional pre-opened h5py.File
        mask_subdir: Subdirectory name ("masks", "cropped_masks", or "aligned_masks")

    Returns:
        Mask array (height, width) with dtype uint8
    """
    if h5_file is not None:
        if "masks" not in h5_file:
            print(f"[DEBUG load_mask] h5_file provided but no 'masks' dataset")
            return np.zeros((height, width), dtype=np.uint8)
        mask = np.array(h5_file["masks"][frame_idx])
        print(f"[DEBUG load_mask] from h5_file: frame={frame_idx}, sum={int(mask.sum())}")
        return mask

    h5_path = _get_video_h5_path(project_path, video_id, mask_subdir)
    print(f"[DEBUG load_mask] h5_path={h5_path}, exists={h5_path.exists()}")
    if not h5_path.exists():
        print(f"[DEBUG load_mask] file doesn't exist, returning zeros")
        return np.zeros((height, width), dtype=np.uint8)

    try:
        with open_video_h5(project_path, video_id, "r", mask_subdir) as f:
            if "masks" not in f:
                print(f"[DEBUG load_mask] 'masks' dataset not in file")
                return np.zeros((height, width), dtype=np.uint8)
            mask = np.array(f["masks"][frame_idx])
            print(f"[DEBUG load_mask] loaded: frame={frame_idx}, sum={int(mask.sum())}, "
                  f"shape={mask.shape}, dtype={mask.dtype}")
            return mask
    except (OSError, KeyError) as e:
        print(f"[DEBUG load_mask] exception: {e}")
        return np.zeros((height, width), dtype=np.uint8)


def load_masks_batch(
    project_path: Path,
    video_id: int,
    start_frame: int,
    count: int,
    num_frames: int,
    height: int,
    width: int,
    h5_file: Optional[h5py.File] = None,
    mask_subdir: str = "masks",
) -> np.ndarray:
    """Load multiple masks efficiently using HDF5 slice indexing.

    Args:
        project_path: Path to the project folder
        video_id: ID of the video
        start_frame: Starting frame index (0-based)
        count: Number of frames to load
        num_frames: Total number of frames in the video
        height: Video height in pixels
        width: Video height in pixels
        h5_file: Optional pre-opened h5py.File
        mask_subdir: Subdirectory name ("masks", "cropped_masks", or "aligned_masks")

    Returns:
        Array of shape (actual_count, height, width) where actual_count
        may be less than count if start_frame + count exceeds num_frames.
    """
    end_frame = min(start_frame + count, num_frames)
    actual_count = end_frame - start_frame

    if h5_file is not None:
        if "masks" not in h5_file:
            return np.zeros((actual_count, height, width), dtype=np.uint8)
        return np.array(h5_file["masks"][start_frame:end_frame])

    h5_path = _get_video_h5_path(project_path, video_id, mask_subdir)
    if not h5_path.exists():
        return np.zeros((actual_count, height, width), dtype=np.uint8)

    try:
        with open_video_h5(project_path, video_id, "r", mask_subdir) as f:
            if "masks" not in f:
                return np.zeros((actual_count, height, width), dtype=np.uint8)
            return np.array(f["masks"][start_frame:end_frame])
    except (OSError, KeyError):
        return np.zeros((actual_count, height, width), dtype=np.uint8)


def clear_mask(
    project_path: Path,
    video_id: int,
    frame_idx: int,
    h5_file: Optional[h5py.File] = None,
    mask_subdir: str = "masks",
) -> None:
    """Clear (zero out) a mask for a specific frame.

    Args:
        project_path: Path to the project folder
        video_id: ID of the video
        frame_idx: Frame index (0-based)
        h5_file: Optional pre-opened h5py.File in write mode
        mask_subdir: Subdirectory name ("masks", "cropped_masks", or "aligned_masks")
    """
    if h5_file is not None:
        if "masks" in h5_file:
            ds = h5_file["masks"]
            ds[frame_idx] = np.zeros((ds.shape[1], ds.shape[2]), dtype=np.uint8)
            h5_file.flush()
        return

    h5_path = _get_video_h5_path(project_path, video_id, mask_subdir)
    if not h5_path.exists():
        return

    with open_video_h5(project_path, video_id, "a", mask_subdir) as f:
        if "masks" in f:
            ds = f["masks"]
            ds[frame_idx] = np.zeros((ds.shape[1], ds.shape[2]), dtype=np.uint8)
            f.flush()


def clear_all_masks(
    project_path: Path, video_id: int, mask_subdir: str = "masks"
) -> None:
    """Delete all masks for a video by removing the HDF5 file.

    Args:
        project_path: Path to the project folder
        video_id: ID of the video
        mask_subdir: Subdirectory name ("masks", "cropped_masks", or "aligned_masks")
    """
    h5_path = _get_video_h5_path(project_path, video_id, mask_subdir)
    lock_path = _get_lock_path(h5_path)

    # Check for lock
    if lock_path.exists():
        raise RuntimeError(
            f"Cannot delete - file is locked by another process: {lock_path}"
        )

    if h5_path.exists():
        h5_path.unlink()


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


def video_has_masks(project_path: Path, video_id: int) -> bool:
    """Check if a video has an HDF5 mask file.

    Args:
        project_path: Path to the project folder
        video_id: ID of the video

    Returns:
        True if the mask file exists, False otherwise
    """
    h5_path = _get_video_h5_path(project_path, video_id)
    return h5_path.exists()
