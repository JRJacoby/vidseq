"""HDF5-based mask storage service."""

from pathlib import Path
from typing import Optional

import h5py
import numpy as np


def _get_h5_path(project_path: Path) -> Path:
    """Get path to the HDF5 file for a project."""
    return project_path / "vidseq.h5"


def get_or_create_mask_dataset(
    project_path: Path,
    video_id: int,
    num_frames: int,
    height: int,
    width: int,
) -> None:
    """Ensure mask dataset exists for a video, creating it with zeros if needed."""
    h5_path = _get_h5_path(project_path)
    dataset_name = f"segmentation_masks/{video_id}"
    
    with h5py.File(h5_path, "a") as h5:
        if dataset_name not in h5:
            h5.create_dataset(
                dataset_name,
                shape=(num_frames, height, width),
                dtype=np.uint8,
                fillvalue=0,
                chunks=(1, height, width),
                compression="gzip",
                compression_opts=4,
            )


def save_mask(
    project_path: Path,
    video_id: int,
    frame_idx: int,
    mask: np.ndarray,
    num_frames: Optional[int] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
) -> None:
    """Save a mask to the HDF5 file."""
    h5_path = _get_h5_path(project_path)
    dataset_name = f"segmentation_masks/{video_id}"
    
    needs_creation = False
    if not h5_path.exists():
        needs_creation = True
    else:
        with h5py.File(h5_path, "r") as h5:
            needs_creation = dataset_name not in h5
    
    if needs_creation:
        if num_frames is None or height is None or width is None:
            raise ValueError(
                "num_frames, height, and width required when dataset doesn't exist"
            )
        get_or_create_mask_dataset(project_path, video_id, num_frames, height, width)
    
    with h5py.File(h5_path, "a") as h5:
        h5[dataset_name][frame_idx] = mask


def load_mask(
    project_path: Path,
    video_id: int,
    frame_idx: int,
    num_frames: int,
    height: int,
    width: int,
) -> np.ndarray:
    """Load a mask from the HDF5 file, returning zeros if not found."""
    h5_path = _get_h5_path(project_path)
    dataset_name = f"segmentation_masks/{video_id}"
    
    if not h5_path.exists():
        get_or_create_mask_dataset(project_path, video_id, num_frames, height, width)
        return np.zeros((height, width), dtype=np.uint8)
    
    with h5py.File(h5_path, "r") as h5:
        if dataset_name not in h5:
            get_or_create_mask_dataset(project_path, video_id, num_frames, height, width)
            return np.zeros((height, width), dtype=np.uint8)
        
        return np.array(h5[dataset_name][frame_idx])


def clear_mask(
    project_path: Path,
    video_id: int,
    frame_idx: int,
) -> None:
    """Clear (zero out) a mask for a specific frame."""
    h5_path = _get_h5_path(project_path)
    dataset_name = f"segmentation_masks/{video_id}"
    
    if not h5_path.exists():
        return
    
    with h5py.File(h5_path, "a") as h5:
        if dataset_name in h5:
            ds = h5[dataset_name]
            ds[frame_idx] = np.zeros((ds.shape[1], ds.shape[2]), dtype=np.uint8)
