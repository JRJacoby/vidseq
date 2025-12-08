"""Mask service - HDF5-based segmentation mask storage with file-based locking."""

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import h5py
import numpy as np


def _get_h5_path(project_path: Path) -> Path:
    """Get path to the HDF5 file for a project."""
    return project_path / "vidseq.h5"


@contextmanager
def open_h5(project_path: Path, mode: str):
    """
    Context manager for HDF5 file access with file-based locking.
    
    Args:
        project_path: Path to the project folder
        mode: File mode - 'r' for read-only, 'a' for append/write
        
    Yields:
        h5py.File: The opened HDF5 file handle
        
    Raises:
        ValueError: If mode is not 'r' or 'a'
        FileNotFoundError: If project_path or h5_path parent directory doesn't exist
        PermissionError: If lock file cannot be created due to permissions
        RuntimeError: If file is locked by another process or thread (write mode)
        RuntimeError: If lock file is missing during cleanup (indicates system compromise)
    """
    if mode not in ('r', 'a'):
        raise ValueError(f"Invalid mode '{mode}'. Must be 'r' (read) or 'a' (append/write)")
    
    if not project_path.exists():
        raise FileNotFoundError(f"Project path does not exist: {project_path}")
    
    h5_path = _get_h5_path(project_path)
    lock_path = h5_path.with_suffix('.h5.lock')
    
    lock_created = False
    
    if mode == 'a':
        if lock_path.exists():
            raise RuntimeError(
                f"HDF5 file is locked by another process OR THREAD. "
                f"Lock file: {lock_path}"
            )
        
        try:
            lock_path.touch(exist_ok=False)
            lock_created = True
        except PermissionError as e:
            raise PermissionError(
                f"Cannot create lock file due to permissions: {lock_path}"
            ) from e
    
    h5_file = None
    try:
        h5_file = h5py.File(h5_path, mode)
        yield h5_file
    finally:
        if h5_file is not None:
            h5_file.close()
        
        if lock_created:
            if not lock_path.exists():
                raise RuntimeError(
                    f"Lock file was removed by another process/thread. "
                    f"This indicates the locking system has been compromised. "
                    f"Lock file: {lock_path}"
                )
            lock_path.unlink()


def get_or_create_mask_dataset(
    project_path: Path,
    video_id: int,
    num_frames: int,
    height: int,
    width: int,
    h5_file: Optional[h5py.File] = None,
) -> None:
    """Ensure mask dataset exists for a video, creating it with zeros if needed."""
    dataset_name = f"segmentation_masks/{video_id}"
    
    if h5_file is not None:
        if dataset_name not in h5_file:
            h5_file.create_dataset(
                dataset_name,
                shape=(num_frames, height, width),
                dtype=np.uint8,
                fillvalue=0,
                chunks=(1, height, width),
                compression=None,
            )
            h5_file.flush()
    else:
        with open_h5(project_path, 'a') as h5_file:
            if dataset_name not in h5_file:
                h5_file.create_dataset(
                    dataset_name,
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
    num_frames: Optional[int] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    h5_file: Optional[h5py.File] = None,
) -> None:
    """Save a mask to the HDF5 file."""
    dataset_name = f"segmentation_masks/{video_id}"
    
    if h5_file is not None:
        if dataset_name not in h5_file:
            if num_frames is None or height is None or width is None:
                raise ValueError(
                    "num_frames, height, and width required when dataset doesn't exist"
                )
            h5_file.create_dataset(
                dataset_name,
                shape=(num_frames, height, width),
                dtype=np.uint8,
                fillvalue=0,
                chunks=(1, height, width),
                compression=None,
            )
        
        h5_file[dataset_name][frame_idx] = mask
        h5_file.flush()
    else:
        with open_h5(project_path, 'a') as h5_file:
            if dataset_name not in h5_file:
                if num_frames is None or height is None or width is None:
                    raise ValueError(
                        "num_frames, height, and width required when dataset doesn't exist"
                    )
                h5_file.create_dataset(
                    dataset_name,
                    shape=(num_frames, height, width),
                    dtype=np.uint8,
                    fillvalue=0,
                    chunks=(1, height, width),
                    compression=None,
                )
            
            h5_file[dataset_name][frame_idx] = mask
            h5_file.flush()


def load_mask(
    project_path: Path,
    video_id: int,
    frame_idx: int,
    num_frames: int,
    height: int,
    width: int,
    h5_file: Optional[h5py.File] = None,
) -> np.ndarray:
    """Load a mask from the HDF5 file, returning zeros if not found."""
    dataset_name = f"segmentation_masks/{video_id}"
    
    if h5_file is not None:
        if dataset_name not in h5_file:
            h5_file.create_dataset(
                dataset_name,
                shape=(num_frames, height, width),
                dtype=np.uint8,
                fillvalue=0,
                chunks=(1, height, width),
                compression=None,
            )
            h5_file.flush()
            return np.zeros((height, width), dtype=np.uint8)
        
        return np.array(h5_file[dataset_name][frame_idx])
    else:
        # Try read mode first (no lock required, can run in parallel with writes)
        try:
            with open_h5(project_path, 'r') as h5_file:
                if dataset_name in h5_file:
                    return np.array(h5_file[dataset_name][frame_idx])
        except (KeyError, OSError):
            pass
        
        # Dataset doesn't exist, need write mode to create it
        with open_h5(project_path, 'a') as h5_file:
            if dataset_name not in h5_file:
                h5_file.create_dataset(
                    dataset_name,
                    shape=(num_frames, height, width),
                    dtype=np.uint8,
                    fillvalue=0,
                    chunks=(1, height, width),
                    compression=None,
                )
                h5_file.flush()
                return np.zeros((height, width), dtype=np.uint8)
            
            return np.array(h5_file[dataset_name][frame_idx])


def load_masks_batch(
    project_path: Path,
    video_id: int,
    start_frame: int,
    count: int,
    num_frames: int,
    height: int,
    width: int,
    h5_file: Optional[h5py.File] = None,
) -> np.ndarray:
    """
    Load multiple masks efficiently using H5 slice indexing.
    
    Returns array of shape (actual_count, height, width) where actual_count
    may be less than count if start_frame + count exceeds num_frames.
    """
    dataset_name = f"segmentation_masks/{video_id}"
    end_frame = min(start_frame + count, num_frames)
    
    if h5_file is not None:
        if dataset_name not in h5_file:
            h5_file.create_dataset(
                dataset_name,
                shape=(num_frames, height, width),
                dtype=np.uint8,
                fillvalue=0,
                chunks=(1, height, width),
                compression=None,
            )
            h5_file.flush()
            return np.zeros((end_frame - start_frame, height, width), dtype=np.uint8)
        
        return np.array(h5_file[dataset_name][start_frame:end_frame])
    else:
        # Try read mode first (no lock required, can run in parallel with writes)
        try:
            with open_h5(project_path, 'r') as h5_file:
                if dataset_name in h5_file:
                    return np.array(h5_file[dataset_name][start_frame:end_frame])
        except (KeyError, OSError):
            pass
        
        # Dataset doesn't exist, need write mode to create it
        with open_h5(project_path, 'a') as h5_file:
            if dataset_name not in h5_file:
                h5_file.create_dataset(
                    dataset_name,
                    shape=(num_frames, height, width),
                    dtype=np.uint8,
                    fillvalue=0,
                    chunks=(1, height, width),
                    compression=None,
                )
                h5_file.flush()
                return np.zeros((end_frame - start_frame, height, width), dtype=np.uint8)
            
            return np.array(h5_file[dataset_name][start_frame:end_frame])


def clear_mask(
    project_path: Path,
    video_id: int,
    frame_idx: int,
    h5_file: Optional[h5py.File] = None,
) -> None:
    """Clear (zero out) a mask for a specific frame."""
    dataset_name = f"segmentation_masks/{video_id}"
    
    if h5_file is not None:
        if dataset_name in h5_file:
            ds = h5_file[dataset_name]
            ds[frame_idx] = np.zeros((ds.shape[1], ds.shape[2]), dtype=np.uint8)
            h5_file.flush()
    else:
        with open_h5(project_path, 'a') as h5_file:
            if dataset_name in h5_file:
                ds = h5_file[dataset_name]
                ds[frame_idx] = np.zeros((ds.shape[1], ds.shape[2]), dtype=np.uint8)
                h5_file.flush()


def clear_all_masks(
    project_path: Path,
    video_id: int,
    h5_file: Optional[h5py.File] = None,
) -> None:
    """Delete all masks for a video by removing the dataset."""
    dataset_name = f"segmentation_masks/{video_id}"
    
    if h5_file is not None:
        if dataset_name in h5_file:
            del h5_file[dataset_name]
            h5_file.flush()
    else:
        with open_h5(project_path, 'a') as h5_file:
            if dataset_name in h5_file:
                del h5_file[dataset_name]
                h5_file.flush()


def get_or_create_frame_types_dataset(
    project_path: Path,
    video_id: int,
    num_frames: int,
    h5_file: Optional[h5py.File] = None,
) -> None:
    """Ensure frame types dataset exists for a video, creating it with empty strings if needed."""
    dataset_name = f"frame_types/{video_id}"
    
    if h5_file is not None:
        if dataset_name not in h5_file:
            h5_file.create_dataset(
                dataset_name,
                shape=(num_frames,),
                dtype=h5py.string_dtype(encoding='utf-8'),
                fillvalue='',
                chunks=(num_frames,),
                compression=None,
            )
            h5_file.flush()
    else:
        with open_h5(project_path, 'a') as h5_file:
            if dataset_name not in h5_file:
                h5_file.create_dataset(
                    dataset_name,
                    shape=(num_frames,),
                    dtype=h5py.string_dtype(encoding='utf-8'),
                    fillvalue='',
                    chunks=(num_frames,),
                    compression=None,
                )
                h5_file.flush()


def mark_frame_type(
    project_path: Path,
    video_id: int,
    frame_idx: int,
    frame_type: str,
    num_frames: Optional[int] = None,
    h5_file: Optional[h5py.File] = None,
) -> None:
    """
    Mark a frame with a type ('train', 'apply', or '' for None).
    
    Args:
        project_path: Path to the project folder
        video_id: Video ID
        frame_idx: Frame index
        frame_type: Frame type ('train', 'apply', or '' for None)
        num_frames: Number of frames (required if dataset doesn't exist)
        h5_file: Optional pre-opened HDF5 file handle
    """
    dataset_name = f"frame_types/{video_id}"
    
    if h5_file is not None:
        if dataset_name not in h5_file:
            if num_frames is None:
                raise ValueError("num_frames required when dataset doesn't exist")
            h5_file.create_dataset(
                dataset_name,
                shape=(num_frames,),
                dtype=h5py.string_dtype(encoding='utf-8'),
                fillvalue='',
                chunks=(num_frames,),
                compression=None,
            )
        
        h5_file[dataset_name][frame_idx] = frame_type
        h5_file.flush()
    else:
        with open_h5(project_path, 'a') as h5_file:
            if dataset_name not in h5_file:
                if num_frames is None:
                    raise ValueError("num_frames required when dataset doesn't exist")
                h5_file.create_dataset(
                    dataset_name,
                    shape=(num_frames,),
                    dtype=h5py.string_dtype(encoding='utf-8'),
                    fillvalue='',
                    chunks=(num_frames,),
                    compression=None,
                )
            
            h5_file[dataset_name][frame_idx] = frame_type
            h5_file.flush()


def get_frame_type(
    project_path: Path,
    video_id: int,
    frame_idx: int,
    num_frames: Optional[int] = None,
    h5_file: Optional[h5py.File] = None,
) -> str:
    """
    Get the frame type for a specific frame.
    
    Args:
        project_path: Path to the project folder
        video_id: Video ID
        frame_idx: Frame index
        num_frames: Number of frames (required if dataset doesn't exist)
        h5_file: Optional pre-opened HDF5 file handle
        
    Returns:
        Frame type string ('train', 'apply', or '' for None)
    """
    dataset_name = f"frame_types/{video_id}"
    
    if h5_file is not None:
        if dataset_name not in h5_file:
            # If file is opened in read mode, can't create dataset - return default
            if h5_file.mode == 'r':
                return ''
            if num_frames is None:
                return ''
            h5_file.create_dataset(
                dataset_name,
                shape=(num_frames,),
                dtype=h5py.string_dtype(encoding='utf-8'),
                fillvalue='',
                chunks=(num_frames,),
                compression=None,
            )
            h5_file.flush()
            return ''
        
        frame_type_bytes = h5_file[dataset_name][frame_idx]
        if isinstance(frame_type_bytes, bytes):
            return frame_type_bytes.decode('utf-8')
        return str(frame_type_bytes) if frame_type_bytes else ''
    else:
        with open_h5(project_path, 'a') as h5_file:
            if dataset_name not in h5_file:
                if num_frames is None:
                    return ''
                h5_file.create_dataset(
                    dataset_name,
                    shape=(num_frames,),
                    dtype=h5py.string_dtype(encoding='utf-8'),
                    fillvalue='',
                    chunks=(num_frames,),
                    compression=None,
                )
                h5_file.flush()
                return ''
            
            frame_type_bytes = h5_file[dataset_name][frame_idx]
            if isinstance(frame_type_bytes, bytes):
                return frame_type_bytes.decode('utf-8')
            return str(frame_type_bytes) if frame_type_bytes else ''


def get_training_frames(
    project_path: Path,
    video_id: int,
    num_frames: int,
) -> list[int]:
    """
    Get list of frame indices marked as 'train'.
    
    Args:
        project_path: Path to the project folder
        video_id: Video ID
        num_frames: Number of frames in video
        
    Returns:
        List of frame indices marked as 'train'
    """
    training_frames = []
    with open_h5(project_path, 'r') as h5_file:
        for frame_idx in range(num_frames):
            frame_type = get_frame_type(project_path, video_id, frame_idx, num_frames, h5_file=h5_file)
            if frame_type == 'train':
                training_frames.append(frame_idx)
    return training_frames


def clear_all_frame_types(
    project_path: Path,
    video_id: int,
    h5_file: Optional[h5py.File] = None,
) -> None:
    """Clear all frame types for a video by removing the dataset."""
    dataset_name = f"frame_types/{video_id}"
    
    if h5_file is not None:
        if dataset_name in h5_file:
            del h5_file[dataset_name]
            h5_file.flush()
    else:
        with open_h5(project_path, 'a') as h5_file:
            if dataset_name in h5_file:
                del h5_file[dataset_name]
                h5_file.flush()


def get_or_create_bbox_dataset(
    project_path: Path,
    video_id: int,
    num_frames: int,
    h5_file: Optional[h5py.File] = None,
) -> None:
    """Ensure bbox dataset exists for a video, creating it with zeros if needed."""
    dataset_name = f"bounding_boxes/{video_id}"
    
    if h5_file is not None:
        if dataset_name not in h5_file:
            h5_file.create_dataset(
                dataset_name,
                shape=(num_frames, 4),
                dtype=np.float32,
                fillvalue=0.0,
                chunks=(1, 4),
                compression=None,
            )
            h5_file.flush()
    else:
        with open_h5(project_path, 'a') as h5_file:
            if dataset_name not in h5_file:
                h5_file.create_dataset(
                    dataset_name,
                    shape=(num_frames, 4),
                    dtype=np.float32,
                    fillvalue=0.0,
                    chunks=(1, 4),
                    compression=None,
                )
                h5_file.flush()


def save_bbox(
    project_path: Path,
    video_id: int,
    frame_idx: int,
    bbox: np.ndarray,
    num_frames: Optional[int] = None,
    h5_file: Optional[h5py.File] = None,
) -> None:
    """
    Save a bounding box to the HDF5 file.
    
    Args:
        project_path: Path to the project folder
        video_id: Video ID
        frame_idx: Frame index
        bbox: Bounding box as numpy array [x1, y1, x2, y2]
        num_frames: Number of frames (required if dataset doesn't exist)
        h5_file: Optional pre-opened HDF5 file handle
    """
    dataset_name = f"bounding_boxes/{video_id}"
    
    if h5_file is not None:
        if dataset_name not in h5_file:
            if num_frames is None:
                raise ValueError("num_frames required when dataset doesn't exist")
            h5_file.create_dataset(
                dataset_name,
                shape=(num_frames, 4),
                dtype=np.float32,
                fillvalue=0.0,
                chunks=(1, 4),
                compression=None,
            )
        
        h5_file[dataset_name][frame_idx] = bbox
        h5_file.flush()
    else:
        with open_h5(project_path, 'a') as h5_file:
            if dataset_name not in h5_file:
                if num_frames is None:
                    raise ValueError("num_frames required when dataset doesn't exist")
                h5_file.create_dataset(
                    dataset_name,
                    shape=(num_frames, 4),
                    dtype=np.float32,
                    fillvalue=0.0,
                    chunks=(1, 4),
                    compression=None,
                )
            
            h5_file[dataset_name][frame_idx] = bbox
            h5_file.flush()


def load_bbox(
    project_path: Path,
    video_id: int,
    frame_idx: int,
    num_frames: Optional[int] = None,
    h5_file: Optional[h5py.File] = None,
) -> Optional[np.ndarray]:
    """
    Load a bounding box from the HDF5 file.
    
    Args:
        project_path: Path to the project folder
        video_id: Video ID
        frame_idx: Frame index
        num_frames: Number of frames (required if dataset doesn't exist)
        h5_file: Optional pre-opened HDF5 file handle
        
    Returns:
        Bounding box as numpy array [x1, y1, x2, y2] or None if no bbox exists
    """
    dataset_name = f"bounding_boxes/{video_id}"
    
    if h5_file is not None:
        if dataset_name not in h5_file:
            if num_frames is None:
                return None
            h5_file.create_dataset(
                dataset_name,
                shape=(num_frames, 4),
                dtype=np.float32,
                fillvalue=0.0,
                chunks=(1, 4),
                compression=None,
            )
            h5_file.flush()
            return None
        
        bbox = np.array(h5_file[dataset_name][frame_idx])
        # Check if bbox is empty (all zeros)
        if np.all(bbox == 0):
            return None
        return bbox
    else:
        # Try read mode first (no lock required, can run in parallel with writes)
        try:
            with open_h5(project_path, 'r') as h5_file:
                if dataset_name in h5_file:
                    bbox = np.array(h5_file[dataset_name][frame_idx])
                    # Check if bbox is empty (all zeros)
                    if np.all(bbox == 0):
                        return None
                    return bbox
        except (KeyError, OSError):
            pass
        
        # Dataset doesn't exist, need write mode to create it
        with open_h5(project_path, 'a') as h5_file:
            if dataset_name not in h5_file:
                if num_frames is None:
                    return None
                h5_file.create_dataset(
                    dataset_name,
                    shape=(num_frames, 4),
                    dtype=np.float32,
                    fillvalue=0.0,
                    chunks=(1, 4),
                    compression=None,
                )
                h5_file.flush()
                return None
            
            bbox = np.array(h5_file[dataset_name][frame_idx])
            # Check if bbox is empty (all zeros)
            if np.all(bbox == 0):
                return None
            return bbox


def load_bboxes_batch(
    project_path: Path,
    video_id: int,
    start_frame: int,
    count: int,
    num_frames: int,
    h5_file: Optional[h5py.File] = None,
) -> np.ndarray:
    """
    Load multiple bounding boxes efficiently using H5 slice indexing.
    
    Returns array of shape (actual_count, 4) where actual_count
    may be less than count if start_frame + count exceeds num_frames.
    """
    dataset_name = f"bounding_boxes/{video_id}"
    end_frame = min(start_frame + count, num_frames)
    
    if h5_file is not None:
        if dataset_name not in h5_file:
            h5_file.create_dataset(
                dataset_name,
                shape=(num_frames, 4),
                dtype=np.float32,
                fillvalue=0.0,
                chunks=(1, 4),
                compression=None,
            )
            h5_file.flush()
            return np.zeros((end_frame - start_frame, 4), dtype=np.float32)
        
        return np.array(h5_file[dataset_name][start_frame:end_frame])
    else:
        # Try read mode first (no lock required, can run in parallel with writes)
        try:
            with open_h5(project_path, 'r') as h5_file:
                if dataset_name in h5_file:
                    return np.array(h5_file[dataset_name][start_frame:end_frame])
        except (KeyError, OSError):
            pass
        
        # Dataset doesn't exist, need write mode to create it
        with open_h5(project_path, 'a') as h5_file:
            if dataset_name not in h5_file:
                h5_file.create_dataset(
                    dataset_name,
                    shape=(num_frames, 4),
                    dtype=np.float32,
                    fillvalue=0.0,
                    chunks=(1, 4),
                    compression=None,
                )
                h5_file.flush()
                return np.zeros((end_frame - start_frame, 4), dtype=np.float32)
            
            return np.array(h5_file[dataset_name][start_frame:end_frame])


def clear_bbox(
    project_path: Path,
    video_id: int,
    frame_idx: int,
    h5_file: Optional[h5py.File] = None,
) -> None:
    """Clear (zero out) a bounding box for a specific frame."""
    dataset_name = f"bounding_boxes/{video_id}"
    
    if h5_file is not None:
        if dataset_name in h5_file:
            h5_file[dataset_name][frame_idx] = np.zeros(4, dtype=np.float32)
            h5_file.flush()
    else:
        with open_h5(project_path, 'a') as h5_file:
            if dataset_name in h5_file:
                h5_file[dataset_name][frame_idx] = np.zeros(4, dtype=np.float32)
                h5_file.flush()


def clear_all_bboxes(
    project_path: Path,
    video_id: int,
    h5_file: Optional[h5py.File] = None,
) -> None:
    """Delete all bounding boxes for a video by removing the dataset."""
    dataset_name = f"bounding_boxes/{video_id}"
    
    if h5_file is not None:
        if dataset_name in h5_file:
            del h5_file[dataset_name]
            h5_file.flush()
    else:
        with open_h5(project_path, 'a') as h5_file:
            if dataset_name in h5_file:
                del h5_file[dataset_name]
                h5_file.flush()
