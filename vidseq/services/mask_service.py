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


def compute_bbox_from_mask(mask: np.ndarray) -> Optional[np.ndarray]:
    """
    Compute bounding box [x1, y1, x2, y2] from a binary mask.
    
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


def _frames_to_ranges(frame_list: list[int]) -> list[tuple[int, int]]:
    """Convert a sorted list of frame indices to contiguous ranges."""
    if not frame_list:
        return []
    
    ranges = []
    start = frame_list[0]
    end = frame_list[0]
    
    for frame in frame_list[1:]:
        if frame == end + 1:
            end = frame
        else:
            ranges.append((start, end))
            start = frame
            end = frame
    
    ranges.append((start, end))
    return ranges


def get_masked_frame_ranges(
    project_path: Path,
    video_id: int,
    num_frames: int,
) -> list[tuple[int, int]]:
    """
    Return list of (start, end) tuples for contiguous masked frame ranges.
    
    A frame is considered "masked" if its mask contains any non-zero pixels.
    """
    masked_frames = []
    dataset_name = f"segmentation_masks/{video_id}"
    
    try:
        with open_h5(project_path, 'r') as h5_file:
            if dataset_name not in h5_file:
                return []
            
            for frame_idx in range(num_frames):
                mask = h5_file[dataset_name][frame_idx]
                if np.any(mask > 0):
                    masked_frames.append(frame_idx)
    except (OSError, FileNotFoundError):
        return []
    
    return _frames_to_ranges(masked_frames)


def get_training_frame_ranges(
    project_path: Path,
    video_id: int,
    num_frames: int,
) -> list[tuple[int, int]]:
    """
    Return list of (start, end) tuples for contiguous training frame ranges.
    """
    training_frames = get_training_frames(project_path, video_id, num_frames)
    return _frames_to_ranges(training_frames)


def validate_frames_have_masks(
    project_path: Path,
    video_id: int,
    start_frame: int,
    end_frame: int,
) -> list[int]:
    """
    Return list of frame indices in range that are missing masks.
    Empty list means all frames in range have valid masks.
    
    Args:
        project_path: Path to the project folder
        video_id: Video ID
        start_frame: Start frame index (inclusive)
        end_frame: End frame index (inclusive)
        
    Returns:
        List of frame indices that are missing masks
    """
    missing_frames = []
    dataset_name = f"segmentation_masks/{video_id}"
    
    try:
        with open_h5(project_path, 'r') as h5_file:
            if dataset_name not in h5_file:
                return list(range(start_frame, end_frame + 1))
            
            for frame_idx in range(start_frame, end_frame + 1):
                mask = h5_file[dataset_name][frame_idx]
                if not np.any(mask > 0):
                    missing_frames.append(frame_idx)
    except (OSError, FileNotFoundError):
        return list(range(start_frame, end_frame + 1))
    
    return missing_frames


def mark_training_range(
    project_path: Path,
    video_id: int,
    start_frame: int,
    end_frame: int,
    num_frames: int,
    height: int,
    width: int,
) -> None:
    """
    Mark frame range as training, computing bboxes from existing masks.
    
    Args:
        project_path: Path to the project folder
        video_id: Video ID
        start_frame: Start frame index (inclusive)
        end_frame: End frame index (inclusive)
        num_frames: Total number of frames in video
        height: Video height in pixels
        width: Video width in pixels
    """
    mask_dataset_name = f"segmentation_masks/{video_id}"
    bbox_dataset_name = f"bounding_boxes/{video_id}"
    frame_type_dataset_name = f"frame_types/{video_id}"
    
    with open_h5(project_path, 'a') as h5_file:
        if bbox_dataset_name not in h5_file:
            h5_file.create_dataset(
                bbox_dataset_name,
                shape=(num_frames, 4),
                dtype=np.float32,
                fillvalue=0.0,
                chunks=(1, 4),
                compression=None,
            )
        
        if frame_type_dataset_name not in h5_file:
            h5_file.create_dataset(
                frame_type_dataset_name,
                shape=(num_frames,),
                dtype=h5py.string_dtype(encoding='utf-8'),
                fillvalue='',
                chunks=(num_frames,),
                compression=None,
            )
        
        for frame_idx in range(start_frame, end_frame + 1):
            if mask_dataset_name in h5_file:
                mask = np.array(h5_file[mask_dataset_name][frame_idx])
                bbox = compute_bbox_from_mask(mask)
                if bbox is not None:
                    h5_file[bbox_dataset_name][frame_idx] = bbox
            
            h5_file[frame_type_dataset_name][frame_idx] = 'train'
        
        h5_file.flush()


def unmark_training_range(
    project_path: Path,
    video_id: int,
    start_frame: int,
    end_frame: int,
    num_frames: int,
) -> None:
    """
    Remove training labels and bboxes for frame range.
    
    Args:
        project_path: Path to the project folder
        video_id: Video ID
        start_frame: Start frame index (inclusive)
        end_frame: End frame index (inclusive)
        num_frames: Total number of frames in video
    """
    bbox_dataset_name = f"bounding_boxes/{video_id}"
    frame_type_dataset_name = f"frame_types/{video_id}"
    
    with open_h5(project_path, 'a') as h5_file:
        for frame_idx in range(start_frame, end_frame + 1):
            if bbox_dataset_name in h5_file:
                h5_file[bbox_dataset_name][frame_idx] = np.zeros(4, dtype=np.float32)
            
            if frame_type_dataset_name in h5_file:
                h5_file[frame_type_dataset_name][frame_idx] = ''
        
        h5_file.flush()
