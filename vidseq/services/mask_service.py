"""Mask service - HDF5-based segmentation mask storage with persistent file handles."""

import atexit
import os
from pathlib import Path
from threading import Lock
from typing import Optional

os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import h5py
import numpy as np


class H5FileManager:
    """Manages persistent HDF5 file handles per project."""
    
    _instance: Optional["H5FileManager"] = None
    _init_lock = Lock()
    
    def __new__(cls) -> "H5FileManager":
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._files: dict[str, h5py.File] = {}
                    instance._locks: dict[str, Lock] = {}
                    instance._global_lock = Lock()
                    atexit.register(instance.close_all)
                    cls._instance = instance
        return cls._instance
    
    def _get_lock(self, path_str: str) -> Lock:
        """Get or create a lock for a specific file."""
        with self._global_lock:
            if path_str not in self._locks:
                self._locks[path_str] = Lock()
            return self._locks[path_str]
    
    def get_file(self, path: Path) -> tuple[h5py.File, Lock]:
        """
        Get an open file handle and its lock.
        
        Returns (file, lock) - caller should use lock for thread safety.
        """
        path_str = str(path)
        lock = self._get_lock(path_str)
        
        with lock:
            # Check if we need to open/reopen the file
            needs_open = (
                path_str not in self._files or 
                not self._files[path_str].id.valid
            )
            
            if needs_open:
                # Create parent directory if needed
                path.parent.mkdir(parents=True, exist_ok=True)
                
                # Close any existing handle before opening
                if path_str in self._files:
                    try:
                        self._files[path_str].close()
                    except Exception:
                        pass
                    del self._files[path_str]
                
                # Try to open the file, handling stuck locks from network disconnects
                try:
                    self._files[path_str] = h5py.File(path, "a")
                except BlockingIOError:
                    # OS-level lock is stuck (e.g., from crashed process)
                    # Retry opening once - if this fails, let the exception propagate
                    self._files[path_str] = h5py.File(path, "a")
            
            return self._files[path_str], lock
    
    def close_file(self, path: Path) -> None:
        """Close a specific file (e.g., when leaving a project)."""
        path_str = str(path)
        with self._global_lock:
            if path_str in self._files:
                try:
                    self._files[path_str].close()
                except Exception:
                    pass
                del self._files[path_str]
    
    def close_all(self) -> None:
        """Close all open files (called on shutdown)."""
        with self._global_lock:
            for f in self._files.values():
                try:
                    f.close()
                except Exception:
                    pass
            self._files.clear()


# Module-level singleton
_h5_manager = H5FileManager()


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
    h5_file, lock = _h5_manager.get_file(h5_path)
    dataset_name = f"segmentation_masks/{video_id}"
    
    with lock:
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
) -> None:
    """Save a mask to the HDF5 file."""
    h5_path = _get_h5_path(project_path)
    h5_file, lock = _h5_manager.get_file(h5_path)
    dataset_name = f"segmentation_masks/{video_id}"
    
    with lock:
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
) -> np.ndarray:
    """Load a mask from the HDF5 file, returning zeros if not found."""
    h5_path = _get_h5_path(project_path)
    h5_file, lock = _h5_manager.get_file(h5_path)
    dataset_name = f"segmentation_masks/{video_id}"
    
    with lock:
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
) -> np.ndarray:
    """
    Load multiple masks efficiently using H5 slice indexing.
    
    Returns array of shape (actual_count, height, width) where actual_count
    may be less than count if start_frame + count exceeds num_frames.
    """
    h5_path = _get_h5_path(project_path)
    h5_file, lock = _h5_manager.get_file(h5_path)
    dataset_name = f"segmentation_masks/{video_id}"
    
    end_frame = min(start_frame + count, num_frames)
    
    with lock:
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
) -> None:
    """Clear (zero out) a mask for a specific frame."""
    h5_path = _get_h5_path(project_path)
    h5_file, lock = _h5_manager.get_file(h5_path)
    dataset_name = f"segmentation_masks/{video_id}"
    
    with lock:
        if dataset_name in h5_file:
            ds = h5_file[dataset_name]
            ds[frame_idx] = np.zeros((ds.shape[1], ds.shape[2]), dtype=np.uint8)
            h5_file.flush()


def clear_all_masks(
    project_path: Path,
    video_id: int,
) -> None:
    """Delete all masks for a video by removing the dataset."""
    h5_path = _get_h5_path(project_path)
    h5_file, lock = _h5_manager.get_file(h5_path)
    dataset_name = f"segmentation_masks/{video_id}"
    
    with lock:
        if dataset_name in h5_file:
            del h5_file[dataset_name]
            h5_file.flush()


def close_project_h5(project_path: Path) -> None:
    """Close the HDF5 file for a project (call when leaving project)."""
    h5_path = _get_h5_path(project_path)
    _h5_manager.close_file(h5_path)


def get_or_create_frame_types_dataset(
    project_path: Path,
    video_id: int,
    num_frames: int,
) -> None:
    """Ensure frame types dataset exists for a video, creating it with empty strings if needed."""
    h5_path = _get_h5_path(project_path)
    h5_file, lock = _h5_manager.get_file(h5_path)
    dataset_name = f"frame_types/{video_id}"
    
    with lock:
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
) -> None:
    """
    Mark a frame with a type ('train', 'apply', or '' for None).
    
    Args:
        project_path: Path to the project folder
        video_id: Video ID
        frame_idx: Frame index
        frame_type: Frame type ('train', 'apply', or '' for None)
        num_frames: Number of frames (required if dataset doesn't exist)
    """
    h5_path = _get_h5_path(project_path)
    h5_file, lock = _h5_manager.get_file(h5_path)
    dataset_name = f"frame_types/{video_id}"
    
    with lock:
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
) -> str:
    """
    Get the frame type for a specific frame.
    
    Args:
        project_path: Path to the project folder
        video_id: Video ID
        frame_idx: Frame index
        num_frames: Number of frames (required if dataset doesn't exist)
        
    Returns:
        Frame type string ('train', 'apply', or '' for None)
    """
    h5_path = _get_h5_path(project_path)
    h5_file, lock = _h5_manager.get_file(h5_path)
    dataset_name = f"frame_types/{video_id}"
    
    with lock:
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
    for frame_idx in range(num_frames):
        frame_type = get_frame_type(project_path, video_id, frame_idx, num_frames)
        if frame_type == 'train':
            training_frames.append(frame_idx)
    return training_frames


def clear_all_frame_types(
    project_path: Path,
    video_id: int,
) -> None:
    """Clear all frame types for a video by removing the dataset."""
    h5_path = _get_h5_path(project_path)
    h5_file, lock = _h5_manager.get_file(h5_path)
    dataset_name = f"frame_types/{video_id}"
    
    with lock:
        if dataset_name in h5_file:
            del h5_file[dataset_name]
            h5_file.flush()
