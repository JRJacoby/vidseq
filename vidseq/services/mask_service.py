"""Mask service - HDF5-based segmentation mask storage with persistent file handles."""

import atexit
from pathlib import Path
from threading import Lock
from typing import Optional

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
            if path_str not in self._files or not self._files[path_str].id.valid:
                # Create parent directory if needed
                path.parent.mkdir(parents=True, exist_ok=True)
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
