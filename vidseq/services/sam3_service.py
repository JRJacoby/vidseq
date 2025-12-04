"""
SAM3 Service with Multiprocessing.

Manages a separate worker process for SAM3 inference to avoid
CUDA/signal conflicts with FastAPI's event loop.
"""

import multiprocessing as mp
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from enum import Enum
import uuid
import time
import queue
import threading

import numpy as np


class SAM3Status(str, Enum):
    NOT_LOADED = "not_loaded"
    LOADING_MODEL = "loading_model"
    READY = "ready"
    ERROR = "error"


@dataclass
class VideoSessionInfo:
    """Info about a video session (stored in main process)."""
    video_id: int
    num_frames: int
    height: int
    width: int
    active_obj_id: Optional[int] = None


class SAM3Service:
    """
    Singleton service for managing SAM3 inference.
    
    Manages a separate worker process to avoid CUDA/signal conflicts
    with FastAPI's event loop.
    """
    
    _instance: Optional["SAM3Service"] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> "SAM3Service":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._worker_process: Optional[Process] = None
        self._command_queue: Optional[Queue] = None
        self._result_queue: Optional[Queue] = None
        self._status = SAM3Status.NOT_LOADED
        self._error_message: Optional[str] = None
        self._sessions: dict[int, VideoSessionInfo] = {}
        self._pending_requests: dict[str, dict | None] = {}
        
        self._initialized = True
    
    @classmethod
    def get_instance(cls) -> "SAM3Service":
        """Get the singleton instance."""
        return cls()
    
    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton (useful for testing)."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance.shutdown()
                cls._instance = None
    
    def _drain_result_queue(self) -> None:
        """Process any pending results from the worker."""
        if self._result_queue is None:
            return
        
        while True:
            try:
                result = self._result_queue.get_nowait()
            except queue.Empty:
                break
            
            result_type = result.get("type")
            
            if result_type == "status":
                status_str = result.get("status")
                if status_str == "loading_model":
                    self._status = SAM3Status.LOADING_MODEL
                elif status_str == "ready":
                    self._status = SAM3Status.READY
                elif status_str == "error":
                    self._status = SAM3Status.ERROR
                    self._error_message = result.get("error")
            
            elif result_type in ("init_session_result", "add_bbox_prompt_result",
                                "add_point_prompt_result", "close_session_result",
                                "remove_object_result", "propagate_forward_result",
                                "inject_mask_result"):
                request_id = result.get("request_id")
                if request_id and request_id in self._pending_requests:
                    self._pending_requests[request_id] = result
    
    def get_status(self) -> dict:
        """Get the current SAM3 loading status."""
        self._drain_result_queue()
        
        # Check if worker died (e.g., uvicorn restart killed it)
        if self._status == SAM3Status.READY and self._worker_process is not None:
            if not self._worker_process.is_alive():
                self._status = SAM3Status.NOT_LOADED
                self._worker_process = None
                self._command_queue = None
                self._result_queue = None
                self._sessions.clear()
        
        return {
            "status": self._status.value,
            "error": self._error_message,
        }
    
    def _start_worker(self) -> None:
        """Start the SAM3 worker process and begin loading the model."""
        if self._worker_process is not None and self._worker_process.is_alive():
            return
        
        ctx = mp.get_context("spawn")
        self._command_queue = ctx.Queue()
        self._result_queue = ctx.Queue()
        
        from vidseq.services.sam3_worker import worker_loop
        
        self._worker_process = ctx.Process(
            target=worker_loop,
            args=(self._command_queue, self._result_queue),
            daemon=True,
        )
        self._worker_process.start()
        self._status = SAM3Status.LOADING_MODEL
        
        self._command_queue.put({"type": "load_model"})
    
    def start_loading_in_background(self) -> None:
        """Start loading SAM3 model in background (via worker process)."""
        if self._status == SAM3Status.NOT_LOADED:
            self._start_worker()
    
    def _ensure_worker_ready(self) -> None:
        """Ensure worker is running and model is loaded."""
        self._drain_result_queue()
        
        if self._status != SAM3Status.READY:
            raise RuntimeError("SAM3 model not loaded. Call preload first.")
        
        if self._worker_process is None or not self._worker_process.is_alive():
            raise RuntimeError("SAM3 worker process not running.")
    
    def _send_and_wait(self, cmd: dict, timeout: float = 120.0) -> dict:
        """Send a command to worker and wait for the result."""
        self._ensure_worker_ready()
        
        request_id = str(uuid.uuid4())
        cmd["request_id"] = request_id
        self._pending_requests[request_id] = None
        
        self._command_queue.put(cmd)
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            self._drain_result_queue()
            
            result = self._pending_requests.get(request_id)
            if result is not None:
                del self._pending_requests[request_id]
                return result
            
            time.sleep(0.01)
        
        del self._pending_requests[request_id]
        raise TimeoutError(f"Timeout waiting for response to {cmd['type']}")
    
    def init_session(self, video_id: int, video_path: Path) -> VideoSessionInfo:
        """Initialize a segmentation session for a video."""
        if video_id in self._sessions:
            return self._sessions[video_id]
        
        result = self._send_and_wait({
            "type": "init_session",
            "video_id": video_id,
            "video_path": str(video_path),
        })
        
        if result.get("status") != "ok":
            raise RuntimeError(result.get("error", "Failed to init session"))
        
        session_info = VideoSessionInfo(
            video_id=video_id,
            num_frames=result["num_frames"],
            height=result["height"],
            width=result["width"],
        )
        self._sessions[video_id] = session_info
        return session_info
    
    def get_session(self, video_id: int) -> Optional[VideoSessionInfo]:
        """Get session info if it exists."""
        return self._sessions.get(video_id)
    
    def close_session(self, video_id: int) -> bool:
        """Close a video session."""
        if video_id not in self._sessions:
            return False
        
        try:
            self._send_and_wait({
                "type": "close_session",
                "video_id": video_id,
            }, timeout=10.0)
        except Exception:
            pass
        
        self._sessions.pop(video_id, None)
        return True
    
    def add_bbox_prompt(
        self,
        video_id: int,
        video_path: Path,
        frame_idx: int,
        bbox: dict,
        text: str,
    ) -> np.ndarray:
        """
        Add a bounding box prompt and get the segmentation mask.
        
        Args:
            video_id: ID of the video
            video_path: Path to video file (used to init session if needed)
            frame_idx: Frame index to segment
            bbox: Dict with x1, y1, x2, y2 in normalized [0,1] coords
            text: Text description of the object to segment (e.g., "dog", "person")
            
        Returns:
            Binary mask as numpy array (height, width), dtype=uint8, values 0 or 255
        """
        session = self.get_session(video_id)
        if session is None:
            session = self.init_session(video_id, video_path)
        
        result = self._send_and_wait({
            "type": "add_bbox_prompt",
            "video_id": video_id,
            "frame_idx": frame_idx,
            "bbox": bbox,
            "text": text,
        }, timeout=120.0)
        
        if result.get("status") != "ok":
            raise RuntimeError(result.get("error", "Failed to add prompt"))
        
        mask_bytes = result["mask_bytes"]
        mask_shape = tuple(result["mask_shape"])
        mask = np.frombuffer(mask_bytes, dtype=np.uint8).reshape(mask_shape)
        
        obj_id = result.get("obj_id")
        if obj_id is not None and session is not None:
            session.active_obj_id = obj_id
        
        return mask
    
    def add_point_prompt(
        self,
        video_id: int,
        video_path: Path,
        frame_idx: int,
        points: list[list[float]],
        labels: list[int],
    ) -> np.ndarray:
        """
        Add point prompts and get the updated segmentation mask.
        
        Args:
            video_id: ID of the video
            video_path: Path to video file (used to init session if needed)
            frame_idx: Frame index to segment
            points: List of [x, y] coordinates in normalized [0,1] coords
            labels: List of labels (1=positive, 0=negative)
            
        Returns:
            Binary mask as numpy array (height, width), dtype=uint8, values 0 or 255
        
        Raises:
            RuntimeError: If no object exists (must add bbox first)
        """
        session = self.get_session(video_id)
        if session is None:
            raise RuntimeError("No session exists. Add a bounding box prompt first.")
        
        if session.active_obj_id is None:
            raise RuntimeError("No active object. Add a bounding box prompt first.")
        
        result = self._send_and_wait({
            "type": "add_point_prompt",
            "video_id": video_id,
            "frame_idx": frame_idx,
            "points": points,
            "labels": labels,
            "obj_id": session.active_obj_id,
        }, timeout=120.0)
        
        if result.get("status") != "ok":
            raise RuntimeError(result.get("error", "Failed to add point prompt"))
        
        mask_bytes = result["mask_bytes"]
        mask_shape = tuple(result["mask_shape"])
        mask = np.frombuffer(mask_bytes, dtype=np.uint8).reshape(mask_shape)
        
        return mask
    
    def remove_object(self, video_id: int) -> bool:
        """
        Remove the active tracked object from SAM3's inference state.
        
        Args:
            video_id: ID of the video
            
        Returns:
            True if successful (or no object to remove)
        """
        session = self.get_session(video_id)
        if session is None or session.active_obj_id is None:
            return True
        
        obj_id = session.active_obj_id
        
        result = self._send_and_wait({
            "type": "remove_object",
            "video_id": video_id,
            "obj_id": obj_id,
        }, timeout=30.0)
        
        if result.get("status") != "ok":
            raise RuntimeError(result.get("error", "Failed to remove object"))
        
        session.active_obj_id = None
        return True
        return True
    
    def propagate_forward(
        self,
        video_id: int,
        start_frame_idx: int,
        max_frames: int,
    ) -> list[tuple[int, np.ndarray]]:
        """
        Propagate tracking forward from a given frame.
        
        Args:
            video_id: ID of the video
            start_frame_idx: Frame index to start propagation from
            max_frames: Maximum number of frames to propagate
            
        Returns:
            List of (frame_idx, mask) tuples
            
        Raises:
            RuntimeError: If no active session exists
        """
        session = self.get_session(video_id)
        if session is None:
            raise RuntimeError("No session exists. Add a bounding box prompt first.")
        
        if session.active_obj_id is None:
            raise RuntimeError("No active object. Add a bounding box prompt first.")
        
        result = self._send_and_wait({
            "type": "propagate_forward",
            "video_id": video_id,
            "start_frame_idx": start_frame_idx,
            "max_frames": max_frames,
        }, timeout=600.0)  # Longer timeout for propagation
        
        if result.get("status") != "ok":
            raise RuntimeError(result.get("error", "Failed to propagate"))
        
        masks = []
        for mask_data in result.get("masks", []):
            frame_idx = mask_data["frame_idx"]
            mask_bytes = mask_data["mask_bytes"]
            mask_shape = tuple(mask_data["mask_shape"])
            mask = np.frombuffer(mask_bytes, dtype=np.uint8).reshape(mask_shape)
            masks.append((frame_idx, mask))
        
        return masks
    
    def inject_mask(
        self,
        video_id: int,
        video_path: Path,
        frame_idx: int,
        mask: np.ndarray,
        obj_id: int = 0,
    ) -> None:
        """
        Inject a mask directly into SAM3's inference state.
        
        This is used to restore conditioning frames after a server restart.
        The mask becomes part of SAM3's memory for tracking/propagation.
        
        Args:
            video_id: ID of the video
            video_path: Path to video file (used to init session if needed)
            frame_idx: Frame index where mask belongs
            mask: Binary mask (height, width), dtype=uint8, values 0 or 255
            obj_id: Object ID to associate with this mask
        """
        session = self.get_session(video_id)
        if session is None:
            session = self.init_session(video_id, video_path)
        
        result = self._send_and_wait({
            "type": "inject_mask",
            "video_id": video_id,
            "frame_idx": frame_idx,
            "mask_bytes": mask.tobytes(),
            "mask_shape": mask.shape,
            "obj_id": obj_id,
        }, timeout=60.0)
        
        if result.get("status") != "ok":
            raise RuntimeError(result.get("error", "Failed to inject mask"))
        
        if session.active_obj_id is None:
            session.active_obj_id = obj_id
    
    def shutdown(self) -> None:
        """Shutdown the worker process gracefully."""
        if self._command_queue is not None:
            try:
                self._command_queue.put({"type": "shutdown"})
            except Exception:
                pass
        
        if self._worker_process is not None:
            self._worker_process.join(timeout=5.0)
            if self._worker_process.is_alive():
                self._worker_process.terminate()
            self._worker_process = None
        
        self._command_queue = None
        self._result_queue = None
        self._status = SAM3Status.NOT_LOADED
        self._sessions.clear()


# Module-level convenience functions that delegate to the singleton.
# These maintain backwards compatibility with existing code.

def get_status() -> dict:
    """Get the current SAM3 loading status."""
    return SAM3Service.get_instance().get_status()


def start_loading_in_background() -> None:
    """Start loading SAM3 model in background (via worker process)."""
    SAM3Service.get_instance().start_loading_in_background()


def init_session(video_id: int, video_path: Path) -> VideoSessionInfo:
    """Initialize a segmentation session for a video."""
    return SAM3Service.get_instance().init_session(video_id, video_path)


def get_session(video_id: int) -> Optional[VideoSessionInfo]:
    """Get session info if it exists."""
    return SAM3Service.get_instance().get_session(video_id)


def close_session(video_id: int) -> bool:
    """Close a video session."""
    return SAM3Service.get_instance().close_session(video_id)


def add_bbox_prompt(
    video_id: int,
    video_path: Path,
    frame_idx: int,
    bbox: dict,
    text: str,
) -> np.ndarray:
    """
    Add a bounding box prompt and get the segmentation mask.
    
    Args:
        video_id: ID of the video
        video_path: Path to video file (used to init session if needed)
        frame_idx: Frame index to segment
        bbox: Dict with x1, y1, x2, y2 in normalized [0,1] coords
        text: Text description of the object to segment (e.g., "dog", "person")
        
    Returns:
        Binary mask as numpy array (height, width), dtype=uint8, values 0 or 255
    """
    return SAM3Service.get_instance().add_bbox_prompt(
        video_id, video_path, frame_idx, bbox, text
    )


def add_point_prompt(
    video_id: int,
    video_path: Path,
    frame_idx: int,
    points: list[list[float]],
    labels: list[int],
) -> np.ndarray:
    """
    Add point prompts and get the updated segmentation mask.
    
    Args:
        video_id: ID of the video
        video_path: Path to video file (used to init session if needed)
        frame_idx: Frame index to segment
        points: List of [x, y] coordinates in normalized [0,1] coords
        labels: List of labels (1=positive, 0=negative)
        
    Returns:
        Binary mask as numpy array (height, width), dtype=uint8, values 0 or 255
    """
    return SAM3Service.get_instance().add_point_prompt(
        video_id, video_path, frame_idx, points, labels
    )


def remove_object(video_id: int) -> bool:
    """
    Remove the active tracked object from SAM3's inference state.
    
    Args:
        video_id: ID of the video
        
    Returns:
        True if successful (or no object to remove)
    """
    return SAM3Service.get_instance().remove_object(video_id)


def propagate_forward(
    video_id: int,
    start_frame_idx: int,
    max_frames: int,
) -> list[tuple[int, np.ndarray]]:
    """
    Propagate tracking forward from a given frame.
    
    Args:
        video_id: ID of the video
        start_frame_idx: Frame index to start propagation from
        max_frames: Maximum number of frames to propagate
        
    Returns:
        List of (frame_idx, mask) tuples
    """
    return SAM3Service.get_instance().propagate_forward(
        video_id, start_frame_idx, max_frames
    )


def inject_mask(
    video_id: int,
    video_path: Path,
    frame_idx: int,
    mask: np.ndarray,
    obj_id: int = 0,
) -> None:
    """
    Inject a mask directly into SAM3's inference state.
    
    This is used to restore conditioning frames after a server restart.
    """
    return SAM3Service.get_instance().inject_mask(
        video_id, video_path, frame_idx, mask, obj_id
    )


def shutdown_worker() -> None:
    """Shutdown the worker process gracefully."""
    SAM3Service.get_instance().shutdown()
