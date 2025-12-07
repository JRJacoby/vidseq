"""
SAM2 Service with Multiprocessing.

Manages a separate worker process for SAM2 inference to avoid
CUDA/signal conflicts with FastAPI's event loop.

Uses SAM2VideoPredictor with point prompts and lazy frame loading.
"""

import multiprocessing as mp
import queue
import threading
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Optional

import numpy as np


class SAM2Status(str, Enum):
    NOT_LOADED = "not_loaded"
    LOADING_MODEL = "loading_model"
    READY = "ready"
    ERROR = "error"


OBJ_ID = 1


@dataclass
class VideoSessionInfo:
    """Info about a video session (stored in main process)."""
    video_id: int
    num_frames: int
    height: int
    width: int
    has_object: bool = False


class SAM2Service:
    """
    Singleton service for managing SAM2 inference.
    
    Manages a separate worker process to avoid CUDA/signal conflicts
    with FastAPI's event loop.
    """
    
    _instance: Optional["SAM2Service"] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> "SAM2Service":
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
        self._status = SAM2Status.NOT_LOADED
        self._error_message: Optional[str] = None
        self._sessions: dict[int, VideoSessionInfo] = {}
        self._pending_requests: dict[str, dict | None] = {}
        self._prompts: dict[int, list[dict]] = {}  # frame_idx -> list of prompts
        
        self._initialized = True
    
    @classmethod
    def get_instance(cls) -> "SAM2Service":
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
                    self._status = SAM2Status.LOADING_MODEL
                elif status_str == "ready":
                    self._status = SAM2Status.READY
                elif status_str == "error":
                    self._status = SAM2Status.ERROR
                    self._error_message = result.get("error")
            
            elif result_type in ("init_session_result", "add_point_prompt_result",
                                "close_session_result", "reset_state_result",
                                "propagate_forward_result", "propagate_backward_result"):
                request_id = result.get("request_id")
                if request_id and request_id in self._pending_requests:
                    self._pending_requests[request_id] = result
    
    def get_status(self) -> dict:
        """Get the current SAM2 loading status."""
        self._drain_result_queue()
        
        if self._status == SAM2Status.READY and self._worker_process is not None:
            if not self._worker_process.is_alive():
                self._status = SAM2Status.NOT_LOADED
                self._worker_process = None
                self._command_queue = None
                self._result_queue = None
                self._sessions.clear()
        
        return {
            "status": self._status.value,
            "error": self._error_message,
        }
    
    def _start_worker(self) -> None:
        """Start the SAM2 worker process and begin loading the model."""
        if self._worker_process is not None and self._worker_process.is_alive():
            return
        
        ctx = mp.get_context("spawn")
        self._command_queue = ctx.Queue()
        self._result_queue = ctx.Queue()
        
        from vidseq.services.sam2_worker import worker_loop
        
        self._worker_process = ctx.Process(
            target=worker_loop,
            args=(self._command_queue, self._result_queue),
            daemon=True,
        )
        self._worker_process.start()
        self._status = SAM2Status.LOADING_MODEL
        
        self._command_queue.put({"type": "load_model"})
    
    def start_loading_in_background(self) -> None:
        """Start loading SAM2 model in background (via worker process)."""
        # Don't start if UNet is training (to avoid GPU memory conflicts)
        try:
            from vidseq.services import unet_service
            unet = unet_service.UNetService.get_instance()
            if unet.is_training():
                return
        except Exception:
            pass
        
        if self._status == SAM2Status.NOT_LOADED:
            self._start_worker()
    
    def _ensure_worker_ready(self) -> None:
        """Ensure worker is running and model is loaded."""
        self._drain_result_queue()
        
        if self._status != SAM2Status.READY:
            raise RuntimeError("SAM2 model not loaded. Call preload first.")
        
        if self._worker_process is None or not self._worker_process.is_alive():
            raise RuntimeError("SAM2 worker process not running.")
    
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
        
        # Clear prompts when initializing a new session
        self._prompts = {}
        
        result = self._send_and_wait({
            "type": "init_session",
            "video_id": video_id,
            "video_path": str(video_path),
        }, timeout=600.0)  # 10 min timeout for first session (torch.compile warmup)
        
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
    
    def add_point_prompt(
        self,
        video_id: int,
        video_path: Path,
        frame_idx: int,
        points: list[list[float]],
        labels: list[int],
    ) -> np.ndarray:
        """
        Add point prompts and get the segmentation mask.
        
        First point creates the tracked object, subsequent points refine it.
        
        Args:
            video_id: ID of the video
            video_path: Path to video file (used to init session if needed)
            frame_idx: Frame index to segment
            points: List of [x, y] coordinates in normalized [0,1] coords
            labels: List of labels (1=positive, 0=negative)
            
        Returns:
            Binary mask as numpy array (height, width), dtype=uint8, values 0 or 255
        """
        session = self.get_session(video_id)
        if session is None:
            session = self.init_session(video_id, video_path)
        
        result = self._send_and_wait({
            "type": "add_point_prompt",
            "video_id": video_id,
            "frame_idx": frame_idx,
            "points": points,
            "labels": labels,
            "obj_id": OBJ_ID,
        }, timeout=120.0)
        
        if result.get("status") != "ok":
            raise RuntimeError(result.get("error", "Failed to add point prompt"))
        
        session.has_object = True
        
        mask_bytes = result["mask_bytes"]
        mask_shape = tuple(result["mask_shape"])
        mask = np.frombuffer(mask_bytes, dtype=np.uint8).reshape(mask_shape)
        
        # Store prompt info
        for i, point in enumerate(points):
            prompt_type = "positive_point" if labels[i] == 1 else "negative_point"
            prompt = {
                "type": prompt_type,
                "x": point[0],
                "y": point[1],
                "frame_idx": frame_idx,
            }
            if frame_idx not in self._prompts:
                self._prompts[frame_idx] = []
            self._prompts[frame_idx].append(prompt)
        
        return mask
    
    def reset_state(self, video_id: int) -> bool:
        """
        Reset the tracking state for a video.
        
        Clears all object tracking memory. User must re-click to define object.
        
        Args:
            video_id: ID of the video
            
        Returns:
            True if successful
        """
        session = self.get_session(video_id)
        if session is None:
            return True
        
        result = self._send_and_wait({
            "type": "reset_state",
            "video_id": video_id,
        }, timeout=30.0)
        
        if result.get("status") != "ok":
            raise RuntimeError(result.get("error", "Failed to reset state"))
        
        session.has_object = False
        # Clear prompts when resetting state
        self._prompts = {}
        return True
    
    def propagate_forward(
        self,
        video_id: int,
        start_frame_idx: int,
        max_frames: int,
    ) -> list[tuple[int, np.ndarray, Optional[list[float]]]]:
        """
        Propagate tracking forward from a given frame.
        
        Args:
            video_id: ID of the video
            start_frame_idx: Frame index to start propagation from
            max_frames: Maximum number of frames to propagate
            
        Returns:
            List of (frame_idx, mask, bbox) tuples where bbox is [x1, y1, x2, y2] or None
            
        Raises:
            RuntimeError: If no object has been tracked
        """
        session = self.get_session(video_id)
        if session is None:
            raise RuntimeError("No session exists. Add a point prompt first.")
        
        if not session.has_object:
            raise RuntimeError("No object tracked. Add a point prompt first.")
        
        result = self._send_and_wait({
            "type": "propagate_forward",
            "video_id": video_id,
            "start_frame_idx": start_frame_idx,
            "max_frames": max_frames,
        }, timeout=600.0)
        
        if result.get("status") != "ok":
            raise RuntimeError(result.get("error", "Failed to propagate"))
        
        masks = []
        for mask_data in result.get("masks", []):
            frame_idx = mask_data["frame_idx"]
            mask_bytes = mask_data["mask_bytes"]
            mask_shape = tuple(mask_data["mask_shape"])
            mask = np.frombuffer(mask_bytes, dtype=np.uint8).reshape(mask_shape)
            bbox = mask_data.get("bbox")  # [x1, y1, x2, y2] or None
            masks.append((frame_idx, mask, bbox))
        
        return masks
    
    def propagate_backward(
        self,
        video_id: int,
        start_frame_idx: int,
        max_frames: int,
    ) -> list[tuple[int, np.ndarray, Optional[list[float]]]]:
        """
        Propagate tracking backward from a given frame.
        
        Args:
            video_id: ID of the video
            start_frame_idx: Frame index to start propagation from
            max_frames: Maximum number of frames to propagate
            
        Returns:
            List of (frame_idx, mask, bbox) tuples where bbox is [x1, y1, x2, y2] or None
            
        Raises:
            RuntimeError: If no object has been tracked
        """
        session = self.get_session(video_id)
        if session is None:
            raise RuntimeError("No session exists. Add a point prompt first.")
        
        if not session.has_object:
            raise RuntimeError("No object tracked. Add a point prompt first.")
        
        result = self._send_and_wait({
            "type": "propagate_backward",
            "video_id": video_id,
            "start_frame_idx": start_frame_idx,
            "max_frames": max_frames,
        }, timeout=600.0)
        
        if result.get("status") != "ok":
            raise RuntimeError(result.get("error", "Failed to propagate backward"))
        
        masks = []
        for mask_data in result.get("masks", []):
            frame_idx = mask_data["frame_idx"]
            mask_bytes = mask_data["mask_bytes"]
            mask_shape = tuple(mask_data["mask_shape"])
            mask = np.frombuffer(mask_bytes, dtype=np.uint8).reshape(mask_shape)
            bbox = mask_data.get("bbox")  # [x1, y1, x2, y2] or None
            masks.append((frame_idx, mask, bbox))
        
        return masks
    
    def get_prompts_for_frame(self, frame_idx: int) -> list[dict]:
        """Get all prompts for a specific frame."""
        return self._prompts.get(frame_idx, [])
    
    def get_all_prompts(self) -> dict[int, list[dict]]:
        """Get all prompts for all frames."""
        return self._prompts.copy()
    
    def clear_prompts_for_frame(self, frame_idx: int) -> None:
        """Clear prompts for a specific frame."""
        if frame_idx in self._prompts:
            del self._prompts[frame_idx]
    
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
        self._status = SAM2Status.NOT_LOADED
        self._sessions.clear()


def get_status() -> dict:
    """Get the current SAM2 loading status."""
    return SAM2Service.get_instance().get_status()


def start_loading_in_background() -> None:
    """Start loading SAM2 model in background (via worker process)."""
    SAM2Service.get_instance().start_loading_in_background()


def init_session(video_id: int, video_path: Path) -> VideoSessionInfo:
    """Initialize a segmentation session for a video."""
    return SAM2Service.get_instance().init_session(video_id, video_path)


def get_session(video_id: int) -> Optional[VideoSessionInfo]:
    """Get session info if it exists."""
    return SAM2Service.get_instance().get_session(video_id)


def close_session(video_id: int) -> bool:
    """Close a video session."""
    return SAM2Service.get_instance().close_session(video_id)


def add_point_prompt(
    video_id: int,
    video_path: Path,
    frame_idx: int,
    points: list[list[float]],
    labels: list[int],
) -> np.ndarray:
    """
    Add point prompts and get the segmentation mask.
    
    First point creates the tracked object, subsequent points refine it.
    """
    return SAM2Service.get_instance().add_point_prompt(
        video_id, video_path, frame_idx, points, labels
    )


def reset_state(video_id: int) -> bool:
    """Reset the tracking state for a video."""
    return SAM2Service.get_instance().reset_state(video_id)


def propagate_forward(
    video_id: int,
    start_frame_idx: int,
    max_frames: int,
) -> list[tuple[int, np.ndarray]]:
    """Propagate tracking forward from a given frame."""
    return SAM2Service.get_instance().propagate_forward(
        video_id, start_frame_idx, max_frames
    )


def propagate_backward(
    video_id: int,
    start_frame_idx: int,
    max_frames: int,
) -> list[tuple[int, np.ndarray]]:
    """Propagate tracking backward from a given frame."""
    return SAM2Service.get_instance().propagate_backward(
        video_id, start_frame_idx, max_frames
    )


def shutdown_worker() -> None:
    """Shutdown the worker process gracefully."""
    SAM2Service.get_instance().shutdown()


def get_prompts_for_frame(frame_idx: int) -> list[dict]:
    """Get all prompts for a specific frame."""
    return SAM2Service.get_instance().get_prompts_for_frame(frame_idx)


def get_all_prompts() -> dict[int, list[dict]]:
    """Get all prompts for all frames."""
    return SAM2Service.get_instance().get_all_prompts()


def clear_prompts_for_frame(frame_idx: int) -> None:
    """Clear prompts for a specific frame."""
    SAM2Service.get_instance().clear_prompts_for_frame(frame_idx)


