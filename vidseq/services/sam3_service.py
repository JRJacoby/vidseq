"""
SAM3 Service with Multiprocessing.

Manages a separate worker process for SAM3 inference to avoid
CUDA/signal conflicts with FastAPI's event loop.
"""

import multiprocessing as mp
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
import uuid
import time
import queue

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


# Process management
_worker_process: Optional[Process] = None
_command_queue: Optional[Queue] = None
_result_queue: Optional[Queue] = None

# State
_status = SAM3Status.NOT_LOADED
_error_message: Optional[str] = None
_sessions: Dict[int, VideoSessionInfo] = {}

# Pending requests waiting for results
_pending_requests: Dict[str, Any] = {}


def _drain_result_queue():
    """Process any pending results from the worker."""
    global _status, _error_message
    
    if _result_queue is None:
        return
    
    while True:
        try:
            result = _result_queue.get_nowait()
        except queue.Empty:
            break
        
        result_type = result.get("type")
        
        if result_type == "status":
            status_str = result.get("status")
            if status_str == "loading_model":
                _status = SAM3Status.LOADING_MODEL
            elif status_str == "ready":
                _status = SAM3Status.READY
            elif status_str == "error":
                _status = SAM3Status.ERROR
                _error_message = result.get("error")
        
        elif result_type in ("init_session_result", "add_bbox_prompt_result", 
                            "close_session_result", "get_video_dimensions_result"):
            request_id = result.get("request_id")
            if request_id and request_id in _pending_requests:
                _pending_requests[request_id] = result


def get_status() -> dict:
    """Get the current SAM3 loading status."""
    _drain_result_queue()
    return {
        "status": _status.value,
        "error": _error_message,
    }


def start_worker():
    """Start the SAM3 worker process and begin loading the model."""
    global _worker_process, _command_queue, _result_queue, _status
    
    if _worker_process is not None and _worker_process.is_alive():
        return  # Already running
    
    # Use spawn to avoid CUDA fork issues
    ctx = mp.get_context("spawn")
    _command_queue = ctx.Queue()
    _result_queue = ctx.Queue()
    
    from vidseq.services.sam3_worker import worker_loop
    
    _worker_process = ctx.Process(
        target=worker_loop,
        args=(_command_queue, _result_queue),
        daemon=True,
    )
    _worker_process.start()
    _status = SAM3Status.LOADING_MODEL
    
    # Send load command
    _command_queue.put({"type": "load_model"})


def start_loading_in_background():
    """Start loading SAM3 model in background (via worker process)."""
    if _status == SAM3Status.NOT_LOADED:
        start_worker()


def ensure_worker_ready():
    """Ensure worker is running and model is loaded."""
    _drain_result_queue()
    
    if _status != SAM3Status.READY:
        raise RuntimeError("SAM3 model not loaded. Call preload first.")
    
    if _worker_process is None or not _worker_process.is_alive():
        raise RuntimeError("SAM3 worker process not running.")


def _send_and_wait(cmd: dict, timeout: float = 120.0) -> dict:
    """Send a command to worker and wait for the result."""
    ensure_worker_ready()
    
    request_id = str(uuid.uuid4())
    cmd["request_id"] = request_id
    _pending_requests[request_id] = None
    
    _command_queue.put(cmd)
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        _drain_result_queue()
        
        result = _pending_requests.get(request_id)
        if result is not None:
            del _pending_requests[request_id]
            return result
        
        time.sleep(0.01)  # Small sleep to avoid busy-waiting
    
    del _pending_requests[request_id]
    raise TimeoutError(f"Timeout waiting for response to {cmd['type']}")


def init_session(video_id: int, video_path: Path) -> VideoSessionInfo:
    """Initialize a segmentation session for a video."""
    # Check if we already have session info locally
    if video_id in _sessions:
        return _sessions[video_id]
    
    result = _send_and_wait({
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
    _sessions[video_id] = session_info
    return session_info


def get_session(video_id: int) -> Optional[VideoSessionInfo]:
    """Get session info if it exists."""
    return _sessions.get(video_id)


def close_session(video_id: int) -> bool:
    """Close a video session."""
    if video_id not in _sessions:
        return False
    
    try:
        result = _send_and_wait({
            "type": "close_session",
            "video_id": video_id,
        }, timeout=10.0)
    except Exception:
        pass  # Best effort
    
    _sessions.pop(video_id, None)
    return True


def add_bbox_prompt(
    video_id: int,
    video_path: Path,
    frame_idx: int,
    bbox: dict,
) -> np.ndarray:
    """
    Add a bounding box prompt and get the segmentation mask.
    
    Args:
        video_id: ID of the video
        video_path: Path to video file (used to init session if needed)
        frame_idx: Frame index to segment
        bbox: Dict with x1, y1, x2, y2 in normalized [0,1] coords
        
    Returns:
        Binary mask as numpy array (height, width), dtype=uint8, values 0 or 255
    """
    # Ensure session exists
    session = get_session(video_id)
    if session is None:
        session = init_session(video_id, video_path)
    
    result = _send_and_wait({
        "type": "add_bbox_prompt",
        "video_id": video_id,
        "frame_idx": frame_idx,
        "bbox": bbox,
    }, timeout=120.0)
    
    if result.get("status") != "ok":
        raise RuntimeError(result.get("error", "Failed to add prompt"))
    
    # Reconstruct mask from bytes
    mask_bytes = result["mask_bytes"]
    mask_shape = tuple(result["mask_shape"])
    mask = np.frombuffer(mask_bytes, dtype=np.uint8).reshape(mask_shape)
    
    return mask


def get_video_dimensions(video_id: int) -> Optional[tuple]:
    """Get (height, width) for a video session."""
    session = get_session(video_id)
    if session is None:
        return None
    return (session.height, session.width)


def shutdown_worker():
    """Shutdown the worker process gracefully."""
    global _worker_process, _command_queue, _result_queue, _status
    
    if _command_queue is not None:
        try:
            _command_queue.put({"type": "shutdown"})
        except Exception:
            pass
    
    if _worker_process is not None:
        _worker_process.join(timeout=5.0)
        if _worker_process.is_alive():
            _worker_process.terminate()
        _worker_process = None
    
    _command_queue = None
    _result_queue = None
    _status = SAM3Status.NOT_LOADED
    _sessions.clear()
