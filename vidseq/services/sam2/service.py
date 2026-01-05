"""
SAM2 Service with TCP IPC.

Manages a separate worker process for SAM2 inference to avoid
CUDA/signal conflicts with FastAPI's event loop.

Uses SAM2VideoPredictor with point prompts and lazy frame loading.
Communicates with worker via TCP sockets.
"""

import base64
import os
import struct
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np

from vidseq.services.sam2.config import (
    get_sam2_port,
    is_sam2_worker_running,
)
from vidseq.services.sam2.client import SAM2TCPClient


def _decode_mask_rle(mask_rle: str, shape: tuple[int, ...], dtype: str = "uint8") -> np.ndarray:
    """
    Decode binary RLE + base64 mask back to numpy array.
    
    Args:
        mask_rle: Base64-encoded RLE string
        shape: Target shape tuple (height, width)
        dtype: Data type (default: "uint8")
        
    Returns:
        Reconstructed mask array
    """
    # Base64 decode
    binary_data = base64.b64decode(mask_rle.encode('utf-8'))
    
    # Parse runs and reconstruct array
    flat_size = int(np.prod(shape))
    flat_array = np.zeros(flat_size, dtype=np.uint8)
    
    i = 0
    offset = 0
    
    while offset < len(binary_data):
        # Unpack: 1 byte value, 4 bytes length (big-endian uint32)
        value, length = struct.unpack_from('>BI', binary_data, offset)
        offset += 5  # 1 byte + 4 bytes
        
        # Fill the run
        flat_array[i:i+length] = value
        i += length
    
    # Reshape to original shape
    return flat_array.reshape(shape).astype(dtype)


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
        
        self._worker_process: Optional[subprocess.Popen] = None
        self._tcp_client: Optional[SAM2TCPClient] = None
        self._status = SAM2Status.NOT_LOADED
        self._error_message: Optional[str] = None
        # Sessions keyed by (project_id, video_id) to avoid collisions across projects
        self._sessions: dict[tuple[int, int], VideoSessionInfo] = {}
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
    
    def get_status(self) -> dict:
        """Get the current SAM2 loading status."""
        # Only check if worker is running if we have a client connection
        # This avoids creating new connections on every status check
        if self._status == SAM2Status.READY:
            # If we have a client, assume it's still connected unless we get an error
            # Only check worker if we don't have a client
            if self._tcp_client is None:
                if not is_sam2_worker_running():
                    self._status = SAM2Status.NOT_LOADED
                    self._worker_process = None
                    self._sessions.clear()
        
        return {
            "status": self._status.value,
            "error": self._error_message,
        }
    
    def _start_worker(self) -> None:
        """Start the SAM2 worker process and begin loading the model."""
        # Check if worker is already running
        if is_sam2_worker_running():
            port = get_sam2_port()
            if port:
                self._tcp_client = SAM2TCPClient()
                try:
                    self._tcp_client.connect("localhost", port)
                    self._status = SAM2Status.READY
                    return
                except Exception as e:
                    print(f"[SAM2 Service] Failed to connect to existing worker: {e}")
                    self._tcp_client = None
        
        # Start new worker process
        # Use -m format for more robust module resolution
        kwargs = {}
        if sys.platform != 'win32':
            kwargs['start_new_session'] = True
        
        self._worker_process = subprocess.Popen(
            [sys.executable, "-m", "vidseq.services.sam2.server.tcp_server"],
            **kwargs
        )
        
        # Wait for port file to appear (poll with timeout)
        timeout = 30.0
        start_time = time.time()
        port = None
        
        while time.time() - start_time < timeout:
            port = get_sam2_port()
            if port is not None:
                break
            time.sleep(0.1)
        
        if port is None:
            self._status = SAM2Status.ERROR
            self._error_message = "Failed to start SAM2 worker (timeout waiting for port file)"
            return
        
        # Connect to worker
        self._tcp_client = SAM2TCPClient()
        try:
            self._tcp_client.connect("localhost", port, timeout=10.0)
        except Exception as e:
            self._status = SAM2Status.ERROR
            self._error_message = f"Failed to connect to SAM2 worker: {e}"
            self._tcp_client = None
            return
        
        self._status = SAM2Status.LOADING_MODEL
        
        # Send load_model command
        try:
            result = self._tcp_client.send_command({"type": "load_model"}, timeout=600.0)
            if result.get("status") == "ready":
                self._status = SAM2Status.READY
            elif result.get("status") == "error":
                self._status = SAM2Status.ERROR
                self._error_message = result.get("error", "Unknown error")
        except Exception as e:
            self._status = SAM2Status.ERROR
            self._error_message = str(e)
    
    def start_loading_in_background(self) -> None:
        """Start loading SAM2 model in background (via worker process)."""
        # Don't start if YOLO is training (to avoid GPU memory conflicts)
        try:
            from vidseq.services import yolo_service
            yolo = yolo_service.YOLOService.get_instance()
            if yolo.is_training():
                return
        except Exception:
            pass
        
        if self._status == SAM2Status.NOT_LOADED:
            self._start_worker()
    
    def _ensure_worker_ready(self) -> None:
        """Ensure worker is running and model is loaded."""
        if self._status != SAM2Status.READY:
            raise RuntimeError("SAM2 model not loaded. Call preload first.")
        
        # Only create client if it doesn't exist
        # Don't check is_connected() here - let send_command() handle connection errors
        if self._tcp_client is None:
            if not is_sam2_worker_running():
                raise RuntimeError("SAM2 worker process not running.")
            port = get_sam2_port()
            if port:
                self._tcp_client = SAM2TCPClient()
                self._tcp_client.connect("localhost", port)
    
    def _send_and_wait(self, cmd: dict, timeout: float = 120.0) -> dict:
        """Send a command to worker and wait for the result."""
        self._ensure_worker_ready()
        
        request_id = str(uuid.uuid4())
        cmd["request_id"] = request_id
        
        try:
            result = self._tcp_client.send_command(cmd, timeout=timeout)
            
            # Update status if we got a status update
            if result.get("type") == "status":
                status_str = result.get("status")
                if status_str == "loading_model":
                    self._status = SAM2Status.LOADING_MODEL
                elif status_str == "ready":
                    self._status = SAM2Status.READY
                elif status_str == "error":
                    self._status = SAM2Status.ERROR
                    self._error_message = result.get("error")
            
            return result
        except Exception as e:
            # If connection error, mark client as disconnected
            if isinstance(e, (ConnectionError, TimeoutError)) or "Connection" in str(e):
                self._tcp_client = None
                # Don't change status - might be temporary connection issue
            raise RuntimeError(f"Failed to communicate with SAM2 worker: {e}") from e
    
    def init_session(self, project_id: int, video_id: int, video_path: Path) -> VideoSessionInfo:
        """Initialize a segmentation session for a video."""
        session_key = (project_id, video_id)
        if session_key in self._sessions:
            return self._sessions[session_key]

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
        self._sessions[session_key] = session_info
        return session_info

    def get_session(self, project_id: int, video_id: int) -> Optional[VideoSessionInfo]:
        """Get session info if it exists."""
        return self._sessions.get((project_id, video_id))

    def close_session(self, project_id: int, video_id: int) -> bool:
        """Close a video session."""
        session_key = (project_id, video_id)
        if session_key not in self._sessions:
            return False

        try:
            self._send_and_wait({
                "type": "close_session",
                "video_id": video_id,
            }, timeout=10.0)
        except Exception:
            pass

        self._sessions.pop(session_key, None)
        return True
    
    def add_point_prompt(
        self,
        project_id: int,
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
            project_id: ID of the project
            video_id: ID of the video
            video_path: Path to video file (used to init session if needed)
            frame_idx: Frame index to segment
            points: List of [x, y] coordinates in normalized [0,1] coords
            labels: List of labels (1=positive, 0=negative)

        Returns:
            Binary mask as numpy array (height, width), dtype=uint8, values 0 or 255
        """
        session = self.get_session(project_id, video_id)
        if session is None:
            session = self.init_session(project_id, video_id, video_path)
        
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
        
        mask_rle = result["mask_rle"]
        mask_shape = tuple(result["mask_shape"])
        mask_dtype = result.get("mask_dtype", "uint8")
        mask = _decode_mask_rle(mask_rle, mask_shape, mask_dtype)
        
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
    
    def reset_state(self, project_id: int, video_id: int) -> bool:
        """
        Reset the tracking state for a video.

        Clears all object tracking memory. User must re-click to define object.

        Args:
            project_id: ID of the project
            video_id: ID of the video

        Returns:
            True if successful
        """
        session = self.get_session(project_id, video_id)
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

    def clear_frame_prompts(self, project_id: int, video_id: int, frame_idx: int, obj_id: int = 1) -> bool:
        """
        Clear all prompts for a specific frame.

        Removes point and mask inputs for the given frame from SAM2's inference state.
        This is useful when resetting a conditioning frame.

        Args:
            project_id: ID of the project
            video_id: ID of the video
            frame_idx: Frame index to clear
            obj_id: Object ID (default: 1)

        Returns:
            True if successful
        """
        session = self.get_session(project_id, video_id)
        if session is None:
            return True

        result = self._send_and_wait({
            "type": "clear_frame_prompts",
            "video_id": video_id,
            "frame_idx": frame_idx,
            "obj_id": obj_id,
        }, timeout=30.0)

        if result.get("status") != "ok":
            raise RuntimeError(result.get("error", "Failed to clear frame prompts"))

        return True
    
    def generate_training_masks(
        self,
        project_id: int,
        video_id: int,
        start_frame_idx: int,
        max_frames: int,
        project_path: Path,
        num_frames: int,
        height: int,
        width: int,
    ) -> int:
        """
        Generate training masks by propagating tracking forward and save to H5.

        Args:
            project_id: ID of the project
            video_id: ID of the video
            start_frame_idx: Frame index to start propagation from
            max_frames: Maximum number of frames to propagate
            project_path: Path to the project folder
            num_frames: Total number of frames in the video
            height: Video height in pixels
            width: Video width in pixels

        Returns:
            Number of frames processed

        Raises:
            RuntimeError: If no object has been tracked
        """
        session = self.get_session(project_id, video_id)
        if session is None:
            raise RuntimeError("No session exists. Add a point prompt first.")
        
        if not session.has_object:
            raise RuntimeError("No object tracked. Add a point prompt first.")
        
        result = self._send_and_wait({
            "type": "generate_training_masks",
            "video_id": video_id,
            "start_frame_idx": start_frame_idx,
            "max_frames": max_frames,
            "project_path": str(project_path),
            "num_frames": num_frames,
            "height": height,
            "width": width,
        }, timeout=600.0)
        
        if result.get("status") != "ok":
            raise RuntimeError(result.get("error", "Failed to generate training masks"))
        
        return result.get("frames_processed", 0)

    async def segment_all_videos(
        self,
        project_id: int,
        project_path: Path,
        videos: list,
        bboxes: dict[int, np.ndarray],
    ) -> list[int]:
        """
        Start batch segmentation for all videos in a project.
        
        Args:
            project_id: ID of the project
            project_path: Path to the project folder
            videos: List of Video model instances
            bboxes: Dict mapping video_id -> bbox array [x1, y1, x2, y2]
            
        Returns:
            List of job IDs created
        """
        import uuid
        
        # Prepare video configurations
        video_configs = []
        for video in videos:
            bbox = bboxes.get(video.id)
            if bbox is None:
                continue
            
            video_configs.append({
                "video_id": video.id,
                "video_path": video.path,
                "bbox": bbox.tolist(),
                "num_frames": video.num_frames,
                "height": video.height,
                "width": video.width,
            })
        
        if not video_configs:
            return []
        
        # Ensure model is loaded
        if self._status == SAM2Status.NOT_LOADED:
            self.start_loading_in_background()
        
        # Wait if still loading
        timeout = 600.0
        start_time = time.time()
        while self._status == SAM2Status.LOADING_MODEL and time.time() - start_time < timeout:
            time.sleep(1.0)
        
        self._ensure_worker_ready()
        
        # Send command to worker (which now handles job creation and all DB updates)
        cmd = {
            "type": "segment_videos_batch",
            "project_id": project_id,
            "project_path": str(project_path),
            "videos": video_configs,
            "request_id": str(uuid.uuid4()),
        }
        
        result = self._tcp_client.send_command(cmd, timeout=10.0)
        
        # Extract job IDs from response
        if result.get("type") == "segment_videos_batch_started":
            return result.get("job_ids", [])
        
        return []
    
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
        if self._tcp_client is not None and self._tcp_client.is_connected():
            try:
                self._tcp_client.send_command({"type": "shutdown"}, timeout=5.0)
            except Exception:
                pass
            self._tcp_client.disconnect()
            self._tcp_client = None
        
        # Note: We don't terminate the worker process here because it might be
        # used by other clients (e.g., job executor). The worker will auto-shutdown
        # when no connections remain.
        self._worker_process = None
        self._status = SAM2Status.NOT_LOADED
        self._sessions.clear()


def get_status() -> dict:
    """Get the current SAM2 loading status."""
    return SAM2Service.get_instance().get_status()


def start_loading_in_background() -> None:
    """Start loading SAM2 model in background (via worker process)."""
    SAM2Service.get_instance().start_loading_in_background()


def init_session(project_id: int, video_id: int, video_path: Path) -> VideoSessionInfo:
    """Initialize a segmentation session for a video."""
    return SAM2Service.get_instance().init_session(project_id, video_id, video_path)


def get_session(project_id: int, video_id: int) -> Optional[VideoSessionInfo]:
    """Get session info if it exists."""
    return SAM2Service.get_instance().get_session(project_id, video_id)


def close_session(project_id: int, video_id: int) -> bool:
    """Close a video session."""
    return SAM2Service.get_instance().close_session(project_id, video_id)


def add_point_prompt(
    project_id: int,
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
        project_id, video_id, video_path, frame_idx, points, labels
    )


def reset_state(project_id: int, video_id: int) -> bool:
    """Reset the tracking state for a video."""
    return SAM2Service.get_instance().reset_state(project_id, video_id)


def clear_frame_prompts(project_id: int, video_id: int, frame_idx: int, obj_id: int = 1) -> bool:
    """Clear all prompts for a specific frame."""
    return SAM2Service.get_instance().clear_frame_prompts(project_id, video_id, frame_idx, obj_id)


def generate_training_masks(
    project_id: int,
    video_id: int,
    start_frame_idx: int,
    max_frames: int,
    project_path: Path,
    num_frames: int,
    height: int,
    width: int,
) -> int:
    """Generate training masks by propagating tracking forward and save to H5."""
    return SAM2Service.get_instance().generate_training_masks(
        project_id, video_id, start_frame_idx, max_frames, project_path, num_frames, height, width
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


async def segment_all_videos(
    project_id: int,
    project_path: Path,
    videos: list,
    bboxes: dict[int, np.ndarray],
) -> list[int]:
    """Start batch segmentation for all videos in a project."""
    return await SAM2Service.get_instance().segment_all_videos(
        project_id, project_path, videos, bboxes
    )


