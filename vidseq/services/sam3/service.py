"""
SAM3 Service with TCP IPC.

Manages a separate worker process for SAM3 inference to avoid
CUDA/signal conflicts with FastAPI's event loop.

Uses SAM3's detector+tracker video model with point prompts and lazy frame loading.
Communicates with worker via TCP sockets.
"""

import base64
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

from vidseq.services.sam3.config import (
    get_sam3_port,
    is_sam3_worker_running,
)
from vidseq.services.sam3.client import SAM3TCPClient


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


class SAM3Status(str, Enum):
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

        self._worker_process: Optional[subprocess.Popen] = None
        self._tcp_client: Optional[SAM3TCPClient] = None
        self._status = SAM3Status.NOT_LOADED
        self._error_message: Optional[str] = None
        # Sessions keyed by (project_id, video_id) to avoid collisions across projects
        self._sessions: dict[tuple[int, int], VideoSessionInfo] = {}

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

    def get_status(self) -> dict:
        """Get the current SAM3 loading status."""
        # Only check if worker is running if we have a client connection
        # This avoids creating new connections on every status check
        if self._status == SAM3Status.READY:
            # If we have a client, assume it's still connected unless we get an error
            # Only check worker if we don't have a client
            if self._tcp_client is None:
                if not is_sam3_worker_running():
                    self._status = SAM3Status.NOT_LOADED
                    self._worker_process = None
                    self._sessions.clear()

        return {
            "status": self._status.value,
            "error": self._error_message,
        }

    def _start_worker(self) -> None:
        """Start the SAM3 worker process and begin loading the model."""
        # Check if worker is already running
        if is_sam3_worker_running():
            port = get_sam3_port()
            if port:
                self._tcp_client = SAM3TCPClient()
                try:
                    self._tcp_client.connect("localhost", port)
                    # Still need to load model - worker may have been started manually
                    self._status = SAM3Status.LOADING_MODEL
                    result = self._tcp_client.send_command({"type": "load_model"}, timeout=600.0)
                    if result.get("status") == "ready":
                        self._status = SAM3Status.READY
                    else:
                        self._status = SAM3Status.ERROR
                        self._error_message = result.get("error", "Unknown error loading model")
                    return
                except Exception as e:
                    print(f"[SAM3 Service] Failed to connect to existing worker: {e}")
                    self._tcp_client = None

        # Start new worker process
        # Use -m format for more robust module resolution
        kwargs = {}
        if sys.platform != 'win32':
            kwargs['start_new_session'] = True

        self._worker_process = subprocess.Popen(
            [sys.executable, "-m", "vidseq.services.sam3.server.tcp_server"],
            **kwargs
        )

        # Wait for port file to appear (poll with timeout)
        timeout = 30.0
        start_time = time.time()
        port = None

        while time.time() - start_time < timeout:
            port = get_sam3_port()
            if port is not None:
                break
            time.sleep(0.1)

        if port is None:
            self._status = SAM3Status.ERROR
            self._error_message = "Failed to start SAM3 worker (timeout waiting for port file)"
            return

        # Connect to worker
        self._tcp_client = SAM3TCPClient()
        try:
            self._tcp_client.connect("localhost", port, timeout=10.0)
        except Exception as e:
            self._status = SAM3Status.ERROR
            self._error_message = f"Failed to connect to SAM3 worker: {e}"
            self._tcp_client = None
            return

        self._status = SAM3Status.LOADING_MODEL

        # Send load_model command
        try:
            result = self._tcp_client.send_command({"type": "load_model"}, timeout=600.0)
            if result.get("status") == "ready":
                self._status = SAM3Status.READY
            elif result.get("status") == "error":
                self._status = SAM3Status.ERROR
                self._error_message = result.get("error", "Unknown error")
        except Exception as e:
            self._status = SAM3Status.ERROR
            self._error_message = str(e)

    def start_loading_in_background(self) -> None:
        """Start loading SAM3 model in background (via worker process)."""
        # Don't start if YOLO is training (to avoid GPU memory conflicts)
        try:
            from vidseq.services import yolo_service
            yolo = yolo_service.YOLOService.get_instance()
            if yolo.is_training():
                return
        except Exception:
            pass

        if self._status == SAM3Status.NOT_LOADED:
            self._start_worker()

    def _ensure_worker_ready(self) -> None:
        """Ensure worker is running and model is loaded."""
        if self._status != SAM3Status.READY:
            raise RuntimeError("SAM3 model not loaded. Call preload first.")

        # Only create client if it doesn't exist
        # Don't check is_connected() here - let send_command() handle connection errors
        if self._tcp_client is None:
            if not is_sam3_worker_running():
                raise RuntimeError("SAM3 worker process not running.")
            port = get_sam3_port()
            if port:
                self._tcp_client = SAM3TCPClient()
                self._tcp_client.connect("localhost", port)

    def _send_and_wait(self, cmd: dict, timeout: float = 120.0) -> dict:
        """Send a command to worker and wait for the result."""
        self._ensure_worker_ready()
        assert self._tcp_client is not None  # Set by _ensure_worker_ready

        request_id = str(uuid.uuid4())
        cmd["request_id"] = request_id

        try:
            result = self._tcp_client.send_command(cmd, timeout=timeout)

            # Update status if we got a status update
            if result.get("type") == "status":
                status_str = result.get("status")
                if status_str == "loading_model":
                    self._status = SAM3Status.LOADING_MODEL
                elif status_str == "ready":
                    self._status = SAM3Status.READY
                elif status_str == "error":
                    self._status = SAM3Status.ERROR
                    self._error_message = result.get("error")

            return result
        except Exception as e:
            # If connection error, mark client as disconnected
            if isinstance(e, (ConnectionError, TimeoutError)) or "Connection" in str(e):
                self._tcp_client = None
                # Don't change status - might be temporary connection issue
            raise RuntimeError(f"Failed to communicate with SAM3 worker: {e}") from e

    def init_session(self, project_id: int, video_id: int, video_path: Path, project_path: Path) -> VideoSessionInfo:
        """Initialize a segmentation session for a video."""
        session_key = (project_id, video_id)
        if session_key in self._sessions:
            return self._sessions[session_key]

        # Query conditioning frames and video metadata from database
        from sqlalchemy import select
        from sqlalchemy.orm import Session
        from vidseq.models.conditioning_frame import ConditioningFrame
        from vidseq.models.video import Video
        from vidseq.services.database_manager import DatabaseManager

        db_manager = DatabaseManager.get_instance()
        project_engine = db_manager.get_project_engine(project_path)

        with Session(project_engine) as db_session:
            # Get video metadata
            video = db_session.execute(
                select(Video).where(Video.id == video_id)
            ).scalar_one()
            num_frames = video.num_frames
            height = video.height
            width = video.width

            # Get conditioning frames
            result = db_session.execute(
                select(ConditioningFrame.frame_idx)
                .where(ConditioningFrame.video_id == video_id)
            )
            cond_frame_indices = list(result.scalars().all())

        result = self._send_and_wait({
            "type": "init_session",
            "video_id": video_id,
            "video_path": str(video_path),
            "project_path": str(project_path),
            "num_frames": num_frames,
            "height": height,
            "width": width,
            "cond_frame_indices": cond_frame_indices,
        }, timeout=600.0)  # 10 min timeout for first session (torch.compile warmup)

        if result.get("status") != "ok":
            raise RuntimeError(result.get("error", "Failed to init session"))

        session_info = VideoSessionInfo(
            video_id=video_id,
            num_frames=num_frames,
            height=height,
            width=width,
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
        project_path: Path,
        frame_idx: int,
        x: float,
        y: float,
        label: int,
    ) -> np.ndarray:
        """
        Add a point prompt and return the mask.

        Args:
            project_id: ID of the project
            video_id: ID of the video
            video_path: Path to video file (used to init session if needed)
            project_path: Path to the project folder
            frame_idx: Frame index to segment
            x: X coordinate in normalized [0, 1] coords
            y: Y coordinate in normalized [0, 1] coords
            label: Label (1=positive, 0=negative)

        Returns:
            Binary mask as numpy array (height, width), dtype=uint8, values 0 or 255
        """
        session = self.get_session(project_id, video_id)
        if session is None:
            session = self.init_session(project_id, video_id, video_path, project_path)

        # Add conditioning frame record to database
        from sqlalchemy import select
        from sqlalchemy.orm import Session
        from vidseq.models.conditioning_frame import ConditioningFrame
        from vidseq.services.database_manager import DatabaseManager

        db_manager = DatabaseManager.get_instance()
        project_engine = db_manager.get_project_engine(project_path)

        with Session(project_engine) as db_session:
            # Check if already exists
            existing = db_session.execute(
                select(ConditioningFrame)
                .where(ConditioningFrame.video_id == video_id)
                .where(ConditioningFrame.frame_idx == frame_idx)
            ).scalar_one_or_none()

            if existing is None:
                db_session.add(ConditioningFrame(video_id=video_id, frame_idx=frame_idx))
                db_session.commit()

        result = self._send_and_wait({
            "type": "add_prompt",
            "video_id": video_id,
            "frame_idx": frame_idx,
            "x": x,
            "y": y,
            "label": label,
        }, timeout=120.0)

        if result.get("status") != "ok":
            raise RuntimeError(result.get("error", "Failed to add point prompt"))

        session.has_object = True

        mask_rle = result["mask_rle"]
        mask_shape = tuple(result["mask_shape"])
        mask_dtype = result.get("mask_dtype", "uint8")
        mask = _decode_mask_rle(mask_rle, mask_shape, mask_dtype)

        return mask

    def refine_mask(
        self,
        project_id: int,
        video_id: int,
        frame_idx: int,
        points: list[dict],
        labels: list[int],
    ) -> np.ndarray:
        """
        Refine an existing mask with point prompt(s).

        Args:
            project_id: ID of the project
            video_id: ID of the video
            frame_idx: Frame index to refine (must have existing mask)
            points: List of points, each {x: float, y: float} in normalized [0, 1] coords
            labels: List of labels (1=positive, 0=negative), one per point

        Returns:
            Refined binary mask as numpy array (height, width), dtype=uint8, values 0 or 255
        """
        session = self.get_session(project_id, video_id)
        if session is None:
            raise RuntimeError("No session exists. Initialize session first.")

        result = self._send_and_wait({
            "type": "refine_mask",
            "video_id": video_id,
            "frame_idx": frame_idx,
            "points": points,
            "labels": labels,
        }, timeout=120.0)

        if result.get("status") != "ok":
            raise RuntimeError(result.get("error", "Failed to refine mask"))

        mask_rle = result["mask_rle"]
        mask_shape = tuple(result["mask_shape"])
        mask_dtype = result.get("mask_dtype", "uint8")
        mask = _decode_mask_rle(mask_rle, mask_shape, mask_dtype)

        return mask

    def propagate(
        self,
        project_id: int,
        video_id: int,
        frame_idx: int,
    ) -> np.ndarray:
        """
        Propagate tracking to a single frame and return the mask.

        Args:
            project_id: ID of the project
            video_id: ID of the video
            frame_idx: Frame index to propagate to

        Returns:
            Binary mask as numpy array (height, width), dtype=uint8, values 0 or 255
        """
        session = self.get_session(project_id, video_id)
        if session is None:
            raise RuntimeError("No session exists. Initialize session first.")

        result = self._send_and_wait({
            "type": "propagate",
            "video_id": video_id,
            "frame_idx": frame_idx,
        }, timeout=120.0)

        if result.get("status") != "ok":
            raise RuntimeError(result.get("error", "Failed to propagate"))

        mask_rle = result["mask_rle"]
        mask_shape = tuple(result["mask_shape"])
        mask_dtype = result.get("mask_dtype", "uint8")
        mask = _decode_mask_rle(mask_rle, mask_shape, mask_dtype)

        return mask

    def reset_frame(
        self,
        project_id: int,
        video_id: int,
        project_path: Path,
        frame_idx: int,
    ) -> None:
        """
        Reset a single frame (clear mask and remove from conditioning frames).

        Args:
            project_id: ID of the project
            video_id: ID of the video
            project_path: Path to the project folder
            frame_idx: Frame index to reset
        """
        # Remove from database
        from sqlalchemy import delete
        from sqlalchemy.orm import Session
        from vidseq.models.conditioning_frame import ConditioningFrame
        from vidseq.services.database_manager import DatabaseManager

        db_manager = DatabaseManager.get_instance()
        project_engine = db_manager.get_project_engine(project_path)

        with Session(project_engine) as db_session:
            db_session.execute(
                delete(ConditioningFrame)
                .where(ConditioningFrame.video_id == video_id)
                .where(ConditioningFrame.frame_idx == frame_idx)
            )
            db_session.commit()

        # Tell worker to reset frame
        session = self.get_session(project_id, video_id)
        if session is not None:
            self._send_and_wait({
                "type": "reset_frame",
                "video_id": video_id,
                "frame_idx": frame_idx,
            }, timeout=30.0)

    def reset_video(
        self,
        project_id: int,
        video_id: int,
        project_path: Path,
    ) -> None:
        """
        Reset entire video (clear all masks and conditioning frames).

        Args:
            project_id: ID of the project
            video_id: ID of the video
            project_path: Path to the project folder
        """
        # Remove all conditioning frames from database
        from sqlalchemy import delete
        from sqlalchemy.orm import Session
        from vidseq.models.conditioning_frame import ConditioningFrame
        from vidseq.services.database_manager import DatabaseManager

        db_manager = DatabaseManager.get_instance()
        project_engine = db_manager.get_project_engine(project_path)

        with Session(project_engine) as db_session:
            db_session.execute(
                delete(ConditioningFrame)
                .where(ConditioningFrame.video_id == video_id)
            )
            db_session.commit()

        # Tell worker to reset video
        session = self.get_session(project_id, video_id)
        if session is not None:
            self._send_and_wait({
                "type": "reset_video",
                "video_id": video_id,
                "project_path": str(project_path),
            }, timeout=30.0)
            session.has_object = False

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
    ) -> list[int]:
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
            List of frame indices that were propagated

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

        return result.get("frame_indices", [])

    async def segment_all_videos(
        self,
        project_id: int,
        project_path: Path,
        videos: list,
    ) -> list[int]:
        """
        Start batch segmentation for all videos in a project using detector-tracker approach.

        For each video:
        1. Initialize session (which loads detector masks)
        2. Run propagate_with_detector command
        3. Close session

        Args:
            project_id: ID of the project
            project_path: Path to the project folder
            videos: List of Video model instances

        Returns:
            List of job IDs created (empty for now - synchronous execution)
        """
        if not videos:
            return []

        # Ensure model is loaded
        if self._status == SAM3Status.NOT_LOADED:
            self.start_loading_in_background()

        # Wait if still loading
        timeout = 600.0
        start_time = time.time()
        while self._status == SAM3Status.LOADING_MODEL and time.time() - start_time < timeout:
            time.sleep(1.0)

        self._ensure_worker_ready()

        # Process each video sequentially
        for video in videos:
            print(f"[SAM3 Service] Segmenting video {video.id} ({video.name})...")

            # Initialize session (this also loads detector masks)
            try:
                result = self._send_and_wait({
                    "type": "init_session",
                    "video_id": video.id,
                    "video_path": video.path,
                    "project_path": str(project_path),
                    "num_frames": video.num_frames,
                    "height": video.height,
                    "width": video.width,
                    "cond_frame_indices": [],  # Fresh session
                }, timeout=600.0)

                if result.get("status") != "ok":
                    print(f"[SAM3 Service] Failed to init session for video {video.id}: {result.get('error')}")
                    continue

                # Run propagate_with_detector
                result = self._send_and_wait({
                    "type": "propagate_with_detector",
                    "video_id": video.id,
                    "num_frames": video.num_frames,
                    "iou_threshold": 0.5,
                }, timeout=3600.0)  # 1 hour timeout for long videos

                if result.get("status") == "ok":
                    frames_corrected = result.get("frames_corrected", 0)
                    print(f"[SAM3 Service] Video {video.id} complete: "
                          f"{video.num_frames} frames, {frames_corrected} corrected by detector")
                else:
                    print(f"[SAM3 Service] Failed to segment video {video.id}: {result.get('error')}")

                # Close session
                self._send_and_wait({
                    "type": "close_session",
                    "video_id": video.id,
                }, timeout=30.0)

            except Exception as e:
                print(f"[SAM3 Service] Error segmenting video {video.id}: {e}")
                # Try to close session on error
                try:
                    self._send_and_wait({
                        "type": "close_session",
                        "video_id": video.id,
                    }, timeout=10.0)
                except Exception:
                    pass

        return []  # No job IDs - synchronous execution

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
        self._status = SAM3Status.NOT_LOADED
        self._sessions.clear()


def get_status() -> dict:
    """Get the current SAM3 loading status."""
    return SAM3Service.get_instance().get_status()


def start_loading_in_background() -> None:
    """Start loading SAM3 model in background (via worker process)."""
    SAM3Service.get_instance().start_loading_in_background()


def init_session(project_id: int, video_id: int, video_path: Path, project_path: Path) -> VideoSessionInfo:
    """Initialize a segmentation session for a video."""
    return SAM3Service.get_instance().init_session(project_id, video_id, video_path, project_path)


def get_session(project_id: int, video_id: int) -> Optional[VideoSessionInfo]:
    """Get session info if it exists."""
    return SAM3Service.get_instance().get_session(project_id, video_id)


def close_session(project_id: int, video_id: int) -> bool:
    """Close a video session."""
    return SAM3Service.get_instance().close_session(project_id, video_id)


def add_point_prompt(
    project_id: int,
    video_id: int,
    video_path: Path,
    project_path: Path,
    frame_idx: int,
    x: float,
    y: float,
    label: int,
) -> np.ndarray:
    """
    Add a point prompt and return the mask.

    Args:
        project_id: ID of the project
        video_id: ID of the video
        video_path: Path to video file (used to init session if needed)
        project_path: Path to the project folder
        frame_idx: Frame index to segment
        x: X coordinate in normalized [0, 1] coords
        y: Y coordinate in normalized [0, 1] coords
        label: Label (1=positive, 0=negative)

    Returns:
        Binary mask as numpy array (height, width), dtype=uint8, values 0 or 255
    """
    return SAM3Service.get_instance().add_point_prompt(
        project_id, video_id, video_path, project_path, frame_idx, x, y, label
    )


def refine_mask(
    project_id: int,
    video_id: int,
    frame_idx: int,
    points: list[dict],
    labels: list[int],
) -> np.ndarray:
    """
    Refine an existing mask with point prompt(s).

    Args:
        project_id: ID of the project
        video_id: ID of the video
        frame_idx: Frame index to refine (must have existing mask)
        points: List of points, each {x: float, y: float} in normalized [0, 1] coords
        labels: List of labels (1=positive, 0=negative), one per point

    Returns:
        Refined binary mask as numpy array (height, width), dtype=uint8, values 0 or 255
    """
    return SAM3Service.get_instance().refine_mask(
        project_id, video_id, frame_idx, points, labels
    )


def propagate(
    project_id: int,
    video_id: int,
    frame_idx: int,
) -> np.ndarray:
    """
    Propagate tracking to a single frame and return the mask.

    Args:
        project_id: ID of the project
        video_id: ID of the video
        frame_idx: Frame index to propagate to

    Returns:
        Binary mask as numpy array (height, width), dtype=uint8, values 0 or 255
    """
    return SAM3Service.get_instance().propagate(project_id, video_id, frame_idx)


def reset_frame(
    project_id: int,
    video_id: int,
    project_path: Path,
    frame_idx: int,
) -> None:
    """
    Reset a single frame (clear mask and remove from conditioning frames).

    Args:
        project_id: ID of the project
        video_id: ID of the video
        project_path: Path to the project folder
        frame_idx: Frame index to reset
    """
    SAM3Service.get_instance().reset_frame(project_id, video_id, project_path, frame_idx)


def reset_video(
    project_id: int,
    video_id: int,
    project_path: Path,
) -> None:
    """
    Reset entire video (clear all masks and conditioning frames).

    Args:
        project_id: ID of the project
        video_id: ID of the video
        project_path: Path to the project folder
    """
    SAM3Service.get_instance().reset_video(project_id, video_id, project_path)


def generate_training_masks(
    project_id: int,
    video_id: int,
    start_frame_idx: int,
    max_frames: int,
    project_path: Path,
    num_frames: int,
    height: int,
    width: int,
) -> list[int]:
    """Generate training masks by propagating tracking forward and save to H5."""
    return SAM3Service.get_instance().generate_training_masks(
        project_id, video_id, start_frame_idx, max_frames, project_path, num_frames, height, width
    )


def shutdown_worker() -> None:
    """Shutdown the worker process gracefully."""
    SAM3Service.get_instance().shutdown()


async def segment_all_videos(
    project_id: int,
    project_path: Path,
    videos: list,
) -> list[int]:
    """Start batch segmentation for all videos using detector-tracker approach."""
    return await SAM3Service.get_instance().segment_all_videos(
        project_id, project_path, videos
    )
