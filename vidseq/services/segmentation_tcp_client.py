"""
Segmentation Service with TCP IPC.

Manages a separate worker process for segmentation inference to avoid
CUDA/signal conflicts with FastAPI's event loop.

Uses detector+tracker video model with point prompts and lazy frame loading.
Communicates with worker via TCP sockets.
"""

import base64
import json
import socket
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

from vidseq.services.segmentation_config import (
    get_segmentation_port,
    is_segmentation_worker_running,
)


# ---------------------------------------------------------------------------
# TCP Client for worker communication
# ---------------------------------------------------------------------------


class SegmentationTCPClient:
    """TCP client for segmentation worker communication."""

    def __init__(self):
        self.socket: Optional[socket.socket] = None
        self.host: Optional[str] = None
        self.port: Optional[int] = None

    def connect(self, host: str, port: int, timeout: float = 10.0) -> None:
        """Connect to segmentation worker server."""
        if self.socket is not None:
            self.disconnect()

        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(timeout)
            self.socket.connect((host, port))
            self.host = host
            self.port = port
        except Exception as e:
            self.socket = None
            raise ConnectionError(f"Failed to connect to segmentation worker at {host}:{port}: {e}")

    def disconnect(self) -> None:
        """Close connection to server."""
        if self.socket is not None:
            try:
                self.socket.close()
            except Exception:
                pass
            self.socket = None
            self.host = None
            self.port = None

    def send_command(self, cmd: dict, timeout: float = 120.0) -> dict:
        """Send command to server and wait for response."""
        if self.socket is None:
            raise ConnectionError("Not connected to segmentation worker")

        old_timeout = self.socket.gettimeout()
        self.socket.settimeout(timeout)

        try:
            cmd_json = json.dumps(cmd)
            cmd_bytes = cmd_json.encode('utf-8')

            length_prefix = struct.pack('>I', len(cmd_bytes))
            self.socket.sendall(length_prefix)
            self.socket.sendall(cmd_bytes)

            length_bytes = self._recv_exact(4)
            if len(length_bytes) != 4:
                raise ConnectionError("Connection closed by server")

            response_length = struct.unpack('>I', length_bytes)[0]
            response_bytes = self._recv_exact(response_length)
            response_json = response_bytes.decode('utf-8')
            return json.loads(response_json)

        except socket.timeout:
            raise TimeoutError(f"Timeout waiting for response to {cmd.get('type', 'unknown')}")
        except Exception as e:
            if isinstance(e, (ConnectionError, TimeoutError)):
                raise
            raise ConnectionError(f"Error communicating with segmentation worker: {e}")
        finally:
            self.socket.settimeout(old_timeout)

    def send_command_streaming(self, cmd: dict, timeout: float = 120.0):
        """Send command to server and yield multiple responses."""
        if self.socket is None:
            raise ConnectionError("Not connected to segmentation worker")

        old_timeout = self.socket.gettimeout()
        self.socket.settimeout(timeout)

        try:
            cmd_json = json.dumps(cmd)
            cmd_bytes = cmd_json.encode('utf-8')

            length_prefix = struct.pack('>I', len(cmd_bytes))
            self.socket.sendall(length_prefix)
            self.socket.sendall(cmd_bytes)

            while True:
                length_bytes = self._recv_exact(4)
                if len(length_bytes) != 4:
                    raise ConnectionError("Connection closed by server")

                response_length = struct.unpack('>I', length_bytes)[0]
                response_bytes = self._recv_exact(response_length)
                response_json = response_bytes.decode('utf-8')
                response = json.loads(response_json)

                yield response

                cmd_type = cmd.get("type", "")
                resp_type = response.get("type", "")
                if resp_type == f"{cmd_type}_result" or resp_type == "error":
                    break

        except socket.timeout:
            raise TimeoutError(f"Timeout waiting for response to {cmd.get('type', 'unknown')}")
        except Exception as e:
            if isinstance(e, (ConnectionError, TimeoutError)):
                raise
            raise ConnectionError(f"Error communicating with segmentation worker: {e}")
        finally:
            self.socket.settimeout(old_timeout)

    def _recv_exact(self, n: int) -> bytes:
        """Receive exactly n bytes from socket."""
        data = b''
        while len(data) < n:
            chunk = self.socket.recv(n - len(data))
            if not chunk:
                raise ConnectionError("Connection closed by server")
            data += chunk
        return data

    def is_connected(self) -> bool:
        """Check if client is connected."""
        return self.socket is not None


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


class SegmentationStatus(str, Enum):
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


class SegmentationService:
    """
    Singleton service for managing segmentation inference.

    Manages a separate worker process to avoid CUDA/signal conflicts
    with FastAPI's event loop.
    """

    _instance: Optional["SegmentationService"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "SegmentationService":
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
        self._tcp_client: Optional[SegmentationTCPClient] = None
        self._status = SegmentationStatus.NOT_LOADED
        self._error_message: Optional[str] = None
        # Sessions keyed by (project_id, video_id) to avoid collisions across projects
        self._sessions: dict[tuple[int, int], VideoSessionInfo] = {}

        self._initialized = True

    @classmethod
    def get_instance(cls) -> "SegmentationService":
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
        """Get the current segmentation loading status."""
        # Only check if worker is running if we have a client connection
        # This avoids creating new connections on every status check
        if self._status == SegmentationStatus.READY:
            # If we have a client, assume it's still connected unless we get an error
            # Only check worker if we don't have a client
            if self._tcp_client is None:
                if not is_segmentation_worker_running():
                    self._status = SegmentationStatus.NOT_LOADED
                    self._worker_process = None
                    self._sessions.clear()

        return {
            "status": self._status.value,
            "error": self._error_message,
        }

    def _load_model_in_background(self) -> None:
        """Send load_model command and update status when done. Runs in a thread."""
        try:
            result = self._tcp_client.send_command({"type": "load_model"}, timeout=600.0)
            if result.get("status") == "ready":
                self._status = SegmentationStatus.READY
            else:
                self._status = SegmentationStatus.ERROR
                self._error_message = result.get("error", "Unknown error loading model")
        except Exception as e:
            self._status = SegmentationStatus.ERROR
            self._error_message = str(e)

    def _start_worker(self) -> None:
        """Start the segmentation worker process and begin loading the model."""
        # Check if worker is already running
        if is_segmentation_worker_running():
            port = get_segmentation_port()
            if port:
                self._tcp_client = SegmentationTCPClient()
                try:
                    self._tcp_client.connect("localhost", port)
                    # Load model in background thread
                    self._status = SegmentationStatus.LOADING_MODEL
                    threading.Thread(target=self._load_model_in_background, daemon=True).start()
                    return
                except Exception as e:
                    print(f"[Segmentation Service] Failed to connect to existing worker: {e}")
                    self._tcp_client = None

        # Start new worker process
        # Use -m format for more robust module resolution
        kwargs = {}
        if sys.platform != 'win32':
            kwargs['start_new_session'] = True

        self._worker_process = subprocess.Popen(
            [sys.executable, "-m", "vidseq.services.segmentation_tcp_server"],
            **kwargs
        )

        # Wait for port file to appear (poll with timeout)
        timeout = 30.0
        start_time = time.time()
        port = None

        while time.time() - start_time < timeout:
            port = get_segmentation_port()
            if port is not None:
                break
            time.sleep(0.1)

        if port is None:
            self._status = SegmentationStatus.ERROR
            self._error_message = "Failed to start segmentation worker (timeout waiting for port file)"
            return

        # Connect to worker
        self._tcp_client = SegmentationTCPClient()
        try:
            self._tcp_client.connect("localhost", port, timeout=10.0)
        except Exception as e:
            self._status = SegmentationStatus.ERROR
            self._error_message = f"Failed to connect to segmentation worker: {e}"
            self._tcp_client = None
            return

        # Load model in background thread
        self._status = SegmentationStatus.LOADING_MODEL
        threading.Thread(target=self._load_model_in_background, daemon=True).start()

    def start_loading_in_background(self) -> None:
        """Start loading segmentation model in background (via worker process).

        Raises:
            RuntimeError: If detector training is in progress.
        """
        from vidseq.services.detector_service import DetectorService
        if DetectorService.get_instance().is_training():
            raise RuntimeError("Cannot load SAM2 while detector training is in progress")
        if self._status == SegmentationStatus.NOT_LOADED:
            self._start_worker()

    def _ensure_worker_ready(self) -> None:
        """Ensure worker is running and model is loaded."""
        if self._status != SegmentationStatus.READY:
            raise RuntimeError("segmentation model not loaded. Call preload first.")

        # Only create client if it doesn't exist
        # Don't check is_connected() here - let send_command() handle connection errors
        if self._tcp_client is None:
            if not is_segmentation_worker_running():
                raise RuntimeError("segmentation worker process not running.")
            port = get_segmentation_port()
            if port:
                self._tcp_client = SegmentationTCPClient()
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
                    self._status = SegmentationStatus.LOADING_MODEL
                elif status_str == "ready":
                    self._status = SegmentationStatus.READY
                elif status_str == "error":
                    self._status = SegmentationStatus.ERROR
                    self._error_message = result.get("error")

            return result
        except Exception as e:
            # If connection error, mark client as disconnected
            if isinstance(e, (ConnectionError, TimeoutError)) or "Connection" in str(e):
                self._tcp_client = None
                # Don't change status - might be temporary connection issue
            raise RuntimeError(f"Failed to communicate with segmentation worker: {e}") from e

    def _send_streaming(self, cmd: dict, timeout: float = 120.0) -> dict:
        """Send a command and stream responses until final result.

        Use this for commands that send progress callbacks.
        """
        self._ensure_worker_ready()
        assert self._tcp_client is not None

        request_id = str(uuid.uuid4())
        cmd["request_id"] = request_id

        try:
            final_result = None
            for response in self._tcp_client.send_command_streaming(cmd, timeout=timeout):
                resp_type = response.get("type", "")
                # Progress messages - just log them
                if resp_type == "progress":
                    frame_idx = response.get("frame_idx", 0)
                    total = response.get("total", 0)
                    print(f"[Segmentation Service] Progress: {frame_idx}/{total}")
                    continue
                # Final result
                final_result = response
            return final_result or {"status": "error", "error": "No response received"}
        except Exception as e:
            if isinstance(e, (ConnectionError, TimeoutError)) or "Connection" in str(e):
                self._tcp_client = None
            raise RuntimeError(f"Failed to communicate with segmentation worker: {e}") from e

    def init_session(
        self,
        project_id: int,
        video_id: int,
        video_path: Path,
        project_path: Path,
        num_frames: int,
        height: int,
        width: int,
        cond_frame_indices: list[int],
    ) -> VideoSessionInfo:
        """Initialize a segmentation session for a video.

        Args:
            project_id: ID of the project
            video_id: ID of the video
            video_path: Path to the video file
            project_path: Path to the project folder
            num_frames: Total number of frames in the video
            height: Video height in pixels
            width: Video width in pixels
            cond_frame_indices: List of frame indices that have conditioning data
        """
        session_key = (project_id, video_id)
        if session_key in self._sessions:
            return self._sessions[session_key]

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
        frame_idx: int,
        x: float,
        y: float,
        label: int,
    ) -> tuple[np.ndarray, float]:
        """
        Add a point prompt and return the mask and confidence score.

        Requires an active session (call init_session first).

        Args:
            project_id: ID of the project
            video_id: ID of the video
            frame_idx: Frame index to segment
            x: X coordinate in normalized [0, 1] coords
            y: Y coordinate in normalized [0, 1] coords
            label: Label (1=positive, 0=negative)

        Returns:
            Tuple of (mask, score) where mask is numpy array (height, width)
            and score is the predicted IoU confidence.

        Raises:
            RuntimeError: If no session exists for this video
        """
        session = self.get_session(project_id, video_id)
        if session is None:
            raise RuntimeError("No session exists. Initialize session first.")

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
        score = result.get("score", -1.0)

        return mask, score

    def add_box_prompt(
        self,
        project_id: int,
        video_id: int,
        frame_idx: int,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
    ) -> tuple[np.ndarray, float]:
        """
        Add a bounding box prompt and return the mask and confidence score.

        Requires an active session (call init_session first).

        Args:
            project_id: ID of the project
            video_id: ID of the video
            frame_idx: Frame index to segment
            x1, y1: Top-left corner in normalized [0, 1] coords
            x2, y2: Bottom-right corner in normalized [0, 1] coords

        Returns:
            Tuple of (mask, score) where mask is numpy array (height, width)
            and score is the predicted IoU confidence.
        """
        session = self.get_session(project_id, video_id)
        if session is None:
            raise RuntimeError("No session exists. Initialize session first.")

        result = self._send_and_wait({
            "type": "add_box_prompt",
            "video_id": video_id,
            "frame_idx": frame_idx,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
        }, timeout=120.0)

        if result.get("status") != "ok":
            raise RuntimeError(result.get("error", "Failed to add box prompt"))

        session.has_object = True

        mask_rle = result["mask_rle"]
        mask_shape = tuple(result["mask_shape"])
        mask_dtype = result.get("mask_dtype", "uint8")
        mask = _decode_mask_rle(mask_rle, mask_shape, mask_dtype)
        score = result.get("score", -1.0)

        return mask, score

    def refine_mask(
        self,
        project_id: int,
        video_id: int,
        frame_idx: int,
        points: list[dict],
        labels: list[int],
    ) -> tuple[np.ndarray, float]:
        """
        Refine an existing mask with point prompt(s).

        Args:
            project_id: ID of the project
            video_id: ID of the video
            frame_idx: Frame index to refine (must have existing mask)
            points: List of points, each {x: float, y: float} in normalized [0, 1] coords
            labels: List of labels (1=positive, 0=negative), one per point

        Returns:
            Tuple of (mask, score) where mask is numpy array (height, width)
            and score is the predicted IoU confidence.
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
        score = result.get("score", -1.0)

        return mask, score

    def propagate(
        self,
        project_id: int,
        video_id: int,
        frame_idx: int,
    ) -> tuple[np.ndarray, float]:
        """
        Propagate tracking to a single frame and return the mask and score.

        Args:
            project_id: ID of the project
            video_id: ID of the video
            frame_idx: Frame index to propagate to

        Returns:
            Tuple of (mask, score) where mask is numpy array (height, width)
            and score is the predicted IoU confidence.
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
        score = result.get("score", -1.0)

        return mask, score

    def reset_frame(
        self,
        project_id: int,
        video_id: int,
        frame_idx: int,
    ) -> None:
        """
        Reset a single frame's SAM state.

        Clears SAM memory for the frame. Does not modify database.
        The caller is responsible for removing conditioning frame records.

        Args:
            project_id: ID of the project
            video_id: ID of the video
            frame_idx: Frame index to reset
        """
        self.reset_frame_memory(project_id, video_id, frame_idx)

    def reset_frame_memory(
        self,
        project_id: int,
        video_id: int,
        frame_idx: int,
    ) -> None:
        """
        Clear SAM memory for a single frame (no database or H5 changes).

        This only sends a command to the TCP worker to clear the frame from
        SAM's in-memory state (cond_frame_outputs, output_dict, etc).
        Does nothing if no active session exists.

        Args:
            project_id: ID of the project
            video_id: ID of the video
            frame_idx: Frame index to clear from memory
        """
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
        Reset video's SAM state.

        Clears SAM memory for the video. Does not modify database.
        The caller is responsible for removing conditioning frame records.

        Args:
            project_id: ID of the project
            video_id: ID of the video
            project_path: Path to the project folder (needed by worker)
        """
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
    ) -> tuple[list[int], list[list]]:
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
            Tuple of (frame_indices, scores) where frame_indices is a list of
            propagated frame indices and scores is a list of [frame_idx, score].

        Raises:
            RuntimeError: If no object has been tracked
        """
        session = self.get_session(project_id, video_id)
        if session is None:
            raise RuntimeError("No session exists. Add a point prompt first.")

        if not session.has_object:
            raise RuntimeError("No object tracked. Add a point prompt first.")

        result = self._send_streaming({
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

        frame_indices = result.get("frame_indices", [])
        scores = result.get("scores", [])
        return frame_indices, scores

    def apply_detector(
        self,
        project_path: Path,
        videos: list,
        detector_type: str = "detector",
    ) -> tuple[dict[int, list[list]], dict[int, list[list]]]:
        """Run trained detector on all frames of given videos.

        Args:
            project_path: Path to the project folder
            videos: List of Video objects with .id, .path, .num_frames
            detector_type: "detector" for axis-aligned boxes, "obb" for oriented boxes

        Returns:
            Tuple of (scores_by_video, bboxes_by_video) where each is
            {video_id: [[frame_idx, ...], ...]}
        """
        result = self._send_streaming({
            "type": "apply_detector",
            "video_ids": [v.id for v in videos],
            "video_paths": [v.path for v in videos],
            "project_path": str(project_path),
            "detector_type": detector_type,
        }, timeout=3600.0)

        if result.get("status") != "ok":
            raise RuntimeError(result.get("error", "Failed to apply detector"))

        score_key = "obb_scores" if detector_type == "obb" else "detector_scores"
        bbox_key = "obb_bboxes" if detector_type == "obb" else "detector_bboxes"
        raw_scores = result.get(score_key, {})
        raw_bboxes = result.get(bbox_key, {})

        scores_by_video = {int(k): v for k, v in raw_scores.items()}
        bboxes_by_video = {int(k): v for k, v in raw_bboxes.items()}
        return scores_by_video, bboxes_by_video

    def propagate_with_associated(
        self,
        video_id: int,
        main_video_id: int,
        project_path: Path,
        num_frames: int,
        confidence_threshold: float,
        main_height: int,
        main_width: int,
        main_video_scores: dict[int, float],
        cond_frame_interval: int = 50,
        max_cond_frames: int = 32,
    ) -> dict:
        """Run associated video propagation via TCP."""
        return self._send_streaming({
            "type": "propagate_with_associated",
            "video_id": video_id,
            "main_video_id": main_video_id,
            "project_path": str(project_path),
            "num_frames": num_frames,
            "confidence_threshold": confidence_threshold,
            "main_height": main_height,
            "main_width": main_width,
            "main_video_scores": main_video_scores,
            "cond_frame_interval": cond_frame_interval,
            "max_cond_frames": max_cond_frames,
        }, timeout=3600.0)  # 1 hour for long videos

    async def segment_all_videos(
        self,
        project_id: int,
        project_path: Path,
        videos: list,
        cond_frames_by_video: dict[int, list[int]] | None = None,
        training_frames_by_video: dict[int, list[int]] | None = None,
        mode: str = "full",
    ) -> tuple[list[int], dict[int, list[list]], dict[int, list[list]], dict[int, list[list]]]:
        """
        Start batch segmentation for all videos in a project using detector-tracker approach.

        For each video:
        1. Initialize session
        2. Run propagate_with_detector command (or simple_propagate_with_detector in simple mode)
        3. Close session

        Args:
            mode: "full" for full propagate_with_detector, "simple" for simple forward-only mode.

        Returns:
            Tuple of (job_ids, scores_by_video, detector_scores_by_video,
            detector_bboxes_by_video).
        """
        if cond_frames_by_video is None:
            cond_frames_by_video = {}
        if not videos:
            return [], {}, {}, {}

        scores_by_video: dict[int, list[list]] = {}
        detector_scores_by_video: dict[int, list[list]] = {}
        detector_bboxes_by_video: dict[int, list[list]] = {}

        # Ensure model is loaded
        if self._status == SegmentationStatus.NOT_LOADED:
            self.start_loading_in_background()

        # Wait if still loading
        timeout = 600.0
        start_time = time.time()
        while self._status == SegmentationStatus.LOADING_MODEL and time.time() - start_time < timeout:
            time.sleep(1.0)

        self._ensure_worker_ready()

        # Process each video sequentially
        for video in videos:
            print(f"[Segmentation Service] Segmenting video {video.id} ({video.name})...")

            # Get conditioning frames from the dict passed by service layer
            cond_frame_indices = cond_frames_by_video.get(video.id, [])

            # Initialize session with conditioning frames
            try:
                result = self._send_and_wait({
                    "type": "init_session",
                    "video_id": video.id,
                    "video_path": video.path,
                    "project_path": str(project_path),
                    "num_frames": video.num_frames,
                    "height": video.height,
                    "width": video.width,
                    "cond_frame_indices": cond_frame_indices,
                }, timeout=600.0)

                if result.get("status") != "ok":
                    print(f"[Segmentation Service] Failed to init session for video {video.id}: {result.get('error')}")
                    continue

                # Run propagation (uses streaming for progress callbacks)
                if mode == "simple":
                    result = self._send_streaming({
                        "type": "simple_propagate_with_detector",
                        "video_id": video.id,
                        "project_path": str(project_path),
                        "num_frames": video.num_frames,
                        "reprompt_interval": int(round(video.fps)),
                    }, timeout=3600.0)
                else:
                    result = self._send_streaming({
                        "type": "propagate_with_detector",
                        "video_id": video.id,
                        "project_path": str(project_path),
                        "num_frames": video.num_frames,
                        "iou_threshold": 0.7,
                        "training_frame_indices": (training_frames_by_video or {}).get(video.id, []),
                    }, timeout=3600.0)  # 1 hour timeout for long videos

                if result.get("status") == "ok":
                    scores_by_video[video.id] = result.get("scores", [])
                    detector_scores_by_video[video.id] = result.get("detector_scores", [])
                    detector_bboxes_by_video[video.id] = result.get("detector_bboxes", [])
                    frames_corrected = result.get("frames_corrected", 0)
                    print(f"[Segmentation Service] Video {video.id} complete: "
                          f"{video.num_frames} frames, {frames_corrected} corrected by detector")
                else:
                    print(f"[Segmentation Service] Failed to segment video {video.id}: {result.get('error')}")

                # Close session
                self._send_and_wait({
                    "type": "close_session",
                    "video_id": video.id,
                }, timeout=30.0)

            except Exception as e:
                print(f"[Segmentation Service] Error segmenting video {video.id}: {e}")
                # Try to close session on error
                try:
                    self._send_and_wait({
                        "type": "close_session",
                        "video_id": video.id,
                    }, timeout=10.0)
                except Exception:
                    pass

        return [], scores_by_video, detector_scores_by_video, detector_bboxes_by_video

    def shutdown(self) -> None:
        """Shutdown the worker process gracefully.

        Raises:
            RuntimeError: If any video sessions are currently active.
        """
        if self._sessions:
            raise RuntimeError(
                "Cannot shutdown SAM2 while sessions are active. "
                "Wait for segmentation to finish first."
            )
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
        self._status = SegmentationStatus.NOT_LOADED
        self._sessions.clear()


def get_status() -> dict:
    """Get the current segmentation loading status."""
    return SegmentationService.get_instance().get_status()


def start_loading_in_background() -> None:
    """Start loading segmentation model in background (via worker process)."""
    SegmentationService.get_instance().start_loading_in_background()


def init_session(
    project_id: int,
    video_id: int,
    video_path: Path,
    project_path: Path,
    num_frames: int,
    height: int,
    width: int,
    cond_frame_indices: list[int],
) -> VideoSessionInfo:
    """Initialize a segmentation session for a video."""
    return SegmentationService.get_instance().init_session(
        project_id, video_id, video_path, project_path,
        num_frames, height, width, cond_frame_indices,
    )


def get_session(project_id: int, video_id: int) -> Optional[VideoSessionInfo]:
    """Get session info if it exists."""
    return SegmentationService.get_instance().get_session(project_id, video_id)


def close_session(project_id: int, video_id: int) -> bool:
    """Close a video session."""
    return SegmentationService.get_instance().close_session(project_id, video_id)


def add_point_prompt(
    project_id: int,
    video_id: int,
    frame_idx: int,
    x: float,
    y: float,
    label: int,
) -> tuple[np.ndarray, float]:
    """Add a point prompt and return the mask and confidence score."""
    return SegmentationService.get_instance().add_point_prompt(
        project_id, video_id, frame_idx, x, y, label
    )


def add_box_prompt(
    project_id: int,
    video_id: int,
    frame_idx: int,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> tuple[np.ndarray, float]:
    """Add a box prompt and return the mask and confidence score."""
    return SegmentationService.get_instance().add_box_prompt(
        project_id, video_id, frame_idx, x1, y1, x2, y2
    )


def refine_mask(
    project_id: int,
    video_id: int,
    frame_idx: int,
    points: list[dict],
    labels: list[int],
) -> tuple[np.ndarray, float]:
    """Refine an existing mask with point prompt(s) and return mask and score."""
    return SegmentationService.get_instance().refine_mask(
        project_id, video_id, frame_idx, points, labels
    )


def propagate(
    project_id: int,
    video_id: int,
    frame_idx: int,
) -> tuple[np.ndarray, float]:
    """Propagate tracking to a single frame and return mask and score."""
    return SegmentationService.get_instance().propagate(project_id, video_id, frame_idx)


def reset_frame(
    project_id: int,
    video_id: int,
    frame_idx: int,
) -> None:
    """
    Reset a single frame's SAM state.

    Clears SAM memory for the frame. Does not modify database.
    The caller is responsible for removing conditioning frame records.

    Args:
        project_id: ID of the project
        video_id: ID of the video
        frame_idx: Frame index to reset
    """
    SegmentationService.get_instance().reset_frame(project_id, video_id, frame_idx)


def reset_frame_memory(
    project_id: int,
    video_id: int,
    frame_idx: int,
) -> None:
    """
    Clear SAM memory for a single frame (no database or H5 changes).

    This only clears the frame from SAM's in-memory state.
    Does nothing if no active session exists.

    Args:
        project_id: ID of the project
        video_id: ID of the video
        frame_idx: Frame index to clear from memory
    """
    SegmentationService.get_instance().reset_frame_memory(project_id, video_id, frame_idx)


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
    SegmentationService.get_instance().reset_video(project_id, video_id, project_path)


def generate_training_masks(
    project_id: int,
    video_id: int,
    start_frame_idx: int,
    max_frames: int,
    project_path: Path,
    num_frames: int,
    height: int,
    width: int,
) -> tuple[list[int], list[list]]:
    """Generate training masks and return frame indices and scores."""
    return SegmentationService.get_instance().generate_training_masks(
        project_id, video_id, start_frame_idx, max_frames, project_path, num_frames, height, width
    )


def apply_detector(
    project_path: Path,
    videos: list,
) -> tuple[dict[int, list[list]], dict[int, list[list]]]:
    """Run trained detector on all frames of given videos."""
    return SegmentationService.get_instance().apply_detector(project_path, videos)


def apply_obb_detector(
    project_path: Path,
    videos: list,
) -> tuple[dict[int, list[list]], dict[int, list[list]]]:
    """Run OBB detector on all frames of given videos."""
    return SegmentationService.get_instance().apply_detector(project_path, videos, detector_type="obb")


def apply_seg_detector(
    project_path: Path,
    videos: list,
) -> tuple[dict[int, list[list]], dict[int, list[list]]]:
    """Run seg detector on all frames of given videos."""
    return SegmentationService.get_instance().apply_detector(project_path, videos, detector_type="seg")


def shutdown_worker() -> None:
    """Shutdown the worker process gracefully."""
    SegmentationService.get_instance().shutdown()


async def segment_all_videos(
    project_id: int,
    project_path: Path,
    videos: list,
    cond_frames_by_video: dict[int, list[int]] | None = None,
    training_frames_by_video: dict[int, list[int]] | None = None,
    mode: str = "full",
) -> tuple[list[int], dict[int, list[list]], dict[int, list[list]], dict[int, list[list]]]:
    """Start batch segmentation for all videos using detector-tracker approach."""
    return await SegmentationService.get_instance().segment_all_videos(
        project_id, project_path, videos, cond_frames_by_video, training_frames_by_video, mode
    )


def propagate_with_associated(
    video_id: int,
    main_video_id: int,
    project_path: Path,
    num_frames: int,
    confidence_threshold: float,
    main_height: int,
    main_width: int,
    main_video_scores: dict[int, float],
    cond_frame_interval: int = 50,
    max_cond_frames: int = 32,
) -> dict:
    """Module-level wrapper for associated video propagation."""
    return SegmentationService.get_instance().propagate_with_associated(
        video_id=video_id,
        main_video_id=main_video_id,
        project_path=project_path,
        num_frames=num_frames,
        confidence_threshold=confidence_threshold,
        main_height=main_height,
        main_width=main_width,
        main_video_scores=main_video_scores,
        cond_frame_interval=cond_frame_interval,
        max_cond_frames=max_cond_frames,
    )
