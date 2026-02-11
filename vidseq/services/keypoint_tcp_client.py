"""
Keypoint Tracking Service with TCP IPC.

Manages a separate worker process for SAM2++ keypoint tracking inference.
Communicates with worker via TCP sockets.
"""

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
from sqlalchemy import delete, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from vidseq.models.keypoint_conditioning_frame import KeypointConditioningFrame
from vidseq.services.array_storage import keypoint_coords, keypoint_logits
from vidseq.services.database_manager import DatabaseManager
from vidseq.services.keypoint_tracking_config import (
    get_keypoint_tracking_port,
    is_keypoint_tracking_worker_running,
)


# ---------------------------------------------------------------------------
# Reuse SegmentationTCPClient (protocol-generic)
# ---------------------------------------------------------------------------

from vidseq.services.segmentation_tcp_client import SegmentationTCPClient as TCPClient


class KeypointTrackingStatus(str, Enum):
    NOT_LOADED = "not_loaded"
    LOADING_MODEL = "loading_model"
    READY = "ready"
    ERROR = "error"


@dataclass
class KeypointSessionInfo:
    """Info about a keypoint tracking session."""
    video_id: int
    num_frames: int
    height: int
    width: int


class KeypointTrackingService:
    """
    Singleton service for managing keypoint tracking inference.

    Manages a separate worker process to avoid CUDA/signal conflicts
    with FastAPI's event loop.
    """

    _instance: Optional["KeypointTrackingService"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "KeypointTrackingService":
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
        self._tcp_client: Optional[TCPClient] = None
        self._status = KeypointTrackingStatus.NOT_LOADED
        self._error_message: Optional[str] = None
        self._sessions: dict[tuple[int, int], KeypointSessionInfo] = {}

        self._initialized = True

    @classmethod
    def get_instance(cls) -> "KeypointTrackingService":
        return cls()

    @classmethod
    def reset_instance(cls) -> None:
        with cls._lock:
            if cls._instance is not None:
                cls._instance.shutdown()
                cls._instance = None

    def get_status(self) -> dict:
        if self._status == KeypointTrackingStatus.READY:
            if self._tcp_client is None:
                if not is_keypoint_tracking_worker_running():
                    self._status = KeypointTrackingStatus.NOT_LOADED
                    self._worker_process = None
                    self._sessions.clear()

        return {
            "status": self._status.value,
            "error": self._error_message,
        }

    def _load_model_in_background(self) -> None:
        try:
            result = self._tcp_client.send_command({"type": "load_model"}, timeout=600.0)
            if result.get("status") == "ready":
                self._status = KeypointTrackingStatus.READY
            else:
                self._status = KeypointTrackingStatus.ERROR
                self._error_message = result.get("error", "Unknown error loading model")
        except Exception as e:
            self._status = KeypointTrackingStatus.ERROR
            self._error_message = str(e)

    def _start_worker(self) -> None:
        # Check if worker is already running
        if is_keypoint_tracking_worker_running():
            port = get_keypoint_tracking_port()
            if port:
                self._tcp_client = TCPClient()
                try:
                    self._tcp_client.connect("localhost", port)
                    self._status = KeypointTrackingStatus.LOADING_MODEL
                    threading.Thread(target=self._load_model_in_background, daemon=True).start()
                    return
                except Exception as e:
                    print(f"[Keypoint Service] Failed to connect to existing worker: {e}")
                    self._tcp_client = None

        # Start new worker process
        kwargs = {}
        if sys.platform != 'win32':
            kwargs['start_new_session'] = True

        self._worker_process = subprocess.Popen(
            [sys.executable, "-m", "vidseq.services.keypoint_tcp_server"],
            **kwargs
        )

        # Wait for port file to appear
        timeout = 30.0
        start_time = time.time()
        port = None
        while time.time() - start_time < timeout:
            port = get_keypoint_tracking_port()
            if port is not None:
                break
            time.sleep(0.1)

        if port is None:
            self._status = KeypointTrackingStatus.ERROR
            self._error_message = "Failed to start keypoint tracking worker (timeout waiting for port file)"
            return

        self._tcp_client = TCPClient()
        try:
            self._tcp_client.connect("localhost", port, timeout=10.0)
        except Exception as e:
            self._status = KeypointTrackingStatus.ERROR
            self._error_message = f"Failed to connect to keypoint tracking worker: {e}"
            self._tcp_client = None
            return

        self._status = KeypointTrackingStatus.LOADING_MODEL
        threading.Thread(target=self._load_model_in_background, daemon=True).start()

    def start_loading_in_background(self) -> None:
        """Start loading keypoint tracking model in background."""
        if self._status == KeypointTrackingStatus.NOT_LOADED:
            self._start_worker()

    def _ensure_worker_ready(self) -> None:
        if self._status != KeypointTrackingStatus.READY:
            raise RuntimeError("Keypoint tracking model not loaded.")
        if self._tcp_client is None:
            if not is_keypoint_tracking_worker_running():
                raise RuntimeError("Keypoint tracking worker process not running.")
            port = get_keypoint_tracking_port()
            if port:
                self._tcp_client = TCPClient()
                self._tcp_client.connect("localhost", port)

    def _send_and_wait(self, cmd: dict, timeout: float = 120.0) -> dict:
        self._ensure_worker_ready()
        assert self._tcp_client is not None

        request_id = str(uuid.uuid4())
        cmd["request_id"] = request_id

        try:
            result = self._tcp_client.send_command(cmd, timeout=timeout)
            if result.get("type") == "status":
                status_str = result.get("status")
                if status_str == "loading_model":
                    self._status = KeypointTrackingStatus.LOADING_MODEL
                elif status_str == "ready":
                    self._status = KeypointTrackingStatus.READY
                elif status_str == "error":
                    self._status = KeypointTrackingStatus.ERROR
                    self._error_message = result.get("error")
            return result
        except Exception as e:
            if isinstance(e, (ConnectionError, TimeoutError)) or "Connection" in str(e):
                self._tcp_client = None
            raise RuntimeError(f"Failed to communicate with keypoint tracking worker: {e}") from e

    def _send_streaming(self, cmd: dict, timeout: float = 120.0) -> dict:
        """Send command and stream responses until final result."""
        self._ensure_worker_ready()
        assert self._tcp_client is not None

        request_id = str(uuid.uuid4())
        cmd["request_id"] = request_id

        try:
            final_result = None
            for response in self._tcp_client.send_command_streaming(cmd, timeout=timeout):
                resp_type = response.get("type", "")
                if resp_type == "progress":
                    frame_idx = response.get("frame_idx", 0)
                    print(f"[Keypoint Service] Progress: frame {frame_idx}")
                    continue
                final_result = response
            return final_result or {"status": "error", "error": "No response received"}
        except Exception as e:
            if isinstance(e, (ConnectionError, TimeoutError)) or "Connection" in str(e):
                self._tcp_client = None
            raise RuntimeError(f"Failed to communicate with keypoint tracking worker: {e}") from e

    # -------------------------------------------------------------------
    # Session management
    # -------------------------------------------------------------------

    def init_session(
        self,
        project_id: int,
        video_id: int,
        video_path: Path,
        project_path: Path,
        num_frames: int,
        height: int,
        width: int,
    ) -> KeypointSessionInfo:
        """Initialize keypoint tracking session for a video.

        Queries KeypointConditioningFrame records and reads H5 coords
        to reconstruct cond_frame_data for the worker.
        """
        session_key = (project_id, video_id)
        if session_key in self._sessions:
            return self._sessions[session_key]

        # Query existing conditioning frames from DB
        db_manager = DatabaseManager.get_instance()
        project_engine = db_manager.get_project_engine(project_path)
        cond_frame_data = []

        with Session(project_engine) as db_session:
            cond_frames = db_session.execute(
                select(KeypointConditioningFrame).where(
                    KeypointConditioningFrame.video_id == video_id
                )
            ).scalars().all()

            if cond_frames:
                # Read coords from H5 for each conditioning frame
                with keypoint_coords(project_path, video_id, "r") as coords_ds:
                    for cf in cond_frames:
                        coords = np.asarray(coords_ds[cf.frame_idx, cf.obj_id])  # (2,)
                        if not np.isnan(coords).any():
                            cond_frame_data.append({
                                "frame_idx": cf.frame_idx,
                                "obj_id": cf.obj_id,
                                "x_norm": float(coords[0]),
                                "y_norm": float(coords[1]),
                            })

        result = self._send_and_wait({
            "type": "init_keypoint_session",
            "video_id": video_id,
            "video_path": str(video_path),
            "project_path": str(project_path),
            "num_frames": num_frames,
            "height": height,
            "width": width,
            "cond_frame_data": cond_frame_data,
        }, timeout=600.0)

        if result.get("status") != "ok":
            raise RuntimeError(result.get("error", "Failed to init keypoint session"))

        session_info = KeypointSessionInfo(
            video_id=video_id,
            num_frames=num_frames,
            height=height,
            width=width,
        )
        self._sessions[session_key] = session_info
        return session_info

    def get_session(self, project_id: int, video_id: int) -> Optional[KeypointSessionInfo]:
        return self._sessions.get((project_id, video_id))

    def close_session(self, project_id: int, video_id: int) -> bool:
        session_key = (project_id, video_id)
        if session_key not in self._sessions:
            return False

        try:
            self._send_and_wait({
                "type": "close_keypoint_session",
                "video_id": video_id,
            }, timeout=10.0)
        except Exception:
            pass

        self._sessions.pop(session_key, None)
        return True

    # -------------------------------------------------------------------
    # Keypoint operations
    # -------------------------------------------------------------------

    def add_keypoint_prompt(
        self,
        project_id: int,
        video_id: int,
        frame_idx: int,
        obj_id: int,
        x_norm: float,
        y_norm: float,
        project_path: Path,
    ) -> dict:
        """Add a keypoint prompt. Returns all keypoints for the frame.

        Also upserts a KeypointConditioningFrame DB record on success.
        """
        session = self.get_session(project_id, video_id)
        if session is None:
            raise RuntimeError("No keypoint session exists. Initialize session first.")

        result = self._send_and_wait({
            "type": "add_keypoint_prompt",
            "video_id": video_id,
            "frame_idx": frame_idx,
            "obj_id": obj_id,
            "x_norm": x_norm,
            "y_norm": y_norm,
        }, timeout=120.0)

        if result.get("status") != "ok":
            raise RuntimeError(result.get("error", "Failed to add keypoint prompt"))

        # Upsert conditioning frame record in DB
        db_manager = DatabaseManager.get_instance()
        project_engine = db_manager.get_project_engine(project_path)
        with Session(project_engine) as db_session:
            stmt = sqlite_insert(KeypointConditioningFrame).values(
                video_id=video_id,
                frame_idx=frame_idx,
                obj_id=obj_id,
            ).on_conflict_do_nothing(
                index_elements=["video_id", "frame_idx", "obj_id"],
            )
            db_session.execute(stmt)
            db_session.commit()

        return result.get("keypoints", {})

    def refine_keypoint(
        self,
        project_id: int,
        video_id: int,
        frame_idx: int,
        obj_id: int,
        x_norm: float,
        y_norm: float,
    ) -> dict:
        """Refine an existing keypoint. Returns all keypoints for the frame."""
        session = self.get_session(project_id, video_id)
        if session is None:
            raise RuntimeError("No keypoint session exists. Initialize session first.")

        result = self._send_and_wait({
            "type": "refine_keypoint",
            "video_id": video_id,
            "frame_idx": frame_idx,
            "obj_id": obj_id,
            "x_norm": x_norm,
            "y_norm": y_norm,
        }, timeout=120.0)

        if result.get("status") != "ok":
            raise RuntimeError(result.get("error", "Failed to refine keypoint"))

        return result.get("keypoints", {})

    def propagate_keypoints(
        self,
        project_id: int,
        video_id: int,
        start_frame_idx: int,
        max_frames: int,
    ) -> dict:
        """Propagate keypoint tracking forward. Returns frames_processed count."""
        session = self.get_session(project_id, video_id)
        if session is None:
            raise RuntimeError("No keypoint session exists. Initialize session first.")

        result = self._send_streaming({
            "type": "propagate_keypoints",
            "video_id": video_id,
            "start_frame_idx": start_frame_idx,
            "max_frames": max_frames,
        }, timeout=3600.0)  # 1 hour for long videos

        if result.get("status") != "ok":
            raise RuntimeError(result.get("error", "Failed to propagate keypoints"))

        return result

    def reset_frame(
        self,
        project_id: int,
        video_id: int,
        frame_idx: int,
        project_path: Path,
    ) -> None:
        """Reset a single frame: clear H5 data, delete DB record, then clear SAM memory."""
        # 1. Zero out H5 coords (set to NaN) and logits (set to 0)
        with keypoint_coords(project_path, video_id, "a") as coords_ds:
            coords_ds[frame_idx] = np.full(coords_ds[frame_idx].shape, np.nan, dtype=np.float32)

        with keypoint_logits(project_path, video_id, "a") as logits_ds:
            logits_ds[frame_idx] = np.zeros(logits_ds[frame_idx].shape, dtype=np.float32)

        # 2. Delete KeypointConditioningFrame records for this frame
        db_manager = DatabaseManager.get_instance()
        project_engine = db_manager.get_project_engine(project_path)
        with Session(project_engine) as db_session:
            db_session.execute(
                delete(KeypointConditioningFrame).where(
                    KeypointConditioningFrame.video_id == video_id,
                    KeypointConditioningFrame.frame_idx == frame_idx,
                )
            )
            db_session.commit()

        # 3. Clear SAM memory (if session active)
        session = self.get_session(project_id, video_id)
        if session is not None:
            self._send_and_wait({
                "type": "reset_keypoint_frame",
                "video_id": video_id,
                "frame_idx": frame_idx,
            }, timeout=30.0)

    def reset_video(
        self,
        project_id: int,
        video_id: int,
        project_path: Path,
    ) -> None:
        """Reset all keypoints for a video: clear H5, delete DB records, clear SAM memory."""
        # 1. Zero out all H5 data
        with keypoint_coords(project_path, video_id, "a") as coords_ds:
            coords_ds[...] = np.nan

        with keypoint_logits(project_path, video_id, "a") as logits_ds:
            logits_ds[...] = 0.0

        # 2. Delete all KeypointConditioningFrame records
        db_manager = DatabaseManager.get_instance()
        project_engine = db_manager.get_project_engine(project_path)
        with Session(project_engine) as db_session:
            db_session.execute(
                delete(KeypointConditioningFrame).where(
                    KeypointConditioningFrame.video_id == video_id,
                )
            )
            db_session.commit()

        # 3. Clear SAM memory (if session active)
        session = self.get_session(project_id, video_id)
        if session is not None:
            self._send_and_wait({
                "type": "reset_keypoint_video",
                "video_id": video_id,
            }, timeout=30.0)

    # -------------------------------------------------------------------
    # Pure H5 reads (no TCP needed)
    # -------------------------------------------------------------------

    def get_keypoints(
        self,
        project_path: Path,
        video_id: int,
        frame_idx: int,
    ) -> dict:
        """Read keypoints for a single frame from H5.

        Returns dict with keys like {0: {x, y}, 1: {x, y}} for non-NaN keypoints.
        """
        with keypoint_coords(project_path, video_id, "r") as coords_ds:
            coords = np.asarray(coords_ds[frame_idx])  # (K, 2)
            result = {}
            for obj_id in range(coords.shape[0]):
                x, y = coords[obj_id]
                if not (np.isnan(x) or np.isnan(y)):
                    result[obj_id] = {"x": float(x), "y": float(y)}
            return result

    def get_keypoints_range(
        self,
        project_path: Path,
        video_id: int,
        start_frame: int,
        count: int,
    ) -> list[dict]:
        """Read keypoints for a range of frames from H5.

        Returns list of {frame_idx, 0: {x, y}, 1: {x, y}, ...} dicts.
        """
        with keypoint_coords(project_path, video_id, "r") as coords_ds:
            num_frames = coords_ds.shape[0]
            end_frame = min(start_frame + count, num_frames)
            results = []
            for frame_idx in range(start_frame, end_frame):
                coords = np.asarray(coords_ds[frame_idx])  # (K, 2)
                entry = {"frame_idx": frame_idx}
                for obj_id in range(coords.shape[0]):
                    x, y = coords[obj_id]
                    if not (np.isnan(x) or np.isnan(y)):
                        entry[str(obj_id)] = {"x": float(x), "y": float(y)}
                results.append(entry)
            return results

    def get_labeled_ranges(
        self,
        project_path: Path,
        video_id: int,
    ) -> dict:
        """Get labeled frame ranges and conditioning frame indices.

        Returns:
            {
                "labeled_ranges": [[start, end], ...],
                "conditioning_frame_indices": {"front": [...], "rear": [...]},
            }
        """
        # Read H5 to find non-NaN frames
        labeled_frames = set()
        with keypoint_coords(project_path, video_id, "r") as coords_ds:
            num_frames = coords_ds.shape[0]
            for frame_idx in range(num_frames):
                coords = np.asarray(coords_ds[frame_idx])  # (K, 2)
                if not np.isnan(coords).all():
                    labeled_frames.add(frame_idx)

        # Convert to ranges
        labeled_ranges = []
        if labeled_frames:
            sorted_frames = sorted(labeled_frames)
            start = sorted_frames[0]
            end = sorted_frames[0]
            for f in sorted_frames[1:]:
                if f == end + 1:
                    end = f
                else:
                    labeled_ranges.append([start, end])
                    start = f
                    end = f
            labeled_ranges.append([start, end])

        # Query conditioning frames from DB
        db_manager = DatabaseManager.get_instance()
        project_engine = db_manager.get_project_engine(project_path)
        cond_indices = {"front": [], "rear": []}

        with Session(project_engine) as db_session:
            cond_frames = db_session.execute(
                select(KeypointConditioningFrame).where(
                    KeypointConditioningFrame.video_id == video_id
                )
            ).scalars().all()

            for cf in cond_frames:
                if cf.obj_id == 0:
                    cond_indices["front"].append(cf.frame_idx)
                elif cf.obj_id == 1:
                    cond_indices["rear"].append(cf.frame_idx)

        cond_indices["front"].sort()
        cond_indices["rear"].sort()

        return {
            "labeled_ranges": labeled_ranges,
            "conditioning_frame_indices": cond_indices,
        }

    def shutdown(self) -> None:
        """Shutdown the worker process gracefully."""
        if self._sessions:
            raise RuntimeError(
                "Cannot shutdown keypoint tracker while sessions are active."
            )
        if self._tcp_client is not None and self._tcp_client.is_connected():
            try:
                self._tcp_client.send_command({"type": "shutdown"}, timeout=5.0)
            except Exception:
                pass
            self._tcp_client.disconnect()
            self._tcp_client = None

        self._worker_process = None
        self._status = KeypointTrackingStatus.NOT_LOADED
        self._sessions.clear()


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------


def get_status() -> dict:
    return KeypointTrackingService.get_instance().get_status()


def start_loading_in_background() -> None:
    KeypointTrackingService.get_instance().start_loading_in_background()


def init_session(
    project_id: int,
    video_id: int,
    video_path: Path,
    project_path: Path,
    num_frames: int,
    height: int,
    width: int,
) -> KeypointSessionInfo:
    return KeypointTrackingService.get_instance().init_session(
        project_id, video_id, video_path, project_path,
        num_frames, height, width,
    )


def get_session(project_id: int, video_id: int) -> Optional[KeypointSessionInfo]:
    return KeypointTrackingService.get_instance().get_session(project_id, video_id)


def close_session(project_id: int, video_id: int) -> bool:
    return KeypointTrackingService.get_instance().close_session(project_id, video_id)


def add_keypoint_prompt(
    project_id: int,
    video_id: int,
    frame_idx: int,
    obj_id: int,
    x_norm: float,
    y_norm: float,
    project_path: Path,
) -> dict:
    return KeypointTrackingService.get_instance().add_keypoint_prompt(
        project_id, video_id, frame_idx, obj_id, x_norm, y_norm, project_path,
    )


def refine_keypoint(
    project_id: int,
    video_id: int,
    frame_idx: int,
    obj_id: int,
    x_norm: float,
    y_norm: float,
) -> dict:
    return KeypointTrackingService.get_instance().refine_keypoint(
        project_id, video_id, frame_idx, obj_id, x_norm, y_norm,
    )


def propagate_keypoints(
    project_id: int,
    video_id: int,
    start_frame_idx: int,
    max_frames: int,
) -> dict:
    return KeypointTrackingService.get_instance().propagate_keypoints(
        project_id, video_id, start_frame_idx, max_frames,
    )


def reset_frame(
    project_id: int,
    video_id: int,
    frame_idx: int,
    project_path: Path,
) -> None:
    KeypointTrackingService.get_instance().reset_frame(
        project_id, video_id, frame_idx, project_path,
    )


def reset_video(
    project_id: int,
    video_id: int,
    project_path: Path,
) -> None:
    KeypointTrackingService.get_instance().reset_video(
        project_id, video_id, project_path,
    )


def get_keypoints(project_path: Path, video_id: int, frame_idx: int) -> dict:
    return KeypointTrackingService.get_instance().get_keypoints(
        project_path, video_id, frame_idx,
    )


def get_keypoints_range(
    project_path: Path, video_id: int, start_frame: int, count: int,
) -> list[dict]:
    return KeypointTrackingService.get_instance().get_keypoints_range(
        project_path, video_id, start_frame, count,
    )


def get_labeled_ranges(project_path: Path, video_id: int) -> dict:
    return KeypointTrackingService.get_instance().get_labeled_ranges(
        project_path, video_id,
    )


def shutdown_worker() -> None:
    KeypointTrackingService.get_instance().shutdown()
