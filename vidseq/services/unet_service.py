"""
UNet Service with Multiprocessing.

Manages a separate worker process for UNet training and inference to avoid
CUDA/signal conflicts with FastAPI's event loop.
"""

import multiprocessing as mp
import queue
import threading
import time
import uuid
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Optional


class UNetService:
    """
    Singleton service for managing UNet training and inference.
    
    Manages a separate worker process to avoid CUDA/signal conflicts
    with FastAPI's event loop.
    """
    
    _instance: Optional["UNetService"] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> "UNetService":
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
        self._is_training = False
        self._is_applying = False
        self._pending_requests: dict[str, dict | None] = {}
        
        self._initialized = True
    
    @classmethod
    def get_instance(cls) -> "UNetService":
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
            
            if result_type == "train_model_result":
                self._is_training = False
            elif result_type == "apply_model_result":
                self._is_applying = False
            elif result_type == "test_apply_model_result":
                self._is_applying = False
            
            request_id = result.get("request_id")
            if request_id and request_id in self._pending_requests:
                self._pending_requests[request_id] = result
    
    def _start_worker(self) -> None:
        """Start the UNet worker process."""
        if self._worker_process is not None and self._worker_process.is_alive():
            return
        
        ctx = mp.get_context("spawn")
        self._command_queue = ctx.Queue()
        self._result_queue = ctx.Queue()
        
        from vidseq.services.unet_worker import worker_loop
        
        self._worker_process = ctx.Process(
            target=worker_loop,
            args=(self._command_queue, self._result_queue),
            daemon=True,
        )
        self._worker_process.start()
    
    def _ensure_worker_ready(self) -> None:
        """Ensure worker is running."""
        if self._worker_process is None or not self._worker_process.is_alive():
            self._start_worker()
    
    def _send_and_wait(self, cmd: dict, timeout: float = 3600.0) -> dict:
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
            
            time.sleep(0.1)
        
        del self._pending_requests[request_id]
        raise TimeoutError(f"Timeout waiting for response to {cmd['type']}")
    
    def train_model(self, project_path: Path) -> None:
        """
        Start training UNet model for a project.
        
        Args:
            project_path: Path to the project folder
            
        Raises:
            RuntimeError: If training is already in progress
        """
        if self._is_training:
            raise RuntimeError("Training already in progress")
        
        # Close HDF5 file in main process so worker can open it
        from vidseq.services import mask_service
        mask_service.close_project_h5(project_path)
        
        self._is_training = True
        self._ensure_worker_ready()
        
        self._command_queue.put({
            "type": "train_model",
            "project_path": str(project_path),
        })
    
    def apply_model(self, project_path: Path, video_id: int) -> None:
        """
        Start applying UNet model to a video.
        
        Args:
            project_path: Path to the project folder
            video_id: ID of the video to apply model to
            
        Raises:
            RuntimeError: If application is already in progress
        """
        if self._is_applying:
            raise RuntimeError("Application already in progress")
        
        # Close HDF5 file in main process so worker can open it
        from vidseq.services import mask_service
        mask_service.close_project_h5(project_path)
        
        self._is_applying = True
        self._ensure_worker_ready()
        
        self._command_queue.put({
            "type": "apply_model",
            "project_path": str(project_path),
            "video_id": video_id,
        })
    
    def test_apply_model(self, project_path: Path, video_id: int, start_frame: int) -> None:
        """
        Start test applying UNet model to a limited range of frames.
        
        Args:
            project_path: Path to the project folder
            video_id: ID of the video to apply model to
            start_frame: Starting frame index
            
        Raises:
            RuntimeError: If application is already in progress
        """
        if self._is_applying:
            raise RuntimeError("Application already in progress")
        
        # Close HDF5 file in main process so worker can open it
        from vidseq.services import mask_service
        mask_service.close_project_h5(project_path)
        
        self._is_applying = True
        self._ensure_worker_ready()
        
        self._command_queue.put({
            "type": "test_apply_model",
            "project_path": str(project_path),
            "video_id": video_id,
            "start_frame": start_frame,
        })
    
    def is_training(self) -> bool:
        """Check if training is in progress."""
        self._drain_result_queue()
        return self._is_training
    
    def is_applying(self) -> bool:
        """Check if application is in progress."""
        self._drain_result_queue()
        return self._is_applying
    
    def get_model_path(self, project_path: Path) -> Path:
        """Get path to the trained model file."""
        return project_path / "segmentation_student_unet.pth"
    
    def model_exists(self, project_path: Path) -> bool:
        """Check if model file exists."""
        return self.get_model_path(project_path).exists()
    
    def shutdown(self) -> None:
        """Shutdown the worker process."""
        if self._worker_process is not None and self._worker_process.is_alive():
            try:
                self._command_queue.put({"type": "shutdown"})
                self._worker_process.join(timeout=5.0)
            except Exception:
                pass
            finally:
                if self._worker_process.is_alive():
                    self._worker_process.terminate()
                    self._worker_process.join(timeout=2.0)
                if self._worker_process.is_alive():
                    self._worker_process.kill()
                self._worker_process = None
        
        self._command_queue = None
        self._result_queue = None
        self._is_training = False
        self._is_applying = False

