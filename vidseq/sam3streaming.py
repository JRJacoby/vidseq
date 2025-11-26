from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from numpy.typing import NDArray

class LazyVideoFrameLoader:
    """
    Lazy frame loader that provides frames on-demand from a video file.
    
    Implements __getitem__ and __len__ to be compatible with SAM3's 
    inference_state["images"] access pattern.
    """
    
    IMAGE_SIZE = 1008
    IMG_MEAN = (0.5, 0.5, 0.5)
    IMG_STD = (0.5, 0.5, 0.5)
    
    def __init__(
        self,
        video_path: Path | str,
        device: torch.device | str = "cuda",
    ):
        """
        Initialize the lazy frame loader.
        
        Args:
            video_path: Path to the video file (.mp4)
            device: Torch device for loaded tensors
        """
        self.video_path = Path(video_path)
        self.device = torch.device(device) if isinstance(device, str) else device
        
        self._cap = cv2.VideoCapture(str(self.video_path))
        if not self._cap.isOpened():
            raise ValueError(f"Could not open video: {self.video_path}")
        
        self._num_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._video_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._video_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._fps = self._cap.get(cv2.CAP_PROP_FPS)
        
        self._img_mean = torch.tensor(self.IMG_MEAN, dtype=torch.float32)[:, None, None]
        self._img_std = torch.tensor(self.IMG_STD, dtype=torch.float32)[:, None, None]
    
    def __len__(self) -> int:
        return self._num_frames
    
    def __getitem__(self, frame_idx: int) -> torch.Tensor:
        """
        Load and preprocess a single frame.
        
        Returns:
            Tensor of shape (3, IMAGE_SIZE, IMAGE_SIZE), normalized and on device
        """
        if frame_idx < 0 or frame_idx >= self._num_frames:
            raise IndexError(f"Frame index {frame_idx} out of range [0, {self._num_frames})")
        return self._load_and_preprocess_frame(frame_idx)
    
    def _load_and_preprocess_frame(self, frame_idx: int) -> torch.Tensor:
        """Load a frame from video and preprocess for SAM3."""
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self._cap.read()
        if not ret:
            raise RuntimeError(f"Failed to read frame {frame_idx} from {self.video_path}")
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (self.IMAGE_SIZE, self.IMAGE_SIZE))
        
        img_tensor = torch.from_numpy(frame_resized).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1)  # HWC -> CHW
        
        img_tensor = (img_tensor - self._img_mean) / self._img_std
        
        return img_tensor.to(self.device, dtype=torch.float16)
    
    def get_raw_frame(self, frame_idx: int) -> NDArray[np.uint8]:
        """Get raw RGB frame without preprocessing (for visualization)."""
        if frame_idx < 0 or frame_idx >= self._num_frames:
            raise IndexError(f"Frame index {frame_idx} out of range [0, {self._num_frames})")
        
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self._cap.read()
        if not ret:
            raise RuntimeError(f"Failed to read frame {frame_idx} from {self.video_path}")
        
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    def close(self) -> None:
        """Release video capture resources."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
    
    def __del__(self):
        self.close()