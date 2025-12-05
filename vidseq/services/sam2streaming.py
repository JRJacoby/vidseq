from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray

from vidseq.services.video_service import VideoMetadata, get_video_metadata


def _preprocess_on_gpu(
    frame_tensor: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    target_size: int,
) -> torch.Tensor:
    x = frame_tensor.permute(2, 0, 1).unsqueeze(0)
    x = F.interpolate(x, size=(target_size, target_size), mode='bilinear', align_corners=False)
    return x.squeeze(0) * scale + shift


class LazyVideoFrameLoader:
    """
    Lazy frame loader that provides frames on-demand from a video file.
    
    Implements __getitem__ and __len__ to be compatible with SAM2's 
    inference_state["images"] access pattern.
    """
    
    IMAGE_SIZE = 1024
    IMG_MEAN = (0.485, 0.456, 0.406)
    IMG_STD = (0.229, 0.224, 0.225)
    
    def __init__(
        self,
        video_path: Path | str,
        offload_to_cpu: bool = False,
        device: torch.device | str = "cuda",
    ):
        """
        Initialize the lazy frame loader.
        
        Args:
            video_path: Path to the video file (.mp4)
            offload_to_cpu: If True, keep tensors on CPU (saves GPU memory)
            device: Torch device for loaded tensors when not offloading
        """
        self.video_path = Path(video_path)
        self.offload_to_cpu = offload_to_cpu
        self.device = torch.device(device) if isinstance(device, str) else device
        
        self._metadata = get_video_metadata(self.video_path)
        
        self._cap = cv2.VideoCapture(str(self.video_path))
        if not self._cap.isOpened():
            raise ValueError(f"Could not open video: {self.video_path}")
        self._current_pos = 0
        
        img_mean = torch.tensor(self.IMG_MEAN, dtype=torch.float32)
        img_std = torch.tensor(self.IMG_STD, dtype=torch.float32)
        self._scale = (1.0 / (255.0 * img_std))[:, None, None]
        self._shift = (-img_mean / img_std)[:, None, None]
        
        if not self.offload_to_cpu:
            self._scale = self._scale.to(self.device)
            self._shift = self._shift.to(self.device)
    
    @property
    def metadata(self) -> VideoMetadata:
        return self._metadata
    
    @property
    def _num_frames(self) -> int:
        return self._metadata.num_frames
    
    @property
    def _video_height(self) -> int:
        return self._metadata.height
    
    @property
    def _video_width(self) -> int:
        return self._metadata.width
    
    @property
    def _fps(self) -> float:
        return self._metadata.fps
    
    def __len__(self) -> int:
        return self._metadata.num_frames
    
    def __getitem__(self, frame_idx: int | torch.Tensor) -> torch.Tensor:
        """
        Load and preprocess frame(s).
        
        SAM2 accesses frames as: inference_state["images"][frame_idx].to(device).float().unsqueeze(0)
        So we return a tensor of shape (3, IMAGE_SIZE, IMAGE_SIZE) that can be .to(), .float(), .unsqueeze()
        
        Args:
            frame_idx: Single int or tensor of indices
            
        Returns:
            Single frame: (3, IMAGE_SIZE, IMAGE_SIZE)
            Multiple frames: (N, 3, IMAGE_SIZE, IMAGE_SIZE)
        """
        if isinstance(frame_idx, torch.Tensor):
            if frame_idx.numel() == 1:
                return self._load_and_preprocess_frame(frame_idx.item())
            return torch.stack([self._load_and_preprocess_frame(i) for i in frame_idx.tolist()])
        
        if frame_idx < 0 or frame_idx >= self._metadata.num_frames:
            raise IndexError(f"Frame index {frame_idx} out of range [0, {self._metadata.num_frames})")
        return self._load_and_preprocess_frame(frame_idx)
    
    def _load_and_preprocess_frame(self, frame_idx: int) -> torch.Tensor:
        """Load a frame from video and preprocess for SAM2."""
        if self.offload_to_cpu:
            return self._load_and_preprocess_cpu(frame_idx)
        
        frame_rgb = self._read_frame(frame_idx)
        frame_tensor = torch.from_numpy(frame_rgb).to(self.device, dtype=torch.float32)
        return _preprocess_on_gpu(frame_tensor, self._scale, self._shift, self.IMAGE_SIZE)
    
    def _read_frame(self, frame_idx: int) -> np.ndarray:
        """Read and decode a frame from the video."""
        if self._current_pos != frame_idx:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self._cap.read()
        self._current_pos = frame_idx + 1
        if not ret:
            raise RuntimeError(f"Failed to read frame {frame_idx} from {self.video_path}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    def _load_and_preprocess_cpu(self, frame_idx: int) -> torch.Tensor:
        """CPU-only preprocessing path."""
        frame_rgb = self._read_frame(frame_idx)
        frame_resized = cv2.resize(frame_rgb, (self.IMAGE_SIZE, self.IMAGE_SIZE))
        img_tensor = torch.from_numpy(frame_resized).float()
        img_tensor = img_tensor.permute(2, 0, 1)
        return img_tensor * self._scale + self._shift
    
    def get_raw_frame(self, frame_idx: int) -> NDArray[np.uint8]:
        """Get raw RGB frame without preprocessing (for visualization)."""
        if frame_idx < 0 or frame_idx >= self._metadata.num_frames:
            raise IndexError(f"Frame index {frame_idx} out of range [0, {self._metadata.num_frames})")
        
        return self._read_frame(frame_idx)
    
    def close(self) -> None:
        """Release video capture resources."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
    
    def __del__(self):
        self.close()


