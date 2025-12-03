"""Video metadata extraction service."""

from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass(frozen=True)
class VideoMetadata:
    """Immutable video metadata."""
    num_frames: int
    height: int
    width: int
    fps: float


class VideoMetadataError(Exception):
    """Error reading video metadata."""
    pass


def get_video_metadata(video_path: Path | str) -> VideoMetadata:
    """
    Extract metadata from a video file.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        VideoMetadata with num_frames, height, width, fps
        
    Raises:
        VideoMetadataError: If the video cannot be opened or metadata cannot be read
    """
    video_path = Path(video_path)
    
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise VideoMetadataError(f"Could not open video: {video_path}")
        
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if fps <= 0:
            raise VideoMetadataError(f"Could not read FPS from video: {video_path}")
        
        if num_frames <= 0:
            raise VideoMetadataError(f"Could not read frame count from video: {video_path}")
        
        if height <= 0 or width <= 0:
            raise VideoMetadataError(f"Could not read dimensions from video: {video_path}")
        
        return VideoMetadata(
            num_frames=num_frames,
            height=height,
            width=width,
            fps=fps,
        )
    finally:
        cap.release()

