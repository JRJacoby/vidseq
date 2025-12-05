"""Segmentation service - orchestrates SAM2 inference with mask storage."""

import io
from pathlib import Path

import numpy as np
from PIL import Image

from vidseq.models.video import Video
from vidseq.services import mask_service, sam2_service, video_service


def mask_to_png(mask: np.ndarray) -> bytes:
    """Convert a numpy mask to PNG bytes."""
    img = Image.fromarray(mask)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def get_mask_png(
    project_path: Path,
    video: Video,
    frame_idx: int,
) -> bytes:
    """
    Load and return mask as PNG bytes.
    
    Args:
        project_path: Path to the project folder
        video: Video model instance
        frame_idx: Frame index
        
    Returns:
        PNG bytes of the mask (zeros if no mask exists)
    """
    video_path = Path(video.path)
    meta = video_service.get_video_metadata(video_path)
    
    mask = mask_service.load_mask(
        project_path=project_path,
        video_id=video.id,
        frame_idx=frame_idx,
        num_frames=meta.num_frames,
        height=meta.height,
        width=meta.width,
    )
    
    return mask_to_png(mask)


def clear_mask(
    project_path: Path,
    video_id: int,
    frame_idx: int,
) -> None:
    """
    Clear (zero out) a mask for a specific frame.
    
    Does not reset SAM2 tracking state.
    
    Args:
        project_path: Path to the project folder
        video_id: Video ID
        frame_idx: Frame index
    """
    mask_service.clear_mask(
        project_path=project_path,
        video_id=video_id,
        frame_idx=frame_idx,
    )


def clear_video(
    project_path: Path,
    video_id: int,
) -> None:
    """
    Clear all masks for a video.
    
    SAM2 tracking state reset is handled separately via sam2_service.reset_state().
    
    Args:
        project_path: Path to the project folder
        video_id: Video ID
    """
    mask_service.clear_all_masks(
        project_path=project_path,
        video_id=video_id,
    )


def propagate_forward_and_save(
    project_path: Path,
    video: Video,
    start_frame_idx: int,
    max_frames: int,
) -> int:
    """
    Propagate tracking forward and save all resulting masks.
    
    Args:
        project_path: Path to the project folder
        video: Video model instance
        start_frame_idx: Frame index to start from
        max_frames: Maximum number of frames to propagate
        
    Returns:
        Number of frames processed
    """
    video_path = Path(video.path)
    meta = video_service.get_video_metadata(video_path)
    
    masks = sam2_service.propagate_forward(
        video_id=video.id,
        start_frame_idx=start_frame_idx,
        max_frames=max_frames,
    )
    
    for frame_idx, mask in masks:
        mask_service.save_mask(
            project_path=project_path,
            video_id=video.id,
            frame_idx=frame_idx,
            mask=mask,
            num_frames=meta.num_frames,
            height=meta.height,
            width=meta.width,
        )
    
    return len(masks)


def propagate_backward_and_save(
    project_path: Path,
    video: Video,
    start_frame_idx: int,
    max_frames: int,
) -> int:
    """
    Propagate tracking backward and save all resulting masks.
    
    Args:
        project_path: Path to the project folder
        video: Video model instance
        start_frame_idx: Frame index to start from
        max_frames: Maximum number of frames to propagate
        
    Returns:
        Number of frames processed
    """
    video_path = Path(video.path)
    meta = video_service.get_video_metadata(video_path)
    
    masks = sam2_service.propagate_backward(
        video_id=video.id,
        start_frame_idx=start_frame_idx,
        max_frames=max_frames,
    )
    
    for frame_idx, mask in masks:
        mask_service.save_mask(
            project_path=project_path,
            video_id=video.id,
            frame_idx=frame_idx,
            mask=mask,
            num_frames=meta.num_frames,
            height=meta.height,
            width=meta.width,
        )
    
    return len(masks)
