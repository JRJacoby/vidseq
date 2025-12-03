"""Segmentation service - orchestrates SAM3 inference with mask storage."""

import io
from pathlib import Path

import numpy as np
from PIL import Image

from vidseq.models.prompt import Prompt
from vidseq.models.video import Video
from vidseq.services import mask_service, sam3_service, video_service


def mask_to_png(mask: np.ndarray) -> bytes:
    """Convert a numpy mask to PNG bytes."""
    img = Image.fromarray(mask)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def run_segmentation_and_save(
    project_path: Path,
    video: Video,
    frame_idx: int,
    prompts: list[Prompt],
) -> bytes:
    """
    Run SAM3 segmentation with prompts and save the mask.
    
    Args:
        project_path: Path to the project folder
        video: Video model instance
        frame_idx: Frame index to segment
        prompts: List of prompts for this frame
        
    Returns:
        PNG bytes of the resulting mask
    """
    video_path = Path(video.path)
    meta = video_service.get_video_metadata(video_path)
    
    if not prompts:
        mask = mask_service.load_mask(
            project_path=project_path,
            video_id=video.id,
            frame_idx=frame_idx,
            num_frames=meta.num_frames,
            height=meta.height,
            width=meta.width,
        )
        return mask_to_png(mask)
    
    bbox_prompt = next((p for p in prompts if p.type == "bbox"), None)
    if bbox_prompt is None:
        mask = mask_service.load_mask(
            project_path=project_path,
            video_id=video.id,
            frame_idx=frame_idx,
            num_frames=meta.num_frames,
            height=meta.height,
            width=meta.width,
        )
        return mask_to_png(mask)
    
    mask = sam3_service.add_bbox_prompt(
        video_id=video.id,
        video_path=video_path,
        frame_idx=frame_idx,
        bbox=bbox_prompt.details,
    )
    
    mask_service.save_mask(
        project_path=project_path,
        video_id=video.id,
        frame_idx=frame_idx,
        mask=mask,
        num_frames=meta.num_frames,
        height=meta.height,
        width=meta.width,
    )
    
    return mask_to_png(mask)


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

