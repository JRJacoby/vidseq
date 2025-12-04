"""Segmentation service - orchestrates SAM3 inference with mask storage."""

import io
from pathlib import Path

import numpy as np
from PIL import Image

from vidseq.models.video import Video
from vidseq.services import mask_service, sam3_service, video_service


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
    Clear (zero out) a mask for a specific frame and remove the object from SAM3.
    
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
    
    try:
        sam3_service.remove_object(video_id=video_id)
    except Exception:
        pass


def clear_video(
    project_path: Path,
    video_id: int,
) -> None:
    """
    Clear all masks for a video and remove the object from SAM3.
    
    Args:
        project_path: Path to the project folder
        video_id: Video ID
    """
    mask_service.clear_all_masks(
        project_path=project_path,
        video_id=video_id,
    )
    
    try:
        sam3_service.remove_object(video_id=video_id)
    except Exception:
        pass


def restore_conditioning_frames(
    project_path: Path,
    video: Video,
    frame_indices: list[int],
) -> int:
    """
    Restore SAM3's inference state by injecting masks from HDF5.
    
    This is called after session init to restore conditioning frames
    so SAM3 has the memory features needed for tracking/propagation.
    
    Args:
        project_path: Path to the project folder
        video: Video model instance
        frame_indices: List of frame indices with conditioning data
        
    Returns:
        Number of frames successfully restored
    """
    if not frame_indices:
        return 0
    
    video_path = Path(video.path)
    meta = video_service.get_video_metadata(video_path)
    
    restored_count = 0
    for frame_idx in sorted(frame_indices):
        mask = mask_service.load_mask(
            project_path=project_path,
            video_id=video.id,
            frame_idx=frame_idx,
            num_frames=meta.num_frames,
            height=meta.height,
            width=meta.width,
        )
        
        if mask.max() == 0:
            continue
        
        try:
            sam3_service.inject_mask(
                video_id=video.id,
                video_path=video_path,
                frame_idx=frame_idx,
                mask=mask,
                obj_id=0,
            )
            restored_count += 1
        except Exception as e:
            print(f"Warning: Failed to inject mask for frame {frame_idx}: {e}")
    
    return restored_count


def propagate_and_save(
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
    
    masks = sam3_service.propagate_forward(
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
