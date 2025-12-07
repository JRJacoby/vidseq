"""Segmentation service - orchestrates SAM2 inference with mask storage."""

import base64
import io
from pathlib import Path

import numpy as np
from PIL import Image

from vidseq.models.video import Video
from vidseq.services import mask_service, sam2_service


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
    mask = mask_service.load_mask(
        project_path=project_path,
        video_id=video.id,
        frame_idx=frame_idx,
        num_frames=video.num_frames,
        height=video.height,
        width=video.width,
    )
    
    return mask_to_png(mask)


def get_masks_batch_json(
    project_path: Path,
    video: Video,
    start_frame: int,
    count: int,
) -> list[dict]:
    """
    Load multiple masks and return as list of dicts with base64-encoded PNGs.
    
    Args:
        project_path: Path to the project folder
        video: Video model instance
        start_frame: Starting frame index
        count: Number of frames to load
        
    Returns:
        List of {"frame_idx": int, "png_base64": str}
    """
    masks = mask_service.load_masks_batch(
        project_path=project_path,
        video_id=video.id,
        start_frame=start_frame,
        count=count,
        num_frames=video.num_frames,
        height=video.height,
        width=video.width,
    )
    
    result = []
    for i, mask in enumerate(masks):
        png_bytes = mask_to_png(mask)
        png_base64 = base64.b64encode(png_bytes).decode('ascii')
        result.append({
            "frame_idx": start_frame + i,
            "png_base64": png_base64,
        })
    
    return result


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
    mask_service.clear_all_frame_types(
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
    masks = sam2_service.propagate_forward(
        video_id=video.id,
        start_frame_idx=start_frame_idx,
        max_frames=max_frames,
    )
    
    from vidseq.services.mask_service import open_h5
    
    with open_h5(project_path, 'a') as h5_file:
        for frame_idx, mask, bbox in masks:
            mask_service.save_mask(
                project_path=project_path,
                video_id=video.id,
                frame_idx=frame_idx,
                mask=mask,
                num_frames=video.num_frames,
                height=video.height,
                width=video.width,
                h5_file=h5_file,
            )
            if bbox is not None:
                mask_service.save_bbox(
                    project_path=project_path,
                    video_id=video.id,
                    frame_idx=frame_idx,
                    bbox=np.array(bbox, dtype=np.float32),
                    num_frames=video.num_frames,
                    h5_file=h5_file,
                )
            mask_service.mark_frame_type(
                project_path=project_path,
                video_id=video.id,
                frame_idx=frame_idx,
                frame_type='train',
                num_frames=video.num_frames,
                h5_file=h5_file,
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
    masks = sam2_service.propagate_backward(
        video_id=video.id,
        start_frame_idx=start_frame_idx,
        max_frames=max_frames,
    )
    
    from vidseq.services.mask_service import open_h5
    
    with open_h5(project_path, 'a') as h5_file:
        for frame_idx, mask, bbox in masks:
            mask_service.save_mask(
                project_path=project_path,
                video_id=video.id,
                frame_idx=frame_idx,
                mask=mask,
                num_frames=video.num_frames,
                height=video.height,
                width=video.width,
                h5_file=h5_file,
            )
            if bbox is not None:
                mask_service.save_bbox(
                    project_path=project_path,
                    video_id=video.id,
                    frame_idx=frame_idx,
                    bbox=np.array(bbox, dtype=np.float32),
                    num_frames=video.num_frames,
                    h5_file=h5_file,
                )
            mask_service.mark_frame_type(
                project_path=project_path,
                video_id=video.id,
                frame_idx=frame_idx,
                frame_type='train',
                num_frames=video.num_frames,
                h5_file=h5_file,
            )
    
    return len(masks)
