"""Segmentation service - orchestrates SAM3 inference with mask storage."""

import base64
import io
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from sqlalchemy.ext.asyncio import AsyncSession

from vidseq.models.video import Video
from vidseq.services import mask_storage


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
    mask = mask_storage.load_mask(
        project_path=project_path,
        video_id=video.id,
        frame_idx=frame_idx,
        num_frames=video.num_frames,
        height=video.height,
        width=video.width,
    )

    # DEBUG: Log mask loading
    print(f"[DEBUG get_mask_png] project={project_path}, video={video.id}, frame={frame_idx}")
    print(f"[DEBUG get_mask_png] mask sum={int(mask.sum())}, shape={mask.shape}, "
          f"min={mask.min()}, max={mask.max()}")

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
    masks = mask_storage.load_masks_batch(
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


async def clear_video(
    project_path: Path,
    video_id: int,
    session: AsyncSession,
    project_id: Optional[int] = None,
    video_path: Optional[Path] = None,
) -> None:
    """
    Clear all masks and frame data for a video.

    Args:
        project_path: Path to the project folder
        video_id: Video ID
        session: Database session for clearing frame data
        project_id: Project ID (required for SAM3 reset)
        video_path: Path to the video file (unused, kept for API compatibility)
    """
    from vidseq.services import frame_data_service, sam3_service

    # Reset via SAM3 service (handles all h5 file operations)
    if project_id is not None:
        sam3_service.reset_video(project_id, video_id, project_path)

    # Clear bboxes, frame_types, and scores from SQLite
    await frame_data_service.clear_all_frame_data(session, video_id)


