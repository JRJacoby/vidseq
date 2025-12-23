"""Segmentation service - orchestrates SAM2 inference with mask storage."""

import base64
import io
from pathlib import Path

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
    mask_storage.clear_mask(
        project_path=project_path,
        video_id=video_id,
        frame_idx=frame_idx,
    )


async def clear_video(
    project_path: Path,
    video_id: int,
    session: AsyncSession,
) -> None:
    """
    Clear all masks and frame data for a video.

    SAM2 tracking state reset is handled separately via sam2_service.reset_state().

    Args:
        project_path: Path to the project folder
        video_id: Video ID
        session: Database session for clearing frame data
    """
    from vidseq.services import frame_data_service

    # Clear masks from per-video HDF5
    mask_storage.clear_all_masks(
        project_path=project_path,
        video_id=video_id,
    )

    # Clear bboxes, frame_types, and scores from SQLite
    await frame_data_service.clear_all_frame_data(session, video_id)


