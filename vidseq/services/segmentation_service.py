"""Segmentation service - orchestrates SAM3 inference with mask storage."""

import base64
import io
from pathlib import Path

import numpy as np
from PIL import Image

from vidseq.models.video import Video
from vidseq.services import h5_storage, segmentation_tcp_client


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
    mask = h5_storage.load_mask(
        project_path=project_path,
        video_id=video.id,
        frame_idx=frame_idx,
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
    masks = h5_storage.load_masks_batch(
        project_path=project_path,
        video_id=video.id,
        start_frame=start_frame,
        count=count,
        num_frames=video.num_frames,
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


def get_final_mask_png(
    project_path: Path,
    video: Video,
    frame_idx: int,
) -> bytes:
    """
    Load and return final (corrected) mask as PNG bytes.

    Args:
        project_path: Path to the project folder
        video: Video model instance
        frame_idx: Frame index

    Returns:
        PNG bytes of the mask (zeros if no mask exists)
    """
    mask = h5_storage.load_final_mask(
        project_path=project_path,
        video_id=video.id,
        frame_idx=frame_idx,
    )
    return mask_to_png(mask)


def get_final_masks_batch_json(
    project_path: Path,
    video: Video,
    start_frame: int,
    count: int,
) -> list[dict]:
    """
    Load multiple final (corrected) masks and return as list of dicts with base64-encoded PNGs.

    Args:
        project_path: Path to the project folder
        video: Video model instance
        start_frame: Starting frame index
        count: Number of frames to load

    Returns:
        List of {"frame_idx": int, "png_base64": str}
    """
    masks = h5_storage.load_final_masks_batch(
        project_path=project_path,
        video_id=video.id,
        start_frame=start_frame,
        count=count,
        num_frames=video.num_frames,
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


def reset_frame_memory(project_id: int, video_id: int, frame_idx: int) -> None:
    """Clear SAM memory for a single frame.

    Only clears the in-memory state on the GPU worker. Does not touch
    H5 files or database. Does nothing if no active session exists.

    Args:
        project_id: ID of the project
        video_id: ID of the video
        frame_idx: Frame index to clear from memory
    """
    segmentation_tcp_client.reset_frame_memory(project_id, video_id, frame_idx)

