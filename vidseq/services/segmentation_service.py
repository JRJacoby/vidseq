"""Segmentation service - orchestrates SAM3 inference with mask storage."""

import base64
import io
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from vidseq.models.video import Video
from vidseq.services import frame_data_service, segmentation_tcp_client
from vidseq.services.array_storage import final_masks, tracker_masks
from vidseq.services.exceptions import MultiPointWithoutMaskError

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


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
    with tracker_masks(project_path, video.id, "r") as masks:
        mask = masks[frame_idx]

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
    with tracker_masks(project_path, video.id, "r") as masks_arr:
        end_frame = min(start_frame + count, video.num_frames)
        masks = np.array(masks_arr[start_frame:end_frame])

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
    with final_masks(project_path, video.id, "r") as masks:
        mask = masks[frame_idx]
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
    with final_masks(project_path, video.id, "r") as masks_arr:
        end_frame = min(start_frame + count, video.num_frames)
        masks = np.array(masks_arr[start_frame:end_frame])

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


async def submit_prompt(
    session: "AsyncSession",
    project_id: int,
    video_id: int,
    video_path: Path,
    project_path: Path,
    frame_idx: int,
    points: list[dict],
    labels: list[int],
) -> np.ndarray:
    """Submit point prompt(s) for segmentation.

    Workflow determined by existing state:
    - 1 point, no existing mask: creates new mask (add_point_prompt)
    - 1 point, existing mask: refines mask with single point
    - 2+ points, existing mask: refines mask with all points
    - 2+ points, no existing mask: ERROR (can't refine without mask)

    Args:
        session: Async database session
        project_id: ID of the project
        video_id: ID of the video
        video_path: Path to the video file
        project_path: Path to the project folder
        frame_idx: Frame index (0-based)
        points: List of {"x": float, "y": float} normalized coords
        labels: List of labels (1=positive, 0=negative)

    Returns:
        Resulting mask as numpy array

    Raises:
        MultiPointWithoutMaskError: If multi-point submitted without existing mask
        RuntimeError: If TCP client fails
    """
    # Check if this frame already has a mask
    has_existing_mask = await frame_data_service.get_has_tracker_mask(
        session, video_id, frame_idx
    )

    # Validate: multi-point requires existing mask
    if len(points) > 1 and not has_existing_mask:
        raise MultiPointWithoutMaskError()

    if has_existing_mask:
        # Refine existing mask using previous logits as dense prompt
        mask = segmentation_tcp_client.refine_mask(
            project_id=project_id,
            video_id=video_id,
            frame_idx=frame_idx,
            points=points,
            labels=labels,
        )
    else:
        # Create new mask on blank frame (single point only, validated above)
        p = points[0]
        label = labels[0]
        mask = segmentation_tcp_client.add_point_prompt(
            project_id=project_id,
            video_id=video_id,
            video_path=video_path,
            project_path=project_path,
            frame_idx=frame_idx,
            x=p["x"],
            y=p["y"],
            label=label,
        )

    # Update mask presence index
    has_content = bool(np.any(mask > 0))
    await frame_data_service.set_has_tracker_mask(
        session, video_id, frame_idx, has_content
    )

    # If refinement resulted in an empty mask, reset the frame's SAM state
    if has_existing_mask and not has_content:
        segmentation_tcp_client.reset_frame(
            project_id=project_id,
            video_id=video_id,
            project_path=project_path,
            frame_idx=frame_idx,
        )

    return mask

