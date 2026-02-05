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
    """Convert a numpy mask to PNG bytes.

    Creates an RGBA PNG with the mask rendered as a semi-transparent blue overlay.
    Masked pixels (value > 0) become light blue, non-masked pixels are transparent.
    """
    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    masked = mask > 0
    rgba[masked, 0] = 102   # R
    rgba[masked, 1] = 179   # G
    rgba[masked, 2] = 255   # B
    rgba[masked, 3] = 102   # A (semi-transparent)
    # Non-masked pixels stay (0, 0, 0, 0) = fully transparent
    img = Image.fromarray(rgba, mode='RGBA')
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


async def init_session(
    session: "AsyncSession",
    project_id: int,
    video: "Video",
    project_path: Path,
) -> "segmentation_tcp_client.VideoSessionInfo":
    """Initialize a SAM session for a video.

    Queries conditioning frames from the database and initializes
    the session on the GPU worker.

    Args:
        session: Async database session
        project_id: ID of the project
        video: Video model instance
        project_path: Path to the project folder

    Returns:
        VideoSessionInfo with video dimensions and frame count
    """
    from sqlalchemy import select
    from vidseq.models.conditioning_frame import ConditioningFrame

    # Query conditioning frames for this video
    result = await session.execute(
        select(ConditioningFrame.frame_idx)
        .where(ConditioningFrame.video_id == video.id)
    )
    cond_frame_indices = list(result.scalars().all())

    return segmentation_tcp_client.init_session(
        project_id=project_id,
        video_id=video.id,
        video_path=Path(video.path),
        project_path=project_path,
        num_frames=video.num_frames,
        height=video.height,
        width=video.width,
        cond_frame_indices=cond_frame_indices,
    )


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
        # First, create conditioning frame record if not exists
        from sqlalchemy import select
        from vidseq.models.conditioning_frame import ConditioningFrame

        existing = await session.execute(
            select(ConditioningFrame)
            .where(ConditioningFrame.video_id == video_id)
            .where(ConditioningFrame.frame_idx == frame_idx)
        )
        if existing.scalar_one_or_none() is None:
            session.add(ConditioningFrame(video_id=video_id, frame_idx=frame_idx))
            await session.commit()

        p = points[0]
        label = labels[0]
        mask = segmentation_tcp_client.add_point_prompt(
            project_id=project_id,
            video_id=video_id,
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
        # Remove conditioning frame record from database
        from sqlalchemy import delete
        from vidseq.models.conditioning_frame import ConditioningFrame

        await session.execute(
            delete(ConditioningFrame)
            .where(ConditioningFrame.video_id == video_id)
            .where(ConditioningFrame.frame_idx == frame_idx)
        )
        await session.commit()

        # Clear SAM memory
        segmentation_tcp_client.reset_frame(
            project_id=project_id,
            video_id=video_id,
            frame_idx=frame_idx,
        )

    return mask


async def propagate(
    session: "AsyncSession",
    project_id: int,
    video_id: int,
    project_path: Path,
    start_frame_idx: int,
    max_frames: int,
    num_frames: int,
    height: int,
    width: int,
) -> int:
    """Propagate segmentation masks forward from a frame.

    Requires an active SAM session with a tracked object.

    Args:
        session: Async database session
        project_id: ID of the project
        video_id: ID of the video
        project_path: Path to the project folder
        start_frame_idx: Frame to start propagation from
        max_frames: Maximum number of frames to propagate
        num_frames: Total frames in video
        height: Video height
        width: Video width

    Returns:
        Number of frames processed

    Raises:
        RuntimeError: If propagation fails (no active session, etc.)
    """
    frame_indices = segmentation_tcp_client.generate_training_masks(
        project_id=project_id,
        video_id=video_id,
        start_frame_idx=start_frame_idx,
        max_frames=max_frames,
        project_path=project_path,
        num_frames=num_frames,
        height=height,
        width=width,
    )

    # Update has_tracker_mask for all propagated frames
    await frame_data_service.set_has_tracker_mask(session, video_id, frame_indices, True)

    return len(frame_indices)


async def segment_all_videos(
    session: "AsyncSession",
    project_id: int,
    project_path: Path,
    videos: list,
) -> list[int]:
    """Segment all videos using detector-tracker approach.

    Runs propagate_with_detector on each video and updates database flags.

    Args:
        session: Async database session
        project_id: ID of the project
        project_path: Path to the project folder
        videos: List of Video model instances

    Returns:
        List of job IDs (empty for now - synchronous execution)
    """
    from sqlalchemy import select
    from vidseq.models.conditioning_frame import ConditioningFrame

    # Query conditioning frames for all videos
    cond_frames_by_video: dict[int, list[int]] = {}
    for video in videos:
        result = await session.execute(
            select(ConditioningFrame.frame_idx)
            .where(ConditioningFrame.video_id == video.id)
        )
        cond_frames_by_video[video.id] = list(result.scalars().all())

    job_ids = await segmentation_tcp_client.segment_all_videos(
        project_id=project_id,
        project_path=project_path,
        videos=videos,
        cond_frames_by_video=cond_frames_by_video,
    )

    # Update database flags for all frames in all videos
    for video in videos:
        all_frames = list(range(video.num_frames))
        await frame_data_service.set_has_tracker_mask(session, video.id, all_frames, True)
        await frame_data_service.set_has_final_mask(session, video.id, all_frames, True)

    return job_ids

