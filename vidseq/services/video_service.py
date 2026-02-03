"""Video service - metadata extraction and database operations."""

from dataclasses import dataclass
from pathlib import Path

import cv2
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from vidseq.models.conditioning_frame import ConditioningFrame
from vidseq.models.frame_data import FrameData
from vidseq.models.video import Video
from vidseq.services import segmentation_service, segmentation_tcp_client
from vidseq.services.array_storage import (
    tracker_masks,
    tracker_logits,
    detector_masks,
    final_masks,
    reset_video_segmentation_arrays,
)


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


async def get_video_by_id(session: AsyncSession, video_id: int) -> Video:
    """
    Get a video by ID.
    
    Args:
        session: Database session
        video_id: ID of the video
        
    Returns:
        Video model instance
        
    Raises:
        LookupError: If video not found
    """
    result = await session.execute(
        select(Video).where(Video.id == video_id)
    )
    video = result.scalar_one_or_none()
    if not video:
        raise LookupError(f"Video {video_id} not found")
    return video


async def get_all_videos(session: AsyncSession) -> list[Video]:
    """Get all videos in the project."""
    result = await session.execute(
        select(Video).order_by(Video.id)
    )
    return list(result.scalars().all())


async def delete_frame_data(
    project_id: int,
    project_path: Path,
    video_id: int,
    frame_idx: int,
    session: AsyncSession,
) -> None:
    """Delete all annotation data for a frame.

    Clears:
    - H5 files: tracker mask/logits, detector mask, final mask
    - Database: conditioning_frame, frame_data tables
    - SAM memory: if session active

    Args:
        project_id: ID of the project
        project_path: Path to the project folder
        video_id: ID of the video
        frame_idx: Frame index (0-based)
        session: Async database session
    """
    # 1. Clear H5 files
    with tracker_masks(project_path, video_id, "a") as masks:
        masks[frame_idx] = 0
    with tracker_logits(project_path, video_id, "a") as logits:
        logits[frame_idx] = 0
    with detector_masks(project_path, video_id, "a") as masks:
        masks[frame_idx] = 0
    with final_masks(project_path, video_id, "a") as masks:
        masks[frame_idx] = 0

    # 2. Clear database tables
    await session.execute(
        delete(ConditioningFrame)
        .where(ConditioningFrame.video_id == video_id)
        .where(ConditioningFrame.frame_idx == frame_idx)
    )
    await session.execute(
        delete(FrameData)
        .where(FrameData.video_id == video_id)
        .where(FrameData.frame_idx == frame_idx)
    )
    await session.commit()

    # 3. Clear SAM memory (if session active)
    segmentation_service.reset_frame_memory(project_id, video_id, frame_idx)


async def reset_video(
    project_id: int,
    project_path: Path,
    video: Video,
    session: AsyncSession,
) -> None:
    """Reset all annotation data for a video.

    Clears:
    - H5 files: tracker masks/logits, detector masks, final masks
    - Database: conditioning_frames, frame_data tables
    - SAM memory: closes and re-opens session if it was open

    Does NOT touch cropped/aligned mask files.

    Args:
        project_id: ID of the project
        project_path: Path to the project folder
        video: Video model instance
        session: Async database session
    """
    video_id = video.id

    # 1. Check if SAM session is currently open
    session_was_open = segmentation_tcp_client.get_session(project_id, video_id) is not None

    # 2. Close SAM session (releases worker file handles)
    if session_was_open:
        segmentation_tcp_client.close_session(project_id, video_id)

    # 3. Reset H5 files (all segmentation H5s: tracker, detector, final)
    reset_video_segmentation_arrays(
        project_path=project_path,
        video_id=video_id,
        num_frames=video.num_frames,
        height=video.height,
        width=video.width,
    )

    # 4. Clear database tables
    await session.execute(
        delete(ConditioningFrame).where(ConditioningFrame.video_id == video_id)
    )
    await session.execute(
        delete(FrameData).where(FrameData.video_id == video_id)
    )
    await session.commit()

    # 5. Re-open SAM session if it was open before
    if session_was_open:
        segmentation_tcp_client.init_session(
            project_id=project_id,
            video_id=video_id,
            video_path=Path(video.path),
            project_path=project_path,
        )

