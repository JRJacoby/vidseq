"""Video service - metadata extraction and database operations."""

import logging
import shutil
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import cv2
from PIL import Image
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from vidseq.models.conditioning_frame import ConditioningFrame
from vidseq.models.frame_data import FrameData
from vidseq.models.video import Video
from vidseq.services.array_storage import (
    create_video_segmentation_arrays,
    tracker_masks,
    tracker_logits,
    detector_masks,
    final_masks,
    reset_video_segmentation_arrays,
)
from vidseq.services.exceptions import (
    DBRecordNotFoundError,
    TextFileParseError,
    VideoFileNotFoundError,
    VideoFileInvalidError,
)
from vidseq.services import segmentation_service, segmentation_tcp_client

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VideoMetadata:
    """Immutable video metadata."""
    num_frames: int
    height: int
    width: int
    fps: float


def get_video_metadata(video_path: Path | str) -> VideoMetadata:
    """
    Extract metadata from a video file.

    Args:
        video_path: Path to the video file

    Returns:
        VideoMetadata with num_frames, height, width, fps

    Raises:
        VideoFileNotFoundError: If the video file doesn't exist
        VideoFileInvalidError: If the video cannot be opened or metadata cannot be read
    """
    video_path = Path(video_path)

    if not video_path.exists():
        raise VideoFileNotFoundError(str(video_path))

    if not video_path.is_file():
        raise VideoFileInvalidError(str(video_path), "path is not a file")

    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise VideoFileInvalidError(str(video_path), "could not open video")

        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps <= 0:
            raise VideoFileInvalidError(str(video_path), "could not read FPS")

        if num_frames <= 0:
            raise VideoFileInvalidError(str(video_path), "could not read frame count")

        if height <= 0 or width <= 0:
            raise VideoFileInvalidError(str(video_path), "could not read dimensions")

        return VideoMetadata(
            num_frames=num_frames,
            height=height,
            width=width,
            fps=fps,
        )
    finally:
        cap.release()


def extract_frame_as_jpeg(
    video_path: Path,
    frame_idx: int,
    num_frames: int | None = None,
    quality: int = 95,
) -> bytes:
    """Extract a specific frame from a video and return as JPEG bytes.

    Args:
        video_path: Path to the video file
        frame_idx: Frame index (0-based)
        num_frames: Total frames in video (for validation). If None, reads from video.
        quality: JPEG quality (1-100)

    Returns:
        JPEG image bytes

    Raises:
        VideoFileNotFoundError: If video file doesn't exist
        FrameIndexOutOfRangeError: If frame_idx is out of bounds
        VideoFileInvalidError: If video cannot be opened or frame cannot be read
    """
    from vidseq.services.exceptions import FrameIndexOutOfRangeError

    if not video_path.exists():
        raise VideoFileNotFoundError(str(video_path))

    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise VideoFileInvalidError(str(video_path), "could not open video")

        # Get frame count if not provided
        if num_frames is None:
            num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Validate frame index
        if frame_idx < 0 or frame_idx >= num_frames:
            raise FrameIndexOutOfRangeError(frame_idx, num_frames)

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            raise VideoFileInvalidError(
                str(video_path), f"could not read frame {frame_idx}"
            )
    finally:
        cap.release()

    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert to PIL Image and then to JPEG bytes
    pil_image = Image.fromarray(frame_rgb)
    img_bytes = BytesIO()
    pil_image.save(img_bytes, format="JPEG", quality=quality)
    img_bytes.seek(0)

    return img_bytes.read()


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


def resolve_video_paths(
    paths: list[Path],
) -> list[tuple[Path, VideoMetadata]]:
    """Resolve a mixed list of video files and text file video lists.

    For each path, try reading as a UTF-8 text file first (video list),
    then fall back to treating it as a video. Text-first avoids FFmpeg's
    tty demuxer, which falsely "opens" text files as ANSI art video.

    Returns a deduplicated list of (path, metadata) tuples.
    """
    resolved: list[tuple[Path, VideoMetadata]] = []
    seen: set[str] = set()

    for path in paths:
        if not path.exists():
            raise VideoFileNotFoundError(str(path))

        # Try text file first — real videos are binary and fail UTF-8 decode
        text_content = _try_read_as_text(path)
        if text_content is not None:
            _parse_and_validate_video_list(path, text_content, resolved, seen)
        else:
            meta = get_video_metadata(path)
            path_str = str(path)
            if path_str not in seen:
                seen.add(path_str)
                resolved.append((path, meta))

    return resolved


def _try_read_as_text(path: Path) -> str | None:
    """Try to read a file as UTF-8 text. Returns content or None if binary."""
    try:
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None


def _parse_and_validate_video_list(
    text_file_path: Path,
    content: str,
    resolved: list[tuple[Path, VideoMetadata]],
    seen: set[str],
) -> None:
    """Parse a text file as a video list and validate each path."""
    lines = [line.strip() for line in content.splitlines()]
    non_empty = [line for line in lines if line]

    if not non_empty:
        raise TextFileParseError(str(text_file_path), "no video paths found")

    for line in non_empty:
        if not line.startswith("/"):
            raise TextFileParseError(
                str(text_file_path),
                f"path is not absolute: {line}",
            )

        video_path = Path(line)
        try:
            meta = get_video_metadata(video_path)
        except VideoFileNotFoundError:
            raise VideoFileNotFoundError(
                f"{line} (listed in {text_file_path})"
            )
        except VideoFileInvalidError as e:
            raise VideoFileInvalidError(
                str(video_path),
                f"{e.reason} (listed in {text_file_path})",
            )

        path_str = str(video_path)
        if path_str not in seen:
            seen.add(path_str)
            resolved.append((video_path, meta))


async def add_videos(
    session: AsyncSession,
    project_path: Path,
    video_paths: list[Path],
) -> list[Video]:
    """Add multiple videos to a project.

    Resolves text file video lists, validates all paths, then adds to DB.

    Args:
        session: Project database session
        project_path: Path to the project folder
        video_paths: List of paths to video files or text file video lists

    Returns:
        List of created Video records

    Raises:
        VideoFileNotFoundError: If a video file doesn't exist
        VideoFileInvalidError: If a video cannot be read
        TextFileParseError: If a text file is malformed
    """
    resolved = resolve_video_paths(video_paths)
    added_videos = []

    for video_path, meta in resolved:
        video = Video(
            name=video_path.name,
            path=str(video_path),
            fps=meta.fps,
            height=meta.height,
            width=meta.width,
            num_frames=meta.num_frames,
        )
        session.add(video)
        added_videos.append(video)

    await session.commit()

    # Create H5 files after commit (need video.id)
    for video in added_videos:
        await session.refresh(video)
        create_video_segmentation_arrays(
            project_path=project_path,
            video_id=video.id,
            num_frames=video.num_frames,
            height=video.height,
            width=video.width,
        )

    return added_videos


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
        # After reset, there are no conditioning frames
        segmentation_tcp_client.init_session(
            project_id=project_id,
            video_id=video_id,
            video_path=Path(video.path),
            project_path=project_path,
            num_frames=video.num_frames,
            height=video.height,
            width=video.width,
            cond_frame_indices=[],
        )


async def delete_videos(
    project_id: int,
    project_path: Path,
    video_ids: list[int],
    session: AsyncSession,
) -> None:
    """Delete videos and all associated data from a project.

    Steps:
    1. Resolve & validate all video IDs
    2. Close any open SAM2 sessions
    3. Delete DB records (child tables first, then Video)
    4. Commit
    5. Delete files (H5 directories, cropped/aligned videos)

    Args:
        project_id: ID of the project
        project_path: Path to the project folder
        video_ids: List of video IDs to delete
        session: Async database session

    Raises:
        DBRecordNotFoundError: If any video_id is not found
    """
    from vidseq.models.alignment_label import AlignmentLabel

    # 1. Resolve & validate — fetch all videos, fail fast if any missing
    videos: list[Video] = []
    for vid in video_ids:
        result = await session.execute(
            select(Video).where(Video.id == vid)
        )
        video = result.scalar_one_or_none()
        if video is None:
            raise DBRecordNotFoundError("Video", vid)
        videos.append(video)

    # 2. Close SAM2 sessions
    for video in videos:
        segmentation_tcp_client.close_session(project_id, video.id)

    # 3. Delete DB records (child tables first)
    for video in videos:
        await session.execute(
            delete(FrameData).where(FrameData.video_id == video.id)
        )
        await session.execute(
            delete(ConditioningFrame).where(ConditioningFrame.video_id == video.id)
        )
        await session.execute(
            delete(AlignmentLabel).where(AlignmentLabel.video_id == video.id)
        )
        await session.delete(video)

    # 4. Commit
    await session.commit()

    # 5. Delete files (after commit — orphaned files are harmless)
    for video in videos:
        # H5 directory
        h5_dir = project_path / "array_data" / str(video.id)
        if h5_dir.exists():
            try:
                shutil.rmtree(h5_dir)
            except OSError:
                logger.warning("Failed to delete H5 directory: %s", h5_dir)

        # Cropped video
        stem = Path(video.name).stem
        cropped = project_path / "cropped_videos" / f"{stem}_cropped.mp4"
        cropped.unlink(missing_ok=True)

        # Aligned video
        aligned = project_path / "aligned_videos" / f"{stem}_cropped_aligned.mp4"
        aligned.unlink(missing_ok=True)

