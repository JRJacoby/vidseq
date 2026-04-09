"""Alignment Service for egocentric alignment (apply only).

Applies pre-computed keypoints from alignment_keypoints.h5 (written by pose service)
to rotate cropped frames and masks to canonical heading orientation.
"""

import asyncio
import collections
import logging
import subprocess
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

# Disable HDF5's internal file locking (we use our own approach)
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import imageio_ffmpeg

from vidseq.services.array_storage import (
    cropped_masks,
    aligned_masks,
    alignment_keypoints,
    create_aligned_masks_array,
)

import cv2
import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from vidseq.models.video import Video
from vidseq.services.cropped_video_service import cropped_video_exists, get_cropped_video_path
from vidseq.services.database_manager import DatabaseManager
from vidseq.services.exceptions import AlignmentTrainingError
from vidseq.services.frame_rotation import rotate_frame, rotate_mask, compute_heading_angles


@dataclass
class AlignmentProgress:
    """Real-time alignment (apply) progress state for SSE streaming."""

    is_aligning: bool = False
    status: str = "idle"  # idle, aligning, completed, failed

    # Video-level progress
    current_video_index: int = 0
    total_videos: int = 0
    current_video_name: str = ""

    # Frame-level progress
    current_frame: int = 0
    total_frames: int = 0

    # Performance metrics
    fps: float = 0.0  # Rolling average
    eta_seconds: float = 0.0

    # Timestamps
    started_at: Optional[float] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


# Configure logger for alignment service
logger = logging.getLogger("vidseq.alignment")
logger.setLevel(logging.DEBUG)

# Add console handler if not already present
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "[%(asctime)s] [Alignment] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class AlignmentService:
    """Singleton service for alignment (apply only)."""

    _instance: Optional["AlignmentService"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "AlignmentService":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
                    logger.info("AlignmentService singleton instance created")
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._is_applying = False

        # Alignment (apply) progress tracking for real-time updates
        self._alignment_progress = AlignmentProgress()
        self._fps_timestamps: collections.deque = collections.deque(maxlen=30)

        self._initialized = True
        logger.info("AlignmentService initialized: is_applying=False")

    @classmethod
    def get_instance(cls) -> "AlignmentService":
        """Get the singleton instance."""
        return cls()

    def is_applying(self) -> bool:
        """Check if applying alignment is in progress."""
        logger.debug(f"is_applying() -> {self._is_applying}")
        return self._is_applying

    def get_alignment_progress(self) -> AlignmentProgress:
        """Get current alignment (apply) progress for SSE streaming."""
        return self._alignment_progress

    def _calculate_rolling_fps(self) -> float:
        """Calculate rolling average FPS from recent frame timestamps."""
        if len(self._fps_timestamps) < 2:
            return 0.0
        time_span = self._fps_timestamps[-1] - self._fps_timestamps[0]
        if time_span <= 0:
            return 0.0
        return (len(self._fps_timestamps) - 1) / time_span

    def _reencode_to_h264(self, input_path: Path, output_path: Path) -> bool:
        """Re-encode video to H.264 for browser compatibility.

        Args:
            input_path: Path to input video (any codec)
            output_path: Path to output video (H.264/MP4)

        Returns:
            True if successful, False otherwise
        """
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

        cmd = [
            ffmpeg_path,
            "-y",  # Overwrite output
            "-i", str(input_path),
            "-c:v", "libx264",  # H.264 codec
            "-preset", "fast",  # Encoding speed/quality tradeoff
            "-crf", "23",  # Quality (lower = better, 18-28 is typical)
            "-pix_fmt", "yuv420p",  # Pixel format for browser compatibility
            "-movflags", "+faststart",  # Move moov atom to start for streaming
            str(output_path),
        ]

        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"_reencode_to_h264: FFmpeg error: {e.stderr}")
            return False

    async def create_videos_alignment(self, project_path: Path, video_ids: list[int]) -> dict:
        """Apply alignment to selected cropped videos.

        Creates aligned videos in <project>/aligned_videos/ folder.
        Starts processing in a background thread.

        Args:
            project_path: Path to the project folder
            video_ids: List of video IDs to apply alignment to

        Returns:
            Dict with status

        Raises:
            AlignmentTrainingError: If alignment already in progress
        """
        if self.is_applying():
            raise AlignmentTrainingError("Alignment already in progress")

        # Get project engine for sync operations
        db_manager = DatabaseManager.get_instance()
        project_engine = db_manager.get_project_engine(project_path)

        logger.info(
            f"create_videos_alignment: starting alignment in background for project at {project_path}"
        )

        # Start alignment in background thread
        asyncio.create_task(
            asyncio.to_thread(
                self.apply_alignment_sync,
                project_path,
                project_engine,
                video_ids,
            )
        )

        return {"status": "started"}

    def apply_alignment_sync(
        self,
        project_path: Path,
        project_engine,
        video_ids: list[int],
    ) -> bool:
        """Apply alignment to selected cropped videos.

        Reads pre-computed keypoints from alignment_keypoints.h5 (written by pose service),
        computes heading angles, rotates frames and masks, and saves aligned output.

        Args:
            project_path: Path to project folder
            project_engine: SQLAlchemy engine for project DB
            video_ids: List of video IDs to apply alignment to

        Returns:
            True if successful
        """
        logger.info(f"apply_alignment_sync: starting, project={project_path.name}")
        self._is_applying = True

        # Initialize alignment progress
        self._alignment_progress = AlignmentProgress(
            is_aligning=True,
            status="aligning",
            started_at=time.time(),
        )
        self._fps_timestamps.clear()
        last_progress_update = time.time()

        try:
            # Get selected videos with cropping completed (check filesystem)
            with Session(project_engine) as session:
                result = session.execute(
                    select(Video).where(Video.id.in_(video_ids))
                )
                all_videos = list(result.scalars().all())

            videos = [v for v in all_videos if cropped_video_exists(project_path, v.name)]
            logger.info(f"apply_alignment_sync: found {len(videos)} cropped videos")

            if not videos:
                logger.warning("apply_alignment_sync: no cropped videos to align")
                return False

            # Create output directory for aligned videos
            output_dir = project_path / "aligned_videos"
            output_dir.mkdir(parents=True, exist_ok=True)

            # Filter out videos that already have aligned files on disk
            videos = [
                v for v in videos
                if not (output_dir / f"{get_cropped_video_path(project_path, v.name).stem}_aligned.mp4").exists()
            ]
            logger.info(f"apply_alignment_sync: {len(videos)} videos need alignment")

            if not videos:
                logger.info("apply_alignment_sync: all videos already aligned")
                self._alignment_progress.status = "completed"
                self._alignment_progress.is_aligning = False
                return True

            self._alignment_progress.total_videos = len(videos)
            logger.info(f"apply_alignment_sync: output_dir={output_dir}")

            for video_idx, video in enumerate(videos):
                logger.info(f"apply_alignment_sync: processing video {video_idx + 1}/{len(videos)}: {video.name}")
                cropped_path = get_cropped_video_path(project_path, video.name)

                if not cropped_path.exists():
                    logger.warning(f"apply_alignment_sync: cropped video not found: {cropped_path}")
                    continue

                # Open input video
                cap = cv2.VideoCapture(str(cropped_path))
                if not cap.isOpened():
                    logger.error(f"apply_alignment_sync: failed to open video: {cropped_path}")
                    continue

                # Get video properties
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                # Update progress for this video
                self._alignment_progress.current_video_index = video_idx + 1
                self._alignment_progress.current_video_name = video.name
                self._alignment_progress.total_frames = frame_count
                self._alignment_progress.current_frame = 0
                self._fps_timestamps.clear()

                # Read pre-computed keypoints from alignment_keypoints.h5
                logger.info(f"apply_alignment_sync: reading keypoints from H5 ({frame_count} frames)")
                with alignment_keypoints(project_path, video.id) as kp_data:
                    keypoints_array = kp_data[:]  # shape: (N, 4) [front_x, front_y, rear_x, rear_y]

                # Compute smoothed heading angles from keypoints
                angles = compute_heading_angles(keypoints_array)
                logger.info(f"apply_alignment_sync: computed {len(angles)} heading angles")

                # Get mask dimensions from cropped masks (may differ from video dimensions)
                with cropped_masks(project_path, video.id) as masks:
                    mask_shape = masks.shape
                    mask_height, mask_width = mask_shape[1], mask_shape[2]

                # Create aligned masks H5 file
                create_aligned_masks_array(project_path, video.id, frame_count, mask_height)

                # Create temp output file (mp4v codec, then re-encode to H.264)
                temp_path = output_dir / f"{cropped_path.stem}_aligned.temp.mp4"
                output_path = output_dir / f"{cropped_path.stem}_aligned.mp4"

                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(str(temp_path), fourcc, fps, (width, height))

                if not writer.isOpened():
                    logger.error(f"apply_alignment_sync: failed to create video writer: {temp_path}")
                    cap.release()
                    continue

                logger.info(f"apply_alignment_sync: rotating frames ({frame_count} frames)")

                with cropped_masks(project_path, video.id) as cropped_mask_data, \
                     aligned_masks(project_path, video.id, "a") as aligned_mask_data:

                    for frame_idx in range(frame_count):
                        ret, frame = cap.read()
                        if not ret:
                            logger.warning(f"apply_alignment_sync: failed to read frame {frame_idx}")
                            break

                        angle = float(angles[frame_idx])

                        # Zero out non-mask pixels before rotating
                        cropped_mask = cropped_mask_data[frame_idx]
                        mask_3ch = np.stack([cropped_mask] * 3, axis=2)
                        frame = np.where(mask_3ch > 0, frame, 0)

                        # Rotate frame
                        rotated = rotate_frame(frame, angle)
                        writer.write(rotated)

                        # Rotate mask and save
                        aligned_mask = rotate_mask(cropped_mask, angle)
                        aligned_mask_data[frame_idx] = aligned_mask

                        # Track frame timestamp for FPS calculation
                        self._fps_timestamps.append(time.time())

                        # Update progress
                        now = time.time()
                        if now - last_progress_update >= 1.0:
                            self._alignment_progress.current_frame = frame_idx + 1
                            self._alignment_progress.fps = self._calculate_rolling_fps()
                            if self._alignment_progress.fps > 0:
                                remaining = frame_count - frame_idx - 1
                                self._alignment_progress.eta_seconds = remaining / self._alignment_progress.fps
                            last_progress_update = now

                    # Final progress update for this video
                    self._alignment_progress.current_frame = frame_count

                # Release video resources outside the H5 context managers
                cap.release()
                writer.release()

                # Re-encode to H.264 for browser compatibility
                # Write to a _tmp file first, then atomically rename so that
                # existence checks only see fully-written files.
                tmp_output_path = output_path.with_name(output_path.stem + "_tmp.mp4")
                if not self._reencode_to_h264(temp_path, tmp_output_path):
                    logger.error(f"apply_alignment_sync: failed to re-encode video {video.id}")
                    temp_path.unlink(missing_ok=True)
                    tmp_output_path.unlink(missing_ok=True)
                    continue

                # Atomic rename to final path
                tmp_output_path.rename(output_path)

                # Clean up mp4v temp file
                temp_path.unlink(missing_ok=True)

            # Mark completion
            self._alignment_progress.status = "completed"
            self._alignment_progress.is_aligning = False
            self._alignment_progress.eta_seconds = 0.0
            logger.info(f"apply_alignment_sync: completed successfully for {len(videos)} videos")
            return True

        except Exception as e:
            logger.error(f"apply_alignment_sync: failed with error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self._alignment_progress.status = "failed"
            self._alignment_progress.is_aligning = False
            return False
        finally:
            self._is_applying = False


# Module-level singleton instance
alignment_service = AlignmentService.get_instance()
