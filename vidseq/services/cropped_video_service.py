"""Cropped Video Service for extracting centered, mask-zeroed videos.

Extracts cropped videos centered on the mask centroid with non-mask pixels zeroed out.
"""

import json
import random
import subprocess
import threading
from pathlib import Path
from typing import Optional

import cv2
import imageio_ffmpeg
import numpy as np

from vidseq.services import h5_storage
from vidseq.services.h5_storage import open_cropped_h5, tracker_h5


# =============================================================================
# Crop Size Persistence
# =============================================================================

def _get_crop_size_path(project_path: Path) -> Path:
    """Get the path to the crop size config file."""
    return project_path / "crop_config.json"


def get_saved_crop_size(project_path: Path) -> int | None:
    """Load the saved crop size for a project, if it exists.

    Args:
        project_path: Path to the project folder

    Returns:
        Saved crop size, or None if not saved yet
    """
    config_path = _get_crop_size_path(project_path)
    if not config_path.exists():
        return None

    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        return config.get("crop_size")
    except Exception as e:
        print(f"[Cropped Video] Warning: Failed to load crop config: {e}")
        return None


def save_crop_size(project_path: Path, crop_size: int) -> None:
    """Save the crop size for a project.

    Args:
        project_path: Path to the project folder
        crop_size: Crop size to save
    """
    config_path = _get_crop_size_path(project_path)
    config = {"crop_size": crop_size}

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"[Cropped Video] Saved crop size {crop_size} to {config_path}")


def clear_crop_size(project_path: Path) -> bool:
    """Clear the saved crop size for a project.

    Use this to force recomputation of crop size on next cropping run.

    Args:
        project_path: Path to the project folder

    Returns:
        True if a config was deleted, False if none existed
    """
    config_path = _get_crop_size_path(project_path)
    if config_path.exists():
        config_path.unlink()
        print(f"[Cropped Video] Cleared saved crop size from {config_path}")
        return True
    return False


def _reencode_to_h264(input_path: Path, output_path: Path) -> bool:
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
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"[Cropped Video] FFmpeg error: {e.stderr}")
        return False


def _get_cropped_videos_dir(project_path: Path) -> Path:
    """Get the cropped_videos directory for a project."""
    return project_path / "cropped_videos"


def _get_cropped_video_path(project_path: Path, video_name: str) -> Path:
    """Get path to the cropped video file."""
    cropped_dir = _get_cropped_videos_dir(project_path)
    # Remove extension and add _cropped.mp4
    stem = Path(video_name).stem
    return cropped_dir / f"{stem}_cropped.mp4"


def cropped_video_exists(project_path: Path, video_name: str) -> bool:
    """Check if cropped video exists for a video."""
    return _get_cropped_video_path(project_path, video_name).exists()


def get_cropped_video_path(project_path: Path, video_name: str) -> Path:
    """Get the path to a cropped video."""
    return _get_cropped_video_path(project_path, video_name)


def compute_centroid(mask: np.ndarray) -> tuple[int, int]:
    """Compute centroid of mask pixels.

    Args:
        mask: Binary mask array (height, width)

    Returns:
        (cx, cy) tuple of centroid coordinates, or frame center if mask is empty
    """
    mask_binary = mask > 0
    if not np.any(mask_binary):
        # Return frame center if no mask pixels
        h, w = mask.shape
        return (w // 2, h // 2)

    # Find all nonzero pixel coordinates
    coords = np.argwhere(mask_binary)  # Returns (row, col) = (y, x)
    cy = int(np.mean(coords[:, 0]))
    cx = int(np.mean(coords[:, 1]))
    return (cx, cy)


def compute_global_crop_size(
    project_path: Path,
    videos: list,
    max_samples: int = 10000,
) -> int:
    """Compute the global crop size by sampling frames across all videos.

    Randomly samples up to max_samples frames, computes bbox for each,
    and returns 1.2x the maximum dimension found.

    Args:
        project_path: Path to the project folder
        videos: List of Video model instances
        max_samples: Maximum number of frames to sample (default 10000)

    Returns:
        Crop size as integer (square crop dimension)
    """
    # Build list of (video_id, frame_idx, height, width) for all frames
    all_frames = []
    for video in videos:
        for frame_idx in range(video.num_frames):
            all_frames.append((video.id, frame_idx, video.height, video.width))

    # Sample if needed
    if len(all_frames) > max_samples:
        sampled_frames = random.sample(all_frames, max_samples)
    else:
        sampled_frames = all_frames

    print(f"[Cropped Video] Sampling {len(sampled_frames)} frames to compute crop size...")

    max_width = 0
    max_height = 0

    # Group by video_id for efficient HDF5 access
    frames_by_video: dict[int, list[tuple[int, int, int]]] = {}
    for video_id, frame_idx, height, width in sampled_frames:
        if video_id not in frames_by_video:
            frames_by_video[video_id] = []
        frames_by_video[video_id].append((frame_idx, height, width))

    for video_id, frame_list in frames_by_video.items():
        # Get video dimensions from first frame
        _, height, width = frame_list[0]

        h5_path = project_path / "masks" / f"{video_id}.h5"
        if not h5_path.exists():
            continue

        try:
            with h5_storage.open_video_h5(project_path, video_id, "r") as h5_file:
                if "masks" not in h5_file:
                    continue

                for frame_idx, height, width in frame_list:
                    mask = np.array(h5_file["masks"][frame_idx])
                    bbox = h5_storage.compute_bbox_from_mask(mask)
                    if bbox is not None:
                        x1, y1, x2, y2 = bbox
                        bbox_w = int(x2 - x1)
                        bbox_h = int(y2 - y1)
                        max_width = max(max_width, bbox_w)
                        max_height = max(max_height, bbox_h)
        except Exception as e:
            print(f"[Cropped Video] Error reading masks for video {video_id}: {e}")
            continue

    # Compute crop size as 1.2x the max dimension
    max_dim = max(max_width, max_height)
    if max_dim == 0:
        # Fallback to reasonable default if no masks found
        print("[Cropped Video] Warning: No masks found, using default crop size of 256")
        return 256

    crop_size = int(max_dim * 1.2)
    print(f"[Cropped Video] Max bbox: {max_width}x{max_height}, crop size: {crop_size}")
    return crop_size


def _process_frames_with_masks(
    cap: cv2.VideoCapture,
    writer: cv2.VideoWriter,
    video,
    crop_size: int,
    mask_dataset,  # h5py.Dataset or None
    cropped_h5_file,
    progress_callback,
) -> bool:
    """Process all frames, reading masks from dataset (or zeros if None).

    Returns:
        True if successful, False otherwise
    """
    try:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Load mask for this frame
            if mask_dataset is not None:
                mask = np.array(mask_dataset[frame_idx])
            else:
                mask = np.zeros((video.height, video.width), dtype=np.uint8)

            # Compute centroid
            cx, cy = compute_centroid(mask)

            # Compute crop bounds
            half_size = crop_size // 2
            x1 = cx - half_size
            y1 = cy - half_size
            x2 = x1 + crop_size
            y2 = y1 + crop_size

            # Create output frame (black background)
            cropped_frame = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)

            # Compute valid source and destination regions
            src_x1 = max(0, x1)
            src_y1 = max(0, y1)
            src_x2 = min(video.width, x2)
            src_y2 = min(video.height, y2)

            dst_x1 = src_x1 - x1
            dst_y1 = src_y1 - y1
            dst_x2 = dst_x1 + (src_x2 - src_x1)
            dst_y2 = dst_y1 + (src_y2 - src_y1)

            # Copy valid region from source frame
            if src_x2 > src_x1 and src_y2 > src_y1:
                cropped_frame[dst_y1:dst_y2, dst_x1:dst_x2] = frame[src_y1:src_y2, src_x1:src_x2]

            # Apply mask - zero out non-mask pixels
            # Need to crop the mask the same way
            cropped_mask = np.zeros((crop_size, crop_size), dtype=np.uint8)
            if src_x2 > src_x1 and src_y2 > src_y1:
                cropped_mask[dst_y1:dst_y2, dst_x1:dst_x2] = mask[src_y1:src_y2, src_x1:src_x2]

            # Save cropped mask to HDF5
            cropped_h5_file["masks"][frame_idx] = cropped_mask

            # Zero out pixels where mask is 0
            mask_3ch = np.stack([cropped_mask, cropped_mask, cropped_mask], axis=2)
            cropped_frame = np.where(mask_3ch > 0, cropped_frame, 0)

            # Write frame
            writer.write(cropped_frame)

            # Progress callback
            if progress_callback is not None and frame_idx % 100 == 0:
                progress_callback(frame_idx)

            frame_idx += 1

        print(f"[Cropped Video] Processed {frame_idx} frames for video {video.id}")
        return True

    except Exception as e:
        print(f"[Cropped Video] Error processing video {video.id}: {e}")
        import traceback
        traceback.print_exc()
        return False


def process_single_video(
    project_path: Path,
    video,
    crop_size: int,
    job_id: int,
    progress_callback=None,
) -> bool:
    """Process a single video to create cropped output.

    Args:
        project_path: Path to the project folder
        video: Video model instance
        crop_size: Square crop dimension
        job_id: Job ID for progress updates
        progress_callback: Optional callback(frame_idx) for progress updates

    Returns:
        True if successful, False otherwise
    """
    video_path = Path(video.path)
    output_path = _get_cropped_video_path(project_path, video.name)
    temp_path = output_path.with_suffix(".temp.mp4")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Open input video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[Cropped Video] Failed to open video: {video_path}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)

    # Create output video writer (to temp file with mp4v codec)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(
        str(temp_path),
        fourcc,
        fps,
        (crop_size, crop_size),
    )

    if not writer.isOpened():
        print(f"[Cropped Video] Failed to create output video: {temp_path}")
        cap.release()
        return False

    # Use context manager for cropped H5 (write operation needs locking)
    with open_cropped_h5(project_path, video.id, "w") as cropped_h5_file:
        # Create dataset for cropped masks
        cropped_h5_file.create_dataset(
            "masks",
            shape=(video.num_frames, crop_size, crop_size),
            dtype=np.uint8,
            fillvalue=0,
            chunks=(1, crop_size, crop_size),
            compression=None,
        )

        # Get mask dataset from tracker H5 (optional - may not exist yet)
        tracker_h5_path = project_path / "masks" / f"{video.id}.h5"
        processed = False
        if tracker_h5_path.exists():
            try:
                # Use tracker_h5 context manager - handle is cached for reads
                with tracker_h5(project_path, video.id, "r") as h5_file:
                    if "masks" in h5_file:
                        mask_dataset = h5_file["masks"]
                        # Process all frames within this context
                        _process_frames_with_masks(
                            cap, writer, video, crop_size, mask_dataset,
                            cropped_h5_file, progress_callback
                        )
                        processed = True
            except FileNotFoundError:
                print(f"[Cropped Video] Warning: Tracker H5 not found for video {video.id}")
            except Exception as e:
                print(f"[Cropped Video] Warning: Could not open mask file: {e}")

        # If we haven't processed yet (no mask file or error), process with zeros
        if not processed:
            _process_frames_with_masks(
                cap, writer, video, crop_size, None,
                cropped_h5_file, progress_callback
            )

        cap.release()
        writer.release()

    # Re-encode to H.264 for browser compatibility
    # Write to a _tmp file first, then atomically rename so that
    # existence checks only see fully-written files.
    print(f"[Cropped Video] Re-encoding to H.264: {output_path.name}")
    tmp_output_path = output_path.with_name(output_path.stem + "_tmp.mp4")
    if not _reencode_to_h264(temp_path, tmp_output_path):
        print(f"[Cropped Video] Failed to re-encode video {video.id}")
        temp_path.unlink(missing_ok=True)
        tmp_output_path.unlink(missing_ok=True)
        return False

    # Atomic rename to final path
    tmp_output_path.rename(output_path)

    # Clean up mp4v temp file
    temp_path.unlink(missing_ok=True)
    print(f"[Cropped Video] Completed video {video.id}: {output_path}")
    return True


class CroppedVideoService:
    """Singleton service for managing cropped video extraction."""

    _instance: Optional["CroppedVideoService"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "CroppedVideoService":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._is_extracting = False
        self._extraction_thread: Optional[threading.Thread] = None

        self._initialized = True

    @classmethod
    def get_instance(cls) -> "CroppedVideoService":
        """Get the singleton instance."""
        return cls()

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton (useful for testing)."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance = None

    def is_extracting(self) -> bool:
        """Check if extraction is in progress."""
        return self._is_extracting

    def extract_all_cropped_videos(
        self,
        project_id: int,
        project_path: Path,
        videos: list,
    ) -> list[int]:
        """Start cropped video extraction for all videos.

        Args:
            project_id: ID of the project
            project_path: Path to the project folder
            videos: List of Video model instances

        Returns:
            List of job IDs created

        Raises:
            RuntimeError: If extraction is already in progress
        """
        if self._is_extracting:
            raise RuntimeError("Extraction already in progress")

        self._is_extracting = True

        # Import here to avoid circular imports
        from sqlalchemy import update
        from sqlalchemy.orm import Session

        from vidseq.models.registry import Job
        from vidseq.models.utils import utc_now
        from vidseq.services.database_manager import DatabaseManager

        db_manager = DatabaseManager.get_instance()
        registry_engine = db_manager.get_registry_engine()

        # Create jobs for all videos
        job_ids = []
        video_configs = []

        with Session(registry_engine) as session:
            for video in videos:
                log_path = project_path / "logs" / f"cropped_video_{video.id}.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)

                job = Job(
                    type="cropped_video_extraction",
                    status="pending",
                    project_id=project_id,
                    details={
                        "video_id": video.id,
                        "video_name": video.name,
                        "current_frame": 0,
                        "total_frames": video.num_frames,
                    },
                    log_path=str(log_path),
                )
                session.add(job)
                session.flush()

                job_ids.append(job.id)
                video_configs.append({
                    "job_id": job.id,
                    "video": video,
                })

            session.commit()

        def _extract():
            try:
                # Pass 1: Get or compute global crop size
                saved_crop_size = get_saved_crop_size(project_path)
                if saved_crop_size is not None:
                    crop_size = saved_crop_size
                    print(f"[Cropped Video] Using saved crop size: {crop_size}")
                else:
                    print(f"[Cropped Video] Computing global crop size from {len(videos)} videos...")
                    crop_size = compute_global_crop_size(project_path, videos)
                    save_crop_size(project_path, crop_size)

                # Pass 2: Process each video
                for config in video_configs:
                    job_id = config["job_id"]
                    video = config["video"]

                    print(f"[Cropped Video] Processing video {video.id}: {video.name}")

                    # Update job status to running
                    with Session(registry_engine) as session:
                        session.execute(
                            update(Job)
                            .where(Job.id == job_id)
                            .values(status="running", updated_at=utc_now())
                        )
                        session.commit()

                    # Progress callback
                    def progress_update(frame_idx):
                        with Session(registry_engine) as session:
                            session.execute(
                                update(Job)
                                .where(Job.id == job_id)
                                .values(
                                    details={
                                        "video_id": video.id,
                                        "video_name": video.name,
                                        "current_frame": frame_idx,
                                        "total_frames": video.num_frames,
                                    },
                                    updated_at=utc_now(),
                                )
                            )
                            session.commit()

                    try:
                        success = process_single_video(
                            project_path=project_path,
                            video=video,
                            crop_size=crop_size,
                            job_id=job_id,
                            progress_callback=progress_update,
                        )

                        if success:
                            # Update job status to completed
                            with Session(registry_engine) as session:
                                session.execute(
                                    update(Job)
                                    .where(Job.id == job_id)
                                    .values(
                                        status="completed",
                                        details={
                                            "video_id": video.id,
                                            "video_name": video.name,
                                            "current_frame": video.num_frames,
                                            "total_frames": video.num_frames,
                                        },
                                        updated_at=utc_now(),
                                    )
                                )
                                session.commit()

                        else:
                            raise RuntimeError("Processing failed")

                    except Exception as e:
                        print(f"[Cropped Video] Error processing video {video.id}: {e}")
                        with Session(registry_engine) as session:
                            session.execute(
                                update(Job)
                                .where(Job.id == job_id)
                                .values(
                                    status="failed",
                                    details={
                                        "video_id": video.id,
                                        "video_name": video.name,
                                        "error": str(e),
                                    },
                                    updated_at=utc_now(),
                                )
                            )
                            session.commit()

                print(f"[Cropped Video] Extraction complete for {len(videos)} videos")

            except Exception as e:
                print(f"[Cropped Video] Extraction failed: {e}")
                import traceback
                traceback.print_exc()
            finally:
                self._is_extracting = False

        self._extraction_thread = threading.Thread(target=_extract, daemon=True)
        self._extraction_thread.start()

        return job_ids
