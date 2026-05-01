"""Cropped Video Service for extracting centered, mask-zeroed videos.

Extracts cropped videos centered on the mask centroid with non-mask pixels zeroed out.
"""

import asyncio
import json
import math
import random
import subprocess
import threading
from pathlib import Path
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from scipy.ndimage import gaussian_filter1d, median_filter

from vidseq.models.video import Video

import cv2
import imageio_ffmpeg
import numpy as np

from vidseq.services.array_storage import (
    compute_bbox_from_mask,
    create_cropped_masks_array,
    cropped_masks,
    final_masks,
)


# =============================================================================
# Crop Size Persistence
# =============================================================================

def _get_crop_size_path(project_path: Path) -> Path:
    """Get the path to the crop size config file."""
    return project_path / "crop_config.json"


def get_saved_crop_size_mask(project_path: Path) -> int | None:
    """Load the saved mask-mode crop size for a project, if it exists.

    Falls back to the legacy ``crop_size`` key for projects that saved their
    mask-mode value before per-mode keys were introduced.

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
        # Prefer the new per-mode key; fall back to legacy bare key
        return config.get("crop_size_mask", config.get("crop_size"))
    except Exception as e:
        print(f"[Cropped Video] Warning: Failed to load crop config: {e}")
        return None


def save_crop_size_mask(project_path: Path, crop_size: int) -> None:
    """Save the mask-mode crop size for a project.

    Reads the existing config (if any) and upserts the ``crop_size_mask`` key
    so that any other per-mode values are preserved.

    Args:
        project_path: Path to the project folder
        crop_size: Crop size to save
    """
    config_path = _get_crop_size_path(project_path)

    # Preserve any existing keys (e.g. crop_size_bbox)
    config: dict = {}
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
        except Exception as e:
            print(f"[Cropped Video] Warning: Failed to read existing crop config, overwriting: {e}")

    config["crop_size_mask"] = crop_size

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"[Cropped Video] Saved mask-mode crop size {crop_size} to {config_path}")


def get_saved_crop_size_bbox(project_path: Path) -> int | None:
    """Load the saved bbox-mode crop size for a project, if it exists.

    Does NOT fall back to any legacy key — bbox mode is new, so an absent
    key simply means the value has not been computed yet.

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
        return config.get("crop_size_bbox")
    except Exception as e:
        print(f"[Cropped Video] Warning: Failed to load crop config: {e}")
        return None


def save_crop_size_bbox(project_path: Path, crop_size: int) -> None:
    """Save the bbox-mode crop size for a project.

    Reads the existing config (if any) and upserts the ``crop_size_bbox`` key
    so that any other per-mode values are preserved.

    Args:
        project_path: Path to the project folder
        crop_size: Crop size to save
    """
    config_path = _get_crop_size_path(project_path)

    # Preserve any existing keys (e.g. crop_size_mask)
    config: dict = {}
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
        except Exception as e:
            print(f"[Cropped Video] Warning: Failed to read existing crop config, overwriting: {e}")

    config["crop_size_bbox"] = crop_size

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"[Cropped Video] Saved bbox-mode crop size {crop_size} to {config_path}")


# ---------------------------------------------------------------------------
# Legacy aliases — kept so any code that imported the old names still works.
# Both delegate to the mask-mode helpers (mask mode was the only mode when
# the bare ``crop_size`` key was introduced).
# ---------------------------------------------------------------------------

def get_saved_crop_size(project_path: Path) -> int | None:
    """Alias for get_saved_crop_size_mask (legacy name)."""
    return get_saved_crop_size_mask(project_path)


def save_crop_size(project_path: Path, crop_size: int) -> None:
    """Alias for save_crop_size_mask (legacy name)."""
    save_crop_size_mask(project_path, crop_size)


def clear_crop_size(project_path: Path) -> bool:
    """Clear all saved crop sizes for a project by deleting crop_config.json.

    Deletes the entire config file rather than individual keys — there is no
    per-mode clear in the UI today, and clearing one mode's cached value while
    leaving the other's would be surprising.  Both modes will recompute their
    crop size on the next extraction run.

    Use this to force recomputation of crop size on next cropping run.

    Args:
        project_path: Path to the project folder

    Returns:
        True if a config was deleted, False if none existed
    """
    config_path = _get_crop_size_path(project_path)
    if config_path.exists():
        config_path.unlink()
        print(f"[Cropped Video] Cleared saved crop sizes from {config_path}")
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
        subprocess.run(
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


# =============================================================================
# Bbox-centroid signal computation
# =============================================================================

async def load_bboxes_for_video(
    session: AsyncSession,
    video_id: int,
    num_frames: int,
) -> np.ndarray:
    """Bulk-load detector bboxes for all frames of a video into a dense array.

    Runs a single SQL query over ``frame_data`` and maps each row into the
    output array by ``frame_idx``.  Frames with no row (or with a NULL bbox)
    are left as NaN.

    Args:
        session: Async SQLAlchemy session bound to the project database.
        video_id: Primary key of the video.
        num_frames: Total frame count — determines the output array length.

    Returns:
        Float32 array of shape ``(num_frames, 4)`` with columns
        ``[x1, y1, x2, y2]`` in pixel coordinates.  Rows for frames that
        have no detector bbox are filled with ``np.nan``.
    """
    from vidseq.models.frame_data import FrameData

    out = np.full((num_frames, 4), np.nan, dtype=np.float32)

    result = await session.execute(
        select(
            FrameData.frame_idx,
            FrameData.detector_bbox_x1,
            FrameData.detector_bbox_y1,
            FrameData.detector_bbox_x2,
            FrameData.detector_bbox_y2,
        )
        .where(
            FrameData.video_id == video_id,
            FrameData.detector_bbox_x1.isnot(None),
            FrameData.detector_bbox_y1.isnot(None),
            FrameData.detector_bbox_x2.isnot(None),
            FrameData.detector_bbox_y2.isnot(None),
        )
        .order_by(FrameData.frame_idx)
    )
    for row in result.all():
        frame_idx, x1, y1, x2, y2 = row
        if 0 <= frame_idx < num_frames:
            out[frame_idx] = (x1, y1, x2, y2)

    return out


# Sigma = ceil(fps / 10) gives ~0.1 s worth of frames.  At 30 fps that is
# 3 frames; at 100 fps it is 10 frames.  Keeps the Gaussian scale
# proportional to real time rather than frame count.
_GAUSSIAN_SIGMA_SECONDS = 0.1


def compute_smoothed_centroids(
    bboxes: np.ndarray,
    fps: float,
) -> np.ndarray:
    """Compute a smoothed per-frame centroid trajectory from detector bboxes.

    Pipeline:
    1. Compute raw centroids (cx, cy) as bbox midpoints.
    2. Raise ``ValueError`` if every frame is NaN.
    3. Linear-interpolate NaN frames (per axis); hold boundary values for
       leading / trailing NaN runs.
    4. Apply a 5-frame median filter (``mode='nearest'``) to kill outliers.
    5. Apply a Gaussian filter with ``sigma = ceil(fps * 0.1)`` frames and
       ``mode='nearest'`` to smooth temporal jitter.

    Args:
        bboxes: Float array of shape ``(N, 4)`` with columns
            ``[x1, y1, x2, y2]``.  NaN rows represent frames with no
            detection.
        fps: Frame rate of the video (used to set the Gaussian sigma so that
            smoothing spans a fixed real-time window).

    Returns:
        Float array of shape ``(N, 2)`` with columns ``[cx, cy]``, fully
        dense (no NaNs).

    Raises:
        ValueError: If every row of *bboxes* is NaN (video has no valid
            bboxes).
    """
    n = len(bboxes)

    # Step 1 — raw centroids
    cx_raw = (bboxes[:, 0] + bboxes[:, 2]) / 2.0
    cy_raw = (bboxes[:, 1] + bboxes[:, 3]) / 2.0
    centroids = np.stack([cx_raw, cy_raw], axis=1)  # (N, 2)

    # Step 2 — guard: all-NaN video is a hard fail
    valid_mask = ~np.isnan(centroids[:, 0])
    if not np.any(valid_mask):
        raise ValueError("video has no valid bboxes")

    # Step 3 — linear interpolation + hold-nearest at boundaries
    all_idxs = np.arange(n, dtype=float)
    valid_idxs = all_idxs[valid_mask]
    for axis in range(2):
        valid_vals = centroids[valid_mask, axis]
        # np.interp clamps to endpoint values outside the range of valid_idxs,
        # which is exactly the hold-nearest behaviour we want for leading/trailing
        # NaN runs.
        centroids[:, axis] = np.interp(all_idxs, valid_idxs, valid_vals)

    # Step 4 — 5-frame median filter (removes single-frame outliers)
    centroids = median_filter(centroids, size=(5, 1), mode="nearest")

    # Step 5 — Gaussian temporal smoothing (~0.1 s window).  Clamp to >=1
    # so videos with missing/zero fps headers don't trigger a ZeroDivisionError
    # inside scipy.
    sigma = max(1, math.ceil(fps * _GAUSSIAN_SIGMA_SECONDS))
    centroids = gaussian_filter1d(centroids, sigma=sigma, axis=0, mode="nearest")

    return centroids.astype(np.float32)


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

        try:
            with final_masks(project_path, video_id, "r") as masks:
                for frame_idx, height, width in frame_list:
                    mask = masks[frame_idx]
                    bbox = compute_bbox_from_mask(mask)
                    if bbox is not None:
                        x1, y1, x2, y2 = bbox
                        bbox_w = int(x2 - x1)
                        bbox_h = int(y2 - y1)
                        max_width = max(max_width, bbox_w)
                        max_height = max(max_height, bbox_h)
        except FileNotFoundError:
            continue
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


async def compute_global_crop_size_bbox(
    session: AsyncSession,
    videos: list,
) -> int:
    """Compute the global square crop size for bbox-centroid mode.

    Pools all non-NULL detector bboxes across the selected videos via a single
    SQL query.  Returns ``ceil(max(p99(widths), p99(heights)) * 1.15)`` as a
    square crop dimension.

    The 1.15 factor adds a +15 % safety margin so that the bounding box of a
    slightly-larger-than-median animal is not clipped at the crop edge.

    Args:
        session: Async SQLAlchemy session bound to the project database.
        videos: List of Video model instances (video IDs are extracted from
            ``v.id`` for each ``v``).

    Returns:
        Square crop size in pixels (integer).

    Raises:
        ValueError: If no non-NULL detector bboxes exist across all selected
            videos.  The caller should gate on detector readiness before
            calling this function, but this guard defends in depth.
    """
    from vidseq.models.frame_data import FrameData

    video_ids = [v.id for v in videos]

    result = await session.execute(
        select(
            FrameData.detector_bbox_x1,
            FrameData.detector_bbox_y1,
            FrameData.detector_bbox_x2,
            FrameData.detector_bbox_y2,
        )
        .where(
            FrameData.video_id.in_(video_ids),
            FrameData.detector_bbox_x1.isnot(None),
            FrameData.detector_bbox_y1.isnot(None),
            FrameData.detector_bbox_x2.isnot(None),
            FrameData.detector_bbox_y2.isnot(None),
        )
    )
    rows = result.all()

    if not rows:
        raise ValueError(
            "no detector bboxes across selected videos; run the detector first"
        )

    arr = np.array(rows, dtype=np.float32)  # shape (N, 4): x1, y1, x2, y2
    widths = arr[:, 2] - arr[:, 0]
    heights = arr[:, 3] - arr[:, 1]

    p99_w = float(np.percentile(widths, 99))
    p99_h = float(np.percentile(heights, 99))
    square = max(p99_w, p99_h)
    # +15 % safety margin so typical animals are not clipped at the crop edge
    crop_size = int(math.ceil(square * 1.15))

    print(
        f"[Cropped Video] bbox crop size: {len(rows)} bboxes across "
        f"{len(video_ids)} video(s), p99_w={p99_w:.1f}, p99_h={p99_h:.1f} "
        f"→ crop_size={crop_size}"
    )
    return crop_size


def _process_frames_with_masks(
    cap: cv2.VideoCapture,
    writer: cv2.VideoWriter,
    video,
    crop_size: int,
    mask_dataset,  # h5py.Dataset or None
    cropped_masks_data,  # h5py.Dataset for output
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
                mask = mask_dataset[frame_idx]
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
            cropped_masks_data[frame_idx] = cropped_mask

            # Write frame (background preserved for labeling; masking deferred to alignment)
            writer.write(cropped_frame)

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
) -> bool:
    """Process a single video to create cropped output.

    Args:
        project_path: Path to the project folder
        video: Video model instance
        crop_size: Square crop dimension

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

    # Create cropped masks array with pre-allocated dataset
    create_cropped_masks_array(project_path, video.id, video.num_frames, crop_size)

    # Use context manager for cropped masks (write operation needs locking)
    with cropped_masks(project_path, video.id, "a") as cropped_masks_data:
        # Get mask dataset from tracker masks (optional - may not exist yet)
        processed = False
        try:
            # Use final_masks context manager - handle is cached for reads
            with final_masks(project_path, video.id, "r") as mask_dataset:
                # Process all frames within this context
                _process_frames_with_masks(
                    cap, writer, video, crop_size, mask_dataset,
                    cropped_masks_data
                )
                processed = True
        except FileNotFoundError:
            print(f"[Cropped Video] Warning: Tracker masks not found for video {video.id}")
        except Exception as e:
            print(f"[Cropped Video] Warning: Could not open mask file: {e}")

        # If we haven't processed yet (no mask file or error), process with zeros
        if not processed:
            _process_frames_with_masks(
                cap, writer, video, crop_size, None,
                cropped_masks_data
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


async def create_videos_extraction(
    session: AsyncSession,
    project_path: Path,
    video_ids: list[int],
) -> dict:
    """Start cropped video extraction for selected videos.

    Fetches videos by ID, validates all are segmented, filters out
    already-cropped, and starts extraction.

    Args:
        session: Async database session
        project_path: Path to the project folder
        video_ids: List of video IDs to extract

    Returns:
        Dict with status and video_count

    Raises:
        ValueError: If no videos found, videos not segmented, or extraction in progress
    """
    # Fetch videos by ID
    result = await session.execute(
        select(Video).where(Video.id.in_(video_ids)).order_by(Video.id)
    )
    videos = list(result.scalars().all())
    if not videos:
        raise ValueError("No videos found for the given IDs")

    # Validate all selected videos are segmented
    unsegmented = [
        v.name for v in videos if v.segmentation_status != "segmented"
    ]
    if unsegmented:
        raise ValueError(
            f"Selected videos must be segmented before extraction. "
            f"Unsegmented: {', '.join(unsegmented)}"
        )

    # Filter out videos that already have cropped files on disk
    uncropped = await asyncio.to_thread(
        lambda: [v for v in videos if not cropped_video_exists(project_path, v.name)]
    )
    if not uncropped:
        return {"status": "skipped", "message": "All selected videos already cropped", "video_count": 0}

    # Start extraction
    service = CroppedVideoService.get_instance()

    if service.is_extracting():
        raise ValueError("Extraction already in progress")

    service.extract_all_cropped_videos(
        project_path=project_path,
        videos=uncropped,
    )

    return {"status": "started", "video_count": len(uncropped)}


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
        project_path: Path,
        videos: list,
    ) -> None:
        """Start cropped video extraction for all videos.

        Args:
            project_path: Path to the project folder
            videos: List of Video model instances

        Raises:
            RuntimeError: If extraction is already in progress
        """
        if self._is_extracting:
            raise RuntimeError("Extraction already in progress")

        self._is_extracting = True

        def _extract():
            try:
                # Pass 1: Get or compute global crop size (mask mode)
                saved_crop_size = get_saved_crop_size_mask(project_path)
                if saved_crop_size is not None:
                    crop_size = saved_crop_size
                    print(f"[Cropped Video] Using saved crop size: {crop_size}")
                else:
                    print(f"[Cropped Video] Computing global crop size from {len(videos)} videos...")
                    crop_size = compute_global_crop_size(project_path, videos)
                    save_crop_size_mask(project_path, crop_size)

                # Pass 2: Process each video
                for video in videos:
                    print(f"[Cropped Video] Processing video {video.id}: {video.name}")

                    try:
                        success = process_single_video(
                            project_path=project_path,
                            video=video,
                            crop_size=crop_size,
                        )

                        if not success:
                            print(f"[Cropped Video] Failed to process video {video.id}")

                    except Exception as e:
                        print(f"[Cropped Video] Error processing video {video.id}: {e}")

                print(f"[Cropped Video] Extraction complete for {len(videos)} videos")

            except Exception as e:
                print(f"[Cropped Video] Extraction failed: {e}")
                import traceback
                traceback.print_exc()
            finally:
                self._is_extracting = False

        self._extraction_thread = threading.Thread(target=_extract, daemon=True)
        self._extraction_thread.start()
