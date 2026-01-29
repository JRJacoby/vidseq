"""SAM3 TCP command handlers using StreamingSegmentor.

Each function handles one command type. The worker maintains a single
StreamingSegmentor instance and manages file handles externally.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import h5py
import numpy as np

from vidseq.services.sam3.frame_source import VideoFrameSource
from vidseq.services.sam3.utils import encode_mask_rle

# Backend selection - can be switched via environment variable
import os
SAM_BACKEND = os.environ.get("SAM_BACKEND", "sam2").lower()

if SAM_BACKEND == "sam2":
    from vidseq.services.sam2.streaming_segmentor import SAM2StreamingSegmentor as StreamingSegmentor
    print(f"[SAM Worker] Using SAM2 backend")
else:
    from vidseq.services.sam3.streaming_segmentor import StreamingSegmentor
    print(f"[SAM Worker] Using SAM3 backend")


@dataclass
class VideoResources:
    """File handles for an open video session."""
    frame_source: VideoFrameSource
    mask_file: h5py.File
    mask_dataset: Any  # h5py.Dataset - binary masks (uint8)
    logits_dataset: Any  # h5py.Dataset - low-res logits (float32) for refinement
    detector_file: Optional[h5py.File] = None  # h5py.File for detector masks
    detector_masks: Any = None  # h5py.Dataset for detector masks


# Global state managed by the worker
_segmentor: StreamingSegmentor | None = None
_video_resources: dict[int, VideoResources] = {}


def handle_load_model(_checkpoint_path: Optional[Path]) -> tuple[dict, StreamingSegmentor]:
    """Load SAM3 model via StreamingSegmentor.

    Returns:
        Tuple of (response_dict, segmentor)
    """
    global _segmentor

    # Skip if already loaded
    if _segmentor is not None:
        print("[SAM Worker] Model already loaded, skipping")
        return {"type": "status", "status": "ready"}, _segmentor

    print(f"[SAM Worker] Loading model via StreamingSegmentor...")

    _segmentor = StreamingSegmentor(device="cuda")

    print("[SAM Worker] Model loaded!")
    return {"type": "status", "status": "ready"}, _segmentor


def _get_mask_path(project_path: Path, video_id: int) -> Path:
    """Get HDF5 mask file path for a video."""
    return project_path / "masks" / f"{video_id}.h5"


def _ensure_mask_dataset(
    mask_path: Path,
    num_frames: int,
    height: int,
    width: int,
    logits_size: int = 288,
) -> tuple[h5py.File, Any, Any]:
    """Open or create HDF5 mask file and datasets.

    Args:
        logits_size: Size of low-res logits (SAM2=256, SAM3=288).

    Returns:
        (h5py.File, mask_dataset, logits_dataset)
    """
    LOGITS_SIZE = logits_size

    mask_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[DEBUG _ensure_mask_dataset] Opening {mask_path}, exists before={mask_path.exists()}")
    mask_file = h5py.File(mask_path, "a")
    print(f"[DEBUG _ensure_mask_dataset] Opened, exists after={mask_path.exists()}, filename={mask_file.filename}")

    # Binary masks for display (full resolution)
    if "masks" not in mask_file:
        mask_file.create_dataset(
            "masks",
            shape=(num_frames, height, width),
            dtype=np.uint8,
            chunks=(1, height, width),
            fillvalue=0,
        )

    # Low-res logits for refinement (256x256)
    if "logits" not in mask_file:
        mask_file.create_dataset(
            "logits",
            shape=(num_frames, LOGITS_SIZE, LOGITS_SIZE),
            dtype=np.float32,
            chunks=(1, LOGITS_SIZE, LOGITS_SIZE),
            fillvalue=0.0,
        )

    # Flush to ensure datasets are written to disk (important for NFS)
    mask_file.flush()

    return mask_file, mask_file["masks"], mask_file["logits"]


def handle_init_session(
    params: dict,
    segmentor: StreamingSegmentor,
) -> dict:
    """Initialize video inference session.

    Args:
        params: Command params with video_id, video_path, project_path,
                num_frames, height, width, cond_frame_indices
        segmentor: StreamingSegmentor instance

    Returns:
        Response dict
    """
    global _video_resources

    video_id = params["video_id"]
    video_path = Path(params["video_path"])
    project_path = Path(params["project_path"])
    num_frames = params["num_frames"]
    height = params["height"]
    width = params["width"]
    cond_frame_indices = set(params.get("cond_frame_indices", []))

    print(f"[SAM3 Worker] Initializing session for video {video_id}...")

    if segmentor is None:
        raise RuntimeError("Model not loaded")

    # Close existing session if any
    if video_id in _video_resources:
        _close_video_resources(video_id, segmentor)

    # Open frame source
    frame_source = VideoFrameSource(video_path)

    # Open/create mask file with both masks and logits datasets
    mask_path = _get_mask_path(project_path, video_id)
    mask_file, mask_dataset, logits_dataset = _ensure_mask_dataset(
        mask_path, num_frames, height, width, logits_size=segmentor.LOGITS_SIZE
    )

    # Open detector masks if available
    detector_h5_path = project_path / "masks" / f"{video_id}_detector.h5"
    detector_file = None
    detector_masks = None
    if detector_h5_path.exists():
        try:
            detector_file = h5py.File(detector_h5_path, "r")
            if "masks" in detector_file:
                detector_masks = detector_file["masks"]
                print(f"[SAM3 Worker] Loaded detector masks from {detector_h5_path}")
        except Exception as e:
            print(f"[SAM3 Worker] Warning: Failed to load detector masks: {e}")
            if detector_file is not None:
                detector_file.close()
                detector_file = None

    # Store resources
    _video_resources[video_id] = VideoResources(
        frame_source=frame_source,
        mask_file=mask_file,
        mask_dataset=mask_dataset,
        logits_dataset=logits_dataset,
        detector_file=detector_file,
        detector_masks=detector_masks,
    )

    # Initialize StreamingSegmentor session
    segmentor.open_video(
        video_id=str(video_id),
        frames=frame_source,
        masks=mask_dataset,
        logits=logits_dataset,
        frame_dims=(height, width),
        cond_frame_indices=cond_frame_indices,
        detector_masks=detector_masks,
    )

    return {
        "type": "init_session_result",
        "status": "ok",
        "video_id": video_id,
        "num_frames": num_frames,
        "height": height,
        "width": width,
    }


def handle_add_prompt(
    params: dict,
    segmentor: StreamingSegmentor,
) -> dict:
    """Add point prompt to a frame.

    Args:
        params: Command params with video_id, frame_idx, x, y, label
        segmentor: StreamingSegmentor instance

    Returns:
        Response dict with mask_rle
    """
    video_id = params["video_id"]
    frame_idx = params["frame_idx"]
    x = params["x"]  # normalized [0, 1]
    y = params["y"]  # normalized [0, 1]
    label = params["label"]  # 1=positive, 0=negative

    if segmentor is None:
        raise RuntimeError("Model not loaded")

    if video_id not in _video_resources:
        raise RuntimeError(f"No session for video {video_id}")

    resources = _video_resources[video_id]
    height, width = resources.mask_dataset.shape[1:]

    # Convert normalized coords to pixel coords
    px = x * width
    py = y * height

    # Get mask before for comparison
    mask_before = np.array(resources.mask_dataset[frame_idx])
    before_sum = int(mask_before.sum())

    # Run segmentation
    segmentor.add_point_prompt(
        video_id=str(video_id),
        frame_idx=frame_idx,
        location=(px, py),
        label=label,
    )

    # Flush HDF5 to ensure write is visible, then read back the mask
    resources.mask_file.flush()
    mask = resources.mask_dataset[frame_idx]
    after_sum = int(mask.sum())

    print(f"[SAM3 Worker] add_prompt frame={frame_idx} label={label} "
          f"point=({px:.1f}, {py:.1f}) mask_sum: {before_sum} -> {after_sum}")

    return {
        "type": "add_prompt_result",
        "status": "ok",
        "mask_rle": encode_mask_rle(mask),
        "mask_shape": mask.shape,
        "mask_dtype": str(mask.dtype),
    }


def handle_refine_mask(
    params: dict,
    segmentor: StreamingSegmentor,
) -> dict:
    """Refine an existing mask with point prompt(s).

    Args:
        params: Command params with video_id, frame_idx, and either:
                - Single point: x, y, label (backward compat)
                - Multiple points: points [{x, y}, ...], labels [int, ...]
        segmentor: StreamingSegmentor instance

    Returns:
        Response dict with mask_rle
    """
    video_id = params["video_id"]
    frame_idx = params["frame_idx"]

    if segmentor is None:
        raise RuntimeError("Model not loaded")

    if video_id not in _video_resources:
        raise RuntimeError(f"No session for video {video_id}")

    resources = _video_resources[video_id]
    height, width = resources.mask_dataset.shape[1:]

    # Support both single point (x, y, label) and arrays (points, labels)
    if "points" in params:
        # New multi-point format
        points = params["points"]  # List of {x, y}
        labels = params["labels"]  # List of int
        locations = [(p["x"] * width, p["y"] * height) for p in points]
    else:
        # Legacy single point format
        x = params["x"]  # normalized [0, 1]
        y = params["y"]  # normalized [0, 1]
        label = params["label"]  # 1=positive, 0=negative
        locations = [(x * width, y * height)]
        labels = [label]

    # Get mask before for comparison
    mask_before = np.array(resources.mask_dataset[frame_idx])
    before_sum = int(mask_before.sum())

    # Run refinement with all points
    segmentor.refine_mask(
        video_id=str(video_id),
        frame_idx=frame_idx,
        location=locations,
        label=labels,
    )

    # Flush HDF5 to ensure write is visible, then read back the mask
    resources.mask_file.flush()
    mask = resources.mask_dataset[frame_idx]
    after_sum = int(mask.sum())

    print(f"[SAM3 Worker] refine_mask frame={frame_idx} "
          f"num_points={len(locations)} mask_sum: {before_sum} -> {after_sum}")

    return {
        "type": "refine_mask_result",
        "status": "ok",
        "mask_rle": encode_mask_rle(mask),
        "mask_shape": mask.shape,
        "mask_dtype": str(mask.dtype),
    }


def handle_propagate(
    params: dict,
    segmentor: StreamingSegmentor,
) -> dict:
    """Propagate tracking to a single frame.

    Args:
        params: Command params with video_id, frame_idx
        segmentor: StreamingSegmentor instance

    Returns:
        Response dict with mask_rle
    """
    video_id = params["video_id"]
    frame_idx = params["frame_idx"]

    if segmentor is None:
        raise RuntimeError("Model not loaded")

    if video_id not in _video_resources:
        raise RuntimeError(f"No session for video {video_id}")

    resources = _video_resources[video_id]

    segmentor.propagate(
        video_id=str(video_id),
        frame_idx=frame_idx,
    )

    # Flush HDF5 to ensure write is visible
    resources.mask_file.flush()
    mask = resources.mask_dataset[frame_idx]

    return {
        "type": "propagate_result",
        "status": "ok",
        "mask_rle": encode_mask_rle(mask),
        "mask_shape": mask.shape,
        "mask_dtype": str(mask.dtype),
    }


def handle_generate_training_masks(
    params: dict,
    segmentor: StreamingSegmentor,
) -> dict:
    """Generate training masks by propagating through video.

    Args:
        params: Command params with video_id, start_frame_idx, max_frames
        segmentor: StreamingSegmentor instance

    Returns:
        Response dict with frames_processed, frame_indices
    """
    video_id = params["video_id"]
    start_frame_idx = params["start_frame_idx"]
    max_frames = params["max_frames"]

    if segmentor is None:
        raise RuntimeError("Model not loaded")

    if video_id not in _video_resources:
        raise RuntimeError(f"No session for video {video_id}")

    frame_indices = segmentor.propagate_sequential(
        video_id=str(video_id),
        start_frame=start_frame_idx,
        num_frames=max_frames,
        progress_interval=50,
    )

    # Flush HDF5 to ensure writes are visible to readers
    resources = _video_resources[video_id]
    resources.mask_file.flush()

    # DEBUG: Verify masks were written
    if frame_indices:
        sample_idx = frame_indices[0]
        mask_data = resources.mask_dataset[sample_idx]
        print(f"[DEBUG] After propagate: frame {sample_idx} mask sum={int(mask_data.sum())}, "
              f"shape={mask_data.shape}, dtype={mask_data.dtype}, "
              f"min={mask_data.min()}, max={mask_data.max()}")
        print(f"[DEBUG] HDF5 file path: {resources.mask_file.filename}")

    return {
        "type": "generate_training_masks_result",
        "status": "ok",
        "frames_processed": len(frame_indices),
        "frame_indices": frame_indices,
    }


def handle_propagate_with_detector(
    params: dict,
    segmentor: StreamingSegmentor,
    response_callback: Callable[[dict], None],
) -> dict:
    """Propagate tracking with detector-based correction.

    Args:
        params: Command params with video_id, num_frames, iou_threshold
        segmentor: StreamingSegmentor instance
        response_callback: Callback for progress updates

    Returns:
        Response dict with frames_processed, frames_corrected, corrected_frame_indices
    """
    video_id = params["video_id"]
    num_frames = params["num_frames"]
    iou_threshold = params.get("iou_threshold", 0.5)

    if segmentor is None:
        raise RuntimeError("Model not loaded")

    if video_id not in _video_resources:
        raise RuntimeError(f"No session for video {video_id}")

    resources = _video_resources[video_id]

    # Check that detector masks are available
    if resources.detector_masks is None:
        raise RuntimeError(f"No detector masks available for video {video_id}")

    def progress_callback(frame_idx: int, total: int) -> None:
        if frame_idx % 50 == 0 or frame_idx == total - 1:
            response_callback({
                "type": "progress",
                "frame_idx": frame_idx,
                "total": total,
            })

    corrected_frames = segmentor.propagate_with_detector(
        video_id=str(video_id),
        num_frames=num_frames,
        iou_threshold=iou_threshold,
        progress_callback=progress_callback,
    )

    # Flush HDF5 to ensure writes are visible
    resources.mask_file.flush()

    return {
        "type": "propagate_with_detector_result",
        "status": "ok",
        "frames_processed": num_frames,
        "frames_corrected": len(corrected_frames),
        "corrected_frame_indices": corrected_frames,
    }


def handle_reset_frame(
    params: dict,
    segmentor: StreamingSegmentor,
) -> dict:
    """Reset a single frame (clear mask and memory).

    Args:
        params: Command params with video_id, frame_idx
        segmentor: StreamingSegmentor instance

    Returns:
        Response dict
    """
    video_id = params["video_id"]
    frame_idx = params["frame_idx"]

    if segmentor is None:
        raise RuntimeError("Model not loaded")

    # Clear from StreamingSegmentor memory
    if video_id in _video_resources:
        segmentor.reset_frame(str(video_id), frame_idx)

        # Clear mask and logits in HDF5
        resources = _video_resources[video_id]
        resources.mask_dataset[frame_idx] = 0
        resources.logits_dataset[frame_idx] = 0.0

    return {
        "type": "reset_frame_result",
        "status": "ok",
        "frame_idx": frame_idx,
    }


def handle_reset_video(
    params: dict,
    segmentor: StreamingSegmentor,
) -> dict:
    """Reset entire video (clear all masks and memory).

    Args:
        params: Command params with video_id, project_path
        segmentor: StreamingSegmentor instance

    Returns:
        Response dict
    """
    video_id = params["video_id"]
    project_path = Path(params["project_path"])

    if segmentor is None:
        raise RuntimeError("Model not loaded")

    if video_id in _video_resources:
        # Close StreamingSegmentor session
        segmentor.close_video(str(video_id))

        resources = _video_resources[video_id]

        # Get dimensions before closing
        num_frames, height, width = resources.mask_dataset.shape
        mask_path = _get_mask_path(project_path, video_id)

        print(f"[DEBUG reset_video] Before close: mask_path={mask_path}, exists={mask_path.exists()}")

        # Close HDF5 file and delete it
        resources.mask_file.close()

        print(f"[DEBUG reset_video] After close: mask_path exists={mask_path.exists()}")

        # Delete all h5 files for this video
        h5_files_to_delete = [
            mask_path,  # tracker masks
            mask_path.parent / f"{video_id}_detector.h5",  # detector masks
            project_path / "cropped_masks" / f"{video_id}.h5",  # cropped masks
            project_path / "aligned_masks" / f"{video_id}.h5",  # aligned masks
        ]
        for h5_path in h5_files_to_delete:
            if h5_path.exists():
                h5_path.unlink()
                print(f"[DEBUG reset_video] Deleted: {h5_path}")

        # Recreate empty HDF5 file so session remains valid
        mask_file, mask_dataset, logits_dataset = _ensure_mask_dataset(
            mask_path, num_frames, height, width, logits_size=segmentor.LOGITS_SIZE
        )

        print(f"[DEBUG reset_video] After _ensure_mask_dataset: mask_path exists={mask_path.exists()}")
        print(f"[DEBUG reset_video] mask_file.filename={mask_file.filename}")

        resources.mask_file = mask_file
        resources.mask_dataset = mask_dataset
        resources.logits_dataset = logits_dataset

        # Re-open StreamingSegmentor session with fresh state
        segmentor.open_video(
            video_id=str(video_id),
            frames=resources.frame_source,
            masks=mask_dataset,
            logits=logits_dataset,
            frame_dims=(height, width),
            cond_frame_indices=set(),  # No conditioning frames after reset
        )

        print(f"[DEBUG reset_video] Complete. Final mask_path exists={mask_path.exists()}")

    return {
        "type": "reset_video_result",
        "status": "ok",
    }


def _close_video_resources(video_id: int, segmentor: StreamingSegmentor) -> None:
    """Close and clean up resources for a video."""
    if video_id in _video_resources:
        segmentor.close_video(str(video_id))

        resources = _video_resources.pop(video_id)
        resources.frame_source.close()
        resources.mask_file.close()
        if resources.detector_file is not None:
            resources.detector_file.close()


def handle_close_session(
    params: dict,
    segmentor: StreamingSegmentor,
) -> dict:
    """Close video session and free resources.

    Args:
        params: Command params with video_id
        segmentor: StreamingSegmentor instance

    Returns:
        Response dict
    """
    video_id = params["video_id"]

    print(f"[SAM3 Worker] Closing session for video {video_id}...")

    if segmentor is not None:
        _close_video_resources(video_id, segmentor)

    return {"type": "close_session_result", "status": "ok"}


def handle_shutdown(segmentor: StreamingSegmentor) -> dict:
    """Shutdown server and clean up all sessions.

    Args:
        segmentor: StreamingSegmentor instance

    Returns:
        Response dict
    """
    global _video_resources

    print("[SAM3 Worker] Shutting down...")

    for video_id in list(_video_resources.keys()):
        try:
            _close_video_resources(video_id, segmentor)
        except Exception:
            pass

    _video_resources.clear()

    return {"type": "shutdown_result", "status": "ok"}


def handle_segment_videos_batch(
    _params: dict,
    _segmentor: StreamingSegmentor,
    _response_callback: Callable[[dict], None],
) -> dict:
    """Segment multiple videos in batch.

    Note: This is a simplified version. The full batch implementation
    with job tracking can be added later if needed.
    """
    # For now, return an error indicating batch is not yet implemented
    # with the new StreamingSegmentor architecture
    return {
        "type": "segment_videos_batch_result",
        "status": "error",
        "error": "Batch segmentation not yet implemented with StreamingSegmentor",
    }
