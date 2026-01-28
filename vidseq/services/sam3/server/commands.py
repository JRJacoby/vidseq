"""SAM3 TCP command handlers using StreamingSegmentor.

Each function handles one command type. The worker maintains a single
StreamingSegmentor instance and manages file handles externally.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import h5py
import numpy as np

from vidseq.services.sam3.frame_source import VideoFrameSource
from vidseq.services.sam3.streaming_segmentor import StreamingSegmentor
from vidseq.services.sam3.utils import encode_mask_rle


@dataclass
class VideoResources:
    """File handles for an open video session."""
    frame_source: VideoFrameSource
    mask_file: h5py.File
    mask_dataset: Any  # h5py.Dataset


# Global state managed by the worker
_segmentor: StreamingSegmentor | None = None
_video_resources: dict[int, VideoResources] = {}


def handle_load_model(_checkpoint_path: Path) -> tuple[dict, StreamingSegmentor]:
    """Load SAM3 model via StreamingSegmentor.

    Returns:
        Tuple of (response_dict, segmentor)
    """
    global _segmentor

    print("[SAM3 Worker] Loading SAM3 model via StreamingSegmentor...")

    _segmentor = StreamingSegmentor(device="cuda")

    print("[SAM3 Worker] SAM3 model loaded!")
    return {"type": "status", "status": "ready"}, _segmentor


def _get_mask_path(project_path: Path, video_id: int) -> Path:
    """Get HDF5 mask file path for a video."""
    return project_path / "masks" / f"{video_id}.h5"


def _ensure_mask_dataset(
    mask_path: Path,
    num_frames: int,
    height: int,
    width: int,
) -> tuple[h5py.File, Any]:
    """Open or create HDF5 mask file and dataset.

    Returns:
        (h5py.File, dataset)
    """
    mask_path.parent.mkdir(parents=True, exist_ok=True)

    mask_file = h5py.File(mask_path, "a")

    if "masks" not in mask_file:
        mask_file.create_dataset(
            "masks",
            shape=(num_frames, height, width),
            dtype=np.uint8,
            chunks=(1, height, width),
            fillvalue=0,
        )

    return mask_file, mask_file["masks"]


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

    # Open/create mask file
    mask_path = _get_mask_path(project_path, video_id)
    mask_file, mask_dataset = _ensure_mask_dataset(
        mask_path, num_frames, height, width
    )

    # Store resources
    _video_resources[video_id] = VideoResources(
        frame_source=frame_source,
        mask_file=mask_file,
        mask_dataset=mask_dataset,
    )

    # Initialize StreamingSegmentor session
    segmentor.open_video(
        video_id=str(video_id),
        frames=frame_source,
        masks=mask_dataset,
        frame_dims=(height, width),
        cond_frame_indices=cond_frame_indices,
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

    # Run segmentation
    segmentor.add_point_prompt(
        video_id=str(video_id),
        frame_idx=frame_idx,
        location=(px, py),
        label=label,
    )

    # Read back the mask for response
    mask = resources.mask_dataset[frame_idx]

    return {
        "type": "add_prompt_result",
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

    return {
        "type": "generate_training_masks_result",
        "status": "ok",
        "frames_processed": len(frame_indices),
        "frame_indices": frame_indices,
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

        # Clear mask in HDF5
        resources = _video_resources[video_id]
        resources.mask_dataset[frame_idx] = 0

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
        params: Command params with video_id
        segmentor: StreamingSegmentor instance

    Returns:
        Response dict
    """
    video_id = params["video_id"]

    if segmentor is None:
        raise RuntimeError("Model not loaded")

    if video_id in _video_resources:
        # Close StreamingSegmentor session
        segmentor.close_video(str(video_id))

        # Clear all masks in HDF5
        resources = _video_resources[video_id]
        resources.mask_dataset[...] = 0

        # Don't close file handles - session may be reopened

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
