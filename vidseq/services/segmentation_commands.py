"""SAM2 TCP command handlers using StreamingSegmentor.

Each function handles one command type. The worker maintains a single
StreamingSegmentor instance and manages file handles externally.
"""

import base64
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import h5py
import numpy as np

from vidseq.services.sam2.streaming_segmentor import SAM2StreamingSegmentor as StreamingSegmentor


# ---------------------------------------------------------------------------
# VideoFrameSource - wrapper for sequential video frame access
# ---------------------------------------------------------------------------


class VideoFrameSource:
    """Wrapper around cv2.VideoCapture with position tracking.

    Implements __getitem__ for random access to video frames, but tracks
    current position to avoid unnecessary seeks during sequential reads.
    """

    def __init__(self, path: str | Path):
        """Open a video file."""
        self._path = str(path)
        self._cap = cv2.VideoCapture(self._path)
        if not self._cap.isOpened():
            raise ValueError(f"Failed to open video: {path}")

        self._pos = 0
        self.frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def __getitem__(self, idx: int) -> np.ndarray:
        """Read a frame by index. Returns BGR numpy array."""
        if idx < 0 or idx >= self.frame_count:
            raise IndexError(f"Frame index {idx} out of range [0, {self.frame_count})")

        if idx != self._pos:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            self._pos = idx

        success, frame = self._cap.read()
        if not success:
            raise IndexError(f"Failed to read frame {idx}")

        self._pos += 1
        return frame

    def __len__(self) -> int:
        return self.frame_count

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


# ---------------------------------------------------------------------------
# Mask encoding utilities
# ---------------------------------------------------------------------------


def encode_mask_rle(mask: np.ndarray) -> str:
    """Encode binary mask as binary RLE + base64 for JSON transport."""
    flat = mask.flatten()
    binary_data = bytearray()
    i = 0

    while i < len(flat):
        value = flat[i]
        length = 1
        while i + length < len(flat) and flat[i + length] == value:
            length += 1
        binary_data.extend(struct.pack(">BI", int(value), length))
        i += length

    return base64.b64encode(bytes(binary_data)).decode("utf-8")


@dataclass
class VideoResources:
    """File handles for an open video session."""
    frame_source: VideoFrameSource
    mask_file: h5py.File
    mask_dataset: Any  # h5py.Dataset - binary masks (uint8)
    logits_dataset: Any  # h5py.Dataset - low-res logits (float32) for refinement
    detector_file: Optional[h5py.File] = None  # h5py.File for detector masks
    detector_masks: Any = None  # h5py.Dataset for detector masks
    final_file: Optional[h5py.File] = None  # h5py.File for final (corrected) masks
    final_masks: Any = None  # h5py.Dataset for final (corrected) masks


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

    mask_file = h5py.File(mask_path, "a")

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
                print(f"[SAM2 Worker] Loaded detector masks from {detector_h5_path}")
            else:
                # No masks dataset - close file handle
                print(f"[SAM2 Worker] Warning: No 'masks' dataset in {detector_h5_path}")
                detector_file.close()
                detector_file = None
        except Exception as e:
            print(f"[SAM2 Worker] Warning: Failed to load detector masks: {e}")
            if detector_file is not None:
                detector_file.close()
                detector_file = None

    # Open/create final masks file if detector masks available
    final_file = None
    final_masks = None
    if detector_masks is not None:
        final_h5_path = project_path / "masks" / f"{video_id}_final.h5"
        try:
            final_file = h5py.File(final_h5_path, "a")
            if "masks" not in final_file:
                final_file.create_dataset(
                    "masks",
                    shape=(num_frames, height, width),
                    dtype=np.uint8,
                    chunks=(1, height, width),
                    fillvalue=0,
                )
            final_masks = final_file["masks"]
            print(f"[SAM2 Worker] Opened final masks file: {final_h5_path}")
        except Exception as e:
            print(f"[SAM2 Worker] Warning: Failed to open final masks file: {e}")
            if final_file is not None:
                final_file.close()
                final_file = None
                final_masks = None

    # Store resources
    _video_resources[video_id] = VideoResources(
        frame_source=frame_source,
        mask_file=mask_file,
        mask_dataset=mask_dataset,
        logits_dataset=logits_dataset,
        detector_file=detector_file,
        detector_masks=detector_masks,
        final_file=final_file,
        final_masks=final_masks,
    )

    # Initialize StreamingSegmentor session
    # Session only holds modeling state - no H5 handles
    segmentor.open_video(
        video_id=str(video_id),
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

    # Get mask before for comparison
    mask_before = np.array(resources.mask_dataset[frame_idx])
    before_sum = int(mask_before.sum())

    # Run segmentation - returns mask and logits
    mask, logits = segmentor.add_point_prompt(
        video_id=str(video_id),
        frame_idx=frame_idx,
        location=(px, py),
        label=label,
        frames=resources.frame_source,
        masks=resources.mask_dataset,
    )

    # Write results to HDF5
    resources.mask_dataset[frame_idx] = mask
    resources.logits_dataset[frame_idx] = logits
    resources.mask_file.flush()
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

    # Read previous logits
    prev_logits = np.array(resources.logits_dataset[frame_idx])

    # Run refinement with all points - returns mask and logits
    mask, logits = segmentor.refine_mask(
        video_id=str(video_id),
        frame_idx=frame_idx,
        location=locations,
        label=labels,
        frames=resources.frame_source,
        masks=resources.mask_dataset,
        prev_logits=prev_logits,
    )

    # Write results to HDF5
    resources.mask_dataset[frame_idx] = mask
    resources.logits_dataset[frame_idx] = logits
    resources.mask_file.flush()
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

    # Read frame from video
    frame = resources.frame_source[frame_idx]

    # Propagate - returns mask and logits
    mask, logits = segmentor.propagate(
        video_id=str(video_id),
        frame_idx=frame_idx,
        frame=frame,
    )

    # Write results to HDF5
    resources.mask_dataset[frame_idx] = mask
    resources.logits_dataset[frame_idx] = logits
    resources.mask_file.flush()

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

    resources = _video_resources[video_id]

    # Callback to write each result to HDF5
    def on_result(frame_idx: int, mask: np.ndarray, logits: np.ndarray) -> None:
        resources.mask_dataset[frame_idx] = mask
        resources.logits_dataset[frame_idx] = logits

    frame_indices = segmentor.propagate_sequential(
        video_id=str(video_id),
        start_frame=start_frame_idx,
        num_frames=max_frames,
        frames=resources.frame_source,
        on_result=on_result,
        progress_interval=50,
    )

    # Flush HDF5 to ensure writes are visible to readers
    resources.mask_file.flush()

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

    # Callback to write tracker results to HDF5
    def on_result(frame_idx: int, mask: np.ndarray, logits: np.ndarray) -> None:
        resources.mask_dataset[frame_idx] = mask
        resources.logits_dataset[frame_idx] = logits

    # Callback to write final (corrected) masks if available
    def on_final_result(frame_idx: int, mask: np.ndarray) -> None:
        if resources.final_masks is not None:
            resources.final_masks[frame_idx] = mask

    # Propagate with detector correction
    propagated_frames, corrected_frames = segmentor.propagate_with_detector(
        video_id=str(video_id),
        num_frames=num_frames,
        frames=resources.frame_source,
        detector_masks=resources.detector_masks,
        on_result=on_result,
        on_final_result=on_final_result if resources.final_masks is not None else None,
        iou_threshold=iou_threshold,
        progress_callback=progress_callback,
    )

    # Flush HDF5 files to ensure writes are visible
    resources.mask_file.flush()
    if resources.final_file is not None:
        resources.final_file.flush()

    return {
        "type": "propagate_with_detector_result",
        "status": "ok",
        "frames_processed": len(propagated_frames),
        "frames_corrected": len(corrected_frames),
        "corrected_frame_indices": corrected_frames,
    }


def handle_reset_frame(
    params: dict,
    segmentor: StreamingSegmentor,
) -> dict:
    """Reset a single frame (clear SAM memory only).

    H5 files are cleared by the FastAPI side (video_service.delete_frame_data)
    before this command is sent. This only clears the in-memory SAM state.

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

    # Clear from StreamingSegmentor memory only
    if str(video_id) in segmentor.sessions:
        segmentor.reset_frame(str(video_id), frame_idx)

    return {
        "type": "reset_frame_result",
        "status": "ok",
        "frame_idx": frame_idx,
    }


def handle_reset_video(
    params: dict,
    segmentor: StreamingSegmentor,
) -> dict:
    """Reset SAM memory for a video.

    H5 file operations are handled by FastAPI side before this command.
    This only clears in-memory SAM state.

    Args:
        params: Command params with video_id
        segmentor: StreamingSegmentor instance

    Returns:
        Response dict
    """
    video_id = params["video_id"]

    if segmentor is None:
        raise RuntimeError("Model not loaded")

    # Only clear SAM memory, no H5 operations
    if str(video_id) in segmentor.sessions:
        segmentor.reset_video(str(video_id))

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
        if resources.final_file is not None:
            resources.final_file.close()


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
