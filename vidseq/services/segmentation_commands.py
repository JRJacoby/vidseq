"""SAM2 TCP command handlers using StreamingSegmentor.

Each function handles one command type. The worker maintains a single
StreamingSegmentor instance and manages file handles externally.
"""

import base64
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from vidseq.services.array_storage import (
    tracker_masks,
    tracker_logits,
    detector_masks,
    final_masks,
)
from vidseq.services.segmentation_model.streaming_segmentor import SAM2StreamingSegmentor as StreamingSegmentor


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
    """Resources for an open video session.

    H5 file handles are NOT cached here - each handler uses context managers
    to open them as needed. Only frame_source and metadata are cached.
    """
    frame_source: VideoFrameSource
    project_path: Path
    video_id: int
    num_frames: int
    height: int
    width: int


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

    frame_source = None
    try:
        frame_source = VideoFrameSource(video_path)

        # Open tracker masks just for session init, then close
        with tracker_masks(project_path, video_id, "r") as mask_data:
            segmentor.open_video(
                video_id=str(video_id),
                frame_dims=(height, width),
                cond_frame_indices=cond_frame_indices,
                frames=frame_source,
                masks=mask_data,
            )

        resources = VideoResources(
            frame_source=frame_source,
            project_path=project_path,
            video_id=video_id,
            num_frames=num_frames,
            height=height,
            width=width,
        )
        _video_resources[video_id] = resources

    except Exception:
        if frame_source is not None:
            frame_source.close()
        raise

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

    # Convert normalized coords to pixel coords
    px = x * resources.width
    py = y * resources.height

    with tracker_masks(resources.project_path, video_id, "a") as mask_data, \
         tracker_logits(resources.project_path, video_id, "a") as logits_data:
        # Get mask before for comparison
        mask_before = np.array(mask_data[frame_idx])
        before_sum = int(mask_before.sum())

        # Run segmentation - returns mask and logits
        mask, logits = segmentor.add_point_prompt(
            video_id=str(video_id),
            frame_idx=frame_idx,
            location=(px, py),
            label=label,
            frames=resources.frame_source,
            masks=mask_data,
        )

        # Write results to HDF5
        mask_data[frame_idx] = mask
        logits_data[frame_idx] = logits

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
    height, width = resources.height, resources.width

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

    with tracker_masks(resources.project_path, video_id, "a") as mask_data, \
         tracker_logits(resources.project_path, video_id, "a") as logits_data:
        # Get mask before for comparison
        mask_before = np.array(mask_data[frame_idx])
        before_sum = int(mask_before.sum())

        # Read previous logits
        prev_logits = np.array(logits_data[frame_idx])

        # Run refinement with all points - returns mask and logits
        mask, logits = segmentor.refine_mask(
            video_id=str(video_id),
            frame_idx=frame_idx,
            location=locations,
            label=labels,
            frames=resources.frame_source,
            masks=mask_data,
            prev_logits=prev_logits,
        )

        # Write results to HDF5
        mask_data[frame_idx] = mask
        logits_data[frame_idx] = logits

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

    with tracker_masks(resources.project_path, video_id, "a") as mask_data, \
         tracker_logits(resources.project_path, video_id, "a") as logits_data:
        # Propagate - returns mask and logits
        mask, logits = segmentor.propagate(
            video_id=str(video_id),
            frame_idx=frame_idx,
            frames=resources.frame_source,
            masks=mask_data,
        )

        # Write results to HDF5
        mask_data[frame_idx] = mask
        logits_data[frame_idx] = logits

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

    with tracker_masks(resources.project_path, video_id, "a") as mask_data, \
         tracker_logits(resources.project_path, video_id, "a") as logits_data:
        # Callback to write each result to HDF5
        def on_result(frame_idx: int, mask: np.ndarray, logits: np.ndarray) -> None:
            mask_data[frame_idx] = mask
            logits_data[frame_idx] = logits

        frame_indices = segmentor.propagate_sequential(
            video_id=str(video_id),
            start_frame=start_frame_idx,
            num_frames=max_frames,
            frames=resources.frame_source,
            masks=mask_data,
            on_result=on_result,
            progress_interval=50,
        )

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
    """Propagate tracking with on-the-fly detection.

    Loads detector model and runs detection every check_interval frames.
    """
    import torch
    from vidseq.services.detector_model import SegFormerDetector

    video_id = params["video_id"]
    num_frames = params["num_frames"]
    project_path = Path(params["project_path"])
    check_interval = params.get("check_interval", 10)
    iou_threshold = params.get("iou_threshold", 0.5)

    if segmentor is None:
        raise RuntimeError("Model not loaded")

    if video_id not in _video_resources:
        raise RuntimeError(f"No session for video {video_id}")

    resources = _video_resources[video_id]

    # Check detector model exists
    model_path = project_path / "models" / "detector.pt"
    if not model_path.exists():
        raise RuntimeError("No trained detector model found. Train first.")

    # Load detector model with cleanup on exit
    detector = None
    try:
        print(f"[SAM Worker] Loading detector model from {model_path}")
        detector = SegFormerDetector(device="cuda")
        detector.load_decoder(str(model_path))
        detector.eval()
        detector = torch.compile(detector, mode="max-autotune", fullgraph=True)

        # GPU preprocessing constants
        IMG_MEAN = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
        IMG_STD = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)

        def get_detector_mask(_frame_idx: int, frame: np.ndarray) -> np.ndarray:
            """Run detector on a single frame."""
            # GPU preprocessing
            frame_gpu = torch.from_numpy(frame).to("cuda")
            pixel_values = frame_gpu[..., [2, 1, 0]].permute(2, 0, 1).float().div_(255.0)
            pixel_values = pixel_values.unsqueeze(0)
            pixel_values = (pixel_values - IMG_MEAN) / IMG_STD

            # Inference
            with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                logits = detector(pixel_values)

            # Post-process: argmax, resize, to numpy
            pred = logits.argmax(dim=1)[0]  # (H/4, W/4)
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(0).unsqueeze(0).float(),
                size=(frame.shape[0], frame.shape[1]),
                mode="nearest"
            )[0, 0]
            mask = (pred * 255).to(torch.uint8).cpu().numpy()
            return mask

        def on_progress(frame_idx: int) -> None:
            if frame_idx % 50 == 0 or frame_idx == num_frames - 1:
                response_callback({
                    "type": "progress",
                    "frame_idx": frame_idx,
                    "total": num_frames,
                })

        # Open arrays with locking using array_storage context managers
        with tracker_masks(project_path, video_id, "a") as trk_mask_data, \
             detector_masks(project_path, video_id, "a") as det_mask_data, \
             final_masks(project_path, video_id, "a") as fin_mask_data:

            segmentor.propagate_with_detector(
                video_id=str(video_id),
                num_frames=num_frames,
                frames=resources.frame_source,
                get_detector_mask=get_detector_mask,
                tracker_masks=trk_mask_data,
                detector_masks=det_mask_data,
                final_masks=fin_mask_data,
                on_progress=on_progress,
                check_interval=check_interval,
                iou_threshold=iou_threshold,
            )

        return {
            "type": "propagate_with_detector_result",
            "status": "ok",
        }

    finally:
        # Clean up detector model to free GPU memory
        if detector is not None:
            del detector
            torch.cuda.empty_cache()


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
        # H5 files are not cached in VideoResources anymore - h5_storage manages them


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
