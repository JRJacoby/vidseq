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
    compute_bbox_from_mask,
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
    """Load SAM2 model via StreamingSegmentor.

    Returns:
        Tuple of (response_dict, segmentor)
    """
    global _segmentor

    # Skip if already loaded
    if _segmentor is not None:
        print("[Segmentation Worker] Model already loaded, skipping")
        return {"type": "status", "status": "ready"}, _segmentor

    print(f"[Segmentation Worker] Loading model via StreamingSegmentor...")

    _segmentor = StreamingSegmentor(device="cuda")

    print("[Segmentation Worker] Model loaded!")
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

    print(f"[Segmentation Worker] Initializing session for video {video_id}...")

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
        mask_before = mask_data[frame_idx]
        before_sum = int(mask_before.sum())

        # Run segmentation - returns mask and logits
        mask, logits, score = segmentor.add_point_prompt(
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
    print(f"[Segmentation Worker] add_prompt frame={frame_idx} label={label} "
          f"point=({px:.1f}, {py:.1f}) mask_sum: {before_sum} -> {after_sum}")

    return {
        "type": "add_prompt_result",
        "status": "ok",
        "mask_rle": encode_mask_rle(mask),
        "mask_shape": mask.shape,
        "mask_dtype": str(mask.dtype),
        "score": score,
    }


def handle_add_box_prompt(
    params: dict,
    segmentor: StreamingSegmentor,
) -> dict:
    """Add bounding box prompt to a frame.

    Args:
        params: Command params with video_id, frame_idx, x1, y1, x2, y2
                (all coordinates normalized [0, 1])
        segmentor: StreamingSegmentor instance

    Returns:
        Response dict with mask_rle
    """
    video_id = params["video_id"]
    frame_idx = params["frame_idx"]
    x1 = params["x1"]  # normalized [0, 1]
    y1 = params["y1"]
    x2 = params["x2"]
    y2 = params["y2"]

    if segmentor is None:
        raise RuntimeError("Model not loaded")

    if video_id not in _video_resources:
        raise RuntimeError(f"No session for video {video_id}")

    resources = _video_resources[video_id]

    # Convert normalized coords to pixel coords
    px1 = x1 * resources.width
    py1 = y1 * resources.height
    px2 = x2 * resources.width
    py2 = y2 * resources.height

    with tracker_masks(resources.project_path, video_id, "a") as mask_data, \
         tracker_logits(resources.project_path, video_id, "a") as logits_data:
        mask_before = mask_data[frame_idx]
        before_sum = int(mask_before.sum())

        mask, logits, score = segmentor.add_box_prompt(
            video_id=str(video_id),
            frame_idx=frame_idx,
            box=(px1, py1, px2, py2),
            frames=resources.frame_source,
            masks=mask_data,
        )

        # Write both mask and logits — logits are required for point refinement
        mask_data[frame_idx] = mask
        logits_data[frame_idx] = logits

    after_sum = int(mask.sum())
    print(f"[Segmentation Worker] add_box_prompt frame={frame_idx} "
          f"box=({px1:.1f}, {py1:.1f}, {px2:.1f}, {py2:.1f}) "
          f"mask_sum: {before_sum} -> {after_sum}")

    return {
        "type": "add_box_prompt_result",
        "status": "ok",
        "mask_rle": encode_mask_rle(mask),
        "mask_shape": mask.shape,
        "mask_dtype": str(mask.dtype),
        "score": score,
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
        mask_before = mask_data[frame_idx]
        before_sum = int(mask_before.sum())

        # Read previous logits
        prev_logits = logits_data[frame_idx]

        # Run refinement with all points - returns mask and logits
        mask, logits, score = segmentor.refine_mask(
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
    print(f"[Segmentation Worker] refine_mask frame={frame_idx} "
          f"num_points={len(locations)} mask_sum: {before_sum} -> {after_sum}")

    return {
        "type": "refine_mask_result",
        "status": "ok",
        "mask_rle": encode_mask_rle(mask),
        "mask_shape": mask.shape,
        "mask_dtype": str(mask.dtype),
        "score": score,
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
        mask, logits, score = segmentor.propagate(
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
        "score": score,
    }


def handle_generate_training_masks(
    params: dict,
    segmentor: StreamingSegmentor,
    response_callback: Callable | None = None,
) -> dict:
    """Generate training masks by propagating through video.

    Args:
        params: Command params with video_id, start_frame_idx, max_frames
        segmentor: StreamingSegmentor instance
        response_callback: Optional callback to send progress messages

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

    scores: list[list] = []
    frames_propagated = 0

    with tracker_masks(resources.project_path, video_id, "a") as mask_data, \
         tracker_logits(resources.project_path, video_id, "a") as logits_data:
        # Callback to write each result to HDF5
        def on_result(frame_idx: int, mask: np.ndarray, logits: np.ndarray, score: float) -> None:
            nonlocal frames_propagated
            mask_data[frame_idx] = mask
            logits_data[frame_idx] = logits
            scores.append([frame_idx, score])
            frames_propagated += 1
            if response_callback and frames_propagated % 50 == 0:
                response_callback({
                    "type": "progress",
                    "frame_idx": frames_propagated,
                    "total": max_frames,
                })

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
        "scores": scores,
    }


def handle_apply_detector(
    params: dict,
    segmentor: StreamingSegmentor,
    response_callback: Callable | None = None,
) -> dict:
    """Run trained detector on every frame of given videos (no SAM2).

    Loads the detector on GPU, iterates videos in batches of 32 frames,
    and returns bboxes + scores grouped by video ID.
    """
    import torch
    from vidseq.services.detector_model import load_finetuned, DETECTION_CONF_THRESHOLD

    video_ids = params["video_ids"]
    project_path = Path(params["project_path"])
    video_paths = params["video_paths"]
    detector_type = params.get("detector_type", "detector")

    weights_name = {"obb": "obb_detector.pt", "seg": "seg_detector.pt"}.get(detector_type, "detector.pt")
    model_path = project_path / "models" / weights_name
    if not model_path.exists():
        raise RuntimeError(f"No trained {'OBB ' if detector_type == 'obb' else ''}detector model found. Train first.")

    detector = None
    try:
        detector = load_finetuned(model_path, device="cuda")

        all_scores: dict[str, list[list]] = {}
        all_bboxes: dict[str, list[list]] = {}

        for video_id, video_path in zip(video_ids, video_paths):
            vid_key = str(video_id)
            scores: list[list] = []
            bboxes: list[list] = []

            with VideoFrameSource(video_path) as frame_source:
                num_frames = frame_source.frame_count
                orig_w = frame_source.width
                orig_h = frame_source.height
                batch_size = 32
                frames_done = 0

                # Open H5 for seg mask writes
                mask_ctx = detector_masks(project_path, video_id, "a") if detector_type == "seg" else None
                mask_data = mask_ctx.__enter__() if mask_ctx else None
                try:
                    for batch_start in range(0, num_frames, batch_size):
                        batch_end = min(batch_start + batch_size, num_frames)
                        batch_frames = [frame_source[i] for i in range(batch_start, batch_end)]
                        batch_indices = list(range(batch_start, batch_end))

                        results = detector(batch_frames, conf=DETECTION_CONF_THRESHOLD, verbose=False)

                        for idx, result in zip(batch_indices, results):
                            if detector_type == "seg":
                                if result.masks is not None and len(result.masks) > 0 and result.boxes is not None and len(result.boxes) > 0:
                                    best_i = result.boxes.conf.argmax()
                                    mask = result.masks.data[best_i].cpu().numpy()
                                    mask_binary = (mask > 0.5).astype(np.uint8) * 255
                                    mask_resized = cv2.resize(mask_binary, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                                    mask_data[idx] = mask_resized
                                    x1, y1, x2, y2 = result.boxes.xyxy[best_i].cpu().tolist()
                                    conf = result.boxes.conf[best_i].item()
                                    scores.append([idx, conf])
                                    bboxes.append([idx, x1, y1, x2, y2])
                                else:
                                    scores.append([idx, 0.0])
                            elif detector_type == "obb":
                                if result.obb is not None and len(result.obb) > 0:
                                    best_i = result.obb.conf.argmax()
                                    corners = result.obb.xyxyxyxy[best_i].cpu().tolist()
                                    conf = result.obb.conf[best_i].item()
                                    scores.append([idx, conf])
                                    bboxes.append([idx, *corners[0], *corners[1], *corners[2], *corners[3]])
                                else:
                                    scores.append([idx, 0.0])
                            else:
                                if result.boxes is not None and len(result.boxes) > 0:
                                    best_i = result.boxes.conf.argmax()
                                    x1, y1, x2, y2 = result.boxes.xyxy[best_i].cpu().tolist()
                                    conf = result.boxes.conf[best_i].item()
                                    scores.append([idx, conf])
                                    bboxes.append([idx, x1, y1, x2, y2])
                                else:
                                    scores.append([idx, 0.0])

                        prev_done = frames_done
                        frames_done += len(batch_frames)
                        if response_callback and (frames_done // 50 > prev_done // 50):
                            response_callback({
                                "type": "progress",
                                "frame_idx": frames_done,
                                "total": num_frames,
                                "video_id": video_id,
                            })
                finally:
                    if mask_ctx:
                        mask_ctx.__exit__(None, None, None)

            all_scores[vid_key] = scores
            all_bboxes[vid_key] = bboxes

        score_key = "obb_scores" if detector_type == "obb" else "detector_scores"
        bbox_key = "obb_bboxes" if detector_type == "obb" else "detector_bboxes"

        return {
            "type": "apply_detector_result",
            "status": "ok",
            score_key: all_scores,
            bbox_key: all_bboxes,
        }

    finally:
        if detector is not None:
            del detector
            torch.cuda.empty_cache()


def handle_propagate_with_detector(
    params: dict,
    segmentor: StreamingSegmentor,
    response_callback: Callable[[dict], None],
) -> dict:
    """Propagate tracking with on-the-fly detector bbox detection."""
    import torch
    from vidseq.services.detector_model import load_finetuned, detect, pick_best_detection

    video_id = params["video_id"]
    num_frames = params["num_frames"]
    project_path = Path(params["project_path"])
    check_interval = params.get("check_interval", 10)
    iou_threshold = params.get("iou_threshold", 0.7)
    training_frame_indices = params.get("training_frame_indices", [])

    if segmentor is None:
        raise RuntimeError("Model not loaded")

    if video_id not in _video_resources:
        raise RuntimeError(f"No session for video {video_id}")

    resources = _video_resources[video_id]

    model_path = project_path / "models" / "detector.pt"
    if not model_path.exists():
        raise RuntimeError("No trained detector model found. Train first.")

    detector = None
    try:
        print(f"[Segmentation Worker] Loading detector from {model_path}")
        detector = load_finetuned(model_path, device="cuda")

        detector_scores: dict[int, float] = {}
        detector_bboxes: dict[int, tuple[float, float, float, float]] = {}

        def get_detector_bbox(
            frame_idx: int, frame: np.ndarray,
            tracker_bbox_hint: tuple | None = None,
        ) -> tuple[tuple[float, float, float, float] | None, float]:
            """Run detector on a single frame, return (bbox, conf) or (None, 0.0)."""
            detections = detect(detector, frame)
            bbox, conf = pick_best_detection(detections, tracker_bbox_hint)
            return bbox, conf

        def on_progress(frame_idx: int) -> None:
            if frame_idx % 50 == 0 or frame_idx == num_frames - 1:
                response_callback({
                    "type": "progress",
                    "frame_idx": frame_idx,
                    "total": num_frames,
                })

        with tracker_masks(project_path, video_id, "a") as trk_mask_data, \
             final_masks(project_path, video_id, "a") as fin_mask_data:

            scores: dict[int, float] = {}
            segmentor.propagate_with_detector(
                video_id=str(video_id),
                num_frames=num_frames,
                frames=resources.frame_source,
                get_detector_bbox=get_detector_bbox,
                tracker_masks=trk_mask_data,
                final_masks=fin_mask_data,
                on_progress=on_progress,
                check_interval=check_interval,
                iou_threshold=iou_threshold,
                scores=scores,
                detector_bboxes=detector_bboxes,
                detector_scores=detector_scores,
                training_frame_indices=training_frame_indices,
            )

        return {
            "type": "propagate_with_detector_result",
            "status": "ok",
            "scores": [[idx, s] for idx, s in scores.items()],
            "detector_scores": [[idx, s] for idx, s in detector_scores.items()],
            "detector_bboxes": [
                [idx, *bbox] for idx, bbox in detector_bboxes.items()
            ],
        }

    finally:
        if detector is not None:
            del detector
            torch.cuda.empty_cache()


def handle_simple_propagate_with_detector(
    params: dict,
    segmentor: StreamingSegmentor,
    response_callback: Callable[[dict], None],
) -> dict:
    """Simple propagate: forward SAM2 with periodic detector re-prompting."""
    import torch
    from vidseq.services.detector_model import load_finetuned, detect, pick_best_detection

    video_id = params["video_id"]
    num_frames = params["num_frames"]
    project_path = Path(params["project_path"])
    reprompt_interval = params.get("reprompt_interval", 30)

    if segmentor is None:
        raise RuntimeError("Model not loaded")
    if video_id not in _video_resources:
        raise RuntimeError(f"No session for video {video_id}")

    resources = _video_resources[video_id]

    model_path = project_path / "models" / "detector.pt"
    if not model_path.exists():
        raise RuntimeError("No trained detector model found. Train first.")

    detector = None
    try:
        detector = load_finetuned(model_path, device="cuda")

        detector_scores: dict[int, float] = {}
        detector_bboxes: dict[int, tuple[float, float, float, float]] = {}

        def get_detector_bbox(
            frame_idx: int, frame: np.ndarray,
        ) -> tuple[tuple[float, float, float, float] | None, float]:
            detections = detect(detector, frame)
            bbox, conf = pick_best_detection(detections, None)
            return bbox, conf

        def on_progress(frame_idx: int) -> None:
            if frame_idx % 50 == 0 or frame_idx == num_frames - 1:
                response_callback({
                    "type": "progress",
                    "frame_idx": frame_idx,
                    "total": num_frames,
                })

        with tracker_masks(project_path, video_id, "a") as trk_mask_data, \
             final_masks(project_path, video_id, "a") as fin_mask_data:

            scores: dict[int, float] = {}
            segmentor.simple_propagate_with_detector(
                video_id=str(video_id),
                num_frames=num_frames,
                frames=resources.frame_source,
                get_detector_bbox=get_detector_bbox,
                tracker_masks=trk_mask_data,
                final_masks=fin_mask_data,
                on_progress=on_progress,
                reprompt_interval=reprompt_interval,
                scores=scores,
                detector_bboxes=detector_bboxes,
                detector_scores=detector_scores,
            )

        return {
            "type": "simple_propagate_with_detector_result",
            "status": "ok",
            "scores": [[idx, s] for idx, s in scores.items()],
            "detector_scores": [[idx, s] for idx, s in detector_scores.items()],
            "detector_bboxes": [
                [idx, *bbox] for idx, bbox in detector_bboxes.items()
            ],
        }

    finally:
        if detector is not None:
            del detector
            torch.cuda.empty_cache()


def handle_propagate_with_associated(
    params: dict,
    segmentor: StreamingSegmentor,
    response_callback: Callable[[dict], None],
) -> dict:
    """Propagate associated video segmentation using main video's bbox prompts.

    The main video's scores determine when to provide box prompts from its masks.
    Uses sparse conditioning + LRU eviction for memory management on long videos.
    """
    from collections import deque

    video_id = params["video_id"]
    main_video_id = params["main_video_id"]
    project_path = Path(params["project_path"])
    confidence_threshold = params["confidence_threshold"]
    main_height = params["main_height"]
    main_width = params["main_width"]
    main_video_scores = params["main_video_scores"]  # {str(frame_idx): float}
    # Note: JSON keys are strings, convert to int
    main_scores = {int(k): v for k, v in main_video_scores.items()}
    num_frames = params["num_frames"]
    cond_frame_interval = params.get("cond_frame_interval", 50)
    max_cond_frames = params.get("max_cond_frames", 32)

    if video_id not in _video_resources:
        raise RuntimeError(f"No session for video {video_id}")

    resources = _video_resources[video_id]

    # Check that at least one frame exceeds threshold
    has_valid = any(
        s > confidence_threshold
        for s in main_scores.values()
        if s > 0
    )
    if not has_valid:
        raise RuntimeError(
            "Main video has no segmentation data above confidence threshold"
        )

    # Compute bbox rescaling factors
    assoc_height, assoc_width = resources.height, resources.width
    x_scale = assoc_width / main_width if main_width != assoc_width else 1.0
    y_scale = assoc_height / main_height if main_height != assoc_height else 1.0
    needs_rescale = x_scale != 1.0 or y_scale != 1.0

    # LRU tracking for conditioning frames
    cond_lru: deque[int] = deque()
    prompted_count = 0  # Counts prompted frames for interval logic

    depth_scores: dict[int, float] = {}

    # Open H5 files: main video read-only, associated video read-write
    # This direct H5 access is a deliberate exception — see spec for justification
    with tracker_masks(project_path, main_video_id, "r") as main_masks, \
         tracker_masks(project_path, video_id, "a") as depth_trk, \
         final_masks(project_path, video_id, "a") as depth_fin:

        frame_source = resources.frame_source

        # 1. Find first frame above threshold
        start_frame = None
        for idx in range(num_frames):
            score = main_scores.get(idx, -1.0)
            if score > 0 and score > confidence_threshold:
                start_frame = idx
                break

        if start_frame is None:
            raise RuntimeError(
                "Main video has no segmentation data above confidence threshold"
            )

        # 2. Get bbox from main video's mask for the start frame
        main_mask = np.asarray(main_masks[start_frame])
        bbox = compute_bbox_from_mask(main_mask)
        if bbox is None:
            raise RuntimeError(f"Main video mask at frame {start_frame} is empty")

        if needs_rescale:
            bbox = np.array([
                bbox[0] * x_scale, bbox[1] * y_scale,
                bbox[2] * x_scale, bbox[3] * y_scale,
            ], dtype=np.float32)

        # Initialize with box prompt (always conditioning).
        # First frame must use fresh segmentation (no memory exists yet).
        frame = frame_source[start_frame]
        mask, _, score = segmentor.propagate_with_box(
            str(video_id), start_frame, frame,
            box_prompt=tuple(bbox),
            add_as_conditioning=True,
            use_memory_with_prompt=False,
        )
        depth_trk[start_frame] = mask
        depth_fin[start_frame] = mask
        depth_scores[start_frame] = score
        cond_lru.append(start_frame)
        prompted_count = 1
        last_anchor_frame = start_frame
        was_below_threshold = False

        if response_callback:
            response_callback({
                "type": "progress",
                "frame_idx": start_frame,
                "total": num_frames,
            })

        # 3. Main loop: propagate forward
        import time as _time
        import json as _json
        _timing_file = open("/tmp/coseg_timing.jsonl", "w")

        for frame_idx in range(start_frame + 1, num_frames):
            _t_total = _time.perf_counter()

            _t0 = _time.perf_counter()
            frame = frame_source[frame_idx]
            _t_frame_read = _time.perf_counter() - _t0

            main_score = main_scores.get(frame_idx, -1.0)
            is_above = main_score > 0 and main_score >= confidence_threshold

            _t_h5_read = 0.0
            _t_bbox_compute = 0.0
            _t_bbox_rescale = 0.0
            _t_propagate = 0.0
            _t_h5_write = 0.0
            _t_backtrack = 0.0
            _frame_type = "coast"

            if is_above:
                # Get bbox from main video
                _t0 = _time.perf_counter()
                main_mask = np.asarray(main_masks[frame_idx])
                _t_h5_read = _time.perf_counter() - _t0

                _t0 = _time.perf_counter()
                bbox = compute_bbox_from_mask(main_mask)
                _t_bbox_compute = _time.perf_counter() - _t0

                if bbox is not None:
                    if needs_rescale:
                        _t0 = _time.perf_counter()
                        bbox = np.array([
                            bbox[0] * x_scale, bbox[1] * y_scale,
                            bbox[2] * x_scale, bbox[3] * y_scale,
                        ], dtype=np.float32)
                        _t_bbox_rescale = _time.perf_counter() - _t0

                    # 4. Check for confidence recovery (backtrack)
                    if was_below_threshold:
                        _frame_type = "recovery"
                        _t0 = _time.perf_counter()
                        mask, _, score = segmentor.propagate_with_box(
                            str(video_id), frame_idx, frame,
                            box_prompt=tuple(bbox),
                            add_as_conditioning=True,
                            use_memory_with_prompt=True,
                        )
                        _t_propagate = _time.perf_counter() - _t0

                        _t0 = _time.perf_counter()
                        depth_fin[frame_idx] = mask
                        _t_h5_write = _time.perf_counter() - _t0

                        depth_scores[frame_idx] = score
                        cond_lru.append(frame_idx)

                        # Backtrack gap frames
                        if last_anchor_frame + 1 <= frame_idx - 1:
                            _t0 = _time.perf_counter()
                            segmentor.backtrack_reprop(
                                str(video_id), frame_source, depth_fin,
                                last_anchor_frame + 1, frame_idx - 1,
                                scores=depth_scores,
                            )
                            _t_backtrack = _time.perf_counter() - _t0

                        was_below_threshold = False
                    else:
                        _frame_type = "prompted"
                        # Normal prompted frame
                        prompted_count += 1
                        add_cond = (prompted_count % cond_frame_interval) == 0

                        _t0 = _time.perf_counter()
                        mask, _, score = segmentor.propagate_with_box(
                            str(video_id), frame_idx, frame,
                            box_prompt=tuple(bbox),
                            add_as_conditioning=add_cond,
                            use_memory_with_prompt=True,
                        )
                        _t_propagate = _time.perf_counter() - _t0

                        if add_cond:
                            cond_lru.append(frame_idx)
                            # LRU eviction
                            while len(cond_lru) > max_cond_frames:
                                oldest = cond_lru.popleft()
                                segmentor.evict_conditioning_frame(
                                    str(video_id), oldest
                                )

                    _t0 = _time.perf_counter()
                    depth_trk[frame_idx] = mask
                    depth_fin[frame_idx] = mask
                    _t_h5_write += _time.perf_counter() - _t0

                    depth_scores[frame_idx] = score
                    last_anchor_frame = frame_idx
                else:
                    _frame_type = "empty_mask"
                    # Main mask is empty despite score above threshold
                    _t0 = _time.perf_counter()
                    mask, _, score = segmentor.propagate_frame(
                        str(video_id), frame_idx, frame
                    )
                    _t_propagate = _time.perf_counter() - _t0

                    _t0 = _time.perf_counter()
                    depth_trk[frame_idx] = mask
                    depth_fin[frame_idx] = mask
                    _t_h5_write = _time.perf_counter() - _t0

                    depth_scores[frame_idx] = score
                    was_below_threshold = True
            else:
                # Coast: propagate without prompt
                _frame_type = "coast"
                _t0 = _time.perf_counter()
                mask, _, score = segmentor.propagate_frame(
                    str(video_id), frame_idx, frame
                )
                _t_propagate = _time.perf_counter() - _t0

                _t0 = _time.perf_counter()
                depth_trk[frame_idx] = mask
                depth_fin[frame_idx] = mask
                _t_h5_write = _time.perf_counter() - _t0

                depth_scores[frame_idx] = score
                was_below_threshold = True

            _t_total_ms = (_time.perf_counter() - _t_total) * 1000
            _timing_file.write(_json.dumps({
                "f": frame_idx,
                "type": _frame_type,
                "total_ms": round(_t_total_ms, 2),
                "frame_read_ms": round(_t_frame_read * 1000, 2),
                "h5_read_ms": round(_t_h5_read * 1000, 2),
                "bbox_compute_ms": round(_t_bbox_compute * 1000, 2),
                "bbox_rescale_ms": round(_t_bbox_rescale * 1000, 2),
                "propagate_ms": round(_t_propagate * 1000, 2),
                "h5_write_ms": round(_t_h5_write * 1000, 2),
                "backtrack_ms": round(_t_backtrack * 1000, 2),
            }) + "\n")
            if frame_idx % 50 == 0:
                _timing_file.flush()

            # Progress callback
            if response_callback and (frame_idx % 50 == 0 or frame_idx == num_frames - 1):
                response_callback({
                    "type": "progress",
                    "frame_idx": frame_idx,
                    "total": num_frames,
                })

        _timing_file.close()

    return {
        "type": "propagate_with_associated_result",
        "status": "ok",
        "scores": [[idx, s] for idx, s in depth_scores.items()],
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
        # H5 files are not cached in VideoResources anymore - array_storage manages them


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

    print(f"[Segmentation Worker] Closing session for video {video_id}...")

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

    print("[Segmentation Worker] Shutting down...")

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
