"""SAM2++ keypoint tracking TCP command handlers.

Each function handles one command type. The worker maintains a single
SAM2PlusKeypointTracker instance and manages file handles externally.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from vidseq.services.array_storage import keypoint_coords, keypoint_logits
from vidseq.services.segmentation_commands import VideoFrameSource
from vidseq.services.keypoint_tracking_model.streaming_keypoint_tracker import SAM2PlusKeypointTracker


# ---------------------------------------------------------------------------
# H5CoordsSource - wrapper for indexable H5 coords access
# ---------------------------------------------------------------------------


class H5CoordsSource:
    """Wraps H5 coords dataset to provide indexable access for the tracker.

    The tracker expects coords_source[idx] -> np.ndarray (K, 2) float32,
    with NaN for empty keypoints. This class provides that interface over
    an h5py dataset handle.
    """

    def __init__(self, coords_dataset):
        self._ds = coords_dataset

    def __getitem__(self, frame_idx):
        return np.asarray(self._ds[frame_idx])  # (K, 2) float32, may contain NaN


# ---------------------------------------------------------------------------
# VideoResources - per-video session state
# ---------------------------------------------------------------------------


@dataclass
class VideoResources:
    """Resources for an open keypoint tracking video session.

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
_tracker: Optional[SAM2PlusKeypointTracker] = None
_video_resources: dict[int, VideoResources] = {}


# ---------------------------------------------------------------------------
# Helper: read all keypoints for a frame from H5 coords
# ---------------------------------------------------------------------------


def _read_frame_keypoints(coords_ds, frame_idx: int) -> dict:
    """Read all keypoints for a frame and return as {obj_id: {x, y}, ...}.

    Only includes keypoints with non-NaN coordinates.

    Args:
        coords_ds: H5 dataset handle for coords, shape (num_frames, K, 2).
        frame_idx: Frame index to read.

    Returns:
        Dict mapping obj_id (int) to {x: float, y: float}.
    """
    coords = np.asarray(coords_ds[frame_idx])  # (K, 2) float32
    keypoints = {}
    for obj_id in range(coords.shape[0]):
        x, y = coords[obj_id]
        if not (np.isnan(x) or np.isnan(y)):
            keypoints[obj_id] = {"x": float(x), "y": float(y)}
    return keypoints


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------


def handle_load_model(params: dict) -> tuple[dict, SAM2PlusKeypointTracker]:
    """Load SAM2++ keypoint tracking model.

    Returns:
        Tuple of (response_dict, tracker)
    """
    global _tracker

    # Skip if already loaded
    if _tracker is not None:
        print("[Keypoint Worker] Model already loaded, skipping")
        return {"type": "status", "status": "ready"}, _tracker

    print("[Keypoint Worker] Loading SAM2++ keypoint tracker...")

    _tracker = SAM2PlusKeypointTracker(compile_model=True)

    print("[Keypoint Worker] Model loaded!")
    return {"type": "status", "status": "ready"}, _tracker


def handle_init_keypoint_session(
    params: dict,
    tracker: SAM2PlusKeypointTracker,
) -> dict:
    """Initialize keypoint tracking session for a video.

    Args:
        params: Command params with video_id, video_path, project_path,
                height, width, num_frames, cond_frame_data
        tracker: SAM2PlusKeypointTracker instance

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
    cond_frame_data = params.get("cond_frame_data", [])

    print(f"[Keypoint Worker] Initializing session for video {video_id}...")

    if tracker is None:
        raise RuntimeError("Model not loaded")

    # Close existing session if any
    if video_id in _video_resources:
        _close_video_resources(video_id, tracker)

    frame_source = None
    try:
        frame_source = VideoFrameSource(video_path)

        # Open H5 coords in read mode for session init (coords_source for memory reconstruction)
        with keypoint_coords(project_path, video_id, "r") as coords_ds:
            tracker.open_video(
                video_id=str(video_id),
                num_frames=num_frames,
                frame_dims=(height, width),
                cond_frame_data=cond_frame_data,
                frames=frame_source,
                coords_source=H5CoordsSource(coords_ds),
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
        "type": "init_keypoint_session_result",
        "status": "ok",
        "video_id": video_id,
        "num_frames": num_frames,
        "height": height,
        "width": width,
    }


def handle_add_keypoint_prompt(
    params: dict,
    tracker: SAM2PlusKeypointTracker,
) -> dict:
    """Add a keypoint prompt to a frame for a specific object.

    Args:
        params: Command params with video_id, frame_idx, obj_id, x_norm, y_norm
        tracker: SAM2PlusKeypointTracker instance

    Returns:
        Response dict with keypoints for this frame
    """
    video_id = params["video_id"]
    frame_idx = params["frame_idx"]
    obj_id = params["obj_id"]
    x_norm = params["x_norm"]
    y_norm = params["y_norm"]

    if tracker is None:
        raise RuntimeError("Model not loaded")

    if video_id not in _video_resources:
        raise RuntimeError(f"No session for video {video_id}")

    resources = _video_resources[video_id]

    # Read frame from VideoFrameSource
    frame = resources.frame_source[frame_idx]

    with keypoint_coords(resources.project_path, video_id, "a") as coords_ds, \
         keypoint_logits(resources.project_path, video_id, "a") as logits_ds:

        coords_source = H5CoordsSource(coords_ds)

        # Run keypoint prompt through tracker
        pred_x, pred_y, logits = tracker.add_keypoint_prompt(
            video_id=str(video_id),
            frame_idx=frame_idx,
            obj_id=obj_id,
            x_norm=x_norm,
            y_norm=y_norm,
            frame=frame,
            frames=resources.frame_source,
            coords_source=coords_source,
        )

        # Write returned coords and logits to H5
        coords_ds[frame_idx, obj_id] = [pred_x, pred_y]
        logits_ds[frame_idx, obj_id] = logits

        # Read ALL keypoints for this frame to return complete state
        keypoints = _read_frame_keypoints(coords_ds, frame_idx)

    print(f"[Keypoint Worker] add_keypoint_prompt frame={frame_idx} obj={obj_id} "
          f"input=({x_norm:.3f}, {y_norm:.3f}) pred=({pred_x:.3f}, {pred_y:.3f})")

    return {
        "type": "add_keypoint_prompt_result",
        "status": "ok",
        "keypoints": keypoints,
    }


def handle_refine_keypoint(
    params: dict,
    tracker: SAM2PlusKeypointTracker,
) -> dict:
    """Refine an existing keypoint with updated coordinates.

    Similar to add_prompt but reads prev_logits from H5 first.

    Args:
        params: Command params with video_id, frame_idx, obj_id, x_norm, y_norm
        tracker: SAM2PlusKeypointTracker instance

    Returns:
        Response dict with keypoints for this frame
    """
    video_id = params["video_id"]
    frame_idx = params["frame_idx"]
    obj_id = params["obj_id"]
    x_norm = params["x_norm"]
    y_norm = params["y_norm"]

    if tracker is None:
        raise RuntimeError("Model not loaded")

    if video_id not in _video_resources:
        raise RuntimeError(f"No session for video {video_id}")

    resources = _video_resources[video_id]

    # Read frame from VideoFrameSource
    frame = resources.frame_source[frame_idx]

    with keypoint_coords(resources.project_path, video_id, "a") as coords_ds, \
         keypoint_logits(resources.project_path, video_id, "a") as logits_ds:

        coords_source = H5CoordsSource(coords_ds)

        # Read previous logits for this keypoint
        prev_logits = np.asarray(logits_ds[frame_idx, obj_id])  # (256, 256)

        # Run refinement through tracker
        pred_x, pred_y, logits = tracker.refine_keypoint(
            video_id=str(video_id),
            frame_idx=frame_idx,
            obj_id=obj_id,
            x_norm=x_norm,
            y_norm=y_norm,
            frame=frame,
            prev_logits=prev_logits,
            frames=resources.frame_source,
            coords_source=coords_source,
        )

        # Write updated coords and logits back to H5
        coords_ds[frame_idx, obj_id] = [pred_x, pred_y]
        logits_ds[frame_idx, obj_id] = logits

        # Read ALL keypoints for this frame to return complete state
        keypoints = _read_frame_keypoints(coords_ds, frame_idx)

    print(f"[Keypoint Worker] refine_keypoint frame={frame_idx} obj={obj_id} "
          f"input=({x_norm:.3f}, {y_norm:.3f}) pred=({pred_x:.3f}, {pred_y:.3f})")

    return {
        "type": "refine_keypoint_result",
        "status": "ok",
        "keypoints": keypoints,
    }


def handle_propagate_keypoints(
    params: dict,
    tracker: SAM2PlusKeypointTracker,
    response_callback: Callable[[dict], None],
) -> dict:
    """Propagate keypoint tracking forward through video frames.

    Args:
        params: Command params with video_id, start_frame_idx, max_frames
        tracker: SAM2PlusKeypointTracker instance
        response_callback: Callback for streaming progress updates

    Returns:
        Response dict with frames_processed count
    """
    video_id = params["video_id"]
    start_frame_idx = params["start_frame_idx"]
    max_frames = params["max_frames"]

    if tracker is None:
        raise RuntimeError("Model not loaded")

    if video_id not in _video_resources:
        raise RuntimeError(f"No session for video {video_id}")

    resources = _video_resources[video_id]

    print(f"[Keypoint Worker] Propagating keypoints from frame {start_frame_idx} "
          f"for up to {max_frames} frames...")

    # Open H5 files in append mode around the whole propagation loop
    with keypoint_coords(resources.project_path, video_id, "a") as coords_ds, \
         keypoint_logits(resources.project_path, video_id, "a") as logits_ds:

        coords_source = H5CoordsSource(coords_ds)
        frames_processed = [0]  # Mutable for closure

        def on_result(frame_idx: int, results_dict: dict[int, tuple[float, float, np.ndarray]]) -> None:
            """Write propagated results to H5 and send progress."""
            for obj_id, (x_norm, y_norm, logits) in results_dict.items():
                coords_ds[frame_idx, obj_id] = [x_norm, y_norm]
                logits_ds[frame_idx, obj_id] = logits

            frames_processed[0] += 1

            # Send progress every 50 frames
            if frames_processed[0] % 50 == 0:
                response_callback({
                    "type": "progress",
                    "frame_idx": frame_idx,
                })

        propagated = tracker.propagate_sequential(
            video_id=str(video_id),
            start_frame=start_frame_idx,
            num_frames=max_frames,
            frames=resources.frame_source,
            coords_source=coords_source,
            on_result=on_result,
        )

    print(f"[Keypoint Worker] Propagation complete: {len(propagated)} frames processed")

    return {
        "type": "propagate_keypoints_result",
        "status": "ok",
        "frames_processed": len(propagated),
    }


def handle_reset_keypoint_frame(
    params: dict,
    tracker: SAM2PlusKeypointTracker,
) -> dict:
    """Reset a single frame (clear SAM memory only).

    H5 files are cleared by the FastAPI side (service layer)
    before this command is sent. This only clears the in-memory SAM state.

    Args:
        params: Command params with video_id, frame_idx
        tracker: SAM2PlusKeypointTracker instance

    Returns:
        Response dict
    """
    video_id = params["video_id"]
    frame_idx = params["frame_idx"]

    if tracker is None:
        raise RuntimeError("Model not loaded")

    # Clear from tracker memory only
    if str(video_id) in tracker.sessions:
        tracker.reset_frame(str(video_id), frame_idx)

    print(f"[Keypoint Worker] Reset frame {frame_idx} for video {video_id}")

    return {
        "type": "reset_keypoint_frame_result",
        "status": "ok",
        "frame_idx": frame_idx,
    }


def handle_reset_keypoint_video(
    params: dict,
    tracker: SAM2PlusKeypointTracker,
) -> dict:
    """Reset SAM memory for a video.

    H5 file operations are handled by FastAPI side before this command.
    This only clears in-memory SAM state.

    Args:
        params: Command params with video_id
        tracker: SAM2PlusKeypointTracker instance

    Returns:
        Response dict
    """
    video_id = params["video_id"]

    if tracker is None:
        raise RuntimeError("Model not loaded")

    # Only clear SAM memory, no H5 operations
    if str(video_id) in tracker.sessions:
        tracker.reset_video(str(video_id))

    print(f"[Keypoint Worker] Reset video {video_id}")

    return {
        "type": "reset_keypoint_video_result",
        "status": "ok",
    }


def _close_video_resources(video_id: int, tracker: SAM2PlusKeypointTracker) -> None:
    """Close and clean up resources for a video."""
    if video_id in _video_resources:
        tracker.close_video(str(video_id))

        resources = _video_resources.pop(video_id)
        resources.frame_source.close()


def handle_close_keypoint_session(
    params: dict,
    tracker: SAM2PlusKeypointTracker,
) -> dict:
    """Close keypoint tracking session and free resources.

    Args:
        params: Command params with video_id
        tracker: SAM2PlusKeypointTracker instance

    Returns:
        Response dict
    """
    video_id = params["video_id"]

    print(f"[Keypoint Worker] Closing session for video {video_id}...")

    if tracker is not None:
        _close_video_resources(video_id, tracker)

    return {"type": "close_keypoint_session_result", "status": "ok"}


def handle_shutdown(tracker: SAM2PlusKeypointTracker) -> dict:
    """Shutdown server and clean up all sessions.

    Args:
        tracker: SAM2PlusKeypointTracker instance

    Returns:
        Response dict
    """
    global _video_resources

    print("[Keypoint Worker] Shutting down...")

    for video_id in list(_video_resources.keys()):
        try:
            _close_video_resources(video_id, tracker)
        except Exception:
            pass

    _video_resources.clear()

    return {"type": "shutdown_result", "status": "ok"}
