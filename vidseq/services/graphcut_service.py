"""Threshold segmentation service.

Thresholds video frames and uses 3D connected component labeling to
produce binary masks. User sets threshold and clicks to select which
component to keep. Runs on CPU — no GPU or TCP worker needed.
"""

import logging
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

from vidseq.services.array_storage import tracker_masks
from vidseq.services.video_io import VideoFrameSource

logger = logging.getLogger(__name__)

CHUNK_SIZE = 150


def run_threshold_segment(
    project_path: Path,
    video_id: int,
    video_path: str,
    start_frame: int,
    end_frame: int,
    threshold: float,
    click_x: int,
    click_y: int,
    click_frame: int,
) -> int:
    """Threshold video chunk and keep the 3D connected component at the click point.

    Args:
        project_path: Path to the project directory.
        video_id: Video ID for H5 storage.
        video_path: Path to the video file.
        start_frame: First frame index (inclusive).
        end_frame: Last frame index (inclusive).
        threshold: Intensity threshold in [0, 255] (uint8 scale).
        click_x: X pixel coordinate of component selection click.
        click_y: Y pixel coordinate of component selection click.
        click_frame: Absolute frame index of the click.

    Returns:
        Number of frames processed.
    """
    # 1. Read frames as grayscale uint8
    frames = _read_frames(video_path, start_frame, end_frame)
    T, H, W = frames.shape

    # 2. Threshold
    binary = (frames > threshold).astype(np.uint8)

    # 3. 3D connected component labeling (6-connectivity: face-adjacent only)
    structure = ndimage.generate_binary_structure(3, 1)  # 6-connected
    labels, num_components = ndimage.label(binary, structure=structure)

    # 4. Find which component the click is in
    t = click_frame - start_frame
    if t < 0 or t >= T or click_y < 0 or click_y >= H or click_x < 0 or click_x >= W:
        raise ValueError(f"Click point ({click_x}, {click_y}, frame {click_frame}) is outside the chunk")

    selected_label = labels[t, click_y, click_x]
    if selected_label == 0:
        raise ValueError(
            f"Click point ({click_x}, {click_y}) is below threshold on frame {click_frame}. "
            "Click on a bright region or lower the threshold."
        )

    # 5. Extract mask for selected component
    masks = (labels == selected_label).astype(np.uint8)

    print(
        f"[threshold] {T} frames, threshold={threshold}, "
        f"{num_components} components, selected={selected_label}, "
        f"fg pixels={int(masks.sum())}, frames with fg={int((masks.sum(axis=(1, 2)) > 0).sum())}/{T}"
    )

    # 6. Write masks to H5
    with tracker_masks(project_path, video_id, mode="a") as h5:
        for t_idx in range(T):
            h5[start_frame + t_idx] = masks[t_idx]

    return T


def _read_frames(video_path: str, start_frame: int, end_frame: int) -> np.ndarray:
    """Read frames as single-channel grayscale uint8."""
    with VideoFrameSource(video_path) as src:
        frame_list = []
        for idx in range(start_frame, end_frame + 1):
            bgr = src[idx]
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            frame_list.append(gray)
    return np.stack(frame_list)  # (T, H, W), uint8
