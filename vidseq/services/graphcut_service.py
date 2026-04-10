"""Graph cut segmentation service.

Builds a 3D spatio-temporal graph over a chunk of video frames and solves
maxflow to produce binary masks. Runs on CPU — no GPU or TCP worker needed.
"""

import logging
from pathlib import Path

import cv2
import maxflow
import numpy as np

from vidseq.services.array_storage import tracker_masks
from vidseq.services.video_io import VideoFrameSource

logger = logging.getLogger(__name__)

GRAPHCUT_CHUNK_SIZE = 150


def run_graphcut(
    project_path: Path,
    video_id: int,
    video_path: str,
    start_frame: int,
    end_frame: int,
    seeds: dict[str, list[dict]],
) -> int:
    """Run spatio-temporal graph cut segmentation on a video chunk.

    Args:
        project_path: Path to the project directory.
        video_id: Video ID for H5 storage.
        video_path: Path to the video file.
        start_frame: First frame index (inclusive).
        end_frame: Last frame index (inclusive).
        seeds: Sparse seed dict {frame_idx_str: [{x, y, label}, ...]}.

    Returns:
        Number of frames processed.
    """
    # 1. Read frames and convert to grayscale
    frames = _read_frames(video_path, start_frame, end_frame)
    T, H, W = frames.shape

    # 2. Build seed volume
    seed_vol = _build_seed_volume(seeds, start_frame, T, H, W)

    # 3. Compute beta from neighbor intensity differences
    beta = _compute_beta(frames)

    # 4. Build graph, solve, extract masks
    masks = _solve_graphcut(frames, seed_vol, beta)

    # 5. Write masks to H5
    with tracker_masks(project_path, video_id, mode="a") as h5:
        for t in range(T):
            h5[start_frame + t] = masks[t]

    logger.info("Graph cut: wrote %d masks for video %d (frames %d-%d)", T, video_id, start_frame, end_frame)
    return T


def _read_frames(video_path: str, start_frame: int, end_frame: int) -> np.ndarray:
    """Read frames and convert to single-channel grayscale float32."""
    with VideoFrameSource(video_path) as src:
        frame_list = []
        for idx in range(start_frame, end_frame + 1):
            bgr = src[idx]
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
            frame_list.append(gray)
    return np.stack(frame_list)  # (T, H, W)


def _build_seed_volume(
    seeds: dict[str, list[dict]],
    start_frame: int,
    T: int,
    H: int,
    W: int,
) -> np.ndarray:
    """Build a 3D seed volume from sparse seed dict.

    Returns array of shape (T, H, W) with 0=no seed, 1=foreground, 2=background.
    """
    vol = np.zeros((T, H, W), dtype=np.uint8)
    for frame_str, points in seeds.items():
        frame_idx = int(frame_str)
        t = frame_idx - start_frame
        if t < 0 or t >= T:
            continue
        for p in points:
            x, y, label = p["x"], p["y"], p["label"]
            if 0 <= x < W and 0 <= y < H:
                vol[t, y, x] = label
    return vol


def _compute_beta(frames: np.ndarray) -> float:
    """Compute beta = 1 / (2 * mean(squared differences)) over all neighbor pairs."""
    diffs_sq = []

    # Spatial horizontal
    dh = (frames[:, :, :-1] - frames[:, :, 1:]) ** 2
    diffs_sq.append(dh.ravel())

    # Spatial vertical
    dv = (frames[:, :-1, :] - frames[:, 1:, :]) ** 2
    diffs_sq.append(dv.ravel())

    # Temporal
    if frames.shape[0] > 1:
        dt = (frames[:-1] - frames[1:]) ** 2
        diffs_sq.append(dt.ravel())

    mean_sq = np.concatenate(diffs_sq).mean()
    if mean_sq < 1e-10:
        return 0.0  # Constant image — all edges get weight 1
    return 1.0 / (2.0 * mean_sq)


def _solve_graphcut(
    frames: np.ndarray,
    seed_vol: np.ndarray,
    beta: float,
) -> np.ndarray:
    """Build 3D graph, solve maxflow, return binary masks.

    Args:
        frames: (T, H, W) float32 grayscale.
        seed_vol: (T, H, W) uint8, 0=none, 1=fg, 2=bg.
        beta: Edge weight parameter.

    Returns:
        (T, H, W) uint8 binary masks (0 or 1).
    """
    T, H, W = frames.shape
    num_nodes = T * H * W

    g = maxflow.Graph[float](num_nodes, num_nodes * 6)
    node_ids = g.add_grid_nodes((T, H, W))

    # PyMaxflow's add_grid_edges expects weights with the same shape as the
    # node grid. The weight at (t,h,w) is used for the edge from node (t,h,w)
    # to its neighbor; boundary nodes with no neighbor are ignored internally.
    # We compute difference-based weights and pad to full grid shape.

    # Spatial horizontal edges (axis=2: W dimension)
    w_horiz_diff = np.exp(-beta * (frames[:, :, :-1] - frames[:, :, 1:]) ** 2)
    w_horiz = np.pad(w_horiz_diff, ((0, 0), (0, 0), (0, 1)), constant_values=0)
    struct_horiz = np.zeros((3, 3, 3), dtype=int)
    struct_horiz[1, 1, 2] = 1  # center to right neighbor
    g.add_grid_edges(node_ids, weights=w_horiz, structure=struct_horiz, symmetric=True)

    # Spatial vertical edges (axis=1: H dimension)
    w_vert_diff = np.exp(-beta * (frames[:, :-1, :] - frames[:, 1:, :]) ** 2)
    w_vert = np.pad(w_vert_diff, ((0, 0), (0, 1), (0, 0)), constant_values=0)
    struct_vert = np.zeros((3, 3, 3), dtype=int)
    struct_vert[1, 2, 1] = 1  # center to bottom neighbor
    g.add_grid_edges(node_ids, weights=w_vert, structure=struct_vert, symmetric=True)

    # Temporal edges (axis=0: T dimension)
    if T > 1:
        w_temp_diff = np.exp(-beta * (frames[:-1] - frames[1:]) ** 2)
        w_temp = np.pad(w_temp_diff, ((0, 1), (0, 0), (0, 0)), constant_values=0)
        struct_temp = np.zeros((3, 3, 3), dtype=int)
        struct_temp[2, 1, 1] = 1  # center to next frame
        g.add_grid_edges(node_ids, weights=w_temp, structure=struct_temp, symmetric=True)

    # Terminal edges (seeds)
    K = 1.0 + 1.0 * 6.0  # 1 + max_edge_weight * max_degree

    source_cap = np.zeros((T, H, W), dtype=np.float64)
    sink_cap = np.zeros((T, H, W), dtype=np.float64)
    source_cap[seed_vol == 1] = K
    sink_cap[seed_vol == 2] = K

    g.add_grid_tedges(node_ids, source_cap, sink_cap)

    g.maxflow()

    # PyMaxflow convention: get_grid_segments returns True for source side,
    # which is BACKGROUND in image segmentation. Invert for foreground mask.
    segments = g.get_grid_segments(node_ids)
    return (~segments).astype(np.uint8)
