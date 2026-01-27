"""SAM3 video propagation - hot path for mask generation.

This module contains the performance-critical propagation loop.
Keep this code tight - no unnecessary abstractions.
"""

from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
from sqlalchemy.orm import Session

from vidseq.services import frame_data_service, mask_storage
from vidseq.services.database_manager import DatabaseManager


def propagate_video(
    model,
    inference_state: dict,
    video_id: int,
    start_frame_idx: int,
    max_frames: int,
    project_path: Path,
    num_frames: int,
    height: int,
    width: int,
    progress_callback: Optional[Callable[[int], None]] = None,
) -> tuple[int, list[int], dict]:
    """
    Run SAM3 propagation and save masks/scores.

    HOT PATH - keep this function tight. Do not add abstraction layers.

    Masks are saved to per-video HDF5 files (masks/{video_id}.h5).
    Scores are buffered and flushed to SQLite every 100 frames.

    Args:
        model: SAM3 model instance (Sam3VideoInferenceWithInstanceInteractivity)
        inference_state: SAM3 inference state dict
        video_id: Database video ID
        start_frame_idx: Frame to start propagation from
        max_frames: Maximum frames to process
        project_path: Path to project directory
        num_frames: Total frames in video
        height: Video height
        width: Video width
        progress_callback: Optional callback called every 10 frames with frame_idx

    Returns:
        Tuple of (frame_count, frame_indices, stats_dict)
    """
    frame_indices = []
    frame_count = 0

    # Score buffer for batch writes to SQLite
    BATCH_SIZE = 100
    score_buffer: list[tuple[int, float]] = []
    # has_mask buffer for batch writes
    has_mask_buffer: list[int] = []

    # Get sync database session for score writes
    db_manager = DatabaseManager.get_instance()
    project_engine = db_manager.get_project_engine(project_path)

    all_scores = []

    with mask_storage.open_video_h5(project_path, video_id, "a") as h5_file:
        # Ensure mask dataset exists
        mask_storage._ensure_dataset(h5_file, num_frames, height, width)

        # === HOT LOOP - DO NOT ADD ABSTRACTION HERE ===
        for frame_idx, postprocessed_out in model.propagate_in_video(
            inference_state,
            start_frame_idx=start_frame_idx,
            max_frame_num_to_track=max_frames,
            reverse=False,
        ):
            # Extract mask from postprocessed output
            mask = _extract_propagation_mask(postprocessed_out, height, width)

            # Write mask directly to per-video HDF5
            h5_file["masks"][frame_idx] = mask

            # Extract confidence score from out_probs
            score = -1.0
            if postprocessed_out is not None:
                probs = postprocessed_out.get("out_probs", np.array([]))
                if len(probs) > 0:
                    score = float(probs[0])
                    all_scores.append(score)

            # Buffer score and has_mask for batch write
            score_buffer.append((frame_idx, score))
            has_mask_buffer.append(frame_idx)

            # Flush to SQLite periodically
            if len(score_buffer) >= BATCH_SIZE:
                with Session(project_engine) as db_session:
                    frame_data_service.save_scores_batch_sync(
                        db_session, video_id, score_buffer
                    )
                    frame_data_service.set_has_mask_batch_sync(
                        db_session, video_id, has_mask_buffer
                    )
                score_buffer.clear()
                has_mask_buffer.clear()

            frame_indices.append(frame_idx)
            frame_count += 1

            if progress_callback and frame_count % 10 == 0:
                progress_callback(frame_idx)
        # === END HOT LOOP ===

        h5_file.flush()

    # Final flush of remaining scores and has_mask
    if score_buffer:
        with Session(project_engine) as db_session:
            frame_data_service.save_scores_batch_sync(db_session, video_id, score_buffer)
            frame_data_service.set_has_mask_batch_sync(db_session, video_id, has_mask_buffer)

    # Compute stats from collected scores
    stats = {}
    if all_scores:
        scores_arr = np.array(all_scores, dtype=np.float32)
        stats = {
            "min": float(np.min(scores_arr)),
            "p50": float(np.median(scores_arr)),
            "p95": float(np.percentile(scores_arr, 95)),
        }

    return frame_count, frame_indices, stats


def _extract_propagation_mask(postprocessed_out, height: int, width: int) -> np.ndarray:
    """Extract single-object mask from SAM3 postprocessed output."""
    if postprocessed_out is None:
        return np.zeros((height, width), dtype=np.uint8)

    masks = postprocessed_out.get("out_binary_masks", np.array([]))
    if len(masks) == 0:
        return np.zeros((height, width), dtype=np.uint8)

    # Take first object's mask (bool array of shape (H, W))
    mask = masks[0]
    if mask.shape != (height, width):
        mask = cv2.resize(
            mask.astype(np.uint8),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )

    return (mask * 255).astype(np.uint8)
