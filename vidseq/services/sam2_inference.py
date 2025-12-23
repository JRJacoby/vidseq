"""SAM2 video propagation - hot path for mask generation.

This module contains the performance-critical propagation loop.
Keep this code tight - no unnecessary abstractions.
"""

from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
from sqlalchemy.orm import Session

from vidseq.services import frame_data_service, mask_storage
from vidseq.services.database_manager import DatabaseManager
from vidseq.services.sam2_utils import extract_mask


def propagate_video(
    predictor,
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
    Run SAM2 propagation and save masks/scores.

    HOT PATH - keep this function tight. Do not add abstraction layers.

    Masks are saved to per-video HDF5 files (masks/{video_id}.h5).
    Scores are buffered and flushed to SQLite every 100 frames.

    Args:
        predictor: CustomSAM2VideoPredictor instance
        inference_state: SAM2 inference state dict
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

    # Get sync database session for score writes
    db_manager = DatabaseManager.get_instance()
    project_engine = db_manager.get_project_engine(project_path)

    with mask_storage.open_video_h5(project_path, video_id, "a") as h5_file:
        # Ensure mask dataset exists
        mask_storage._ensure_dataset(h5_file, num_frames, height, width)

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            # Clear previous IoU scores
            if hasattr(predictor, "frame_ious"):
                predictor.frame_ious = {}

            iterator = predictor.propagate_in_video(
                inference_state=inference_state,
                start_frame_idx=start_frame_idx,
                max_frame_num_to_track=max_frames,
                reverse=False,
            )

            # === HOT LOOP - DO NOT ADD ABSTRACTION HERE ===
            for frame_idx, out_obj_ids, video_res_masks in iterator:
                mask = extract_mask(video_res_masks, out_obj_ids, height, width)

                # Write mask directly to per-video HDF5
                h5_file["masks"][frame_idx] = mask

                # Extract confidence score (IoU)
                score = -1.0
                if hasattr(predictor, "frame_ious"):
                    score = predictor.frame_ious.get(frame_idx, -1.0)

                # Buffer score for batch write
                score_buffer.append((frame_idx, score))

                # Flush scores to SQLite periodically
                if len(score_buffer) >= BATCH_SIZE:
                    with Session(project_engine) as db_session:
                        frame_data_service.save_scores_batch_sync(
                            db_session, video_id, score_buffer
                        )
                    score_buffer.clear()

                frame_indices.append(frame_idx)
                frame_count += 1

                if progress_callback and frame_count % 10 == 0:
                    progress_callback(frame_idx)
            # === END HOT LOOP ===

        h5_file.flush()

    # Final flush of remaining scores
    if score_buffer:
        with Session(project_engine) as db_session:
            frame_data_service.save_scores_batch_sync(db_session, video_id, score_buffer)

    # Calculate stats from collected scores (ignoring -1.0)
    stats = _compute_score_stats(predictor)

    return frame_count, frame_indices, stats


def _compute_score_stats(predictor) -> dict:
    """Compute min/p50/p95 stats from predictor's frame_ious."""
    stats = {}
    if hasattr(predictor, "frame_ious") and predictor.frame_ious:
        scores = list(predictor.frame_ious.values())
        if scores:
            scores_arr = np.array(scores, dtype=np.float32)
            stats = {
                "min": float(np.min(scores_arr)),
                "p50": float(np.median(scores_arr)),
                "p95": float(np.percentile(scores_arr, 95)),
            }
    return stats
