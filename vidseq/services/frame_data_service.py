"""Frame data service - SQLite operations for bboxes, frame types, and scores.

This module provides CRUD operations for frame-level metadata stored in SQLite,
replacing the previous HDF5-based storage for this data.
"""

from typing import Optional

import numpy as np
from sqlalchemy import and_, delete, select, update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from vidseq.models.frame_data import FrameData
from vidseq.services.utils import frames_to_ranges


# =============================================================================
# BOUNDING BOX OPERATIONS
# =============================================================================


async def save_bbox(
    session: AsyncSession,
    video_id: int,
    frame_idx: int,
    bbox: np.ndarray,
) -> None:
    """Save or update a bounding box (upsert).

    Args:
        session: Async database session
        video_id: ID of the video
        frame_idx: Frame index (0-based)
        bbox: Bounding box as numpy array [x1, y1, x2, y2]
    """
    stmt = sqlite_insert(FrameData).values(
        video_id=video_id,
        frame_idx=frame_idx,
        bbox_x1=float(bbox[0]),
        bbox_y1=float(bbox[1]),
        bbox_x2=float(bbox[2]),
        bbox_y2=float(bbox[3]),
    ).on_conflict_do_update(
        index_elements=["video_id", "frame_idx"],
        set_={
            "bbox_x1": float(bbox[0]),
            "bbox_y1": float(bbox[1]),
            "bbox_x2": float(bbox[2]),
            "bbox_y2": float(bbox[3]),
        }
    )
    await session.execute(stmt)
    await session.commit()


def save_bbox_sync(
    session: Session,
    video_id: int,
    frame_idx: int,
    bbox: np.ndarray,
) -> None:
    """Synchronous version of save_bbox for worker processes.

    Args:
        session: Sync database session
        video_id: ID of the video
        frame_idx: Frame index (0-based)
        bbox: Bounding box as numpy array [x1, y1, x2, y2]
    """
    stmt = sqlite_insert(FrameData).values(
        video_id=video_id,
        frame_idx=frame_idx,
        bbox_x1=float(bbox[0]),
        bbox_y1=float(bbox[1]),
        bbox_x2=float(bbox[2]),
        bbox_y2=float(bbox[3]),
    ).on_conflict_do_update(
        index_elements=["video_id", "frame_idx"],
        set_={
            "bbox_x1": float(bbox[0]),
            "bbox_y1": float(bbox[1]),
            "bbox_x2": float(bbox[2]),
            "bbox_y2": float(bbox[3]),
        }
    )
    session.execute(stmt)
    session.commit()


async def load_bbox(
    session: AsyncSession,
    video_id: int,
    frame_idx: int,
) -> Optional[np.ndarray]:
    """Load a bounding box for a specific frame.

    Args:
        session: Async database session
        video_id: ID of the video
        frame_idx: Frame index (0-based)

    Returns:
        Bounding box as numpy array [x1, y1, x2, y2], or None if not found
    """
    result = await session.execute(
        select(FrameData.bbox_x1, FrameData.bbox_y1, FrameData.bbox_x2, FrameData.bbox_y2)
        .where(FrameData.video_id == video_id, FrameData.frame_idx == frame_idx)
    )
    row = result.first()
    if row is None or row[0] is None:
        return None
    return np.array([row[0], row[1], row[2], row[3]], dtype=np.float32)


def load_bbox_sync(
    session: Session,
    video_id: int,
    frame_idx: int,
) -> Optional[np.ndarray]:
    """Synchronous version of load_bbox for worker processes.

    Args:
        session: Sync database session
        video_id: ID of the video
        frame_idx: Frame index (0-based)

    Returns:
        Bounding box as numpy array [x1, y1, x2, y2], or None if not found
    """
    result = session.execute(
        select(FrameData.bbox_x1, FrameData.bbox_y1, FrameData.bbox_x2, FrameData.bbox_y2)
        .where(FrameData.video_id == video_id, FrameData.frame_idx == frame_idx)
    )
    row = result.first()
    if row is None or row[0] is None:
        return None
    return np.array([row[0], row[1], row[2], row[3]], dtype=np.float32)


async def load_bboxes_batch(
    session: AsyncSession,
    video_id: int,
    start_frame: int,
    count: int,
) -> list[tuple[int, Optional[np.ndarray]]]:
    """Load bounding boxes for a range of frames.

    Args:
        session: Async database session
        video_id: ID of the video
        start_frame: Starting frame index
        count: Number of frames to load

    Returns:
        List of (frame_idx, bbox) tuples. bbox is None if not found.
    """
    result = await session.execute(
        select(
            FrameData.frame_idx,
            FrameData.bbox_x1,
            FrameData.bbox_y1,
            FrameData.bbox_x2,
            FrameData.bbox_y2,
        )
        .where(
            FrameData.video_id == video_id,
            FrameData.frame_idx >= start_frame,
            FrameData.frame_idx < start_frame + count,
        )
        .order_by(FrameData.frame_idx)
    )
    rows = result.all()

    # Build lookup dict
    row_dict = {r[0]: r for r in rows}

    # Return list with all frames in range (None for missing)
    bboxes = []
    for frame_idx in range(start_frame, start_frame + count):
        if frame_idx in row_dict:
            r = row_dict[frame_idx]
            if r[1] is not None:
                bboxes.append((frame_idx, np.array([r[1], r[2], r[3], r[4]], dtype=np.float32)))
            else:
                bboxes.append((frame_idx, None))
        else:
            bboxes.append((frame_idx, None))
    return bboxes


async def clear_bbox(
    session: AsyncSession,
    video_id: int,
    frame_idx: int,
) -> None:
    """Clear the bounding box for a specific frame (sets to NULL).

    Args:
        session: Async database session
        video_id: ID of the video
        frame_idx: Frame index (0-based)
    """
    await session.execute(
        update(FrameData)
        .where(FrameData.video_id == video_id, FrameData.frame_idx == frame_idx)
        .values(bbox_x1=None, bbox_y1=None, bbox_x2=None, bbox_y2=None)
    )
    await session.commit()


# =============================================================================
# FRAME TYPE OPERATIONS
# =============================================================================


async def mark_frame_type(
    session: AsyncSession,
    video_id: int,
    frame_idx: int,
    frame_type: Optional[str],
) -> None:
    """Set the frame type for a specific frame.

    Args:
        session: Async database session
        video_id: ID of the video
        frame_idx: Frame index (0-based)
        frame_type: Type to set ('train', 'apply', or None to clear)
    """
    stmt = sqlite_insert(FrameData).values(
        video_id=video_id,
        frame_idx=frame_idx,
        frame_type=frame_type,
    ).on_conflict_do_update(
        index_elements=["video_id", "frame_idx"],
        set_={"frame_type": frame_type}
    )
    await session.execute(stmt)
    await session.commit()


async def get_frame_type(
    session: AsyncSession,
    video_id: int,
    frame_idx: int,
) -> Optional[str]:
    """Get the frame type for a specific frame.

    Args:
        session: Async database session
        video_id: ID of the video
        frame_idx: Frame index (0-based)

    Returns:
        Frame type string ('train', 'apply') or None if not set
    """
    result = await session.execute(
        select(FrameData.frame_type)
        .where(FrameData.video_id == video_id, FrameData.frame_idx == frame_idx)
    )
    row = result.first()
    return row[0] if row else None


async def get_training_frames(
    session: AsyncSession,
    video_id: int,
) -> list[int]:
    """Get all frame indices marked as 'train'.

    This uses an index lookup and is O(log n + k) where k is the number of
    training frames, much faster than O(n) iteration through all frames.

    Args:
        session: Async database session
        video_id: ID of the video

    Returns:
        List of frame indices marked as 'train', sorted ascending
    """
    result = await session.execute(
        select(FrameData.frame_idx)
        .where(FrameData.video_id == video_id, FrameData.frame_type == "train")
        .order_by(FrameData.frame_idx)
    )
    return [row[0] for row in result.all()]


def get_training_frames_sync(
    session: Session,
    video_id: int,
) -> list[int]:
    """Synchronous version of get_training_frames for worker processes.

    Args:
        session: Sync database session
        video_id: ID of the video

    Returns:
        List of frame indices marked as 'train', sorted ascending
    """
    result = session.execute(
        select(FrameData.frame_idx)
        .where(FrameData.video_id == video_id, FrameData.frame_type == "train")
        .order_by(FrameData.frame_idx)
    )
    return [row[0] for row in result.all()]


async def get_training_frame_ranges(
    session: AsyncSession,
    video_id: int,
) -> list[tuple[int, int]]:
    """Get contiguous ranges of training frames.

    Args:
        session: Async database session
        video_id: ID of the video

    Returns:
        List of (start_frame, end_frame) tuples for contiguous training ranges
    """
    training_frames = await get_training_frames(session, video_id)
    return frames_to_ranges(training_frames)


async def mark_training_range(
    session: AsyncSession,
    video_id: int,
    start_frame: int,
    end_frame: int,
) -> None:
    """Mark a range of frames as training data.

    Args:
        session: Async database session
        video_id: ID of the video
        start_frame: First frame of the range (inclusive)
        end_frame: Last frame of the range (inclusive)
    """
    if start_frame > end_frame:
        return

    values = [
        {"video_id": video_id, "frame_idx": frame_idx, "frame_type": "train"}
        for frame_idx in range(start_frame, end_frame + 1)
    ]
    stmt = sqlite_insert(FrameData).values(values)
    stmt = stmt.on_conflict_do_update(
        index_elements=["video_id", "frame_idx"],
        set_={"frame_type": stmt.excluded.frame_type}
    )
    await session.execute(stmt)
    await session.commit()


async def unmark_training_range(
    session: AsyncSession,
    video_id: int,
    start_frame: int,
    end_frame: int,
) -> None:
    """Remove training labels from a range of frames.

    Also clears bounding boxes for the range.

    Args:
        session: Async database session
        video_id: ID of the video
        start_frame: First frame of the range (inclusive)
        end_frame: Last frame of the range (inclusive)
    """
    await session.execute(
        update(FrameData)
        .where(
            FrameData.video_id == video_id,
            FrameData.frame_idx >= start_frame,
            FrameData.frame_idx <= end_frame,
        )
        .values(frame_type=None, bbox_x1=None, bbox_y1=None, bbox_x2=None, bbox_y2=None)
    )
    await session.commit()


# =============================================================================
# SCORE OPERATIONS
# =============================================================================


async def save_score(
    session: AsyncSession,
    video_id: int,
    frame_idx: int,
    score: float,
) -> None:
    """Save or update a confidence score (upsert).

    Args:
        session: Async database session
        video_id: ID of the video
        frame_idx: Frame index (0-based)
        score: Confidence score (-1.0 to 1.0)
    """
    stmt = sqlite_insert(FrameData).values(
        video_id=video_id,
        frame_idx=frame_idx,
        score=score,
    ).on_conflict_do_update(
        index_elements=["video_id", "frame_idx"],
        set_={"score": score}
    )
    await session.execute(stmt)
    await session.commit()


def save_scores_batch_sync(
    session: Session,
    video_id: int,
    scores: list[tuple[int, float]],
) -> None:
    """Batch insert/update scores for efficiency during propagation.

    Args:
        session: Sync database session
        video_id: ID of the video
        scores: List of (frame_idx, score) tuples
    """
    if not scores:
        return

    # Prepare values for bulk upsert
    values = [
        {"video_id": video_id, "frame_idx": frame_idx, "score": score}
        for frame_idx, score in scores
    ]

    # SQLite upsert
    stmt = sqlite_insert(FrameData).values(values)
    stmt = stmt.on_conflict_do_update(
        index_elements=["video_id", "frame_idx"],
        set_={"score": stmt.excluded.score}
    )
    session.execute(stmt)
    session.commit()


async def load_scores_batch(
    session: AsyncSession,
    video_id: int,
    start_frame: int,
    count: int,
) -> list[dict]:
    """Load scores for a range of frames.

    Args:
        session: Async database session
        video_id: ID of the video
        start_frame: Starting frame index
        count: Number of frames to load

    Returns:
        List of dicts with 'frame_idx' and 'score' keys
    """
    result = await session.execute(
        select(FrameData.frame_idx, FrameData.score)
        .where(
            FrameData.video_id == video_id,
            FrameData.frame_idx >= start_frame,
            FrameData.frame_idx < start_frame + count,
        )
        .order_by(FrameData.frame_idx)
    )
    return [{"frame_idx": r[0], "score": r[1]} for r in result.all()]


async def load_scores_in_range(
    session: AsyncSession,
    video_id: int,
    start_frame: int,
    end_frame: Optional[int] = None,
) -> list[dict]:
    """Load valid scores (> -1.0) in a frame range for LTTB downsampling.

    This is optimized for the confidence graph visualization. It only returns
    frames that have a valid score (not the default -1.0).

    Args:
        session: Async database session
        video_id: ID of the video
        start_frame: Starting frame index (inclusive)
        end_frame: Ending frame index (inclusive), or None for no upper bound

    Returns:
        List of dicts with 'frame_idx' and 'score' keys, sorted by frame_idx
    """
    conditions = [
        FrameData.video_id == video_id,
        FrameData.frame_idx >= start_frame,
        FrameData.score > -1.0,
    ]
    if end_frame is not None:
        conditions.append(FrameData.frame_idx <= end_frame)

    result = await session.execute(
        select(FrameData.frame_idx, FrameData.score)
        .where(and_(*conditions))
        .order_by(FrameData.frame_idx)
    )
    return [{"frame_idx": r[0], "score": r[1]} for r in result.all()]


# =============================================================================
# TRACKER MASK PRESENCE OPERATIONS
# =============================================================================


async def set_has_tracker_mask(
    session: AsyncSession,
    video_id: int,
    frame_idx: int,
    has_mask: bool,
) -> None:
    """Set the has_tracker_mask flag for a specific frame (upsert).

    Args:
        session: Async database session
        video_id: ID of the video
        frame_idx: Frame index (0-based)
        has_mask: True if frame has a tracker mask, False otherwise
    """
    stmt = sqlite_insert(FrameData).values(
        video_id=video_id,
        frame_idx=frame_idx,
        has_tracker_mask=1 if has_mask else 0,
    ).on_conflict_do_update(
        index_elements=["video_id", "frame_idx"],
        set_={"has_tracker_mask": 1 if has_mask else 0}
    )
    await session.execute(stmt)
    await session.commit()


async def get_has_tracker_mask(
    session: AsyncSession,
    video_id: int,
    frame_idx: int,
) -> bool:
    """Get the has_tracker_mask flag for a specific frame.

    Args:
        session: Async database session
        video_id: ID of the video
        frame_idx: Frame index (0-based)

    Returns:
        True if frame has a tracker mask, False otherwise (or if no record exists)
    """
    result = await session.execute(
        select(FrameData.has_tracker_mask)
        .where(FrameData.video_id == video_id)
        .where(FrameData.frame_idx == frame_idx)
    )
    row = result.scalar_one_or_none()
    return bool(row) if row is not None else False


def set_has_tracker_mask_sync(
    session: Session,
    video_id: int,
    frame_idx: int,
    has_mask: bool,
) -> None:
    """Synchronous version of set_has_tracker_mask for worker processes.

    Args:
        session: Sync database session
        video_id: ID of the video
        frame_idx: Frame index (0-based)
        has_mask: True if frame has a tracker mask, False otherwise
    """
    stmt = sqlite_insert(FrameData).values(
        video_id=video_id,
        frame_idx=frame_idx,
        has_tracker_mask=1 if has_mask else 0,
    ).on_conflict_do_update(
        index_elements=["video_id", "frame_idx"],
        set_={"has_tracker_mask": 1 if has_mask else 0}
    )
    session.execute(stmt)
    session.commit()


def set_has_tracker_mask_batch_sync(
    session: Session,
    video_id: int,
    frame_indices: list[int],
    has_mask: bool = True,
) -> None:
    """Batch update has_tracker_mask flag for multiple frames (sync version).

    Optimized for propagation hot path where many frames are processed.

    Args:
        session: Sync database session
        video_id: ID of the video
        frame_indices: List of frame indices
        has_mask: Value to set (default True)
    """
    if not frame_indices:
        return

    values = [
        {"video_id": video_id, "frame_idx": frame_idx, "has_tracker_mask": 1 if has_mask else 0}
        for frame_idx in frame_indices
    ]
    stmt = sqlite_insert(FrameData).values(values)
    stmt = stmt.on_conflict_do_update(
        index_elements=["video_id", "frame_idx"],
        set_={"has_tracker_mask": stmt.excluded.has_tracker_mask}
    )
    session.execute(stmt)
    session.commit()


async def get_tracker_masked_frames(
    session: AsyncSession,
    video_id: int,
) -> list[int]:
    """Get all frame indices that have tracker masks.

    Uses indexed query - O(log n + k) where k = number of masked frames.

    Args:
        session: Async database session
        video_id: ID of the video

    Returns:
        List of frame indices that have tracker masks, sorted ascending
    """
    result = await session.execute(
        select(FrameData.frame_idx)
        .where(FrameData.video_id == video_id, FrameData.has_tracker_mask == 1)
        .order_by(FrameData.frame_idx)
    )
    return [row[0] for row in result.all()]


async def get_tracker_masked_frame_ranges(
    session: AsyncSession,
    video_id: int,
) -> list[tuple[int, int]]:
    """Get contiguous ranges of frames that have tracker masks.

    Args:
        session: Async database session
        video_id: ID of the video

    Returns:
        List of (start_frame, end_frame) tuples for contiguous masked ranges
    """
    masked_frames = await get_tracker_masked_frames(session, video_id)
    return frames_to_ranges(masked_frames)


async def get_missing_tracker_masks_in_range(
    session: AsyncSession,
    video_id: int,
    start_frame: int,
    end_frame: int,
) -> list[int]:
    """Get frame indices in range that are missing tracker masks.

    Returns frames where has_tracker_mask is NULL or 0.

    Args:
        session: Async database session
        video_id: ID of the video
        start_frame: Start frame index (inclusive)
        end_frame: End frame index (inclusive)

    Returns:
        List of frame indices that are missing tracker masks, sorted ascending
    """
    # Get frames that DO have tracker masks in range
    result = await session.execute(
        select(FrameData.frame_idx)
        .where(
            FrameData.video_id == video_id,
            FrameData.frame_idx >= start_frame,
            FrameData.frame_idx <= end_frame,
            FrameData.has_tracker_mask == 1,
        )
    )
    masked_in_range = set(row[0] for row in result.all())

    # Return frames NOT in the masked set
    all_in_range = set(range(start_frame, end_frame + 1))
    return sorted(all_in_range - masked_in_range)


async def clear_has_tracker_mask(
    session: AsyncSession,
    video_id: int,
    frame_idx: int,
) -> None:
    """Clear the has_tracker_mask flag for a frame (set to 0).

    Args:
        session: Async database session
        video_id: ID of the video
        frame_idx: Frame index (0-based)
    """
    await session.execute(
        update(FrameData)
        .where(FrameData.video_id == video_id, FrameData.frame_idx == frame_idx)
        .values(has_tracker_mask=0)
    )
    await session.commit()


async def clear_all_has_tracker_mask(
    session: AsyncSession,
    video_id: int,
) -> None:
    """Clear all has_tracker_mask flags for a video (set to 0).

    Args:
        session: Async database session
        video_id: ID of the video
    """
    await session.execute(
        update(FrameData)
        .where(FrameData.video_id == video_id)
        .values(has_tracker_mask=0)
    )
    await session.commit()


# =============================================================================
# DETECTOR MASK PRESENCE OPERATIONS
# =============================================================================


async def set_has_detector_mask(
    session: AsyncSession,
    video_id: int,
    frame_idx: int,
    has_mask: bool,
) -> None:
    """Set the has_detector_mask flag for a specific frame (upsert)."""
    stmt = sqlite_insert(FrameData).values(
        video_id=video_id,
        frame_idx=frame_idx,
        has_detector_mask=1 if has_mask else 0,
    ).on_conflict_do_update(
        index_elements=["video_id", "frame_idx"],
        set_={"has_detector_mask": 1 if has_mask else 0}
    )
    await session.execute(stmt)
    await session.commit()


def set_has_detector_mask_batch_sync(
    session: Session,
    video_id: int,
    frame_indices: list[int],
    has_mask: bool = True,
) -> None:
    """Batch update has_detector_mask flag for multiple frames (sync version)."""
    if not frame_indices:
        return
    values = [
        {"video_id": video_id, "frame_idx": frame_idx, "has_detector_mask": 1 if has_mask else 0}
        for frame_idx in frame_indices
    ]
    stmt = sqlite_insert(FrameData).values(values)
    stmt = stmt.on_conflict_do_update(
        index_elements=["video_id", "frame_idx"],
        set_={"has_detector_mask": stmt.excluded.has_detector_mask}
    )
    session.execute(stmt)
    session.commit()


async def clear_all_has_detector_mask(
    session: AsyncSession,
    video_id: int,
) -> None:
    """Clear all has_detector_mask flags for a video (set to 0)."""
    await session.execute(
        update(FrameData)
        .where(FrameData.video_id == video_id)
        .values(has_detector_mask=0)
    )
    await session.commit()


# =============================================================================
# FINAL MASK PRESENCE OPERATIONS
# =============================================================================


async def set_has_final_mask(
    session: AsyncSession,
    video_id: int,
    frame_idx: int,
    has_mask: bool,
) -> None:
    """Set the has_final_mask flag for a specific frame (upsert)."""
    stmt = sqlite_insert(FrameData).values(
        video_id=video_id,
        frame_idx=frame_idx,
        has_final_mask=1 if has_mask else 0,
    ).on_conflict_do_update(
        index_elements=["video_id", "frame_idx"],
        set_={"has_final_mask": 1 if has_mask else 0}
    )
    await session.execute(stmt)
    await session.commit()


def set_has_final_mask_batch_sync(
    session: Session,
    video_id: int,
    frame_indices: list[int],
    has_mask: bool = True,
) -> None:
    """Batch update has_final_mask flag for multiple frames (sync version)."""
    if not frame_indices:
        return
    values = [
        {"video_id": video_id, "frame_idx": frame_idx, "has_final_mask": 1 if has_mask else 0}
        for frame_idx in frame_indices
    ]
    stmt = sqlite_insert(FrameData).values(values)
    stmt = stmt.on_conflict_do_update(
        index_elements=["video_id", "frame_idx"],
        set_={"has_final_mask": stmt.excluded.has_final_mask}
    )
    session.execute(stmt)
    session.commit()


async def clear_all_has_final_mask(
    session: AsyncSession,
    video_id: int,
) -> None:
    """Clear all has_final_mask flags for a video (set to 0)."""
    await session.execute(
        update(FrameData)
        .where(FrameData.video_id == video_id)
        .values(has_final_mask=0)
    )
    await session.commit()


async def video_has_any_final_masks(
    session: AsyncSession,
    video_id: int,
) -> bool:
    """Check if any frames in a video have final masks written."""
    result = await session.execute(
        select(FrameData.id)
        .where(FrameData.video_id == video_id, FrameData.has_final_mask == 1)
        .limit(1)
    )
    return result.scalar_one_or_none() is not None


# =============================================================================
# CLEAR OPERATIONS
# =============================================================================


async def clear_frame_data(
    session: AsyncSession,
    video_id: int,
    frame_idx: int,
) -> None:
    """Clear all data for a specific frame.

    Args:
        session: Async database session
        video_id: ID of the video
        frame_idx: Frame index (0-based)
    """
    await session.execute(
        delete(FrameData)
        .where(FrameData.video_id == video_id, FrameData.frame_idx == frame_idx)
    )
    await session.commit()


async def clear_all_frame_data(
    session: AsyncSession,
    video_id: int,
) -> None:
    """Clear all frame data for a video.

    Args:
        session: Async database session
        video_id: ID of the video
    """
    await session.execute(
        delete(FrameData).where(FrameData.video_id == video_id)
    )
    await session.commit()


def clear_all_frame_data_sync(
    session: Session,
    video_id: int,
) -> None:
    """Synchronous version of clear_all_frame_data for worker processes.

    Args:
        session: Sync database session
        video_id: ID of the video
    """
    session.execute(
        delete(FrameData).where(FrameData.video_id == video_id)
    )
    session.commit()


# =============================================================================
# QUERY OPERATIONS (for mask_storage integration)
# =============================================================================


async def get_frames_with_bboxes(
    session: AsyncSession,
    video_id: int,
) -> list[tuple[int, np.ndarray]]:
    """Get all frames that have bounding boxes.

    Useful for YOLO dataset generation.

    Args:
        session: Async database session
        video_id: ID of the video

    Returns:
        List of (frame_idx, bbox) tuples
    """
    result = await session.execute(
        select(
            FrameData.frame_idx,
            FrameData.bbox_x1,
            FrameData.bbox_y1,
            FrameData.bbox_x2,
            FrameData.bbox_y2,
        )
        .where(
            FrameData.video_id == video_id,
            FrameData.bbox_x1.isnot(None),
        )
        .order_by(FrameData.frame_idx)
    )
    return [
        (r[0], np.array([r[1], r[2], r[3], r[4]], dtype=np.float32))
        for r in result.all()
    ]


def get_training_frames_with_bboxes_sync(
    session: Session,
    video_id: int,
) -> list[tuple[int, np.ndarray]]:
    """Get all training frames that have bounding boxes (sync version).

    Useful for YOLO dataset generation in worker processes.

    Args:
        session: Sync database session
        video_id: ID of the video

    Returns:
        List of (frame_idx, bbox) tuples
    """
    result = session.execute(
        select(
            FrameData.frame_idx,
            FrameData.bbox_x1,
            FrameData.bbox_y1,
            FrameData.bbox_x2,
            FrameData.bbox_y2,
        )
        .where(
            FrameData.video_id == video_id,
            FrameData.frame_type == "train",
            FrameData.bbox_x1.isnot(None),
        )
        .order_by(FrameData.frame_idx)
    )
    return [
        (r[0], np.array([r[1], r[2], r[3], r[4]], dtype=np.float32))
        for r in result.all()
    ]
