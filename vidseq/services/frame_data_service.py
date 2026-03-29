"""Frame data service - SQLite operations for bboxes, frame types, and scores.

This module provides CRUD operations for frame-level metadata stored in SQLite,
replacing the previous HDF5-based storage for this data.
"""

from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np
from sqlalchemy import and_, delete, select, update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from vidseq.models.frame_data import FrameData
from vidseq.services.utils import frames_to_ranges

# SQLite has a max of 32,766 bound parameters per statement.
# Compute chunk size from actual column count so it stays safe as columns are added.
_SQLITE_MAX_PARAMS = 32_766
_UPSERT_CHUNK_SIZE = _SQLITE_MAX_PARAMS // len(FrameData.__table__.columns)


async def _chunked_upsert(
    session: AsyncSession,
    values: list[dict],
    conflict_columns: list[str],
    update_columns: list[str] | dict,
) -> None:
    """Execute a chunked INSERT ... ON CONFLICT DO UPDATE.

    Splits values into chunks to stay under SQLite's parameter limit.

    Args:
        values: List of row dicts to upsert
        conflict_columns: Index columns for ON CONFLICT
        update_columns: Either a list of column names (uses excluded ref)
                       or a dict of {column: static_value}
    """
    for i in range(0, len(values), _UPSERT_CHUNK_SIZE):
        chunk = values[i : i + _UPSERT_CHUNK_SIZE]
        stmt = sqlite_insert(FrameData).values(chunk)
        if isinstance(update_columns, list):
            set_ = {col: getattr(stmt.excluded, col) for col in update_columns}
        else:
            set_ = update_columns
        stmt = stmt.on_conflict_do_update(index_elements=conflict_columns, set_=set_)
        await session.execute(stmt)
    await session.commit()


def _chunked_upsert_sync(
    session: Session,
    values: list[dict],
    conflict_columns: list[str],
    update_columns: list[str] | dict,
) -> None:
    """Sync version of _chunked_upsert."""
    for i in range(0, len(values), _UPSERT_CHUNK_SIZE):
        chunk = values[i : i + _UPSERT_CHUNK_SIZE]
        stmt = sqlite_insert(FrameData).values(chunk)
        if isinstance(update_columns, list):
            set_ = {col: getattr(stmt.excluded, col) for col in update_columns}
        else:
            set_ = update_columns
        stmt = stmt.on_conflict_do_update(index_elements=conflict_columns, set_=set_)
        session.execute(stmt)
    session.commit()


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
    await _chunked_upsert(session, values, ["video_id", "frame_idx"], ["frame_type"])


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


async def create_training_range(
    session: AsyncSession,
    project_path: Path,
    video_id: int,
    start_frame: int,
    end_frame: int,
) -> None:
    """Create a training range by validating masks exist and marking frames.

    Validates that all frames in range have tracker masks, then marks them as training.

    Args:
        session: Async database session
        project_path: Path to the project folder (unused, kept for API compatibility)
        video_id: ID of the video
        start_frame: First frame of range (inclusive)
        end_frame: Last frame of range (inclusive)

    Raises:
        MissingMasksError: If any frames in range are missing masks
    """
    from vidseq.services.exceptions import MissingMasksError

    # Validate all frames have masks
    missing_frames = await get_missing_tracker_masks_in_range(
        session, video_id, start_frame, end_frame
    )
    if missing_frames:
        raise MissingMasksError(missing_frames)

    # Mark frames as training
    await mark_training_range(session, video_id, start_frame, end_frame)


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


async def save_scores_batch(
    session: AsyncSession,
    video_id: int,
    scores: list[tuple[int, float]] | list[list],
) -> None:
    """Batch insert/update scores (async version).

    Args:
        session: Async database session
        video_id: ID of the video
        scores: List of (frame_idx, score) tuples or [frame_idx, score] lists
    """
    if not scores:
        return

    values = [
        {"video_id": video_id, "frame_idx": int(frame_idx), "score": float(score)}
        for frame_idx, score in scores
    ]

    await _chunked_upsert(session, values, ["video_id", "frame_idx"], ["score"])


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

    _chunked_upsert_sync(session, values, ["video_id", "frame_idx"], ["score"])


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


async def load_all_scores(
    session: AsyncSession,
    video_id: int,
) -> dict[int, float]:
    """Load all scores for a video as {frame_idx: score}.

    Returns all FrameData rows for the video as a dict. Frames without
    FrameData rows are absent from the dict. The caller should treat
    missing keys the same as scores <= 0 (below threshold).
    """
    result = await session.execute(
        select(FrameData.frame_idx, FrameData.score)
        .where(FrameData.video_id == video_id)
    )
    return {row.frame_idx: row.score for row in result.all()}


async def save_detector_scores_batch(
    session: AsyncSession,
    video_id: int,
    scores: list[tuple[int, float]] | list[list],
) -> None:
    """Batch insert/update detector confidence scores."""
    if not scores:
        return
    values = [
        {"video_id": video_id, "frame_idx": int(frame_idx), "detector_score": float(score)}
        for frame_idx, score in scores
    ]
    await _chunked_upsert(session, values, ["video_id", "frame_idx"], ["detector_score"])


async def load_detector_scores_in_range(
    session: AsyncSession,
    video_id: int,
    start_frame: int,
    end_frame: Optional[int] = None,
) -> list[dict]:
    """Load valid detector scores (> -1.0) in a frame range for LTTB downsampling."""
    conditions = [
        FrameData.video_id == video_id,
        FrameData.frame_idx >= start_frame,
        FrameData.detector_score > -1.0,
    ]
    if end_frame is not None:
        conditions.append(FrameData.frame_idx <= end_frame)
    result = await session.execute(
        select(FrameData.frame_idx, FrameData.detector_score)
        .where(and_(*conditions))
        .order_by(FrameData.frame_idx)
    )
    return [{"frame_idx": r[0], "score": r[1]} for r in result.all()]


async def get_scores_downsampled(
    session: AsyncSession,
    video_id: int,
    num_frames: int,
    max_samples: int = 800,
    start_frame: int = 0,
    end_frame: int | None = None,
) -> dict:
    """Get LTTB-downsampled tracker confidence scores for visualization."""
    from vidseq.services import lttb

    if end_frame is None:
        end_frame = num_frames - 1

    scores = await load_scores_in_range(session, video_id, start_frame, end_frame)
    downsampled = lttb.downsample_scores(scores, max_samples)
    return {"scores": downsampled, "total_count": len(scores)}


async def get_detector_scores_downsampled(
    session: AsyncSession,
    video_id: int,
    num_frames: int,
    max_samples: int = 800,
    start_frame: int = 0,
    end_frame: int | None = None,
) -> dict:
    """Get LTTB-downsampled detector confidence scores for visualization."""
    from vidseq.services import lttb

    if end_frame is None:
        end_frame = num_frames - 1

    scores = await load_detector_scores_in_range(session, video_id, start_frame, end_frame)
    downsampled = lttb.downsample_scores(scores, max_samples)
    return {"scores": downsampled, "total_count": len(scores)}


async def load_obb_scores_in_range(
    session: AsyncSession,
    video_id: int,
    start_frame: int,
    end_frame: int,
) -> list[dict]:
    """Load OBB scores for a frame range."""
    result = await session.execute(
        select(FrameData.frame_idx, FrameData.obb_score)
        .where(
            FrameData.video_id == video_id,
            FrameData.frame_idx >= start_frame,
            FrameData.frame_idx <= end_frame,
            FrameData.obb_score > -1.0,
        )
        .order_by(FrameData.frame_idx)
    )
    return [{"frame_idx": row[0], "score": row[1]} for row in result.all()]


async def get_obb_scores_downsampled(
    session: AsyncSession,
    video_id: int,
    num_frames: int,
    max_samples: int = 800,
    start_frame: int = 0,
    end_frame: int | None = None,
) -> dict:
    """Get LTTB-downsampled OBB confidence scores for visualization."""
    from vidseq.services import lttb

    if end_frame is None:
        end_frame = num_frames - 1

    scores = await load_obb_scores_in_range(session, video_id, start_frame, end_frame)
    downsampled = lttb.downsample_scores(scores, max_samples)
    return {"scores": downsampled, "total_count": len(scores)}


# =============================================================================
# DETECTOR BBOX OPERATIONS
# =============================================================================


async def save_detector_bboxes_batch(
    session: AsyncSession,
    video_id: int,
    bboxes: list[tuple[int, float, float, float, float]] | list[list],
) -> None:
    """Batch insert/update detector bounding boxes.

    Args:
        session: Async database session.
        video_id: Video ID.
        bboxes: List of (frame_idx, x1, y1, x2, y2) tuples or lists.
    """
    if not bboxes:
        return
    rows = [
        {
            "video_id": video_id,
            "frame_idx": int(frame_idx),
            "detector_bbox_x1": float(x1),
            "detector_bbox_y1": float(y1),
            "detector_bbox_x2": float(x2),
            "detector_bbox_y2": float(y2),
        }
        for frame_idx, x1, y1, x2, y2 in bboxes
    ]
    await _chunked_upsert(session, rows, ["video_id", "frame_idx"],
        ["detector_bbox_x1", "detector_bbox_y1", "detector_bbox_x2", "detector_bbox_y2"])


async def save_obb_bboxes_batch(
    session: AsyncSession,
    video_id: int,
    bboxes: list[list],
) -> None:
    """Batch insert/update OBB bounding boxes.

    Args:
        session: Async database session.
        video_id: Video ID.
        bboxes: List of [frame_idx, x1, y1, x2, y2, x3, y3, x4, y4] lists.
    """
    if not bboxes:
        return
    rows = [
        {
            "video_id": video_id,
            "frame_idx": int(b[0]),
            "obb_x1": float(b[1]), "obb_y1": float(b[2]),
            "obb_x2": float(b[3]), "obb_y2": float(b[4]),
            "obb_x3": float(b[5]), "obb_y3": float(b[6]),
            "obb_x4": float(b[7]), "obb_y4": float(b[8]),
        }
        for b in bboxes
    ]
    await _chunked_upsert(session, rows, ["video_id", "frame_idx"],
        ["obb_x1", "obb_y1", "obb_x2", "obb_y2", "obb_x3", "obb_y3", "obb_x4", "obb_y4"])


async def save_obb_scores_batch(
    session: AsyncSession,
    video_id: int,
    scores: list[list],
) -> None:
    """Batch insert/update OBB confidence scores."""
    if not scores:
        return
    values = [
        {"video_id": video_id, "frame_idx": int(frame_idx), "obb_score": float(score)}
        for frame_idx, score in scores
    ]
    await _chunked_upsert(session, values, ["video_id", "frame_idx"], ["obb_score"])


async def get_detector_bbox(
    session: AsyncSession,
    video_id: int,
    frame_idx: int,
) -> dict | None:
    """Get detector bbox for a single frame.

    Returns:
        Dict with x1, y1, x2, y2 keys, or None if no detection.
    """
    result = await session.execute(
        select(
            FrameData.detector_bbox_x1,
            FrameData.detector_bbox_y1,
            FrameData.detector_bbox_x2,
            FrameData.detector_bbox_y2,
        ).where(
            FrameData.video_id == video_id,
            FrameData.frame_idx == frame_idx,
            FrameData.detector_bbox_x1.isnot(None),
        )
    )
    row = result.first()
    if row is None:
        return None
    return {"x1": row[0], "y1": row[1], "x2": row[2], "y2": row[3]}


async def get_detector_bboxes_batch(
    session: AsyncSession,
    video_id: int,
    start_frame: int,
    count: int = 100,
) -> list[dict]:
    """Get detector bboxes for a range of frames.

    Returns:
        List of dicts with frame_idx, x1, y1, x2, y2 keys.
    """
    result = await session.execute(
        select(
            FrameData.frame_idx,
            FrameData.detector_bbox_x1,
            FrameData.detector_bbox_y1,
            FrameData.detector_bbox_x2,
            FrameData.detector_bbox_y2,
        ).where(
            FrameData.video_id == video_id,
            FrameData.frame_idx >= start_frame,
            FrameData.frame_idx < start_frame + count,
            FrameData.detector_bbox_x1.isnot(None),
        ).order_by(FrameData.frame_idx)
    )
    return [
        {"frame_idx": row[0], "x1": row[1], "y1": row[2], "x2": row[3], "y2": row[4]}
        for row in result.all()
    ]


async def get_obb_bbox(
    session: AsyncSession,
    video_id: int,
    frame_idx: int,
) -> dict | None:
    """Get OBB bbox for a single frame.

    Returns:
        Dict with corners key containing [[x1,y1],[x2,y2],[x3,y3],[x4,y4]], or None.
    """
    result = await session.execute(
        select(
            FrameData.obb_x1, FrameData.obb_y1,
            FrameData.obb_x2, FrameData.obb_y2,
            FrameData.obb_x3, FrameData.obb_y3,
            FrameData.obb_x4, FrameData.obb_y4,
        ).where(
            FrameData.video_id == video_id,
            FrameData.frame_idx == frame_idx,
            FrameData.obb_x1.isnot(None),
        )
    )
    row = result.first()
    if row is None:
        return None
    return {
        "corners": [
            [row[0], row[1]], [row[2], row[3]],
            [row[4], row[5]], [row[6], row[7]],
        ]
    }


async def get_obb_bboxes_batch(
    session: AsyncSession,
    video_id: int,
    start_frame: int,
    count: int = 100,
) -> list[dict]:
    """Get OBB bboxes for a range of frames."""
    result = await session.execute(
        select(
            FrameData.frame_idx,
            FrameData.obb_x1, FrameData.obb_y1,
            FrameData.obb_x2, FrameData.obb_y2,
            FrameData.obb_x3, FrameData.obb_y3,
            FrameData.obb_x4, FrameData.obb_y4,
        ).where(
            FrameData.video_id == video_id,
            FrameData.frame_idx >= start_frame,
            FrameData.frame_idx < start_frame + count,
            FrameData.obb_x1.isnot(None),
        ).order_by(FrameData.frame_idx)
    )
    return [
        {
            "frame_idx": row[0],
            "corners": [
                [row[1], row[2]], [row[3], row[4]],
                [row[5], row[6]], [row[7], row[8]],
            ],
        }
        for row in result.all()
    ]


async def obb_bboxes_exist(
    session: AsyncSession,
    video_id: int,
) -> bool:
    """Check if any OBB bbox data exists for a video."""
    result = await session.execute(
        select(FrameData.id)
        .where(
            FrameData.video_id == video_id,
            FrameData.obb_x1.isnot(None),
        )
        .limit(1)
    )
    return result.first() is not None


# =============================================================================
# TRACKER MASK PRESENCE OPERATIONS
# =============================================================================


async def set_has_tracker_mask(
    session: AsyncSession,
    video_id: int,
    frame_indices: Union[int, Iterable[int]],
    has_mask: bool,
) -> None:
    """Set the has_tracker_mask flag for one or more frames (upsert).

    Args:
        session: Async database session
        video_id: ID of the video
        frame_indices: Single frame index or iterable of frame indices (0-based)
        has_mask: True if frames have a tracker mask, False otherwise
    """
    # Normalize to list
    if isinstance(frame_indices, int):
        indices = [frame_indices]
    else:
        indices = list(frame_indices)

    if not indices:
        return

    mask_value = 1 if has_mask else 0
    values = [
        {"video_id": video_id, "frame_idx": idx, "has_tracker_mask": mask_value}
        for idx in indices
    ]

    await _chunked_upsert(session, values, ["video_id", "frame_idx"],
        {"has_tracker_mask": mask_value})


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
    _chunked_upsert_sync(session, values, ["video_id", "frame_idx"], ["has_tracker_mask"])


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


# Alias for backwards compatibility
get_missing_masks_in_range = get_missing_tracker_masks_in_range


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
    _chunked_upsert_sync(session, values, ["video_id", "frame_idx"], ["has_detector_mask"])


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
    frame_indices: Union[int, Iterable[int]],
    has_mask: bool,
) -> None:
    """Set the has_final_mask flag for one or more frames (upsert)."""
    # Normalize to list
    if isinstance(frame_indices, int):
        indices = [frame_indices]
    else:
        indices = list(frame_indices)

    if not indices:
        return

    mask_value = 1 if has_mask else 0
    values = [
        {"video_id": video_id, "frame_idx": idx, "has_final_mask": mask_value}
        for idx in indices
    ]

    await _chunked_upsert(session, values, ["video_id", "frame_idx"],
        {"has_final_mask": mask_value})


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
    _chunked_upsert_sync(session, values, ["video_id", "frame_idx"], ["has_final_mask"])


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
