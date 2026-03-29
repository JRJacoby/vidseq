"""Frame data retrieval endpoints - bboxes, scores, ranges."""

from pathlib import Path

import numpy as np
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from vidseq.api.dependencies import get_project_folder, get_project_session, get_video
from vidseq.models.video import Video
from vidseq.services import frame_data_service

router = APIRouter()


@router.get(
    "/projects/{project_id}/videos/{video_id}/frame-ranges",
)
async def get_frame_ranges(
    video: Video = Depends(get_video),
    session: AsyncSession = Depends(get_project_session),
    project_path: Path = Depends(get_project_folder),
):
    """
    Get tracker masked and training frame ranges for data track visualization.

    Returns contiguous ranges as [start, end] pairs (inclusive).
    Tracker masks are from SAM2 segmentation; training ranges are user-marked.
    """
    # Get tracker masked ranges from SQLite (indexed query)
    tracker_masked_ranges = await frame_data_service.get_tracker_masked_frame_ranges(
        session, video.id
    )

    # Get training ranges from SQLite
    training_ranges = await frame_data_service.get_training_frame_ranges(
        session, video.id
    )

    return {
        "tracker_masked_ranges": [[r[0], r[1]] for r in tracker_masked_ranges],
        "training_ranges": [[r[0], r[1]] for r in training_ranges],
    }


# ----- Bbox endpoints (deprecated, to be removed) -----

@router.get(
    "/projects/{project_id}/videos/{video_id}/bbox/{frame_idx}",
)
async def get_bbox(
    frame_idx: int,
    video: Video = Depends(get_video),
    session: AsyncSession = Depends(get_project_session),
):
    """
    Get the bounding box for a specific frame.

    Returns JSON with bbox coordinates or null if no bbox exists.

    DEPRECATED: Bboxes are no longer used for training. Will be removed.
    """
    bbox = await frame_data_service.load_bbox(session, video.id, frame_idx)

    if bbox is None:
        return None

    return {
        "x1": float(bbox[0]),
        "y1": float(bbox[1]),
        "x2": float(bbox[2]),
        "y2": float(bbox[3]),
    }


@router.get(
    "/projects/{project_id}/videos/{video_id}/bboxes-batch",
)
async def get_bboxes_batch(
    start_frame: int,
    count: int = 100,
    video: Video = Depends(get_video),
    session: AsyncSession = Depends(get_project_session),
):
    """
    Get bounding boxes for a batch of frames.

    Returns JSON array of {frame_idx, bbox} objects where bbox is [x1, y1, x2, y2] or null.

    DEPRECATED: Bboxes are no longer used for training. Will be removed.
    """
    # Clamp count to not exceed video length
    actual_count = min(count, video.num_frames - start_frame)
    if actual_count <= 0:
        return {"bboxes": []}

    bboxes = await frame_data_service.load_bboxes_batch(
        session, video.id, start_frame, actual_count
    )

    result = []
    for frame_idx, bbox in bboxes:
        if bbox is None or np.all(bbox == 0):
            result.append({"frame_idx": frame_idx, "bbox": None})
        else:
            result.append({
                "frame_idx": frame_idx,
                "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
            })

    return {"bboxes": result}


# ----- Score endpoints (under /segmentation/) -----

@router.get(
    "/projects/{project_id}/videos/{video_id}/segmentation/scores",
)
async def get_scores(
    start_frame: int,
    count: int = 100,
    video: Video = Depends(get_video),
    session: AsyncSession = Depends(get_project_session),
):
    """
    Get segmentation scores (IoU) for a batch of frames.

    Returns JSON array of {frame_idx, score} objects. Score is -1.0 if no valid score exists.
    """
    # Clamp count to not exceed video length
    actual_count = min(count, video.num_frames - start_frame)
    if actual_count <= 0:
        return {"scores": []}

    scores = await frame_data_service.load_scores_batch(
        session, video.id, start_frame, actual_count
    )

    return {"scores": scores}


@router.get(
    "/projects/{project_id}/videos/{video_id}/segmentation/scores-downsampled",
)
async def get_scores_downsampled(
    max_samples: int = 800,
    start_frame: int = 0,
    end_frame: int | None = None,
    video: Video = Depends(get_video),
    session: AsyncSession = Depends(get_project_session),
):
    """Get LTTB-downsampled confidence scores for visualization."""
    return await frame_data_service.get_scores_downsampled(
        session, video.id, video.num_frames, max_samples, start_frame, end_frame
    )


@router.get(
    "/projects/{project_id}/videos/{video_id}/segmentation/detector-scores-downsampled",
)
async def get_detector_scores_downsampled(
    max_samples: int = 800,
    start_frame: int = 0,
    end_frame: int | None = None,
    video: Video = Depends(get_video),
    session: AsyncSession = Depends(get_project_session),
):
    """Get LTTB-downsampled detector confidence scores for visualization."""
    return await frame_data_service.get_detector_scores_downsampled(
        session, video.id, video.num_frames, max_samples, start_frame, end_frame
    )


@router.get(
    "/projects/{project_id}/videos/{video_id}/segmentation/obb-scores-downsampled",
)
async def get_obb_scores_downsampled(
    max_samples: int = 800,
    start_frame: int = 0,
    end_frame: int | None = None,
    video: Video = Depends(get_video),
    session: AsyncSession = Depends(get_project_session),
):
    """Get LTTB-downsampled OBB confidence scores for visualization."""
    return await frame_data_service.get_obb_scores_downsampled(
        session, video.id, video.num_frames, max_samples, start_frame, end_frame
    )
