"""Segmentation inference endpoints - point prompts and propagation."""

import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from sqlalchemy.ext.asyncio import AsyncSession

from vidseq.api.dependencies import get_project_folder, get_project_session, get_video
from vidseq.models.video import Video
from vidseq.schemas.segmentation import (
    BoxPromptRequest,
    PromptRequest,
    PropagateRequest,
    PropagateResponse,
)
from vidseq.services import segmentation_service

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/projects/{project_id}/videos/{video_id}/prompt/{frame_idx}")
async def submit_prompt(
    project_id: int,
    frame_idx: int,
    request: PromptRequest,
    video: Video = Depends(get_video),
    session: AsyncSession = Depends(get_project_session),
):
    """
    Submit point prompt(s) for segmentation.

    Point coords should be normalized [0,1].

    Workflow is determined by existing state:
    - 1 point, no existing mask: creates new mask (add_point_prompt)
    - 1 point, existing mask: refines mask with single point
    - 2+ points, existing mask: refines mask with all points
    - 2+ points, no existing mask: ERROR (can't refine without mask)
    """
    # Convert points to backend format
    points = [{"x": p.x, "y": p.y} for p in request.points]
    labels = [1 if p.type == "positive_point" else 0 for p in request.points]

    try:
        mask, n_cond_used, n_non_cond_used = await segmentation_service.submit_prompt(
            session=session,
            project_id=project_id,
            video_id=video.id,
            frame_idx=frame_idx,
            points=points,
            labels=labels,
            use_cond_memory=request.use_cond_memory,
            use_non_cond_memory=request.use_non_cond_memory,
            working_range_start=request.working_range_start,
            working_range_end=request.working_range_end,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    mask_png = segmentation_service.mask_to_png(mask)
    return Response(
        content=mask_png,
        media_type="image/png",
        headers={
            "X-Cond-Frames-Used": str(n_cond_used),
            "X-Non-Cond-Frames-Used": str(n_non_cond_used),
        },
    )


@router.post("/projects/{project_id}/videos/{video_id}/box-prompt/{frame_idx}")
async def submit_box_prompt(
    project_id: int,
    frame_idx: int,
    request: BoxPromptRequest,
    video: Video = Depends(get_video),
    session: AsyncSession = Depends(get_project_session),
):
    """Submit a bounding box prompt to create an initial segmentation mask.

    Box coords should be normalized [0,1].
    """
    try:
        mask, n_cond_used, n_non_cond_used = await segmentation_service.submit_box_prompt(
            session=session,
            project_id=project_id,
            video_id=video.id,
            frame_idx=frame_idx,
            x1=request.x1,
            y1=request.y1,
            x2=request.x2,
            y2=request.y2,
            use_cond_memory=request.use_cond_memory,
            use_non_cond_memory=request.use_non_cond_memory,
            working_range_start=request.working_range_start,
            working_range_end=request.working_range_end,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    mask_png = segmentation_service.mask_to_png(mask)
    return Response(
        content=mask_png,
        media_type="image/png",
        headers={
            "X-Cond-Frames-Used": str(n_cond_used),
            "X-Non-Cond-Frames-Used": str(n_non_cond_used),
        },
    )


@router.post(
    "/projects/{project_id}/videos/{video_id}/propagation",
    response_model=PropagateResponse,
)
async def propagate(
    project_id: int,
    request: PropagateRequest,
    video: Video = Depends(get_video),
    session: AsyncSession = Depends(get_project_session),
    project_path: Path = Depends(get_project_folder),
):
    """
    Propagate segmentation mask forward from the given frame.

    Requires an active SAM session with a tracked object (submit a point prompt first).
    Saves only masks to HDF5. Does NOT mark frames as training.
    Use POST /training-range to mark frames for training.
    """
    try:
        frames_processed = await segmentation_service.propagate(
            session=session,
            project_id=project_id,
            video_id=video.id,
            project_path=project_path,
            start_frame_idx=request.start_frame_idx,
            max_frames=request.max_frames,
            num_frames=video.num_frames,
            height=video.height,
            width=video.width,
            working_range_start=request.working_range_start,
            working_range_end=request.working_range_end,
        )
    except RuntimeError as e:
        logger.error(f"Propagation failed: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in propagation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    return PropagateResponse(frames_processed=frames_processed)


@router.post(
    "/projects/{project_id}/videos/{video_id}/propagation-without-memory",
    response_model=PropagateResponse,
)
async def propagate_without_memory(
    project_id: int,
    request: PropagateRequest,
    video: Video = Depends(get_video),
    session: AsyncSession = Depends(get_project_session),
    project_path: Path = Depends(get_project_folder),
):
    """Propagate using only conditioning frame memories (no temporal window).

    Prevents drift by ignoring the sliding window of recent propagated frames.
    Each frame is segmented independently using only user-prompted conditioning frames.
    """
    try:
        frames_processed = await segmentation_service.propagate_without_memory(
            session=session,
            project_id=project_id,
            video_id=video.id,
            project_path=project_path,
            start_frame_idx=request.start_frame_idx,
            max_frames=request.max_frames,
            num_frames=video.num_frames,
            height=video.height,
            width=video.width,
            working_range_start=request.working_range_start,
            working_range_end=request.working_range_end,
        )
    except RuntimeError as e:
        logger.error(f"Propagation without memory failed: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in propagation without memory: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    return PropagateResponse(frames_processed=frames_processed)
