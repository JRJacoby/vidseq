"""Segmentation inference endpoints - point prompts and propagation."""

import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from sqlalchemy.ext.asyncio import AsyncSession

from vidseq.api.dependencies import get_project_folder, get_project_session, get_video
from vidseq.models.video import Video
from vidseq.schemas.segmentation import (
    PromptRequest,
    PropagateRequest,
    PropagateResponse,
)
from vidseq.services import (
    frame_data_service,
    segmentation_service,
    segmentation_tcp_client,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/projects/{project_id}/videos/{video_id}/prompt/{frame_idx}")
async def submit_prompt(
    project_id: int,
    frame_idx: int,
    request: PromptRequest,
    video: Video = Depends(get_video),
    session: AsyncSession = Depends(get_project_session),
    project_path: Path = Depends(get_project_folder),
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
    video_path = Path(video.path)

    # Convert points to backend format
    points = [{"x": p.x, "y": p.y} for p in request.points]
    labels = [1 if p.type == "positive_point" else 0 for p in request.points]

    try:
        mask = await segmentation_service.submit_prompt(
            session=session,
            project_id=project_id,
            video_id=video.id,
            video_path=video_path,
            project_path=project_path,
            frame_idx=frame_idx,
            points=points,
            labels=labels,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    mask_png = segmentation_service.mask_to_png(mask)
    return Response(content=mask_png, media_type="image/png")


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
        frame_indices = segmentation_tcp_client.generate_training_masks(
            project_id=project_id,
            video_id=video.id,
            start_frame_idx=request.start_frame_idx,
            max_frames=request.max_frames,
            project_path=project_path,
            num_frames=video.num_frames,
            height=video.height,
            width=video.width,
        )
    except RuntimeError as e:
        logger.error(f"Propagation failed: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in propagation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    # Update has_tracker_mask for all propagated frames
    for frame_idx in frame_indices:
        await frame_data_service.set_has_tracker_mask(session, video.id, frame_idx, True)

    return PropagateResponse(frames_processed=len(frame_indices))
