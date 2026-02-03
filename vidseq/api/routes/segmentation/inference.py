"""Segmentation inference endpoints - point prompts and propagation."""

import logging
from pathlib import Path

import numpy as np
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

    # Check if this frame already has a mask
    has_existing_mask = await frame_data_service.get_has_tracker_mask(
        session, video.id, frame_idx
    )

    # Validate: multi-point requires existing mask
    if len(request.points) > 1 and not has_existing_mask:
        raise HTTPException(
            status_code=400,
            detail="Cannot submit multiple points without an existing mask. Submit a single point first to create a mask."
        )

    # Convert points to backend format
    points = [{"x": p.x, "y": p.y} for p in request.points]
    labels = [1 if p.type == "positive_point" else 0 for p in request.points]

    try:
        if has_existing_mask:
            # Refine existing mask using previous logits as dense prompt
            mask = segmentation_tcp_client.refine_mask(
                project_id=project_id,
                video_id=video.id,
                frame_idx=frame_idx,
                points=points,
                labels=labels,
            )
        else:
            # Create new mask on blank frame (single point only, validated above)
            p = request.points[0]
            label = 1 if p.type == "positive_point" else 0
            mask = segmentation_tcp_client.add_point_prompt(
                project_id=project_id,
                video_id=video.id,
                video_path=video_path,
                project_path=project_path,
                frame_idx=frame_idx,
                x=p.x,
                y=p.y,
                label=label,
            )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Update mask presence index
    has_content = bool(np.any(mask > 0))
    await frame_data_service.set_has_tracker_mask(
        session, video.id, frame_idx, has_content
    )

    # If refinement resulted in an empty mask, reset the frame's SAM state
    if has_existing_mask and not has_content:
        segmentation_tcp_client.reset_frame(
            project_id=project_id,
            video_id=video.id,
            project_path=project_path,
            frame_idx=frame_idx,
        )

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
