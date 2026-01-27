"""Segmentation inference endpoints - point prompts and propagation."""

import logging
from pathlib import Path

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from sqlalchemy.ext.asyncio import AsyncSession

from vidseq.api.dependencies import get_project_folder, get_project_session, get_video
from vidseq.models.video import Video
from vidseq.schemas.segmentation import SegmentRequest, PropagateRequest, PropagateResponse
from vidseq.services import (
    conditioning_service,
    frame_data_service,
    mask_storage,
    sam3_service,
    segmentation_service,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/projects/{project_id}/videos/{video_id}/segment")
async def run_segmentation(
    project_id: int,
    segment_request: SegmentRequest,
    video: Video = Depends(get_video),
    session: AsyncSession = Depends(get_project_session),
    project_path: Path = Depends(get_project_folder),
):
    """
    Run segmentation with a point prompt.

    Point coords should be normalized [0,1].
    First point creates the tracked object, subsequent points refine it.
    Marks the frame as a conditioning frame.
    Returns the mask as PNG.
    """
    video_path = Path(video.path)

    label = 1 if segment_request.type == "positive_point" else 0

    try:
        mask = sam3_service.add_point_prompt(
            project_id=project_id,
            video_id=video.id,
            video_path=video_path,
            frame_idx=segment_request.frame_idx,
            points=[[segment_request.details["x"], segment_request.details["y"]]],
            labels=[label],
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    mask_storage.save_mask(
        project_path=project_path,
        video_id=video.id,
        frame_idx=segment_request.frame_idx,
        mask=mask,
        num_frames=video.num_frames,
        height=video.height,
        width=video.width,
    )

    # Update mask presence index
    has_content = bool(np.any(mask > 0))
    await frame_data_service.set_has_mask(
        session, video.id, segment_request.frame_idx, has_content
    )

    await conditioning_service.add_conditioning_frame(
        session=session,
        video_id=video.id,
        frame_idx=segment_request.frame_idx,
    )

    mask_png = segmentation_service.mask_to_png(mask)
    return Response(content=mask_png, media_type="image/png")


@router.post(
    "/projects/{project_id}/videos/{video_id}/propagate-mask",
    response_model=PropagateResponse,
)
async def propagate_mask(
    project_id: int,
    request: PropagateRequest,
    video: Video = Depends(get_video),
    project_path: Path = Depends(get_project_folder),
):
    """
    Propagate segmentation mask forward from the given frame.

    Requires an active SAM2 session with a tracked object (add a point prompt first).
    Saves only masks to HDF5. Does NOT mark frames as training or compute bounding boxes.
    Use mark-training endpoint to explicitly mark frames for YOLO training.
    """
    try:
        frames_processed = sam3_service.generate_training_masks(
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
        logger.error(f"Propagate mask failed: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in propagate mask: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    return PropagateResponse(frames_processed=frames_processed)


@router.get(
    "/projects/{project_id}/videos/{video_id}/prompts/{frame_idx}",
)
async def get_prompts_for_frame(
    frame_idx: int,
    video: Video = Depends(get_video),
):
    """Get all prompts for a specific frame."""
    prompts = sam3_service.get_prompts_for_frame(frame_idx)
    return prompts


@router.get(
    "/projects/{project_id}/videos/{video_id}/prompts",
)
async def get_all_prompts(
    video: Video = Depends(get_video),
):
    """Get all prompts for all frames."""
    prompts = sam3_service.get_all_prompts()
    return prompts
