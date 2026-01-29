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
    SegmentRequest,
    MultiPointSegmentRequest,
    PropagateRequest,
    PropagateResponse,
)
from vidseq.services import (
    frame_data_service,
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

    If the frame already has a mask, this will refine it using the existing
    mask as context. Otherwise, it creates a new mask from scratch.
    """
    video_path = Path(video.path)
    label = 1 if segment_request.type == "positive_point" else 0
    frame_idx = segment_request.frame_idx

    # Check if this frame already has a mask (to decide add_point_prompt vs refine_mask)
    has_existing_mask = await frame_data_service.get_has_mask(
        session, video.id, frame_idx
    )

    try:
        if has_existing_mask:
            # Refine existing mask using previous logits as dense prompt
            # Convert single point to array format
            mask = sam3_service.refine_mask(
                project_id=project_id,
                video_id=video.id,
                frame_idx=frame_idx,
                points=[{"x": segment_request.details["x"], "y": segment_request.details["y"]}],
                labels=[label],
            )
        else:
            # Create new mask on blank frame
            mask = sam3_service.add_point_prompt(
                project_id=project_id,
                video_id=video.id,
                video_path=video_path,
                project_path=project_path,
                frame_idx=frame_idx,
                x=segment_request.details["x"],
                y=segment_request.details["y"],
                label=label,
            )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Note: mask is already saved to HDF5 by the TCP worker

    # Update mask presence index
    has_content = bool(np.any(mask > 0))
    await frame_data_service.set_has_mask(
        session, video.id, frame_idx, has_content
    )

    # If refinement resulted in an empty mask, reset the frame's SAM state
    # to avoid corrupted memory state on next interaction
    if has_existing_mask and not has_content:
        sam3_service.reset_frame(
            project_id=project_id,
            video_id=video.id,
            project_path=project_path,
            frame_idx=frame_idx,
        )

    # Note: conditioning_service.add_conditioning_frame is now handled by sam3_service

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

    Requires an active SAM3 session with a tracked object (add a point prompt first).
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


@router.post("/projects/{project_id}/videos/{video_id}/refine-mask")
async def refine_mask_multipoint(
    project_id: int,
    request: MultiPointSegmentRequest,
    video: Video = Depends(get_video),
    session: AsyncSession = Depends(get_project_session),
    project_path: Path = Depends(get_project_folder),
):
    """
    Refine an existing mask with multiple accumulated point prompts.

    All accumulated points are sent together so SAM3 has the full context
    of user intent (e.g., "this area is foreground, but NOT this part").

    Point coords should be normalized [0,1].
    """
    frame_idx = request.frame_idx

    # Convert to backend format
    points = [{"x": p.x, "y": p.y} for p in request.points]
    labels = [1 if p.type == "positive_point" else 0 for p in request.points]

    try:
        mask = sam3_service.refine_mask(
            project_id=project_id,
            video_id=video.id,
            frame_idx=frame_idx,
            points=points,
            labels=labels,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Update mask presence index
    has_content = bool(np.any(mask > 0))
    await frame_data_service.set_has_mask(
        session, video.id, frame_idx, has_content
    )

    # If refinement resulted in an empty mask, reset the frame's SAM state
    if not has_content:
        sam3_service.reset_frame(
            project_id=project_id,
            video_id=video.id,
            project_path=project_path,
            frame_idx=frame_idx,
        )

    mask_png = segmentation_service.mask_to_png(mask)
    return Response(content=mask_png, media_type="image/png")


