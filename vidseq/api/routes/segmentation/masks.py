"""Mask retrieval endpoints."""

from pathlib import Path

from fastapi import APIRouter, Depends
from fastapi.responses import Response
from sqlalchemy.ext.asyncio import AsyncSession

from vidseq.api.dependencies import get_project_folder, get_project_session, get_video
from vidseq.models.video import Video
from vidseq.services import frame_data_service, segmentation_service

router = APIRouter()


# ----- Tracker masks -----

@router.get(
    "/projects/{project_id}/videos/{video_id}/segmentation/tracker-mask/{frame_idx}",
)
async def get_tracker_mask(
    frame_idx: int,
    video: Video = Depends(get_video),
    project_path: Path = Depends(get_project_folder),
):
    """
    Get the tracker segmentation mask for a specific frame.

    Returns PNG binary. If no mask exists, returns a transparent (all zeros) mask.
    """
    mask_png = segmentation_service.get_mask_png(
        project_path=project_path,
        video=video,
        frame_idx=frame_idx,
    )
    return Response(content=mask_png, media_type="image/png")


@router.get(
    "/projects/{project_id}/videos/{video_id}/segmentation/tracker-masks",
)
async def get_tracker_masks(
    start_frame: int,
    count: int = 100,
    video: Video = Depends(get_video),
    project_path: Path = Depends(get_project_folder),
):
    """
    Get multiple tracker segmentation masks in a single request.

    Returns JSON with base64-encoded PNG masks for efficient batch transfer.
    """
    masks = segmentation_service.get_masks_batch_json(
        project_path=project_path,
        video=video,
        start_frame=start_frame,
        count=count,
    )
    return {"masks": masks}


# ----- Final masks -----
# Final masks are the tracker-detector fusion result

@router.get(
    "/projects/{project_id}/videos/{video_id}/segmentation/final-mask/{frame_idx}",
)
async def get_final_mask(
    frame_idx: int,
    video: Video = Depends(get_video),
    project_path: Path = Depends(get_project_folder),
):
    """
    Get the final (corrected) mask for a specific frame.

    Final masks are: tracker output when IoU >= 0.5, or detector-corrected when IoU < 0.5.
    Returns PNG binary. If no mask exists, returns a transparent (all zeros) mask.
    """
    mask_png = segmentation_service.get_final_mask_png(
        project_path=project_path,
        video=video,
        frame_idx=frame_idx,
    )
    return Response(content=mask_png, media_type="image/png")


@router.get(
    "/projects/{project_id}/videos/{video_id}/segmentation/final-masks",
)
async def get_final_masks(
    start_frame: int,
    count: int = 100,
    video: Video = Depends(get_video),
    project_path: Path = Depends(get_project_folder),
):
    """
    Get multiple final (corrected) masks in a single request.

    Returns JSON with base64-encoded PNG masks for efficient batch transfer.
    """
    masks = segmentation_service.get_final_masks_batch_json(
        project_path=project_path,
        video=video,
        start_frame=start_frame,
        count=count,
    )
    return {"masks": masks}


@router.get(
    "/projects/{project_id}/videos/{video_id}/segmentation/final-masks/exists",
)
async def final_masks_exist(
    video: Video = Depends(get_video),
    session: AsyncSession = Depends(get_project_session),
):
    """
    Check if final (corrected) masks exist for a video.

    Returns {"exists": true/false}.
    """
    exists = await frame_data_service.video_has_any_final_masks(
        session=session,
        video_id=video.id,
    )
    return {"exists": exists}
