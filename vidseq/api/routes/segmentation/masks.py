"""Mask retrieval endpoints."""

from pathlib import Path

from fastapi import APIRouter, Depends
from fastapi.responses import Response

from vidseq.api.dependencies import get_project_folder, get_video
from vidseq.models.video import Video
from vidseq.services import segmentation_service

router = APIRouter()


@router.get(
    "/projects/{project_id}/videos/{video_id}/mask/{frame_idx}",
)
async def get_mask(
    frame_idx: int,
    video: Video = Depends(get_video),
    project_path: Path = Depends(get_project_folder),
):
    """
    Get the segmentation mask for a specific frame.

    Returns PNG binary. If no mask exists, returns a transparent (all zeros) mask.
    """
    mask_png = segmentation_service.get_mask_png(
        project_path=project_path,
        video=video,
        frame_idx=frame_idx,
    )
    return Response(content=mask_png, media_type="image/png")


@router.get(
    "/projects/{project_id}/videos/{video_id}/masks-batch",
)
async def get_masks_batch(
    start_frame: int,
    count: int = 100,
    video: Video = Depends(get_video),
    project_path: Path = Depends(get_project_folder),
):
    """
    Get multiple segmentation masks in a single request.

    Returns JSON with base64-encoded PNG masks for efficient batch transfer.
    """
    masks = segmentation_service.get_masks_batch_json(
        project_path=project_path,
        video=video,
        start_frame=start_frame,
        count=count,
    )
    return {"masks": masks}
