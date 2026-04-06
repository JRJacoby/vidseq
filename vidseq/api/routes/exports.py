"""API routes for data export."""

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from vidseq.api.dependencies import get_project_folder, get_project_session
from vidseq.api.schemas import VideoSelectionRequest
from vidseq.schemas.export import ExportResponse
from vidseq.services import export_service

router = APIRouter()


@router.post(
    "/projects/{project_id}/exports/detector-bboxes",
    response_model=ExportResponse,
)
async def create_detector_bboxes_export(
    request: VideoSelectionRequest,
    project_path: Path = Depends(get_project_folder),
    session: AsyncSession = Depends(get_project_session),
):
    """Export detector bounding boxes to CSV for selected videos."""
    if not request.video_ids:
        raise HTTPException(status_code=400, detail="video_ids must not be empty")

    try:
        csv_path, row_count = await export_service.export_detector_bboxes(
            session=session,
            project_path=project_path,
            video_ids=request.video_ids,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return ExportResponse(path=csv_path, row_count=row_count)


@router.post(
    "/projects/{project_id}/exports/detector-masks",
    response_model=ExportResponse,
)
async def create_detector_masks_export(
    request: VideoSelectionRequest,
    project_path: Path = Depends(get_project_folder),
    session: AsyncSession = Depends(get_project_session),
):
    """Export seg detector masks to H5 for selected videos."""
    if not request.video_ids:
        raise HTTPException(status_code=400, detail="video_ids must not be empty")

    try:
        h5_path, frame_count = await export_service.export_detector_masks(
            session=session,
            project_path=project_path,
            video_ids=request.video_ids,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return ExportResponse(path=h5_path, row_count=frame_count)
