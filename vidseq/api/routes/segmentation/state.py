"""State management endpoints - reset operations and conditioning frames."""

from pathlib import Path

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from vidseq.api.dependencies import get_project_folder, get_project_session, get_video
from vidseq.models.video import Video
from vidseq.services import (
    conditioning_service,
    frame_data_service,
    sam3_service,
    segmentation_service,
)

router = APIRouter()


@router.delete(
    "/projects/{project_id}/videos/{video_id}/frame/{frame_idx}",
)
async def reset_frame(
    project_id: int,
    frame_idx: int,
    video: Video = Depends(get_video),
    session: AsyncSession = Depends(get_project_session),
    project_path: Path = Depends(get_project_folder),
):
    """
    Reset a frame: clear the mask, bounding box, and SAM3 state for this frame.
    """
    # Reset in SAM3 (clears mask in HDF5 and conditioning frame in DB)
    sam3_service.reset_frame(project_id, video.id, project_path, frame_idx)

    # Clear mask from per-video HDF5 (belt and suspenders)
    segmentation_service.clear_mask(
        project_path=project_path,
        video_id=video.id,
        frame_idx=frame_idx,
    )

    # Clear bbox, frame_type, and score from SQLite
    await frame_data_service.clear_frame_data(session, video.id, frame_idx)

    return {"message": "Frame reset"}


@router.delete(
    "/projects/{project_id}/videos/{video_id}/all-frames",
)
async def reset_video(
    project_id: int,
    video: Video = Depends(get_video),
    session: AsyncSession = Depends(get_project_session),
    project_path: Path = Depends(get_project_folder),
):
    """
    Reset entire video: clear all masks, conditioning frames, and SAM3 tracking state.
    """
    # Reset in SAM3 (clears all masks in HDF5 and conditioning frames in DB)
    sam3_service.reset_video(project_id, video.id, project_path)

    # Clear frame data from SQLite (bboxes, scores, types)
    await segmentation_service.clear_video(
        project_path=project_path,
        video_id=video.id,
        session=session,
    )

    return {"message": "Video reset"}


@router.get(
    "/projects/{project_id}/videos/{video_id}/conditioning-frames",
)
async def get_conditioning_frames(
    video: Video = Depends(get_video),
    session: AsyncSession = Depends(get_project_session),
):
    """Get list of conditioning frame indices for a video."""
    frames = await conditioning_service.get_conditioning_frames(session, video.id)
    return {"conditioning_frames": frames}
