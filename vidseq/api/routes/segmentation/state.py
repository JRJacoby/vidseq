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
    Reset a frame: clear the mask, bounding box, conditioning frame record, and SAM2 prompts.
    """
    sam3_service.clear_prompts_for_frame(frame_idx)
    sam3_service.clear_frame_prompts(project_id, video.id, frame_idx)

    # Clear mask from per-video HDF5
    segmentation_service.clear_mask(
        project_path=project_path,
        video_id=video.id,
        frame_idx=frame_idx,
    )

    # Clear bbox, frame_type, and score from SQLite
    await frame_data_service.clear_frame_data(session, video.id, frame_idx)

    await conditioning_service.remove_conditioning_frame(
        session=session,
        video_id=video.id,
        frame_idx=frame_idx,
    )

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
    Reset entire video: clear all masks, conditioning frames, frame type labels, and SAM2 tracking state.
    """
    # Clear masks from HDF5 and frame data from SQLite
    await segmentation_service.clear_video(
        project_path=project_path,
        video_id=video.id,
        session=session,
    )

    deleted_count = await conditioning_service.clear_conditioning_frames(
        session=session,
        video_id=video.id,
    )

    sam3_service.reset_state(project_id, video.id)

    return {"message": "Video reset", "conditioning_frames_cleared": deleted_count}


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
