"""Training range endpoints - validation, marking, and unmarking."""

from pathlib import Path

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from vidseq.api.dependencies import get_project_folder, get_project_session, get_video
from vidseq.models.video import Video
from vidseq.services import frame_data_service

router = APIRouter()


@router.get(
    "/projects/{project_id}/videos/{video_id}/training-range/validation",
)
async def validate_training_range(
    start_frame: int,
    end_frame: int,
    video: Video = Depends(get_video),
    session: AsyncSession = Depends(get_project_session),
):
    """
    Check if all frames in range have masks.

    Returns {"valid": true} if all frames have masks,
    or {"valid": false, "missing_frames": [...]} if some are missing.
    """
    missing_frames = await frame_data_service.get_missing_masks_in_range(
        session, video.id, start_frame, end_frame
    )

    if missing_frames:
        return {"valid": False, "missing_frames": missing_frames}
    return {"valid": True}


@router.post(
    "/projects/{project_id}/videos/{video_id}/training-range",
    status_code=204,
)
async def create_training_range(
    start_frame: int,
    end_frame: int,
    video: Video = Depends(get_video),
    session: AsyncSession = Depends(get_project_session),
    project_path: Path = Depends(get_project_folder),
):
    """
    Mark frame range as training (computes bboxes from masks).

    All frames in range must have masks. Use GET /training-range/validation first.
    """
    await frame_data_service.create_training_range(
        session=session,
        project_path=project_path,
        video_id=video.id,
        start_frame=start_frame,
        end_frame=end_frame,
    )
    return None


@router.delete(
    "/projects/{project_id}/videos/{video_id}/training-range",
    status_code=204,
)
async def delete_training_range(
    start_frame: int,
    end_frame: int,
    video: Video = Depends(get_video),
    session: AsyncSession = Depends(get_project_session),
):
    """
    Remove training labels and bboxes for frame range.
    """
    await frame_data_service.unmark_training_range(
        session, video.id, start_frame, end_frame
    )

    return None
