"""State management endpoints - conditioning frames."""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from vidseq.api.dependencies import get_project_session, get_video
from vidseq.models.video import Video
from vidseq.services import conditioning_service

router = APIRouter()


# Note: reset_frame endpoint moved to videos.py as DELETE /videos/{id}/frame-data/{idx}
# Note: reset_video endpoint moved to videos.py as DELETE /videos/{id}/frame-data


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
