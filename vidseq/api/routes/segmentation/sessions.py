"""Session lifecycle and model status endpoints."""

import asyncio
import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from vidseq.api.dependencies import get_project_folder, get_project_session, get_video
from vidseq.models.video import Video
from vidseq.services import sam3_service, video_service

router = APIRouter()


@router.post("/projects/{project_id}/segment-all-videos")
async def segment_all_videos_route(
    project_id: int,
    session: AsyncSession = Depends(get_project_session),
    project_path: Path = Depends(get_project_folder),
):
    """
    Start batch segmentation for all videos in the project.

    Validates that the detector model exists. Detection is performed on-the-fly
    during segmentation, so pre-computed detector masks are no longer required.
    """
    videos = await video_service.get_all_videos(session)
    if not videos:
        raise HTTPException(status_code=400, detail="No videos found in project")

    # Check that detector model exists
    detector_model_path = project_path / "models" / "detector.pt"
    if not detector_model_path.exists():
        raise HTTPException(
            status_code=400,
            detail="Detector model not found. Please train the detector first."
        )

    job_ids = await sam3_service.segment_all_videos(
        project_id=project_id,
        project_path=project_path,
        videos=videos,
    )

    return {"job_ids": job_ids}


@router.get("/segmentation/status")
async def get_segmentation_status():
    """Get the current SAM3 model loading status."""
    return sam3_service.get_status()


@router.get("/segmentation/status/stream")
async def stream_segmentation_status():
    """SSE endpoint for real-time SAM3 status updates."""
    async def event_generator():
        last_status_str = None
        while True:
            current_status = sam3_service.get_status()
            current_status_str = json.dumps(current_status)

            if current_status_str != last_status_str:
                yield f"data: {current_status_str}\n\n"
                last_status_str = current_status_str

            await asyncio.sleep(0.3)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@router.post("/segmentation/preload")
async def preload_segmentation():
    """Start loading SAM3 model in background."""
    sam3_service.start_loading_in_background()
    return {"message": "Loading started"}


@router.post("/projects/{project_id}/videos/{video_id}/session")
async def init_video_session(
    project_id: int,
    video: Video = Depends(get_video),
    project_path: Path = Depends(get_project_folder),
):
    """
    Initialize a SAM3 session for a video.

    Creates the tracker state and frame loader.
    Call this when entering the video detail view.
    Returns 503 if SAM3 model isn't loaded yet.
    """
    video_path = Path(video.path)

    try:
        session_info = sam3_service.init_session(project_id, video.id, video_path, project_path)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return {
        "video_id": video.id,
        "num_frames": session_info.num_frames,
        "height": session_info.height,
        "width": session_info.width,
    }


@router.delete("/projects/{project_id}/videos/{video_id}/session")
async def close_video_session(
    project_id: int,
    video_id: int,
):
    """
    Close a SAM3 session for a video.

    Frees GPU memory. Call this when leaving the video detail view.
    """
    closed = sam3_service.close_session(project_id, video_id)
    return {"closed": closed}
