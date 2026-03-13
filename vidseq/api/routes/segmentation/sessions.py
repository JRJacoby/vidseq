"""Session lifecycle and model status endpoints."""

import asyncio
import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from vidseq.api.dependencies import get_project_folder, get_project_session, get_video
from vidseq.api.schemas import CoSegmentationRequest, VideoSelectionRequest
from vidseq.models.video import Video
from vidseq.services import segmentation_service, segmentation_tcp_client

router = APIRouter()


@router.post("/projects/{project_id}/videos/segmentation")
async def create_videos_segmentation(
    project_id: int,
    request: VideoSelectionRequest,
    session: AsyncSession = Depends(get_project_session),
    project_path: Path = Depends(get_project_folder),
):
    """Start batch segmentation for selected videos."""
    try:
        job_ids = await segmentation_service.segment_all_videos(
            session=session,
            project_id=project_id,
            project_path=project_path,
            video_ids=request.video_ids,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"job_ids": job_ids}


@router.post("/projects/{project_id}/videos/associated/segmentation")
async def co_segment_associated_videos(
    project_id: int,
    request: CoSegmentationRequest,
    session: AsyncSession = Depends(get_project_session),
    project_path: Path = Depends(get_project_folder),
):
    """Start co-segmentation for associated videos of selected main videos."""
    if not request.video_ids:
        raise HTTPException(status_code=400, detail="No video IDs provided")

    try:
        await segmentation_service.co_segment_videos(
            session=session,
            project_id=project_id,
            project_path=project_path,
            video_ids=request.video_ids,
            confidence_threshold=request.confidence_threshold,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"status": "ok"}


@router.get("/segmentation/status")
async def get_segmentation_status():
    """Get the current SAM model loading status."""
    return segmentation_tcp_client.get_status()


@router.get("/segmentation/status/stream")
async def stream_segmentation_status():
    """SSE endpoint for real-time SAM status updates."""
    async def event_generator():
        last_status_str = None
        while True:
            current_status = segmentation_tcp_client.get_status()
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


@router.post("/segmentation/loaded-model")
async def create_segmentation_loaded_model():
    """Start loading SAM model in background."""
    segmentation_tcp_client.start_loading_in_background()
    return {"status": "started"}


@router.post("/projects/{project_id}/videos/{video_id}/session")
async def init_video_session(
    project_id: int,
    video: Video = Depends(get_video),
    project_path: Path = Depends(get_project_folder),
    session: AsyncSession = Depends(get_project_session),
):
    """
    Initialize a SAM session for a video.

    Creates the tracker state and frame loader.
    Call this when entering the video detail view.
    Returns 503 if SAM model isn't loaded yet.
    """
    try:
        session_info = await segmentation_service.init_session(
            session, project_id, video, project_path
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return {
        "video_id": video.id,
        "num_frames": session_info.num_frames,
        "height": session_info.height,
        "width": session_info.width,
    }


@router.delete("/projects/{project_id}/videos/{video_id}/session", status_code=204)
async def close_video_session(
    project_id: int,
    video_id: int,
):
    """
    Close a SAM session for a video.

    Frees GPU memory. Call this when leaving the video detail view.
    """
    segmentation_tcp_client.close_session(project_id, video_id)
    return None
