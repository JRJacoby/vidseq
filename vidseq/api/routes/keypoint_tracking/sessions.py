"""Keypoint tracking session lifecycle and model status endpoints."""

import asyncio
import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from vidseq.api.dependencies import get_project_folder, get_video
from vidseq.models.video import Video
from vidseq.services import keypoint_tcp_client
from vidseq.services.alignment_service import init_keypoint_session as _init_keypoint_session

router = APIRouter()


@router.post("/keypoint-tracking/model")
async def create_keypoint_model():
    """Start loading SAM2++ keypoint tracking model in background."""
    keypoint_tcp_client.start_loading_in_background()
    return {"status": "started"}


@router.get("/keypoint-tracking/status")
async def get_keypoint_status():
    """Get the current keypoint tracking model loading status."""
    return keypoint_tcp_client.get_status()


@router.get("/keypoint-tracking/status/stream")
async def stream_keypoint_status():
    """SSE endpoint for real-time keypoint tracking status updates."""
    async def event_generator():
        last_status_str = None
        while True:
            current_status = keypoint_tcp_client.get_status()
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
        },
    )


@router.post("/projects/{project_id}/videos/{video_id}/keypoint-tracking/session")
async def init_keypoint_session(
    project_id: int,
    video: Video = Depends(get_video),
    project_path: Path = Depends(get_project_folder),
):
    """Initialize a keypoint tracking session for a video."""
    try:
        session_info = _init_keypoint_session(
            project_id=project_id,
            video_id=video.id,
            video_name=video.name,
            project_path=project_path,
            num_frames=video.num_frames,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return {
        "video_id": video.id,
        "num_frames": session_info.num_frames,
        "height": session_info.height,
        "width": session_info.width,
    }


@router.delete(
    "/projects/{project_id}/videos/{video_id}/keypoint-tracking/session",
    status_code=204,
)
async def close_keypoint_session(
    project_id: int,
    video_id: int,
):
    """Close a keypoint tracking session for a video."""
    keypoint_tcp_client.close_session(project_id, video_id)
    return None
