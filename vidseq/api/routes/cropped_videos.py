"""API routes for cropped video extraction and streaming."""

import mimetypes
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from vidseq.api.dependencies import get_project_folder, get_project_session, get_video
from vidseq.api.schemas import VideoSelectionRequest
from vidseq.models.video import Video
from vidseq.services import cropped_video_service, video_service

router = APIRouter()


@router.post("/projects/{project_id}/videos/extraction")
async def create_videos_extraction(
    project_id: int,
    request: VideoSelectionRequest,
    session: AsyncSession = Depends(get_project_session),
    project_path: Path = Depends(get_project_folder),
):
    """Start cropped video extraction for selected videos."""
    try:
        result = await cropped_video_service.create_videos_extraction(
            session=session,
            project_path=project_path,
            video_ids=request.video_ids,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return result


@router.post("/projects/{project_id}/videos/bbox-extraction")
async def create_videos_bbox_extraction(
    project_id: int,
    request: VideoSelectionRequest,
    session: AsyncSession = Depends(get_project_session),
    project_path: Path = Depends(get_project_folder),
):
    """Start bbox-centroid cropped video extraction for selected videos."""
    try:
        result = await cropped_video_service.create_videos_extraction_bbox(
            session=session,
            project_path=project_path,
            video_ids=request.video_ids,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return result


@router.get("/projects/{project_id}/videos/{video_id}/cropped-video/exists")
async def get_cropped_video_exists(
    video: Video = Depends(get_video),
    project_path: Path = Depends(get_project_folder),
):
    """Check if cropped video exists for a video."""
    exists = cropped_video_service.cropped_video_exists(project_path, video.name)
    path = None
    if exists:
        path = str(cropped_video_service.get_cropped_video_path(project_path, video.name))

    return {"exists": exists, "path": path}


@router.get("/projects/{project_id}/videos/{video_id}/cropped-video/stream")
async def stream_cropped_video(
    request: Request,
    video: Video = Depends(get_video),
    project_path: Path = Depends(get_project_folder),
):
    """Stream cropped video with HTTP range support."""
    cropped_path = cropped_video_service.get_cropped_video_path(project_path, video.name)

    if not cropped_path.exists():
        raise HTTPException(status_code=404, detail="Cropped video not found")

    file_size = cropped_path.stat().st_size
    content_type = mimetypes.guess_type(str(cropped_path))[0] or "video/mp4"

    range_header = request.headers.get("range")
    if range_header:
        range_match = range_header.replace("bytes=", "").split("-")
        start = int(range_match[0])
        end = int(range_match[1]) if range_match[1] else file_size - 1

        chunk_size = end - start + 1

        def iter_file():
            with open(cropped_path, "rb") as f:
                f.seek(start)
                remaining = chunk_size
                while remaining > 0:
                    read_size = min(8192, remaining)
                    data = f.read(read_size)
                    if not data:
                        break
                    remaining -= len(data)
                    yield data

        return StreamingResponse(
            iter_file(),
            status_code=206,
            media_type=content_type,
            headers={
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(chunk_size),
            },
        )

    return FileResponse(
        cropped_path,
        media_type=content_type,
        headers={"Accept-Ranges": "bytes"},
    )


@router.get("/projects/{project_id}/videos/{video_id}/cropped-video/frame/{frame_idx}")
async def get_cropped_video_frame(
    frame_idx: int,
    video: Video = Depends(get_video),
    project_path: Path = Depends(get_project_folder),
):
    """Extract a specific frame from a cropped video and return it as a JPEG image."""
    cropped_path = cropped_video_service.get_cropped_video_path(project_path, video.name)
    jpeg_bytes = video_service.extract_frame_as_jpeg(
        video_path=cropped_path,
        frame_idx=frame_idx,
    )
    return Response(content=jpeg_bytes, media_type="image/jpeg")
