import mimetypes
from pathlib import Path
import cv2
import numpy as np
from io import BytesIO

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse, Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from PIL import Image

from vidseq.api.dependencies import get_project_session
from vidseq.models.video import Video
from vidseq.schemas.video import VideoCreate, VideoResponse
from vidseq.services.video_service import (
    VideoMetadataError,
    get_video_by_id,
    get_video_metadata,
)

router = APIRouter()


@router.get("/projects/{project_id}/videos", response_model=list[VideoResponse])
async def get_videos(
    session: AsyncSession = Depends(get_project_session),
):
    result = await session.execute(
        select(Video).order_by(Video.id)
    )
    return result.scalars().all()


@router.post("/projects/{project_id}/videos", response_model=list[VideoResponse], status_code=201)
async def add_videos(
    video_data: VideoCreate,
    session: AsyncSession = Depends(get_project_session),
):
    added_videos = []
    
    for path_str in video_data.paths:
        path = Path(path_str)
        
        if not path.exists():
            raise HTTPException(status_code=400, detail=f"Path does not exist: {path}")
        
        if not path.is_file():
            raise HTTPException(status_code=400, detail=f"Path is not a file: {path}")
        
        try:
            meta = get_video_metadata(path)
        except VideoMetadataError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        video = Video(
            name=path.name,
            path=str(path),
            fps=meta.fps,
            height=meta.height,
            width=meta.width,
            num_frames=meta.num_frames,
        )
        session.add(video)
        added_videos.append(video)
    
    await session.commit()
    
    for video in added_videos:
        await session.refresh(video)
    
    return added_videos


@router.get("/projects/{project_id}/videos/{video_id}")
async def get_video(
    video_id: int,
    session: AsyncSession = Depends(get_project_session),
) -> VideoResponse:
    try:
        video = await get_video_by_id(session, video_id)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return video


@router.get("/projects/{project_id}/videos/{video_id}/stream")
async def stream_video(
    video_id: int,
    request: Request,
    session: AsyncSession = Depends(get_project_session),
):
    try:
        video = await get_video_by_id(session, video_id)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    
    video_path = Path(video.path)
    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"Video file not found: {video.path}")
    
    file_size = video_path.stat().st_size
    content_type = mimetypes.guess_type(str(video_path))[0] or "video/mp4"
    
    range_header = request.headers.get("range")
    if range_header:
        range_match = range_header.replace("bytes=", "").split("-")
        start = int(range_match[0])
        end = int(range_match[1]) if range_match[1] else file_size - 1
        
        chunk_size = end - start + 1
        
        def iter_file():
            with open(video_path, "rb") as f:
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
        video_path,
        media_type=content_type,
        headers={"Accept-Ranges": "bytes"},
    )


@router.get("/projects/{project_id}/videos/{video_id}/frame/{frame_idx}")
async def get_frame(
    video_id: int,
    frame_idx: int,
    session: AsyncSession = Depends(get_project_session),
):
    """
    Extract a specific frame from a video and return it as a JPEG image.
    
    Args:
        video_id: ID of the video
        frame_idx: Frame index to extract (0-based)
        
    Returns:
        JPEG image bytes
    """
    try:
        video = await get_video_by_id(session, video_id)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    
    if frame_idx < 0 or frame_idx >= video.num_frames:
        raise HTTPException(status_code=400, detail=f"Frame index {frame_idx} out of range [0, {video.num_frames})")
    
    video_path = Path(video.path)
    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"Video file not found: {video.path}")
    
    # Extract frame using OpenCV
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise HTTPException(status_code=500, detail=f"Failed to open video: {video.path}")
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise HTTPException(status_code=500, detail=f"Failed to read frame {frame_idx} from video")
    
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image and then to JPEG bytes
    pil_image = Image.fromarray(frame_rgb)
    img_bytes = BytesIO()
    pil_image.save(img_bytes, format='JPEG', quality=95)
    img_bytes.seek(0)
    
    return Response(content=img_bytes.read(), media_type="image/jpeg")
