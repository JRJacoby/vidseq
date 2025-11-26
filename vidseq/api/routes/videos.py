from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from pathlib import Path
import mimetypes
from vidseq.database import get_project_session
from vidseq.models.project import Video
from vidseq.schemas.video import VideoCreate, VideoResponse

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
        
        video = Video(
            name=path.name,
            path=str(path)
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
    result = await session.execute(
        select(Video).where(Video.id == video_id)
    )
    video = result.scalar_one_or_none()
    if not video:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found")
    return video


@router.get("/projects/{project_id}/videos/{video_id}/stream")
async def stream_video(
    video_id: int,
    request: Request,
    session: AsyncSession = Depends(get_project_session),
):
    result = await session.execute(
        select(Video).where(Video.id == video_id)
    )
    video = result.scalar_one_or_none()
    if not video:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found")
    
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

