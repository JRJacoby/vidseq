from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from pathlib import Path
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

