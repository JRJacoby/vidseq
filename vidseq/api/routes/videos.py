from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from pathlib import Path
from vidseq.database import get_project_db
from vidseq.api.routes.projects import get_project_folder
from vidseq.models.project import Video
from vidseq.schemas.video import VideoCreate, VideoResponse

router = APIRouter()

@router.get("/projects/{project_id}/videos", response_model=list[VideoResponse])
async def get_videos(
    project_folder: Path = Depends(get_project_folder),
):
    get_db = get_project_db(project_folder)
    async for session in get_db():
        result = await session.execute(
            select(Video).order_by(Video.id)
        )
        return result.scalars().all()

@router.post("/projects/{project_id}/videos", response_model=list[VideoResponse], status_code=201)
async def add_videos(
    video_data: VideoCreate,
    project_folder: Path = Depends(get_project_folder),
):
    
    added_videos = []
    
    get_db = get_project_db(project_folder)
    async for session in get_db():
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

