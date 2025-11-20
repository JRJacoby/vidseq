import asyncio
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pathlib import Path
from vidseq.database import get_registry_db, get_project_folder
from vidseq.models.registry import Job
from vidseq.models.project import Video
from vidseq.schemas.segmentation import SegmentationRequest
from vidseq.schemas.job import JobResponse
from vidseq.jobs.runner import run_segmentation_job
from vidseq.database import get_project_session

router = APIRouter()

@router.post("/projects/{project_id}/segmentation", response_model=list[JobResponse])
async def run_segmentation(
    project_id: int,
    request: SegmentationRequest,
    registry_db: AsyncSession = Depends(get_registry_db),
    project_folder: Path = Depends(get_project_folder)
):
    created_jobs = []
    
    try:
        async for project_session in get_project_session(project_folder)():
            for video_id in request.video_ids:
                result = await project_session.execute(
                    select(Video).where(Video.id == video_id)
                )
                video = result.scalar_one_or_none()
                
                if not video:
                    for job in created_jobs:
                        job.status = "failed"
                    await registry_db.commit()
                    raise HTTPException(
                        status_code=400,
                        detail=f"Video {video_id} not found in project {project_id}"
                    )
                
                jobs_dir = project_folder / "jobs"
                jobs_dir.mkdir(exist_ok=True)
                
                job = Job(
                    type="segmentation",
                    status="pending",
                    project_id=project_id,
                    details={"video_id": video_id, "prompt": request.prompt},
                    log_path=str(jobs_dir / f"job_{{id}}.log")
                )
                registry_db.add(job)
                await registry_db.flush()
                
                job.log_path = str(jobs_dir / f"job_{job.id}.log")
                created_jobs.append(job)
            
            await registry_db.commit()
            
            for job in created_jobs:
                await registry_db.refresh(job)
            
            for job in created_jobs:
                asyncio.create_task(run_segmentation_job(job.id, project_id))
            
            return created_jobs
            
    except HTTPException:
        raise
    except Exception as e:
        for job in created_jobs:
            job.status = "failed"
        await registry_db.commit()
        raise HTTPException(status_code=500, detail=f"Failed to start segmentation: {str(e)}")


