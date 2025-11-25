import asyncio
import json
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pathlib import Path
from vidseq.database import get_registry_db
from vidseq.models.registry import Job
from vidseq.schemas.job import JobResponse
from vidseq import database

router = APIRouter()

@router.get("/jobs", response_model=list[JobResponse])
async def get_jobs(
    db: AsyncSession = Depends(get_registry_db)
):
    result = await db.execute(
        select(Job).order_by(Job.created_at.desc())
    )
    return result.scalars().all()

@router.get("/jobs/stream")
async def stream_jobs():
    async def event_generator():
        last_states = None  # Use None to ensure first emission
        
        while True:
            async with database.RegistrySessionLocal() as session:
                result = await session.execute(
                    select(Job).order_by(Job.created_at.desc())
                )
                jobs = result.scalars().all()
                
                current_states = {job.id: job.status for job in jobs}
                
                if current_states != last_states:
                    jobs_data = [
                        {
                            "id": job.id,
                            "type": job.type,
                            "status": job.status,
                            "project_id": job.project_id,
                            "details": job.details,
                            "log_path": job.log_path,
                            "created_at": job.created_at.isoformat(),
                            "updated_at": job.updated_at.isoformat(),
                        }
                        for job in jobs
                    ]
                    yield f"data: {json.dumps(jobs_data)}\n\n"
                    last_states = current_states
            
            await asyncio.sleep(1)
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )

@router.get("/jobs/{job_id}/logs/stream")
async def stream_logs(
    job_id: int,
    db: AsyncSession = Depends(get_registry_db)
):
    result = await db.execute(select(Job).where(Job.id == job_id))
    job = result.scalar_one_or_none()
    
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    log_path = Path(job.log_path)
    
    async def event_generator():
        if not log_path.exists():
            yield "data: Waiting for log file...\n\n"
            while not log_path.exists() and job.status == "pending":
                await asyncio.sleep(0.5)
                result = await db.execute(select(Job).where(Job.id == job_id))
                job_fresh = result.scalar_one()
                if job_fresh.status != "pending":
                    break
        
        if not log_path.exists():
            yield "data: Log file not found\n\n"
            return
        
        with open(log_path, 'r') as f:
            existing = f.read()
            if existing:
                for line in existing.splitlines():
                    yield f"data: {line}\n\n"
            
            while True:
                result = await db.execute(select(Job).where(Job.id == job_id))
                current_job = result.scalar_one()
                
                line = f.readline()
                if line:
                    yield f"data: {line.rstrip()}\n\n"
                elif current_job.status in ["completed", "failed"]:
                    break
                else:
                    await asyncio.sleep(0.1)
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )


