import asyncio
from pathlib import Path
from sqlalchemy import select
from vidseq.models.registry import Job
from vidseq import database

async def run_segmentation_job(job_id: int, project_id: int):
    async with database.RegistrySessionLocal() as session:
        result = await session.execute(select(Job).where(Job.id == job_id))
        job = result.scalar_one()
        
        log_file = Path(job.log_path)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            job.status = "running"
            await session.commit()
            
            with open(log_file, 'w') as f:
                f.write(f"Starting segmentation job {job_id}\n")
                f.write(f"Video ID: {job.details['video_id']}\n")
                f.write(f"Prompt: {job.details['prompt']}\n")
                f.write("Processing...\n")
                f.flush()
                
                await asyncio.sleep(10)
                
                f.write("Segmentation complete!\n")
                f.flush()
            
            job.status = "completed"
            await session.commit()
            
        except Exception as e:
            job.status = "failed"
            with open(log_file, 'a') as f:
                f.write(f"\nError: {str(e)}\n")
            await session.commit()


