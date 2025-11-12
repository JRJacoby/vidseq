from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from vidseq.database import get_registry_db
from vidseq.models.registry import Project
from vidseq.schemas.project import ProjectResponse

router = APIRouter()

@router.get("/projects", response_model=list[ProjectResponse])
async def get_projects(
    db: AsyncSession = Depends(get_registry_db)
):
    result = await db.execute(
        select(Project).order_by(Project.updated_at.desc())
    )
    return result.scalars().all()