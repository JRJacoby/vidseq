from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pathlib import Path
from vidseq.database import get_registry_db, init_project_db
from vidseq.models.registry import Project
from vidseq.schemas.project import ProjectCreate, ProjectResponse

router = APIRouter()

async def get_project_folder(
    project_id: int,
    db: AsyncSession = Depends(get_registry_db)
) -> Path:
    result = await db.execute(
        select(Project).where(Project.id == project_id)
    )
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail=f"Project {project_id} not found")
    return Path(project.path)

@router.get("/projects", response_model=list[ProjectResponse])
async def get_projects(
    db: AsyncSession = Depends(get_registry_db)
):
    result = await db.execute(
        select(Project).order_by(Project.updated_at.desc())
    )
    return result.scalars().all()

@router.post("/projects", response_model=ProjectResponse, status_code=201)
async def create_project(
    project_data: ProjectCreate,
    db: AsyncSession = Depends(get_registry_db)
):
    parent_dir = Path(project_data.path)
    
    if not parent_dir.exists():
        raise HTTPException(status_code=400, detail=f"Directory does not exist: {parent_dir}")
    
    if not parent_dir.is_dir():
        raise HTTPException(status_code=400, detail=f"Path is not a directory: {parent_dir}")
    
    project_dir = parent_dir / project_data.name
    
    try:
        project_dir.mkdir(exist_ok=False)
    except FileExistsError:
        raise HTTPException(status_code=400, detail=f"A project already exists at {project_dir}")
    except PermissionError:
        raise HTTPException(status_code=403, detail=f"Permission denied creating directory at {project_dir}")
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Failed to create directory: {str(e)}")
    
    await init_project_db(project_dir)
    
    project = Project(
        name=project_data.name,
        path=str(project_dir)
    )
    db.add(project)
    await db.commit()
    await db.refresh(project)
    
    return project