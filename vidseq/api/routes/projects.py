import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from vidseq.api.dependencies import get_project, get_registry_session
from vidseq.models.registry import Project
from vidseq.schemas.project import ProjectCreate, ProjectResponse
from vidseq.services import mask_service
from vidseq.services.database_manager import DatabaseManager

router = APIRouter()


@router.get("/projects", response_model=list[ProjectResponse])
async def get_projects(
    db: AsyncSession = Depends(get_registry_session),
):
    result = await db.execute(
        select(Project).order_by(Project.updated_at.desc())
    )
    return result.scalars().all()


@router.get("/projects/{project_id}", response_model=ProjectResponse)
async def get_project_route(
    project: Project = Depends(get_project),
):
    return project


@router.post("/projects", response_model=ProjectResponse, status_code=201)
async def create_project(
    project_data: ProjectCreate,
    db: AsyncSession = Depends(get_registry_session),
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
    
    db_manager = DatabaseManager.get_instance()
    await db_manager.init_project(project_dir)
    
    project = Project(
        name=project_data.name,
        path=str(project_dir),
    )
    db.add(project)
    await db.commit()
    await db.refresh(project)
    
    return project


@router.delete("/projects/{project_id}", status_code=204)
async def delete_project(
    project: Project = Depends(get_project),
    db: AsyncSession = Depends(get_registry_session),
):
    project_path = Path(project.path)
    
    db_manager = DatabaseManager.get_instance()
    await db_manager.dispose_project(project_path)
    
    # Close any open h5 file handles before deleting
    mask_service.close_project_h5(project_path)
    
    await db.delete(project)
    await db.commit()
    
    if project_path.exists():
        shutil.rmtree(project_path)
