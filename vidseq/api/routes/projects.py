"""Project routes - thin HTTP wrappers around project_service."""

from pathlib import Path

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from vidseq.api.dependencies import get_registry_session
from vidseq.schemas.project import ProjectCreate, ProjectResponse
from vidseq.services import project_service

router = APIRouter()


@router.get("/projects", response_model=list[ProjectResponse])
async def get_projects(
    db: AsyncSession = Depends(get_registry_session),
):
    return await project_service.get_all_projects(db)


@router.get("/projects/{project_id}", response_model=ProjectResponse)
async def get_project_route(
    project_id: int,
    db: AsyncSession = Depends(get_registry_session),
):
    return await project_service.get_project(db, project_id)


@router.post("/projects", response_model=ProjectResponse, status_code=201)
async def create_project(
    project_data: ProjectCreate,
    db: AsyncSession = Depends(get_registry_session),
):
    return await project_service.create_project(
        db, project_data.name, Path(project_data.path)
    )


@router.delete("/projects/{project_id}", status_code=204)
async def delete_project(
    project_id: int,
    db: AsyncSession = Depends(get_registry_session),
):
    await project_service.delete_project(db, project_id)
