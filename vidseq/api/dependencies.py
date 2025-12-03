"""FastAPI dependency injection functions."""

from pathlib import Path

from fastapi import Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from vidseq.models.registry import Project
from vidseq.services.database_manager import DatabaseManager


async def get_registry_session():
    """Yield a registry database session."""
    db = DatabaseManager.get_instance()
    factory = db.get_registry_session_factory()
    async with factory() as session:
        yield session


async def get_project(
    project_id: int,
    session: AsyncSession = Depends(get_registry_session),
) -> Project:
    """Get project by ID from path parameter, raise 404 if not found."""
    result = await session.execute(
        select(Project).where(Project.id == project_id)
    )
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail=f"Project {project_id} not found")
    return project


async def get_project_folder(
    project: Project = Depends(get_project),
) -> Path:
    """Get the project folder path."""
    return Path(project.path)


async def get_project_session(
    project_folder: Path = Depends(get_project_folder),
):
    """Yield a project database session."""
    db = DatabaseManager.get_instance()
    
    if not db.is_project_initialized(project_folder):
        db_path = project_folder / "vidseq.db"
        if not db_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Project database not found at {db_path}. Project may not be properly initialized.",
            )
        await db.init_project(project_folder)
    
    factory = db.get_project_session_factory(project_folder)
    async with factory() as session:
        yield session

