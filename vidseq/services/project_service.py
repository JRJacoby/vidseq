"""Project service - business logic for project management."""

import shutil
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from vidseq.models.registry import Project
from vidseq.services.database_manager import DatabaseManager
from vidseq.services.exceptions import (
    DBRecordNotFoundError,
    ParentDirectoryNotFoundError,
    PathNotDirectoryError,
    PermissionDeniedError,
    ProjectAlreadyExistsError,
)


async def get_all_projects(session: AsyncSession) -> list[Project]:
    """Get all projects ordered by most recently updated.

    Args:
        session: Registry database session

    Returns:
        List of all projects
    """
    result = await session.execute(
        select(Project).order_by(Project.updated_at.desc())
    )
    return list(result.scalars().all())


async def get_project(session: AsyncSession, project_id: int) -> Project:
    """Get a project by ID.

    Args:
        session: Registry database session
        project_id: ID of the project

    Returns:
        The project

    Raises:
        DBRecordNotFoundError: If project not found
    """
    result = await session.execute(
        select(Project).where(Project.id == project_id)
    )
    project = result.scalar_one_or_none()
    if project is None:
        raise DBRecordNotFoundError("project", project_id)
    return project


async def create_project(
    session: AsyncSession,
    name: str,
    parent_path: Path,
) -> Project:
    """Create a new project.

    Creates the project directory, initializes the project database,
    and registers the project in the registry.

    Args:
        session: Registry database session
        name: Name of the project (used as directory name)
        parent_path: Parent directory where project folder will be created

    Returns:
        The created project

    Raises:
        ParentDirectoryNotFoundError: If parent directory doesn't exist
        PathNotDirectoryError: If parent path is not a directory
        ProjectAlreadyExistsError: If project directory already exists
        PermissionDeniedError: If permission denied creating directory
    """
    if not parent_path.exists():
        raise ParentDirectoryNotFoundError(str(parent_path))

    if not parent_path.is_dir():
        raise PathNotDirectoryError(str(parent_path))

    project_dir = parent_path / name

    try:
        project_dir.mkdir(exist_ok=False)
    except FileExistsError:
        raise ProjectAlreadyExistsError(str(project_dir))
    except PermissionError:
        raise PermissionDeniedError(str(project_dir), "create directory")
    except OSError as e:
        raise PermissionDeniedError(str(project_dir), f"create directory ({e})")

    # Initialize project database
    db_manager = DatabaseManager.get_instance()
    await db_manager.init_project(project_dir)

    # Create and persist project record
    project = Project(name=name, path=str(project_dir))
    session.add(project)
    await session.commit()
    await session.refresh(project)

    return project


async def delete_project(session: AsyncSession, project_id: int) -> None:
    """Delete a project.

    Disposes the project database, removes from registry, and deletes
    the project directory.

    Args:
        session: Registry database session
        project_id: ID of the project to delete

    Raises:
        DBRecordNotFoundError: If project not found
    """
    project = await get_project(session, project_id)
    project_path = Path(project.path)

    # Dispose project database connections
    db_manager = DatabaseManager.get_instance()
    await db_manager.dispose_project(project_path)

    # Remove from registry
    await session.delete(project)
    await session.commit()

    # Delete project directory
    if project_path.exists():
        shutil.rmtree(project_path)
