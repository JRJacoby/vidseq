import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select
from pathlib import Path
from os import PathLike
from platformdirs import user_data_dir
from fastapi import Depends, HTTPException
from vidseq.models.registry import Base as RegistryBase, Project, Job
from vidseq.models.project import Base as ProjectBase
from vidseq.models.prompt import Prompt  # noqa: F401 - needed for table creation

APP_DATA_DIR = Path(user_data_dir("vidseq"))
REGISTRY_DB_PATH = APP_DATA_DIR / "registry.db"

registry_engine = None
RegistrySessionLocal = None

project_engines = {}
project_sessions = {}

async def init_registry_db():
    global registry_engine, RegistrySessionLocal

    REGISTRY_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    registry_engine = create_async_engine(
        f"sqlite+aiosqlite:///{REGISTRY_DB_PATH}",
        echo=True,
    )

    RegistrySessionLocal = sessionmaker(
        registry_engine, class_=AsyncSession, expire_on_commit=False
    )

    async with registry_engine.begin() as conn:
        await conn.run_sync(RegistryBase.metadata.create_all)

async def get_registry_db():
    if RegistrySessionLocal is None:
        raise RuntimeError("Registry database not initialized. Call init_registry_db() first.")

    async with RegistrySessionLocal() as session:
        yield session

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

def get_project_db_path(project_folder: Path) -> Path:
    return project_folder / "vidseq.db"

async def init_project_db(project_folder: PathLike):
    project_folder = Path(project_folder)
    db_path = get_project_db_path(project_folder)
    
    project_folder.mkdir(parents=True, exist_ok=True)

    engine = create_async_engine(
        f'sqlite+aiosqlite:///{db_path}',
        echo=True,
    )

    session_factory = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )

    project_engines[str(project_folder)] = engine
    project_sessions[str(project_folder)] = session_factory

    async with engine.begin() as conn:
        await conn.run_sync(ProjectBase.metadata.create_all)

def get_project_db(project_folder: PathLike):
    project_folder = Path(project_folder)
    project_key = str(project_folder)

    async def get_session():
        if project_key not in project_sessions:
            await init_project_db(project_folder)
        
        session_factory = project_sessions[project_key]
        async with session_factory() as session:
            yield session

    return get_session

async def get_project_session(
    project_folder: Path = Depends(get_project_folder)
):
    project_key = str(project_folder)
    
    if project_key not in project_sessions:
        db_path = get_project_db_path(project_folder)
        if not db_path.exists():
            raise HTTPException(
                status_code=404, 
                detail=f"Project database not found at {db_path}. Project may not be properly initialized."
            )
        await init_project_db(project_folder)
    
    session_factory = project_sessions[project_key]
    async with session_factory() as session:
        yield session