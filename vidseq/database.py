import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from pathlib import Path
from os import PathLike
from platformdirs import user_data_dir
from vidseq.models.registry import Base as RegistryBase
from vidseq.models.project import Base as ProjectBase

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