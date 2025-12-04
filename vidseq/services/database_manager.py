"""Database engine and session management."""

import threading
from pathlib import Path
from typing import Optional

from platformdirs import user_data_dir
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from vidseq.models.project_db import Base as ProjectBase
from vidseq.models.video import Video  # noqa: F401 - registers with ProjectBase.metadata
from vidseq.models.conditioning_frame import ConditioningFrame  # noqa: F401 - registers with ProjectBase.metadata
from vidseq.models.registry import Base as RegistryBase


APP_DATA_DIR = Path(user_data_dir("vidseq"))
REGISTRY_DB_PATH = APP_DATA_DIR / "registry.db"


class DatabaseManager:
    """
    Singleton manager for database engines and session factories.
    
    Handles both the global registry database and per-project databases.
    """
    
    _instance: Optional["DatabaseManager"] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> "DatabaseManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._registry_engine: Optional[AsyncEngine] = None
        self._registry_session_factory: Optional[sessionmaker] = None
        self._project_engines: dict[str, AsyncEngine] = {}
        self._project_session_factories: dict[str, sessionmaker] = {}
        
        self._initialized = True
    
    @classmethod
    def get_instance(cls) -> "DatabaseManager":
        """Get the singleton instance."""
        return cls()
    
    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton (useful for testing)."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance = None
    
    @property
    def registry_engine(self) -> Optional[AsyncEngine]:
        """Get the registry engine (for server shutdown)."""
        return self._registry_engine
    
    @property
    def project_engines(self) -> dict[str, AsyncEngine]:
        """Get all project engines (for server shutdown)."""
        return self._project_engines
    
    async def init_registry(self) -> None:
        """Initialize the global registry database."""
        if self._registry_engine is not None:
            return
        
        REGISTRY_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        self._registry_engine = create_async_engine(
            f"sqlite+aiosqlite:///{REGISTRY_DB_PATH}",
            echo=True,
        )
        
        self._registry_session_factory = sessionmaker(
            self._registry_engine, class_=AsyncSession, expire_on_commit=False
        )
        
        async with self._registry_engine.begin() as conn:
            await conn.run_sync(RegistryBase.metadata.create_all)
    
    def get_registry_session_factory(self) -> sessionmaker:
        """Get the registry session factory."""
        if self._registry_session_factory is None:
            raise RuntimeError("Registry not initialized. Call init_registry() first.")
        return self._registry_session_factory
    
    async def init_project(self, project_folder: Path) -> None:
        """Initialize a project-specific database."""
        key = str(project_folder)
        if key in self._project_engines:
            return
        
        db_path = project_folder / "vidseq.db"
        project_folder.mkdir(parents=True, exist_ok=True)
        
        engine = create_async_engine(
            f"sqlite+aiosqlite:///{db_path}",
            echo=True,
        )
        
        self._project_engines[key] = engine
        self._project_session_factories[key] = sessionmaker(
            engine, class_=AsyncSession, expire_on_commit=False
        )
        
        async with engine.begin() as conn:
            await conn.run_sync(ProjectBase.metadata.create_all)
    
    def get_project_session_factory(self, project_folder: Path) -> sessionmaker:
        """Get the session factory for a project."""
        key = str(project_folder)
        if key not in self._project_session_factories:
            raise RuntimeError(f"Project database not initialized: {project_folder}")
        return self._project_session_factories[key]
    
    def is_project_initialized(self, project_folder: Path) -> bool:
        """Check if a project database is initialized."""
        return str(project_folder) in self._project_session_factories
    
    async def dispose_project(self, project_folder: Path) -> None:
        """Dispose a project's database engine and remove from cache."""
        key = str(project_folder)
        
        if key in self._project_engines:
            engine = self._project_engines.pop(key)
            await engine.dispose()
        
        self._project_session_factories.pop(key, None)
    
    async def shutdown(self) -> None:
        """Dispose all database engines."""
        if self._registry_engine:
            await self._registry_engine.dispose()
            self._registry_engine = None
            self._registry_session_factory = None
        
        for engine in self._project_engines.values():
            await engine.dispose()
        
        self._project_engines.clear()
        self._project_session_factories.clear()

