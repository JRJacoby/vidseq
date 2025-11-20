from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, DateTime, JSON
from datetime import datetime, timezone

def now():
    return datetime.now(timezone.utc)

class Base(DeclarativeBase):
    pass

class Project(Base):
    __tablename__ = "projects"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String)
    path: Mapped[str] = mapped_column(String, unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=now)

class Job(Base):
    __tablename__ = "jobs"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    type: Mapped[str] = mapped_column(String)
    status: Mapped[str] = mapped_column(String)
    project_id: Mapped[int] = mapped_column()
    details: Mapped[dict] = mapped_column(JSON)
    log_path: Mapped[str] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=now)