"""Video model for per-project database."""
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import String, Float, Integer

from vidseq.models.project_db import Base


class Video(Base):
    __tablename__ = "videos"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String)
    path: Mapped[str] = mapped_column(String)
    fps: Mapped[float] = mapped_column(Float)
    height: Mapped[int] = mapped_column(Integer)
    width: Mapped[int] = mapped_column(Integer)
    num_frames: Mapped[int] = mapped_column(Integer)
