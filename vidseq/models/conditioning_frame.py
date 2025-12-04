"""ConditioningFrame model for tracking frames with user prompts."""
from sqlalchemy import ForeignKey, Integer, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from vidseq.models.project_db import Base


class ConditioningFrame(Base):
    __tablename__ = "conditioning_frames"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    video_id: Mapped[int] = mapped_column(Integer, ForeignKey("videos.id"), nullable=False)
    frame_idx: Mapped[int] = mapped_column(Integer, nullable=False)
    
    __table_args__ = (
        UniqueConstraint("video_id", "frame_idx", name="uq_video_frame"),
    )

