"""Keypoint conditioning frame model — tracks which frames have user-provided keypoint prompts."""

from sqlalchemy import Column, Integer, UniqueConstraint, ForeignKey
from vidseq.models.project_db import Base


class KeypointConditioningFrame(Base):
    __tablename__ = "keypoint_conditioning_frames"

    id = Column(Integer, primary_key=True)
    video_id = Column(Integer, ForeignKey("videos.id"), nullable=False)
    frame_idx = Column(Integer, nullable=False)
    obj_id = Column(Integer, nullable=False)  # 0=front, 1=rear

    __table_args__ = (
        UniqueConstraint("video_id", "frame_idx", "obj_id", name="uq_kp_cond_frame"),
    )
