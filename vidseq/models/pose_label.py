"""PoseLabel model for storing front/rear keypoint annotations."""
from sqlalchemy import Float, ForeignKey, Index, Integer
from sqlalchemy.orm import Mapped, mapped_column

from vidseq.models.project_db import Base


class PoseLabel(Base):
    """Per-frame pose labels with front and rear keypoint coordinates.

    Used for training the YOLO pose model. Coordinates are
    normalized (0-1) relative to original video frame dimensions.
    """
    __tablename__ = "pose_labels"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    video_id: Mapped[int] = mapped_column(Integer, ForeignKey("videos.id"), nullable=False)
    frame_idx: Mapped[int] = mapped_column(Integer, nullable=False)

    # Front point (nose) - normalized coordinates (0-1)
    front_x: Mapped[float] = mapped_column(Float, nullable=False)
    front_y: Mapped[float] = mapped_column(Float, nullable=False)

    # Rear point (tail) - normalized coordinates (0-1)
    rear_x: Mapped[float] = mapped_column(Float, nullable=False)
    rear_y: Mapped[float] = mapped_column(Float, nullable=False)

    __table_args__ = (
        # Unique constraint: one label per video/frame pair
        Index("ix_pose_label_video_frame", "video_id", "frame_idx", unique=True),
    )
