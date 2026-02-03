"""FrameData model for per-frame storage of bboxes, frame types, and scores."""
from sqlalchemy import Float, ForeignKey, Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from vidseq.models.project_db import Base


class FrameData(Base):
    """Per-frame data including bounding boxes, frame types, and confidence scores.

    This table stores frame-level metadata that was previously stored in HDF5.
    Each row represents one frame of one video.
    """
    __tablename__ = "frame_data"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    video_id: Mapped[int] = mapped_column(Integer, ForeignKey("videos.id"), nullable=False)
    frame_idx: Mapped[int] = mapped_column(Integer, nullable=False)

    # Bounding box coordinates (NULL = no bbox)
    bbox_x1: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
    bbox_y1: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
    bbox_x2: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
    bbox_y2: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)

    # Frame type: 'train', 'apply', or NULL (no type assigned)
    frame_type: Mapped[str | None] = mapped_column(String(10), nullable=True, default=None)

    # Segmentation confidence score (-1.0 = not computed)
    score: Mapped[float] = mapped_column(Float, nullable=False, default=-1.0)

    # Mask presence flags: 1 = has mask, 0 = no mask, NULL = unknown
    # Using Integer for SQLite compatibility (no native boolean type)
    has_tracker_mask: Mapped[int | None] = mapped_column(Integer, nullable=True, default=None)
    has_detector_mask: Mapped[int | None] = mapped_column(Integer, nullable=True, default=None)
    has_final_mask: Mapped[int | None] = mapped_column(Integer, nullable=True, default=None)

    __table_args__ = (
        # Primary lookup: video + frame (unique constraint)
        Index("ix_frame_data_video_frame", "video_id", "frame_idx", unique=True),
        # Training frame queries: find all frames with a specific type
        Index("ix_frame_data_video_type", "video_id", "frame_type"),
        # Mask presence queries: find all frames with masks for a video
        Index("ix_frame_data_video_has_tracker_mask", "video_id", "has_tracker_mask"),
        Index("ix_frame_data_video_has_detector_mask", "video_id", "has_detector_mask"),
        Index("ix_frame_data_video_has_final_mask", "video_id", "has_final_mask"),
    )
