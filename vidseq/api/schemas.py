"""Shared Pydantic request schemas for API routes."""

from pydantic import BaseModel


class VideoSelectionRequest(BaseModel):
    """Request body containing a list of video IDs to operate on."""

    video_ids: list[int]


class AlignmentTrainingRequest(BaseModel):
    """Request body for alignment training."""

    video_ids: list[int]
    epochs: int = 100
    augment: bool = True
    early_stop_patience: int = 5
    lr_patience: int = 3


class PCARequest(BaseModel):
    """Request body for PCA computation."""

    video_ids: list[int]
    n_components: int = 20
