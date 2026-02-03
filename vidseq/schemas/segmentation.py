"""Pydantic schemas for segmentation API."""

from pydantic import BaseModel, field_validator


class PointPrompt(BaseModel):
    """A single point prompt with normalized coordinates."""
    x: float
    y: float
    type: str  # "positive_point" or "negative_point"

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        if v not in ("positive_point", "negative_point"):
            raise ValueError("type must be 'positive_point' or 'negative_point'")
        return v


class PromptRequest(BaseModel):
    """Request to submit point prompts for segmentation.

    Accepts 1 or more points. Backend determines workflow:
    - 1 point, no existing mask: add_point_prompt (new mask)
    - 1 point, existing mask: refine (single point refinement)
    - 2+ points, existing mask: refine (multi-point refinement)
    - 2+ points, no existing mask: ERROR (can't refine without mask)
    """
    points: list[PointPrompt]

    @field_validator("points")
    @classmethod
    def validate_points(cls, v: list[PointPrompt]) -> list[PointPrompt]:
        if len(v) == 0:
            raise ValueError("At least one point is required")
        return v


class PropagateRequest(BaseModel):
    start_frame_idx: int
    max_frames: int = 1000


class PropagateResponse(BaseModel):
    frames_processed: int


# Backward compatibility aliases (deprecated, will be removed)
# These allow inference.py to continue working until Task 2 refactors it
SegmentRequest = PromptRequest  # type: ignore[assignment]
MultiPointSegmentRequest = PromptRequest  # type: ignore[assignment]
