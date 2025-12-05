"""Pydantic schemas for segmentation API."""

from typing import Any, Literal

from pydantic import BaseModel, field_validator


class SegmentRequest(BaseModel):
    """Request to run segmentation with a point prompt."""
    frame_idx: int
    type: Literal["positive_point", "negative_point"]
    details: dict[str, Any]
    
    @field_validator("details")
    @classmethod
    def validate_details(cls, v: dict, info) -> dict:
        required = {"x", "y"}
        if not required.issubset(v.keys()):
            raise ValueError(f"point requires {required}")
        return v


class PropagateRequest(BaseModel):
    start_frame_idx: int
    max_frames: int = 100


class PropagateResponse(BaseModel):
    frames_processed: int
