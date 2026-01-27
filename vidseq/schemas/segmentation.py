"""Pydantic schemas for segmentation API."""

from typing import Any, Literal

from pydantic import BaseModel, field_validator


class SegmentRequest(BaseModel):
    """Request to run segmentation with a point or bounding box prompt."""
    frame_idx: int
    type: Literal["positive_point", "negative_point", "bounding_box"]
    details: dict[str, Any]

    @field_validator("details")
    @classmethod
    def validate_details(cls, v: dict, info) -> dict:
        prompt_type = info.data.get("type")
        if prompt_type in ("positive_point", "negative_point"):
            required = {"x", "y"}
            if not required.issubset(v.keys()):
                raise ValueError(f"point requires {required}")
        elif prompt_type == "bounding_box":
            required = {"x1", "y1", "x2", "y2"}
            if not required.issubset(v.keys()):
                raise ValueError(f"bounding_box requires {required}")
        return v


class PropagateRequest(BaseModel):
    start_frame_idx: int
    max_frames: int = 1000


class PropagateResponse(BaseModel):
    frames_processed: int
