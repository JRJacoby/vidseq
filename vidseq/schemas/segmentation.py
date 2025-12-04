"""Pydantic schemas for segmentation API."""

from typing import Any, Literal

from pydantic import BaseModel, field_validator


class SegmentRequest(BaseModel):
    """Request to run segmentation with a prompt."""
    frame_idx: int
    type: Literal["bbox", "positive_point", "negative_point"]
    details: dict[str, Any]
    text: str
    
    @field_validator("details")
    @classmethod
    def validate_details(cls, v: dict, info) -> dict:
        prompt_type = info.data.get("type")
        if prompt_type == "bbox":
            required = {"x1", "y1", "x2", "y2"}
            if not required.issubset(v.keys()):
                raise ValueError(f"bbox requires {required}")
        elif prompt_type in ("positive_point", "negative_point"):
            required = {"x", "y"}
            if not required.issubset(v.keys()):
                raise ValueError(f"point requires {required}")
        return v


class PropagateRequest(BaseModel):
    start_frame_idx: int
    max_frames: int = 100


class PropagateResponse(BaseModel):
    frames_processed: int
