"""Pydantic schemas for threshold segmentation API."""

from pydantic import BaseModel, Field


class ThresholdSegmentRequest(BaseModel):
    start_frame: int
    end_frame: int
    threshold: int = Field(ge=0, le=255)
    click_x: int
    click_y: int
    click_frame: int


class ThresholdSegmentResponse(BaseModel):
    frames_processed: int
