"""Pydantic schemas for segmentation API."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel


class PromptCreate(BaseModel):
    frame_idx: int
    type: str
    details: dict[str, Any]


class PromptResponse(BaseModel):
    id: int
    video_id: int
    frame_idx: int
    type: str
    details: dict[str, Any]
    created_at: datetime

    class Config:
        from_attributes = True

