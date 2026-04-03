"""Schemas for export endpoints."""

from pydantic import BaseModel


class ExportResponse(BaseModel):
    """Response from an export operation."""

    path: str
    row_count: int
