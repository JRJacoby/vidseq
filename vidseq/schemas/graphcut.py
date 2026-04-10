"""Pydantic schemas for graph cut segmentation API."""

from pydantic import BaseModel, field_validator


class SeedPoint(BaseModel):
    x: int
    y: int
    label: int  # 1 = foreground, 2 = background


class GraphCutRequest(BaseModel):
    start_frame: int
    end_frame: int
    seeds: dict[str, list[SeedPoint]]  # frame_idx (as string) -> list of seed points

    @field_validator("seeds")
    @classmethod
    def validate_seeds(cls, v: dict[str, list[SeedPoint]]) -> dict[str, list[SeedPoint]]:
        all_labels = set()
        total_points = 0
        for points in v.values():
            for p in points:
                if p.label not in (1, 2):
                    raise ValueError("Seed label must be 1 (foreground) or 2 (background)")
                all_labels.add(p.label)
                total_points += 1
        if total_points == 0:
            raise ValueError("At least one seed point is required")
        if len(all_labels) < 2:
            raise ValueError("Both foreground (1) and background (2) seeds are required")
        return v


class GraphCutResponse(BaseModel):
    frames_processed: int
