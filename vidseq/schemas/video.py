from pydantic import BaseModel

class VideoCreate(BaseModel):
    paths: list[str]

class VideoResponse(BaseModel):
    id: int
    name: str
    path: str
    fps: float
    segmentation_status: str | None = None
    # New fields for associated video support
    is_associated: bool = False
    associated_with_id: int | None = None
    associated_video_id: int | None = None  # Populated by service layer, not from_attributes

    class Config:
        from_attributes = True

