from pydantic import BaseModel

class VideoCreate(BaseModel):
    paths: list[str]

class VideoResponse(BaseModel):
    id: int
    name: str
    path: str
    has_segmentation: bool

    class Config:
        from_attributes = True

