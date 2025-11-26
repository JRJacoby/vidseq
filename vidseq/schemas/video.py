from pydantic import BaseModel

class VideoCreate(BaseModel):
    paths: list[str]

class VideoResponse(BaseModel):
    id: int
    name: str
    path: str
    fps: float

    class Config:
        from_attributes = True

