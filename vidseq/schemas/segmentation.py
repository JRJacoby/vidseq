from pydantic import BaseModel

class SegmentationRequest(BaseModel):
    video_ids: list[int]
    prompt: str


