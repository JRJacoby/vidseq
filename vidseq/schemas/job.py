from pydantic import BaseModel
from datetime import datetime

class JobCreate(BaseModel):
    type: str
    project_id: int
    details: dict

class JobResponse(BaseModel):
    id: int
    type: str
    status: str
    project_id: int
    details: dict
    log_path: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


