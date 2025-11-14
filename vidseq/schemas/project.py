from pydantic import BaseModel
from datetime import datetime

class ProjectCreate(BaseModel):
    name: str
    path: str

class ProjectResponse(BaseModel):
    id: int
    name: str
    path: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True