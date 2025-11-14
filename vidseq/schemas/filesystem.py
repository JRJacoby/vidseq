from pydantic import BaseModel, Field

class DirectoryEntry(BaseModel):
    name: str
    path: str
    is_directory: bool = Field(alias="isDirectory")
    
    class Config:
        populate_by_name = True

