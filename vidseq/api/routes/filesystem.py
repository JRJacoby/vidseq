from fastapi import APIRouter, HTTPException, Query
from pathlib import Path
from vidseq.schemas.filesystem import DirectoryEntry

router = APIRouter()

@router.get("/filesystem/list", response_model=list[DirectoryEntry])
async def list_directory(path: str = Query(..., description="Directory path to list")):
    dir_path = Path(path)
    
    if not dir_path.exists():
        raise HTTPException(status_code=404, detail=f"Path does not exist: {path}")
    
    if not dir_path.is_dir():
        raise HTTPException(status_code=400, detail=f"Path is not a directory: {path}")
    
    entries = []
    try:
        for item in dir_path.iterdir():
            entries.append(DirectoryEntry(
                name=item.name,
                path=str(item.absolute()),
                is_directory=item.is_dir()
            ))
    except PermissionError:
        raise HTTPException(status_code=403, detail=f"Permission denied accessing directory: {path}")
    
    entries.sort(key=lambda x: (not x.is_directory, x.name.lower()))
    
    return entries

