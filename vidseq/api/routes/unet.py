"""UNet API routes."""

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from vidseq.api.dependencies import get_project_folder, get_project_session
from vidseq.services import sam2_service, unet_service, video_service

router = APIRouter()


@router.post("/projects/{project_id}/train-model")
async def train_model(
    project_id: int,
    project_path: Path = Depends(get_project_folder),
):
    """
    Start training UNet model for the project.
    
    Uses frames marked as 'train' from all videos in the project.
    """
    from vidseq.services import mask_service
    
    service = unet_service.UNetService.get_instance()
    
    if service.is_training():
        raise HTTPException(status_code=400, detail="Training already in progress")
    
    # Shutdown SAM2 to free GPU memory before training
    sam2_status = sam2_service.get_status()
    if sam2_status["status"] == "ready":
        print("[UNet API] Shutting down SAM2 to free GPU memory for training...")
        sam2_service.shutdown_worker()
    
    try:
        service.train_model(project_path)
        return {"message": "Training started"}
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/projects/{project_id}/videos/{video_id}/apply-model")
async def apply_model(
    project_id: int,
    video_id: int,
    session: AsyncSession = Depends(get_project_session),
    project_path: Path = Depends(get_project_folder),
):
    """
    Apply trained UNet model to a video.
    
    Predicts masks for frames that don't already have masks.
    """
    try:
        await video_service.get_video_by_id(session, video_id)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    
    service = unet_service.UNetService.get_instance()
    
    if service.is_applying():
        raise HTTPException(status_code=400, detail="Application already in progress")
    
    if not service.model_exists(project_path):
        raise HTTPException(status_code=404, detail="Model not found. Train model first.")
    
    try:
        service.apply_model(project_path, video_id)
        return {"message": "Application started"}
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/projects/{project_id}/videos/{video_id}/test-apply-model")
async def test_apply_model(
    project_id: int,
    video_id: int,
    start_frame: int = Query(0, description="Starting frame index"),
    session: AsyncSession = Depends(get_project_session),
    project_path: Path = Depends(get_project_folder),
):
    """
    Test apply trained UNet model to a limited range of frames.
    
    Applies model to current frame plus next 1000 frames, skipping frames labeled 'train'.
    """
    try:
        await video_service.get_video_by_id(session, video_id)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    
    service = unet_service.UNetService.get_instance()
    
    if service.is_applying():
        raise HTTPException(status_code=400, detail="Application already in progress")
    
    if not service.model_exists(project_path):
        raise HTTPException(status_code=404, detail="Model not found. Train model first.")
    
    try:
        service.test_apply_model(project_path, video_id, start_frame)
        return {"message": "Test application started"}
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/projects/{project_id}/model-status")
async def get_model_status(
    project_id: int,
    project_path: Path = Depends(get_project_folder),
):
    """Check if UNet model exists for the project."""
    service = unet_service.UNetService.get_instance()
    exists = service.model_exists(project_path)
    model_path = str(service.get_model_path(project_path)) if exists else None
    
    return {
        "exists": exists,
        "model_path": model_path,
        "is_training": service.is_training(),
        "is_applying": service.is_applying(),
    }

