"""YOLO API routes. Uses YOLOv11-nano for bounding box detection."""

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from vidseq.api.dependencies import get_project_folder
from vidseq.services import sam2_service, yolo_service

router = APIRouter()


@router.post("/projects/{project_id}/train-model")
async def train_model(
    project_id: int,
    project_path: Path = Depends(get_project_folder),
):
    """
    Start training YOLO model for the project.
    
    Uses frames marked as 'train' from all videos in the project.
    Trains the initial detection model.
    """
    service = yolo_service.YOLOService.get_instance()
    
    if service.is_training():
        raise HTTPException(status_code=400, detail="Training already in progress")
    
    # Shutdown SAM2 to free GPU memory before training
    sam2_status = sam2_service.get_status()
    if sam2_status["status"] == "ready":
        print("[YOLO API] Shutting down SAM2 to free GPU memory for training...")
        sam2_service.shutdown_worker()
    
    try:
        service.train_model(project_path)
        return {"message": "Training started"}
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/projects/{project_id}/run-initial-detection")
async def run_initial_detection(
    project_id: int,
    project_path: Path = Depends(get_project_folder),
):
    """
    Run initial detection on all videos in the project.
    
    Applies trained YOLO model to frame 0 of each video in the project.
    Skips videos that already have a bounding box on frame 0.
    """
    service = yolo_service.YOLOService.get_instance()
    
    if service.is_applying():
        raise HTTPException(status_code=400, detail="Application already in progress")
    
    if not service.model_exists(project_path):
        raise HTTPException(status_code=404, detail="Model not found. Train model first.")
    
    try:
        service.run_initial_detection(project_path)
        return {"message": "Initial detection started"}
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/projects/{project_id}/model-status")
async def get_model_status(
    project_id: int,
    project_path: Path = Depends(get_project_folder),
):
    """Check if YOLO model exists for the project."""
    service = yolo_service.YOLOService.get_instance()
    exists = service.model_exists(project_path)
    model_path = str(service.get_model_path(project_path)) if exists else None
    
    return {
        "exists": exists,
        "model_path": model_path,
        "is_training": service.is_training(),
        "is_applying": service.is_applying(),
    }

