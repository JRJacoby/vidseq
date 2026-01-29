"""API routes for detector training and inference."""

import asyncio
import json
import logging
from pathlib import Path

import h5py
import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

from vidseq.api.dependencies import get_project_folder, get_video
from vidseq.models.video import Video
from vidseq.services.detector_service import DetectorService
from vidseq.services import segmentation_service

logger = logging.getLogger(__name__)

router = APIRouter()


class DetectorStatusResponse(BaseModel):
    """Response for detector status endpoint."""

    model_exists: bool
    is_training: bool


class TrainRequest(BaseModel):
    """Request body for training endpoint."""

    max_epochs: int = 1000
    batch_size: int = 4
    lr: float = 1e-4
    lr_patience: int = 10
    lr_factor: float = 0.25
    early_stop_patience: int = 20


class DetectorMasksExistsResponse(BaseModel):
    """Response for detector masks existence check."""

    exists: bool


@router.get(
    "/projects/{project_id}/detector/status",
    response_model=DetectorStatusResponse,
)
async def get_detector_status(
    project_path: Path = Depends(get_project_folder),
):
    """
    Get detector model status.

    Returns whether a trained model exists and if training is in progress.
    """
    service = DetectorService.get_instance()
    return DetectorStatusResponse(
        model_exists=service.model_exists(project_path),
        is_training=service.is_training(),
    )


@router.post("/projects/{project_id}/detector/train")
async def start_detector_training(
    request: TrainRequest,
    project_path: Path = Depends(get_project_folder),
):
    """
    Start detector training.

    Returns immediately. Training runs asynchronously in a background thread.
    Use the training status endpoints to monitor progress.
    """
    service = DetectorService.get_instance()

    if service.is_training():
        raise HTTPException(status_code=409, detail="Training already in progress")

    try:
        service.train(
            project_path=project_path,
            max_epochs=request.max_epochs,
            batch_size=request.batch_size,
            lr=request.lr,
            lr_patience=request.lr_patience,
            lr_factor=request.lr_factor,
            early_stop_patience=request.early_stop_patience,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"message": "Training started"}


@router.post("/projects/{project_id}/detector/stop")
async def stop_detector_training():
    """
    Stop detector training or apply.

    Requests the current operation to stop gracefully.
    """
    service = DetectorService.get_instance()

    if not service.is_training():
        raise HTTPException(status_code=400, detail="No training or apply in progress")

    stopped = service.stop_training()
    return {"stopped": stopped}


@router.post("/projects/{project_id}/detector/apply-all")
async def apply_detector_to_all(
    project_path: Path = Depends(get_project_folder),
):
    """
    Apply detector to all frames of all videos in the project.

    Runs in background. Skips frames that already have detector masks.
    Use /training/status or /training/stream to monitor progress.
    """
    service = DetectorService.get_instance()

    try:
        service.apply_to_all(project_path)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"message": "Apply started"}


@router.get("/projects/{project_id}/detector/training/status")
async def get_training_status():
    """
    Get current training progress.

    Returns detailed progress information including epoch, loss, and status.
    """
    service = DetectorService.get_instance()
    progress = service.get_training_progress()
    return progress.to_dict()


@router.get("/projects/{project_id}/detector/training/stream")
async def stream_training_status():
    """
    SSE stream for real-time training updates.

    Emits JSON updates when epoch or status changes.
    Stream ends when training completes, fails, or is stopped.
    """

    async def event_generator():
        service = DetectorService.get_instance()
        last_progress_str = None

        while True:
            progress = service.get_training_progress()
            progress_dict = progress.to_dict()
            progress_str = json.dumps(progress_dict)

            # Emit if changed
            if progress_str != last_progress_str:
                yield f"data: {progress_str}\n\n"
                last_progress_str = progress_str

            # Stop streaming when training ends
            if progress.status in ("completed", "failed", "stopped", "idle"):
                # If we just emitted, give client a moment to process
                if not progress.is_training:
                    break

            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@router.get("/projects/{project_id}/videos/{video_id}/detector-mask/{frame_idx}")
async def get_detector_mask(
    frame_idx: int,
    video: Video = Depends(get_video),
    project_path: Path = Depends(get_project_folder),
):
    """
    Get detector mask for a specific frame.

    Returns PNG binary. If no detector mask exists, returns a transparent (all zeros) mask.
    """
    h5_path = project_path / "masks" / f"{video.id}_detector.h5"

    if not h5_path.exists():
        # Return empty mask
        mask = np.zeros((video.height, video.width), dtype=np.uint8)
        mask_png = segmentation_service.mask_to_png(mask)
        return Response(content=mask_png, media_type="image/png")

    try:
        with h5py.File(h5_path, "r") as f:
            if "masks" not in f:
                # No masks dataset
                mask = np.zeros((video.height, video.width), dtype=np.uint8)
            else:
                mask = np.array(f["masks"][frame_idx])
    except (OSError, KeyError) as e:
        logger.warning(f"Error reading detector mask: {e}")
        mask = np.zeros((video.height, video.width), dtype=np.uint8)

    mask_png = segmentation_service.mask_to_png(mask)
    return Response(content=mask_png, media_type="image/png")


@router.get("/projects/{project_id}/videos/{video_id}/detector-masks-batch")
async def get_detector_masks_batch(
    start_frame: int,
    count: int = 100,
    video: Video = Depends(get_video),
    project_path: Path = Depends(get_project_folder),
):
    """
    Get multiple detector masks in a single request.

    Returns JSON with base64-encoded PNG masks for efficient batch transfer.
    """
    import base64

    h5_path = project_path / "masks" / f"{video.id}_detector.h5"
    end_frame = min(start_frame + count, video.num_frames)
    actual_count = end_frame - start_frame

    masks_list = []

    if not h5_path.exists():
        # Return empty masks
        for i in range(actual_count):
            mask = np.zeros((video.height, video.width), dtype=np.uint8)
            png_bytes = segmentation_service.mask_to_png(mask)
            png_base64 = base64.b64encode(png_bytes).decode('ascii')
            masks_list.append({"frame_idx": start_frame + i, "png_base64": png_base64})
    else:
        try:
            with h5py.File(h5_path, "r") as f:
                if "masks" not in f:
                    # No masks dataset - return empty
                    for i in range(actual_count):
                        mask = np.zeros((video.height, video.width), dtype=np.uint8)
                        png_bytes = segmentation_service.mask_to_png(mask)
                        png_base64 = base64.b64encode(png_bytes).decode('ascii')
                        masks_list.append({"frame_idx": start_frame + i, "png_base64": png_base64})
                else:
                    masks = np.array(f["masks"][start_frame:end_frame])
                    for i, mask in enumerate(masks):
                        png_bytes = segmentation_service.mask_to_png(mask)
                        png_base64 = base64.b64encode(png_bytes).decode('ascii')
                        masks_list.append({"frame_idx": start_frame + i, "png_base64": png_base64})
        except (OSError, KeyError) as e:
            logger.warning(f"Error reading detector masks batch: {e}")
            for i in range(actual_count):
                mask = np.zeros((video.height, video.width), dtype=np.uint8)
                png_bytes = segmentation_service.mask_to_png(mask)
                png_base64 = base64.b64encode(png_bytes).decode('ascii')
                masks_list.append({"frame_idx": start_frame + i, "png_base64": png_base64})

    return {"masks": masks_list}


@router.get(
    "/projects/{project_id}/videos/{video_id}/detector-masks/exists",
    response_model=DetectorMasksExistsResponse,
)
async def check_detector_masks_exist(
    video: Video = Depends(get_video),
    project_path: Path = Depends(get_project_folder),
):
    """
    Check if detector masks exist for a video.

    Returns true if the detector h5 file exists.
    """
    h5_path = project_path / "masks" / f"{video.id}_detector.h5"

    return DetectorMasksExistsResponse(exists=h5_path.exists())
