"""API routes for detector training and inference."""

import asyncio
import json
from pathlib import Path

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

from sqlalchemy.ext.asyncio import AsyncSession

from vidseq.api.dependencies import get_project_folder, get_project_session, get_video
from vidseq.models.video import Video
from vidseq.services.detector_service import DetectorService
from vidseq.services import frame_data_service, segmentation_service
from vidseq.services.array_storage import detector_masks

router = APIRouter()


class DetectorStatusResponse(BaseModel):
    """Response for detector status endpoint."""

    model_exists: bool
    is_training: bool


class TrainRequest(BaseModel):
    """Request body for training endpoint."""

    model_config = {"extra": "ignore"}

    video_ids: list[int]
    max_epochs: int = 100
    batch_size: int = 4
    lr: float = 1e-4
    early_stop_patience: int = 20


class DetectorMasksExistsResponse(BaseModel):
    """Response for detector masks existence check."""

    exists: bool


@router.get(
    "/projects/{project_id}/detection/status",
    response_model=DetectorStatusResponse,
)
async def get_detection_status(
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


@router.post("/projects/{project_id}/detection/training")
async def create_detection_training(
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
            video_ids=request.video_ids,
            max_epochs=request.max_epochs,
            batch_size=request.batch_size,
            lr=request.lr,
            early_stop_patience=request.early_stop_patience,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"status": "started"}


@router.delete("/projects/{project_id}/detection/training", status_code=204)
async def delete_detection_training():
    """
    Stop detector training or apply.

    Requests the current operation to stop gracefully.
    """
    service = DetectorService.get_instance()

    if not service.is_training():
        raise HTTPException(status_code=400, detail="No training or apply in progress")

    service.stop_training()
    return None


@router.get("/projects/{project_id}/detection/training")
async def get_detection_training():
    """
    Get current training progress.

    Returns detailed progress information including epoch, loss, and status.
    """
    service = DetectorService.get_instance()
    progress = service.get_training_progress()
    return progress.to_dict()


@router.get("/projects/{project_id}/detection/training/stream")
async def stream_detection_training():
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
    h5_path = project_path / "array_data" / str(video.id) / "detector_masks.h5"

    return DetectorMasksExistsResponse(exists=h5_path.exists())


@router.get("/projects/{project_id}/videos/{video_id}/detector-masks/{frame_idx}")
async def get_detector_mask(
    frame_idx: int,
    video: Video = Depends(get_video),
    project_path: Path = Depends(get_project_folder),
):
    """
    Get detector mask for a specific frame.

    Returns PNG binary.
    """
    with detector_masks(project_path, video.id, "r") as masks:
        mask = masks[frame_idx]

    mask_png = segmentation_service.mask_to_png(mask)
    return Response(content=mask_png, media_type="image/png")


@router.get("/projects/{project_id}/videos/{video_id}/detector-masks")
async def get_detector_masks(
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

    end_frame = min(start_frame + count, video.num_frames)
    masks_list = []

    with detector_masks(project_path, video.id, "r") as masks_ds:
        masks_array = np.array(masks_ds[start_frame:end_frame])
        for i, mask in enumerate(masks_array):
            png_bytes = segmentation_service.mask_to_png(mask)
            png_base64 = base64.b64encode(png_bytes).decode('ascii')
            masks_list.append({"frame_idx": start_frame + i, "png_base64": png_base64})

    return {"masks": masks_list}


@router.get("/projects/{project_id}/videos/{video_id}/detector-bboxes/{frame_idx}")
async def get_detector_bbox_endpoint(
    frame_idx: int,
    video: Video = Depends(get_video),
    session: AsyncSession = Depends(get_project_session),
):
    """Get detector bbox for a single frame."""
    bbox = await frame_data_service.get_detector_bbox(session, video.id, frame_idx)
    return {"bbox": bbox}


@router.get("/projects/{project_id}/videos/{video_id}/detector-bboxes")
async def get_detector_bboxes_endpoint(
    start_frame: int,
    count: int = 100,
    video: Video = Depends(get_video),
    session: AsyncSession = Depends(get_project_session),
):
    """Get detector bboxes for a range of frames."""
    bboxes = await frame_data_service.get_detector_bboxes_batch(
        session, video.id, start_frame, count
    )
    return {"bboxes": bboxes}
