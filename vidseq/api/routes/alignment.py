"""API routes for egocentric alignment."""

import asyncio
import json
import logging
import mimetypes
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from vidseq.api.dependencies import get_project_folder, get_project_session, get_video
from vidseq.api.schemas import AlignmentTrainingRequest, VideoSelectionRequest
from vidseq.models.video import Video
from vidseq.services import alignment_service
from vidseq.services.alignment_service import AlignmentService


def get_aligned_video_path(project_path: Path, video_name: str) -> Path:
    """Get path to the aligned video file."""
    stem = Path(video_name).stem
    return project_path / "aligned_videos" / f"{stem}_cropped_aligned.mp4"

router = APIRouter()

# Configure logger for alignment API
logger = logging.getLogger("vidseq.alignment.api")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "[%(asctime)s] [Alignment API] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


# --- Schemas ---


class AlignmentLabelCreate(BaseModel):
    """Request body for creating an alignment label."""
    video_id: int
    frame_idx: int
    front_x: float
    front_y: float
    rear_x: float
    rear_y: float


class AlignmentLabelResponse(BaseModel):
    """Response for an alignment label."""
    id: int
    video_id: int
    frame_idx: int
    front_x: float
    front_y: float
    rear_x: float
    rear_y: float


class RandomFrameResponse(BaseModel):
    """Response for a random frame request."""
    video_id: int
    frame_idx: int


class AlignmentStatusResponse(BaseModel):
    """Response for alignment status."""
    label_count: int
    model_trained: bool
    is_training: bool
    is_applying: bool
    all_videos_cropped: bool


# --- Routes ---


@router.get("/projects/{project_id}/alignment/status")
async def get_alignment_status(
    project_id: int,
    session: AsyncSession = Depends(get_project_session),
    project_path: Path = Depends(get_project_folder),
) -> AlignmentStatusResponse:
    """Get current alignment training status."""
    service = AlignmentService.get_instance()
    status = await service.get_alignment_status(session, project_path)
    return AlignmentStatusResponse(**status)


@router.get("/projects/{project_id}/alignment/random-frame")
async def get_random_frame(
    project_id: int,
    session: AsyncSession = Depends(get_project_session),
    project_path: Path = Depends(get_project_folder),
) -> RandomFrameResponse:
    """Get a random unlabeled frame from cropped videos."""
    logger.info(f"GET /alignment/random-frame: project_id={project_id}")

    service = AlignmentService.get_instance()

    result = await service.get_random_unlabeled_frame(session, project_path)
    if result is None:
        logger.warning(f"GET /alignment/random-frame: no unlabeled frames available")
        raise HTTPException(
            status_code=404,
            detail="No unlabeled frames available. Either no videos have cropping completed, or all frames are labeled.",
        )

    video_id, frame_idx = result
    logger.info(f"GET /alignment/random-frame: returning video_id={video_id}, frame_idx={frame_idx}")
    return RandomFrameResponse(video_id=video_id, frame_idx=frame_idx)


@router.post("/projects/{project_id}/alignment/labels")
async def save_alignment_label(
    project_id: int,
    label: AlignmentLabelCreate,
    session: AsyncSession = Depends(get_project_session),
) -> AlignmentLabelResponse:
    """Save front/rear keypoint label for a frame."""
    logger.info(
        f"POST /alignment/labels: project_id={project_id}, "
        f"video_id={label.video_id}, frame_idx={label.frame_idx}, "
        f"front=({label.front_x:.4f}, {label.front_y:.4f}), "
        f"rear=({label.rear_x:.4f}, {label.rear_y:.4f})"
    )

    service = AlignmentService.get_instance()

    saved = await service.save_label(
        session,
        video_id=label.video_id,
        frame_idx=label.frame_idx,
        front_x=label.front_x,
        front_y=label.front_y,
        rear_x=label.rear_x,
        rear_y=label.rear_y,
    )

    response = AlignmentLabelResponse(
        id=saved.id,
        video_id=saved.video_id,
        frame_idx=saved.frame_idx,
        front_x=saved.front_x,
        front_y=saved.front_y,
        rear_x=saved.rear_x,
        rear_y=saved.rear_y,
    )
    logger.info(f"POST /alignment/labels: created/updated label id={saved.id}")
    logger.debug(f"POST /alignment/labels: response={response.model_dump()}")
    return response


@router.get("/projects/{project_id}/alignment/labels")
async def get_all_labels(
    project_id: int,
    session: AsyncSession = Depends(get_project_session),
) -> list[AlignmentLabelResponse]:
    """Get all alignment labels."""
    logger.info(f"GET /alignment/labels: project_id={project_id}")

    service = AlignmentService.get_instance()

    labels = await service.get_all_labels(session)

    response = [
        AlignmentLabelResponse(
            id=l.id,
            video_id=l.video_id,
            frame_idx=l.frame_idx,
            front_x=l.front_x,
            front_y=l.front_y,
            rear_x=l.rear_x,
            rear_y=l.rear_y,
        )
        for l in labels
    ]

    logger.info(f"GET /alignment/labels: returning {len(response)} labels")
    return response


@router.delete("/projects/{project_id}/alignment/labels")
async def delete_all_labels(
    project_id: int,
    session: AsyncSession = Depends(get_project_session),
):
    """Delete all alignment labels for the project."""
    logger.info(f"DELETE /alignment/labels: project_id={project_id}")

    service = AlignmentService.get_instance()

    deleted_count = await service.delete_all_labels(session)

    logger.info(f"DELETE /alignment/labels: deleted {deleted_count} labels")
    return {"deleted_count": deleted_count}


@router.delete("/projects/{project_id}/alignment/model", status_code=204)
async def delete_model(
    project_id: int,
    project_path: Path = Depends(get_project_folder),
):
    """Delete the alignment model for the project."""
    logger.info(f"DELETE /alignment/model: project_id={project_id}")

    service = AlignmentService.get_instance()

    service.delete_model(project_path)

    logger.info(f"DELETE /alignment/model: completed")
    return None


@router.post("/projects/{project_id}/alignment/training")
async def create_alignment_training(
    project_id: int,
    request: AlignmentTrainingRequest,
    project_path: Path = Depends(get_project_folder),
    session: AsyncSession = Depends(get_project_session),
):
    """Start alignment model training (fire-and-forget)."""
    service = AlignmentService.get_instance()
    return await service.create_alignment_training(
        session=session,
        project_path=project_path,
        video_ids=request.video_ids,
        epochs=request.epochs,
        augment=request.augment,
        early_stop_patience=request.early_stop_patience,
        lr_patience=request.lr_patience,
    )


@router.delete("/projects/{project_id}/alignment/training", status_code=204)
async def delete_alignment_training(project_id: int):
    """Stop alignment training gracefully."""
    service = AlignmentService.get_instance()
    if not service.is_training():
        raise HTTPException(status_code=400, detail="No training in progress")
    service.stop_training()
    return None


@router.get("/projects/{project_id}/alignment/training")
async def get_alignment_training(
    project_id: int,
):
    """Get current training progress (non-streaming).

    Returns training progress state including current epoch, loss, LR,
    patience counters, and loss history.
    """
    logger.info(f"GET /alignment/training/status: project_id={project_id}")

    service = AlignmentService.get_instance()
    progress = service.get_training_progress()

    logger.debug(f"GET /alignment/training/status: status={progress.status}, epoch={progress.current_epoch}")
    return progress.to_dict()


@router.get("/projects/{project_id}/alignment/training/stream")
async def stream_training_progress(
    project_id: int,
):
    """Stream training progress via Server-Sent Events.

    Streams real-time updates during training. The stream will close
    automatically when training completes, stops early, or fails.
    """
    logger.info(f"GET /alignment/training/stream: project_id={project_id} - SSE connection started")

    service = AlignmentService.get_instance()

    async def event_generator():
        last_epoch = -1
        last_status = None

        while True:
            progress = service.get_training_progress()

            # Send update if epoch changed or status changed
            if progress.current_epoch != last_epoch or progress.status != last_status:
                last_epoch = progress.current_epoch
                last_status = progress.status

                data = json.dumps(progress.to_dict())
                yield f"data: {data}\n\n"

                logger.debug(f"SSE: sent update - epoch={progress.current_epoch}, status={progress.status}")

                # Stop streaming if training finished
                if progress.status in ("completed", "stopped", "failed"):
                    logger.info(f"GET /alignment/training/stream: closing - status={progress.status}")
                    break

            await asyncio.sleep(0.5)  # Poll every 500ms

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )


@router.post("/projects/{project_id}/videos/alignment")
async def create_videos_alignment(
    project_id: int,
    request: VideoSelectionRequest,
    project_path: Path = Depends(get_project_folder),
):
    """Apply alignment to selected cropped videos."""
    service = AlignmentService.get_instance()
    return await service.create_videos_alignment(project_path, request.video_ids)


@router.get("/projects/{project_id}/videos/alignment/status")
async def get_videos_alignment_status(
    project_id: int,
):
    """Get current alignment (apply) progress status."""
    service = AlignmentService.get_instance()
    progress = service.get_alignment_progress()
    return progress.to_dict()


@router.get("/projects/{project_id}/videos/alignment/stream")
async def stream_videos_alignment(
    project_id: int,
):
    """Stream alignment (apply) progress via Server-Sent Events.

    Streams real-time updates during alignment. The stream will close
    automatically when alignment completes or fails.
    """
    logger.info(f"GET /alignment/apply/stream: project_id={project_id} - SSE connection started")

    service = AlignmentService.get_instance()

    async def event_generator():
        last_frame = -1
        last_status = None

        while True:
            progress = service.get_alignment_progress()

            # Send update if frame changed or status changed
            if progress.current_frame != last_frame or progress.status != last_status:
                last_frame = progress.current_frame
                last_status = progress.status

                data = json.dumps(progress.to_dict())
                yield f"data: {data}\n\n"

                # Stop streaming if alignment finished
                if progress.status in ("completed", "failed"):
                    logger.info(f"GET /alignment/apply/stream: closing - status={progress.status}")
                    break

            await asyncio.sleep(1.0)  # Poll every second

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )


@router.get("/projects/{project_id}/videos/{video_id}/aligned-video/exists")
async def get_aligned_video_exists(
    video: Video = Depends(get_video),
    project_path: Path = Depends(get_project_folder),
):
    """Check if aligned video exists for a video."""
    aligned_path = get_aligned_video_path(project_path, video.name)
    exists = aligned_path.exists()
    path = str(aligned_path) if exists else None

    logger.info(f"GET /aligned-video/exists: video_id={video.id}, exists={exists}")
    return {"exists": exists, "path": path}


@router.get("/projects/{project_id}/videos/{video_id}/aligned-video/stream")
async def stream_aligned_video(
    request: Request,
    video: Video = Depends(get_video),
    project_path: Path = Depends(get_project_folder),
):
    """Stream aligned video with HTTP range support."""
    aligned_path = get_aligned_video_path(project_path, video.name)

    if not aligned_path.exists():
        raise HTTPException(status_code=404, detail="Aligned video not found")

    file_size = aligned_path.stat().st_size
    content_type = mimetypes.guess_type(str(aligned_path))[0] or "video/mp4"

    range_header = request.headers.get("range")
    if range_header:
        range_match = range_header.replace("bytes=", "").split("-")
        start = int(range_match[0])
        end = int(range_match[1]) if range_match[1] else file_size - 1

        chunk_size = end - start + 1

        def iter_file():
            with open(aligned_path, "rb") as f:
                f.seek(start)
                remaining = chunk_size
                while remaining > 0:
                    read_size = min(8192, remaining)
                    data = f.read(read_size)
                    if not data:
                        break
                    remaining -= len(data)
                    yield data

        return StreamingResponse(
            iter_file(),
            status_code=206,
            headers={
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(chunk_size),
                "Content-Type": content_type,
            },
        )
    else:
        def iter_full_file():
            with open(aligned_path, "rb") as f:
                while chunk := f.read(8192):
                    yield chunk

        return StreamingResponse(
            iter_full_file(),
            headers={
                "Content-Length": str(file_size),
                "Content-Type": content_type,
                "Accept-Ranges": "bytes",
            },
        )


@router.get("/projects/{project_id}/videos/{video_id}/alignment-labels")
async def get_video_alignment_labels(
    video_id: int,
    session: AsyncSession = Depends(get_project_session),
):
    """Get frame indices that have alignment labels for a specific video."""
    frames = await alignment_service.get_video_alignment_label_frames(session, video_id)
    return {"frame_indices": frames}


@router.delete("/projects/{project_id}/videos/{video_id}/alignment-labels/{frame_idx}", status_code=204)
async def delete_video_alignment_label(
    video_id: int,
    frame_idx: int,
    session: AsyncSession = Depends(get_project_session),
):
    """Delete a single alignment label for a specific frame."""
    await alignment_service.delete_video_alignment_label(session, video_id, frame_idx)


@router.delete("/projects/{project_id}/videos/{video_id}/alignment-labels", status_code=204)
async def delete_video_alignment_labels(
    video_id: int,
    session: AsyncSession = Depends(get_project_session),
):
    """Delete all alignment labels for a specific video."""
    await alignment_service.delete_video_alignment_labels(session, video_id)
