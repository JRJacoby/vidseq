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
