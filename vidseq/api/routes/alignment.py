"""API routes for egocentric alignment."""

import asyncio
import json
import logging
import mimetypes
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from vidseq.api.dependencies import get_project_folder, get_project_session, get_video
from vidseq.models.alignment_label import AlignmentLabel
from vidseq.models.video import Video
from vidseq.services.alignment_service import (
    AlignmentService,
    heatmap_to_png,
    load_prediction,
    predictions_exist,
)
from vidseq.services.database_manager import DatabaseManager


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
    logger.info(f"GET /alignment/status: project_id={project_id}")

    service = AlignmentService.get_instance()

    label_count = await service.get_label_count(session)
    model_trained = service.is_model_trained(project_path)
    is_training = service.is_training()
    is_applying = service.is_applying()

    # Check if all videos have cropping completed
    result = await session.execute(select(Video))
    videos = list(result.scalars().all())
    video_count = len(videos)
    cropped_count = sum(1 for v in videos if v.cropping_status == "completed")
    all_cropped = video_count > 0 and cropped_count == video_count

    logger.info(
        f"GET /alignment/status: label_count={label_count}, model_trained={model_trained}, "
        f"is_training={is_training}, is_applying={is_applying}, "
        f"videos={video_count}, cropped={cropped_count}, all_cropped={all_cropped}"
    )

    response = AlignmentStatusResponse(
        label_count=label_count,
        model_trained=model_trained,
        is_training=is_training,
        is_applying=is_applying,
        all_videos_cropped=all_cropped,
    )
    logger.debug(f"GET /alignment/status: response={response.model_dump()}")
    return response


@router.get("/projects/{project_id}/alignment/random-frame")
async def get_random_frame(
    project_id: int,
    session: AsyncSession = Depends(get_project_session),
) -> RandomFrameResponse:
    """Get a random unlabeled frame from cropped videos."""
    logger.info(f"GET /alignment/random-frame: project_id={project_id}")

    service = AlignmentService.get_instance()

    result = await service.get_random_unlabeled_frame(session)
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


@router.delete("/projects/{project_id}/alignment/model")
async def delete_model(
    project_id: int,
    project_path: Path = Depends(get_project_folder),
):
    """Delete the alignment model for the project."""
    logger.info(f"DELETE /alignment/model: project_id={project_id}")

    service = AlignmentService.get_instance()

    deleted = service.delete_model(project_path)

    logger.info(f"DELETE /alignment/model: deleted={deleted}")
    return {"deleted": deleted}


def _run_training_in_background(
    service: AlignmentService,
    project_path: Path,
    labels: list,
    video_name_map: dict,
    epochs: int,
):
    """Run training synchronously (called from background task)."""
    service.train_model_sync(project_path, labels, video_name_map, epochs)


@router.post("/projects/{project_id}/alignment/train")
async def train_alignment_model(
    project_id: int,
    epochs: int = 100,
    project_path: Path = Depends(get_project_folder),
    session: AsyncSession = Depends(get_project_session),
):
    """Start alignment model training (fire-and-forget).

    Training will run for up to `epochs` (max), but may stop early if loss
    plateaus. Learning rate is automatically reduced on plateau.

    Returns immediately after starting training. Use the SSE stream endpoint
    (/alignment/training/stream) to monitor progress, or the status endpoint
    (/alignment/training/status) to check current state.
    """
    logger.info(f"POST /alignment/train: project_id={project_id}, max_epochs={epochs}")

    service = AlignmentService.get_instance()

    if service.is_training():
        logger.warning(f"POST /alignment/train: training already in progress")
        raise HTTPException(status_code=400, detail="Training already in progress")

    # Fetch all labels
    labels = await service.get_all_labels(session)
    logger.info(f"POST /alignment/train: label_count={len(labels)}")

    if len(labels) == 0:
        logger.warning(f"POST /alignment/train: no labels available for training")
        raise HTTPException(
            status_code=400, detail="No labels available for training"
        )

    # Build video_name_map: video_id -> video.name
    video_ids = list({l.video_id for l in labels})
    result = await session.execute(select(Video).where(Video.id.in_(video_ids)))
    videos = list(result.scalars().all())
    video_name_map = {v.id: v.name for v in videos}
    logger.info(f"POST /alignment/train: video_name_map has {len(video_name_map)} videos")

    # Start training in background thread (fire-and-forget)
    # Use asyncio.to_thread but don't await it - let it run in background
    logger.info(f"POST /alignment/train: starting training in background...")
    asyncio.create_task(
        asyncio.to_thread(
            _run_training_in_background,
            service,
            project_path,
            labels,
            video_name_map,
            epochs,
        )
    )

    logger.info(f"POST /alignment/train: training started, returning immediately")
    return {"message": "Training started", "max_epochs": epochs}


@router.get("/projects/{project_id}/alignment/training/status")
async def get_training_status(
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


@router.get("/projects/{project_id}/alignment/predict/{video_id}/{frame_idx}")
async def get_alignment_prediction(
    project_id: int,
    video_id: int,
    frame_idx: int,
    session: AsyncSession = Depends(get_project_session),
    project_path: Path = Depends(get_project_folder),
):
    """Get model prediction heatmap for a frame.

    Returns PNG image with R channel = front probability, G channel = rear probability.
    """
    logger.info(f"GET /alignment/predict: project_id={project_id}, video_id={video_id}, frame_idx={frame_idx}")

    service = AlignmentService.get_instance()

    if not service.is_model_trained(project_path):
        logger.warning(f"GET /alignment/predict: model not trained yet")
        raise HTTPException(status_code=404, detail="Model not trained yet")

    # Get video info
    result = await session.execute(select(Video).where(Video.id == video_id))
    video = result.scalar_one_or_none()
    if video is None:
        logger.warning(f"GET /alignment/predict: video {video_id} not found")
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found")

    logger.debug(f"GET /alignment/predict: found video name={video.name}")

    # Load frame from cropped video
    from vidseq.services.cropped_video_service import get_cropped_video_path
    import cv2

    cropped_path = get_cropped_video_path(project_path, video.name)
    if not cropped_path.exists():
        logger.warning(f"GET /alignment/predict: cropped video not found: {cropped_path}")
        raise HTTPException(
            status_code=404, detail=f"Cropped video not found for video {video_id}"
        )

    # Read the specific frame
    cap = cv2.VideoCapture(str(cropped_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        logger.warning(f"GET /alignment/predict: failed to read frame {frame_idx}")
        raise HTTPException(
            status_code=404, detail=f"Failed to read frame {frame_idx} from video {video_id}"
        )

    height, width = frame.shape[:2]
    logger.info(f"GET /alignment/predict: loaded frame size={width}x{height}")

    # Generate prediction heatmap
    png_bytes = service.predict_to_png(project_path, frame)

    logger.info(f"GET /alignment/predict: returning PNG, size={len(png_bytes)} bytes")

    return Response(content=png_bytes, media_type="image/png")


@router.post("/projects/{project_id}/alignment/apply")
async def apply_alignment(
    project_id: int,
    project_path: Path = Depends(get_project_folder),
):
    """Apply alignment to all cropped videos.

    Creates aligned videos in <project>/aligned_videos/ folder.
    """
    logger.info(f"POST /alignment/apply: project_id={project_id}")

    service = AlignmentService.get_instance()

    if service.is_applying():
        logger.warning(f"POST /alignment/apply: alignment already in progress")
        raise HTTPException(status_code=400, detail="Alignment already in progress")

    if not service.is_model_trained(project_path):
        logger.warning(f"POST /alignment/apply: model not trained yet")
        raise HTTPException(status_code=404, detail="Model not trained yet")

    # Get project engine for sync operations
    db_manager = DatabaseManager.get_instance()
    project_engine = db_manager.get_project_engine(project_path)

    logger.info(f"POST /alignment/apply: starting alignment...")
    success = service.apply_alignment_sync(project_path, project_engine)

    if not success:
        logger.error(f"POST /alignment/apply: alignment failed")
        raise HTTPException(status_code=500, detail="Alignment failed")

    logger.info(f"POST /alignment/apply: alignment complete")
    return {"message": "Alignment complete"}


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


@router.get("/projects/{project_id}/videos/{video_id}/alignment-predictions/exists")
async def get_alignment_predictions_exist(
    video: Video = Depends(get_video),
    project_path: Path = Depends(get_project_folder),
):
    """Check if stored alignment predictions exist for a video."""
    exists = predictions_exist(project_path, video.id)
    logger.info(f"GET /alignment-predictions/exists: video_id={video.id}, exists={exists}")
    return {"exists": exists}


@router.get("/projects/{project_id}/videos/{video_id}/alignment-prediction/{frame_idx}")
async def get_alignment_prediction_frame(
    frame_idx: int,
    video: Video = Depends(get_video),
    project_path: Path = Depends(get_project_folder),
):
    """Get stored prediction heatmap for a specific frame.

    Returns PNG image with R=front, G=rear (same format as /predict endpoint).
    """
    logger.info(f"GET /alignment-prediction/{frame_idx}: video_id={video.id}")

    heatmap = load_prediction(project_path, video.id, frame_idx)
    if heatmap is None:
        logger.warning(f"GET /alignment-prediction/{frame_idx}: prediction not found")
        raise HTTPException(
            status_code=404,
            detail=f"Prediction not found for video {video.id} frame {frame_idx}"
        )

    png_bytes = heatmap_to_png(heatmap)
    logger.info(f"GET /alignment-prediction/{frame_idx}: returning PNG, size={len(png_bytes)} bytes")

    return Response(content=png_bytes, media_type="image/png")


@router.get("/projects/{project_id}/videos/{video_id}/alignment-labels")
async def get_video_alignment_labels(
    video: Video = Depends(get_video),
    session: AsyncSession = Depends(get_project_session),
):
    """Get frame indices that have alignment labels for a specific video."""
    logger.info(f"GET /videos/{video.id}/alignment-labels")

    result = await session.execute(
        select(AlignmentLabel.frame_idx)
        .where(AlignmentLabel.video_id == video.id)
        .order_by(AlignmentLabel.frame_idx)
    )
    frame_indices = list(result.scalars().all())

    logger.info(f"GET /videos/{video.id}/alignment-labels: found {len(frame_indices)} labels")
    return {"frame_indices": frame_indices}


@router.delete("/projects/{project_id}/videos/{video_id}/alignment-labels/{frame_idx}")
async def delete_video_alignment_label(
    frame_idx: int,
    video: Video = Depends(get_video),
    session: AsyncSession = Depends(get_project_session),
):
    """Delete a single alignment label for a specific frame."""
    logger.info(f"DELETE /videos/{video.id}/alignment-labels/{frame_idx}")

    result = await session.execute(
        delete(AlignmentLabel)
        .where(AlignmentLabel.video_id == video.id)
        .where(AlignmentLabel.frame_idx == frame_idx)
    )
    await session.commit()

    deleted = result.rowcount > 0
    logger.info(f"DELETE /videos/{video.id}/alignment-labels/{frame_idx}: deleted={deleted}")
    return {"deleted": deleted}


@router.delete("/projects/{project_id}/videos/{video_id}/alignment-labels")
async def delete_video_alignment_labels(
    video: Video = Depends(get_video),
    session: AsyncSession = Depends(get_project_session),
):
    """Delete all alignment labels for a specific video."""
    logger.info(f"DELETE /videos/{video.id}/alignment-labels (all)")

    result = await session.execute(
        delete(AlignmentLabel).where(AlignmentLabel.video_id == video.id)
    )
    await session.commit()

    deleted_count = result.rowcount
    logger.info(f"DELETE /videos/{video.id}/alignment-labels: deleted_count={deleted_count}")
    return {"deleted_count": deleted_count}
