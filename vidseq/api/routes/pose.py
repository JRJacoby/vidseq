"""API routes for pose labeling, training, and inference."""

import asyncio
import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from vidseq.api.dependencies import get_project_folder, get_project_session, get_video
from vidseq.models.frame_data import FrameData
from vidseq.models.pose_label import PoseLabel
from vidseq.models.video import Video
from vidseq.services import frame_data_service
from vidseq.services.pose_service import PoseService

router = APIRouter()


# --- Schemas ---


class PoseLabelRequest(BaseModel):
    front_x: float = Field(ge=0.0, le=1.0)
    front_y: float = Field(ge=0.0, le=1.0)
    rear_x: float = Field(ge=0.0, le=1.0)
    rear_y: float = Field(ge=0.0, le=1.0)


class PoseLabelResponse(BaseModel):
    frame_idx: int
    front_x: float
    front_y: float
    rear_x: float
    rear_y: float


class PoseTrainingRequest(BaseModel):
    video_ids: list[int]
    max_epochs: int = 300


class PoseStatusResponse(BaseModel):
    model_exists: bool
    is_training: bool


class VideoSelectionRequest(BaseModel):
    video_ids: list[int]


# --- Label Routes ---


@router.get("/projects/{project_id}/videos/{video_id}/pose/labels")
async def list_pose_labels(
    video_id: int,
    session: AsyncSession = Depends(get_project_session),
) -> list[PoseLabelResponse]:
    """List all pose labels for a video, ordered by frame_idx."""
    result = await session.execute(
        select(PoseLabel)
        .where(PoseLabel.video_id == video_id)
        .order_by(PoseLabel.frame_idx)
    )
    labels = result.scalars().all()
    return [
        PoseLabelResponse(
            frame_idx=label.frame_idx,
            front_x=label.front_x,
            front_y=label.front_y,
            rear_x=label.rear_x,
            rear_y=label.rear_y,
        )
        for label in labels
    ]


@router.post("/projects/{project_id}/videos/{video_id}/pose/labels/{frame_idx}")
async def upsert_pose_label(
    video_id: int,
    frame_idx: int,
    body: PoseLabelRequest,
    session: AsyncSession = Depends(get_project_session),
) -> PoseLabelResponse:
    """Upsert a pose label for a specific frame (delete existing then insert)."""
    # Delete existing label if present
    await session.execute(
        delete(PoseLabel).where(
            PoseLabel.video_id == video_id,
            PoseLabel.frame_idx == frame_idx,
        )
    )

    label = PoseLabel(
        video_id=video_id,
        frame_idx=frame_idx,
        front_x=body.front_x,
        front_y=body.front_y,
        rear_x=body.rear_x,
        rear_y=body.rear_y,
    )
    session.add(label)
    await session.commit()

    return PoseLabelResponse(
        frame_idx=label.frame_idx,
        front_x=label.front_x,
        front_y=label.front_y,
        rear_x=label.rear_x,
        rear_y=label.rear_y,
    )


@router.delete(
    "/projects/{project_id}/videos/{video_id}/pose/labels/{frame_idx}",
    status_code=204,
)
async def delete_pose_label(
    video_id: int,
    frame_idx: int,
    session: AsyncSession = Depends(get_project_session),
):
    """Delete pose label for a specific frame."""
    await session.execute(
        delete(PoseLabel).where(
            PoseLabel.video_id == video_id,
            PoseLabel.frame_idx == frame_idx,
        )
    )
    await session.commit()
    return None


@router.get("/projects/{project_id}/videos/{video_id}/pose/label-count")
async def get_pose_label_count(
    video_id: int,
    session: AsyncSession = Depends(get_project_session),
):
    """Return count of pose labels for a video."""
    result = await session.execute(
        select(PoseLabel).where(PoseLabel.video_id == video_id)
    )
    count = len(result.scalars().all())
    return {"count": count}


# --- Prediction Route ---


@router.get("/projects/{project_id}/videos/{video_id}/pose/prediction/{frame_idx}")
async def get_pose_prediction(
    video_id: int,
    frame_idx: int,
    session: AsyncSession = Depends(get_project_session),
):
    """Get pose prediction for a specific frame from FrameData."""
    result = await session.execute(
        select(
            FrameData.pose_front_x,
            FrameData.pose_front_y,
            FrameData.pose_rear_x,
            FrameData.pose_rear_y,
            FrameData.pose_score,
        ).where(
            FrameData.video_id == video_id,
            FrameData.frame_idx == frame_idx,
            FrameData.pose_score >= 0,
        )
    )
    row = result.first()
    if row is None or row[0] is None:
        return {"prediction": None}

    return {
        "prediction": {
            "front_x": row[0],
            "front_y": row[1],
            "rear_x": row[2],
            "rear_y": row[3],
            "score": row[4],
        }
    }


# --- Training Routes ---


@router.get(
    "/projects/{project_id}/pose/status",
    response_model=PoseStatusResponse,
)
async def get_pose_status(
    project_path: Path = Depends(get_project_folder),
):
    """Get pose model status: model_exists and is_training."""
    service = PoseService.get_instance()
    return PoseStatusResponse(
        model_exists=service.model_exists(project_path),
        is_training=service.is_training(),
    )


@router.post("/projects/{project_id}/pose/training")
async def create_pose_training(
    request: PoseTrainingRequest,
    project_path: Path = Depends(get_project_folder),
):
    """Start pose training (409 if already running)."""
    service = PoseService.get_instance()

    if service.is_training():
        raise HTTPException(status_code=409, detail="Training already in progress")

    try:
        service.train(
            project_path=project_path,
            video_ids=request.video_ids,
            max_epochs=request.max_epochs,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"status": "started"}


@router.delete("/projects/{project_id}/pose/training", status_code=204)
async def delete_pose_training():
    """Stop pose training gracefully."""
    service = PoseService.get_instance()

    if not service.is_training():
        raise HTTPException(status_code=400, detail="No training or apply in progress")

    service.stop_training()
    return None


@router.get("/projects/{project_id}/pose/training")
async def get_pose_training():
    """Get current training progress."""
    service = PoseService.get_instance()
    progress = service.get_training_progress()
    return progress.to_dict()


@router.get("/projects/{project_id}/pose/training/stream")
async def stream_pose_training():
    """SSE stream for real-time training updates."""

    async def event_generator():
        service = PoseService.get_instance()
        last_progress_str = None

        while True:
            progress = service.get_training_progress()
            progress_dict = progress.to_dict()
            progress_str = json.dumps(progress_dict)

            if progress_str != last_progress_str:
                yield f"data: {progress_str}\n\n"
                last_progress_str = progress_str

            if progress.status in ("completed", "failed", "stopped", "idle"):
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


# --- Apply Route ---


@router.post("/projects/{project_id}/videos/pose")
async def create_videos_pose(
    request: VideoSelectionRequest,
    project_path: Path = Depends(get_project_folder),
):
    """Apply pose model to selected videos (409 if busy)."""
    service = PoseService.get_instance()

    if service.is_training():
        raise HTTPException(status_code=409, detail="Training or apply already in progress")

    try:
        service.apply(
            project_path=project_path,
            video_ids=request.video_ids,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"status": "started"}


# --- Scores Route ---


@router.get("/projects/{project_id}/videos/{video_id}/pose-scores-downsampled")
async def get_pose_scores_downsampled(
    max_samples: int = 800,
    start_frame: int = 0,
    end_frame: int | None = None,
    video: Video = Depends(get_video),
    session: AsyncSession = Depends(get_project_session),
):
    """Get LTTB-downsampled pose confidence scores for visualization."""
    from vidseq.services import lttb

    if end_frame is None:
        end_frame = video.num_frames - 1

    result = await session.execute(
        select(FrameData.frame_idx, FrameData.pose_score)
        .where(
            FrameData.video_id == video.id,
            FrameData.frame_idx >= start_frame,
            FrameData.frame_idx <= end_frame,
            FrameData.pose_score >= 0,
        )
        .order_by(FrameData.frame_idx)
    )
    scores = [{"frame_idx": row[0], "score": row[1]} for row in result.all()]
    downsampled = lttb.downsample_scores(scores, max_samples)
    return {"scores": downsampled, "total_count": len(scores)}
