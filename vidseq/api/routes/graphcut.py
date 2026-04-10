"""Threshold segmentation endpoint."""

import asyncio
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from vidseq.api.dependencies import get_project_folder, get_project_session, get_video
from vidseq.models.video import Video
from vidseq.schemas.graphcut import ThresholdSegmentRequest, ThresholdSegmentResponse
from vidseq.services import frame_data_service, graphcut_service

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/projects/{project_id}/videos/{video_id}/threshold-masks",
    response_model=ThresholdSegmentResponse,
)
async def create_threshold_masks(
    project_id: int,
    request: ThresholdSegmentRequest,
    video: Video = Depends(get_video),
    session: AsyncSession = Depends(get_project_session),
    project_path: Path = Depends(get_project_folder),
):
    """Threshold video chunk and keep selected connected component."""
    end_frame = min(request.end_frame, video.num_frames - 1)
    if request.start_frame > end_frame:
        raise HTTPException(status_code=400, detail="start_frame is past end of video")

    try:
        frames_processed = await asyncio.to_thread(
            graphcut_service.run_threshold_segment,
            project_path,
            video.id,
            video.path,
            request.start_frame,
            end_frame,
            request.threshold,
            request.click_x,
            request.click_y,
            request.click_frame,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    frame_indices = list(range(request.start_frame, end_frame + 1))
    await frame_data_service.set_has_tracker_mask(session, video.id, frame_indices, True)

    return ThresholdSegmentResponse(frames_processed=frames_processed)
