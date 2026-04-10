"""Graph cut segmentation endpoint."""

import asyncio
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from vidseq.api.dependencies import get_project_folder, get_video
from vidseq.models.video import Video
from vidseq.schemas.graphcut import GraphCutRequest, GraphCutResponse
from vidseq.services import graphcut_service

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/projects/{project_id}/videos/{video_id}/graphcut-masks",
    response_model=GraphCutResponse,
)
async def create_graphcut_masks(
    project_id: int,
    request: GraphCutRequest,
    video: Video = Depends(get_video),
    project_path: Path = Depends(get_project_folder),
):
    """Run graph cut segmentation on a video chunk."""
    end_frame = min(request.end_frame, video.num_frames - 1)
    if request.start_frame > end_frame:
        raise HTTPException(status_code=400, detail="start_frame is past end of video")

    seeds = {
        k: [{"x": p.x, "y": p.y, "label": p.label} for p in v]
        for k, v in request.seeds.items()
    }

    try:
        frames_processed = await asyncio.to_thread(
            graphcut_service.run_graphcut,
            project_path,
            video.id,
            video.path,
            request.start_frame,
            end_frame,
            seeds,
        )
    except MemoryError:
        raise HTTPException(
            status_code=500,
            detail="Out of memory building graph cut. Try a smaller chunk or lower-resolution video.",
        )

    return GraphCutResponse(frames_processed=frames_processed)
