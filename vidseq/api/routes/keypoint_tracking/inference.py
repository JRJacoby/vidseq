"""Keypoint tracking inference endpoints - prompts, propagation, reset."""

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from vidseq.api.dependencies import get_project_folder, get_video
from vidseq.models.video import Video
from vidseq.services import keypoint_tcp_client

router = APIRouter()

KEYPOINT_MAP = {"front": 0, "rear": 1}


class KeypointPromptRequest(BaseModel):
    x: float
    y: float
    keypoint: str  # "front" or "rear"
    label: int = 1


class PropagateKeypointsRequest(BaseModel):
    start_frame_idx: int
    max_frames: int = 1000


def _format_keypoints(raw: dict) -> dict:
    """Convert internal {obj_id: {x, y}} to API {front_x, front_y, rear_x, rear_y}."""
    result: dict = {
        "front_x": None,
        "front_y": None,
        "rear_x": None,
        "rear_y": None,
    }
    # Keys may be int or str depending on JSON round-trip
    for key, val in raw.items():
        obj_id = int(key)
        if obj_id == 0:
            result["front_x"] = val["x"]
            result["front_y"] = val["y"]
        elif obj_id == 1:
            result["rear_x"] = val["x"]
            result["rear_y"] = val["y"]
    return result


@router.post(
    "/projects/{project_id}/videos/{video_id}/keypoint-tracking/prompt/{frame_idx}"
)
async def submit_keypoint_prompt(
    project_id: int,
    frame_idx: int,
    request: KeypointPromptRequest,
    video: Video = Depends(get_video),
    project_path: Path = Depends(get_project_folder),
):
    """Submit a keypoint prompt for a specific frame and keypoint type."""
    obj_id = KEYPOINT_MAP.get(request.keypoint)
    if obj_id is None:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid keypoint type: {request.keypoint}. Must be 'front' or 'rear'.",
        )

    try:
        raw_keypoints = keypoint_tcp_client.add_keypoint_prompt(
            project_id=project_id,
            video_id=video.id,
            frame_idx=frame_idx,
            obj_id=obj_id,
            x_norm=request.x,
            y_norm=request.y,
            project_path=project_path,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return _format_keypoints(raw_keypoints)


@router.post(
    "/projects/{project_id}/videos/{video_id}/keypoint-tracking/propagate"
)
async def propagate_keypoints(
    project_id: int,
    request: PropagateKeypointsRequest,
    video: Video = Depends(get_video),
):
    """Propagate keypoint tracking forward from a frame."""
    try:
        result = keypoint_tcp_client.propagate_keypoints(
            project_id=project_id,
            video_id=video.id,
            start_frame_idx=request.start_frame_idx,
            max_frames=request.max_frames,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"frames_processed": result.get("frames_processed", 0)}


@router.delete(
    "/projects/{project_id}/videos/{video_id}/keypoint-tracking/frames/{frame_idx}"
)
async def delete_keypoint_frame(
    project_id: int,
    video_id: int,
    frame_idx: int,
    project_path: Path = Depends(get_project_folder),
):
    """Reset keypoints for a single frame."""
    try:
        keypoint_tcp_client.reset_frame(
            project_id=project_id,
            video_id=video_id,
            frame_idx=frame_idx,
            project_path=project_path,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"status": "ok"}


@router.delete(
    "/projects/{project_id}/videos/{video_id}/keypoint-tracking/frames"
)
async def delete_keypoint_all_frames(
    project_id: int,
    video_id: int,
    project_path: Path = Depends(get_project_folder),
):
    """Reset all keypoints for a video."""
    try:
        keypoint_tcp_client.reset_video(
            project_id=project_id,
            video_id=video_id,
            project_path=project_path,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"status": "ok"}
