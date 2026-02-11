"""Keypoint tracking data read endpoints."""

from pathlib import Path

from fastapi import APIRouter, Depends, Query

from vidseq.api.dependencies import get_project_folder
from vidseq.services import keypoint_tcp_client

router = APIRouter()


def _format_keypoints(raw: dict) -> dict:
    """Convert internal {obj_id: {x, y}} to API {front_x, front_y, rear_x, rear_y}."""
    result: dict = {
        "front_x": None,
        "front_y": None,
        "rear_x": None,
        "rear_y": None,
    }
    for key, val in raw.items():
        obj_id = int(key)
        if obj_id == 0:
            result["front_x"] = val["x"]
            result["front_y"] = val["y"]
        elif obj_id == 1:
            result["rear_x"] = val["x"]
            result["rear_y"] = val["y"]
    return result


@router.get(
    "/projects/{project_id}/videos/{video_id}/keypoint-tracking/keypoints/{frame_idx}"
)
async def get_frame_keypoints(
    project_id: int,
    video_id: int,
    frame_idx: int,
    project_path: Path = Depends(get_project_folder),
):
    """Get keypoints for a single frame."""
    raw = keypoint_tcp_client.get_keypoints(project_path, video_id, frame_idx)
    return _format_keypoints(raw)


@router.get(
    "/projects/{project_id}/videos/{video_id}/keypoint-tracking/keypoints"
)
async def get_keypoints_range(
    project_id: int,
    video_id: int,
    start_frame: int = Query(default=0),
    count: int = Query(default=100),
    project_path: Path = Depends(get_project_folder),
):
    """Get keypoints for a range of frames."""
    raw_list = keypoint_tcp_client.get_keypoints_range(
        project_path, video_id, start_frame, count,
    )
    keypoints = []
    for entry in raw_list:
        formatted: dict = {"frame_idx": entry["frame_idx"]}
        # Extract keypoint data (non-frame_idx keys) and format
        raw_kps = {k: v for k, v in entry.items() if k != "frame_idx"}
        formatted.update(_format_keypoints(raw_kps))
        keypoints.append(formatted)

    return {"keypoints": keypoints}


@router.get(
    "/projects/{project_id}/videos/{video_id}/keypoint-tracking/labeled-ranges"
)
async def get_labeled_ranges(
    project_id: int,
    video_id: int,
    project_path: Path = Depends(get_project_folder),
):
    """Get labeled frame ranges and conditioning frame indices."""
    return keypoint_tcp_client.get_labeled_ranges(project_path, video_id)
