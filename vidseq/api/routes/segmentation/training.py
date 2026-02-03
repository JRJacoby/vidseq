"""Training range endpoints - validation, marking, and unmarking."""

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from vidseq.api.dependencies import get_project_folder, get_project_session, get_video
from vidseq.models.video import Video
from vidseq.services import frame_data_service
from vidseq.services.array_storage import tracker_masks, compute_bbox_from_mask

router = APIRouter()


@router.get(
    "/projects/{project_id}/videos/{video_id}/training-range/validation",
)
async def validate_training_range(
    start_frame: int,
    end_frame: int,
    video: Video = Depends(get_video),
    session: AsyncSession = Depends(get_project_session),
):
    """
    Check if all frames in range have masks.

    Returns {"valid": true} if all frames have masks,
    or {"valid": false, "missing_frames": [...]} if some are missing.
    """
    missing_frames = await frame_data_service.get_missing_masks_in_range(
        session, video.id, start_frame, end_frame
    )

    if missing_frames:
        return {"valid": False, "missing_frames": missing_frames}
    return {"valid": True}


@router.post(
    "/projects/{project_id}/videos/{video_id}/training-range",
    status_code=204,
)
async def create_training_range(
    start_frame: int,
    end_frame: int,
    video: Video = Depends(get_video),
    session: AsyncSession = Depends(get_project_session),
    project_path: Path = Depends(get_project_folder),
):
    """
    Mark frame range as training (computes bboxes from masks).

    All frames in range must have masks. Use GET /training-range/validation first.
    """
    missing_frames = await frame_data_service.get_missing_masks_in_range(
        session, video.id, start_frame, end_frame
    )

    if missing_frames:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Some frames are missing masks",
                "missing_frames": missing_frames,
            }
        )

    # Load masks and compute bboxes, then save to SQLite
    with tracker_masks(project_path, video.id, "r") as masks:
        for frame_idx in range(start_frame, end_frame + 1):
            mask = masks[frame_idx]
            bbox = compute_bbox_from_mask(mask)
            if bbox is not None:
                await frame_data_service.save_bbox(session, video.id, frame_idx, bbox)

    # Mark frames as training
    await frame_data_service.mark_training_range(session, video.id, start_frame, end_frame)

    return None


@router.delete(
    "/projects/{project_id}/videos/{video_id}/training-range",
    status_code=204,
)
async def delete_training_range(
    start_frame: int,
    end_frame: int,
    video: Video = Depends(get_video),
    session: AsyncSession = Depends(get_project_session),
):
    """
    Remove training labels and bboxes for frame range.
    """
    await frame_data_service.unmark_training_range(
        session, video.id, start_frame, end_frame
    )

    return None
