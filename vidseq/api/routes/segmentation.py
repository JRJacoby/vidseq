"""Segmentation API routes."""

import asyncio
import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response, StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from vidseq.api.dependencies import get_project_folder, get_project_session
from vidseq.schemas.segmentation import SegmentRequest, PropagateRequest, PropagateResponse
from vidseq.services import conditioning_service, sam2_service, segmentation_service, video_service

router = APIRouter()


@router.get("/segmentation/status")
async def get_segmentation_status():
    """Get the current SAM2 model loading status."""
    return sam2_service.get_status()


@router.get("/segmentation/status/stream")
async def stream_segmentation_status():
    """SSE endpoint for real-time SAM2 status updates."""
    async def event_generator():
        last_status_str = None
        while True:
            current_status = sam2_service.get_status()
            current_status_str = json.dumps(current_status)
            
            if current_status_str != last_status_str:
                yield f"data: {current_status_str}\n\n"
                last_status_str = current_status_str
            
            await asyncio.sleep(0.3)
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@router.post("/segmentation/preload")
async def preload_segmentation():
    """Start loading SAM2 model in background."""
    sam2_service.start_loading_in_background()
    return {"message": "Loading started"}


@router.post("/projects/{project_id}/videos/{video_id}/session")
async def init_video_session(
    video_id: int,
    session: AsyncSession = Depends(get_project_session),
    project_path: Path = Depends(get_project_folder),
):
    """
    Initialize a SAM2 session for a video.
    
    Creates the tracker state and frame loader.
    Call this when entering the video detail view.
    Returns 503 if SAM2 model isn't loaded yet.
    """
    try:
        video = await video_service.get_video_by_id(session, video_id)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    video_path = Path(video.path)
    
    try:
        session_info = sam2_service.init_session(video_id, video_path)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    
    return {
        "video_id": video_id,
        "num_frames": session_info.num_frames,
        "height": session_info.height,
        "width": session_info.width,
    }


@router.delete("/projects/{project_id}/videos/{video_id}/session")
async def close_video_session(
    video_id: int,
):
    """
    Close a SAM2 session for a video.
    
    Frees GPU memory. Call this when leaving the video detail view.
    """
    closed = sam2_service.close_session(video_id)
    return {"closed": closed}


@router.post("/projects/{project_id}/videos/{video_id}/segment")
async def run_segmentation(
    video_id: int,
    segment_request: SegmentRequest,
    session: AsyncSession = Depends(get_project_session),
    project_path: Path = Depends(get_project_folder),
):
    """
    Run segmentation with a point prompt.
    
    Point coords should be normalized [0,1].
    First point creates the tracked object, subsequent points refine it.
    Marks the frame as a conditioning frame.
    Returns the mask as PNG.
    """
    try:
        video = await video_service.get_video_by_id(session, video_id)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    
    video_path = Path(video.path)
    
    label = 1 if segment_request.type == "positive_point" else 0
    
    try:
        mask = sam2_service.add_point_prompt(
            video_id=video_id,
            video_path=video_path,
            frame_idx=segment_request.frame_idx,
            points=[[segment_request.details["x"], segment_request.details["y"]]],
            labels=[label],
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    from vidseq.services import mask_service
    mask_service.save_mask(
        project_path=project_path,
        video_id=video_id,
        frame_idx=segment_request.frame_idx,
        mask=mask,
        num_frames=video.num_frames,
        height=video.height,
        width=video.width,
    )
    mask_service.mark_frame_type(
        project_path=project_path,
        video_id=video_id,
        frame_idx=segment_request.frame_idx,
        frame_type='train',
        num_frames=video.num_frames,
    )
    
    await conditioning_service.add_conditioning_frame(
        session=session,
        video_id=video_id,
        frame_idx=segment_request.frame_idx,
    )
    
    mask_png = segmentation_service.mask_to_png(mask)
    return Response(content=mask_png, media_type="image/png")


@router.delete(
    "/projects/{project_id}/videos/{video_id}/frame/{frame_idx}",
)
async def reset_frame(
    video_id: int,
    frame_idx: int,
    session: AsyncSession = Depends(get_project_session),
    project_path: Path = Depends(get_project_folder),
):
    """
    Reset a frame: clear the mask and remove conditioning frame record.
    Does not reset SAM2 tracking state.
    """
    try:
        await video_service.get_video_by_id(session, video_id)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    
    try:
        video = await video_service.get_video_by_id(session, video_id)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    
    sam2_service.clear_prompts_for_frame(frame_idx)
    
    segmentation_service.clear_mask(
        project_path=project_path,
        video_id=video_id,
        frame_idx=frame_idx,
    )
    
    from vidseq.services import mask_service
    mask_service.mark_frame_type(
        project_path=project_path,
        video_id=video_id,
        frame_idx=frame_idx,
        frame_type='',
        num_frames=video.num_frames,
    )
    
    await conditioning_service.remove_conditioning_frame(
        session=session,
        video_id=video_id,
        frame_idx=frame_idx,
    )
    
    return {"message": "Frame reset"}


@router.delete(
    "/projects/{project_id}/videos/{video_id}/all-frames",
)
async def reset_video(
    video_id: int,
    session: AsyncSession = Depends(get_project_session),
    project_path: Path = Depends(get_project_folder),
):
    """
    Reset entire video: clear all masks, conditioning frames, frame type labels, and SAM2 tracking state.
    """
    try:
        await video_service.get_video_by_id(session, video_id)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    
    segmentation_service.clear_video(
        project_path=project_path,
        video_id=video_id,
    )
    
    deleted_count = await conditioning_service.clear_conditioning_frames(
        session=session,
        video_id=video_id,
    )
    
    from vidseq.services import mask_service
    mask_service.clear_all_frame_types(
        project_path=project_path,
        video_id=video_id,
    )
    
    sam2_service.reset_state(video_id)
    
    return {"message": "Video reset", "conditioning_frames_cleared": deleted_count}


@router.get(
    "/projects/{project_id}/videos/{video_id}/conditioning-frames",
)
async def get_conditioning_frames(
    video_id: int,
    session: AsyncSession = Depends(get_project_session),
):
    """Get list of conditioning frame indices for a video."""
    try:
        await video_service.get_video_by_id(session, video_id)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    
    frames = await conditioning_service.get_conditioning_frames(session, video_id)
    return {"conditioning_frames": frames}


@router.post(
    "/projects/{project_id}/videos/{video_id}/propagate-forward",
    response_model=PropagateResponse,
)
async def propagate_forward(
    video_id: int,
    request: PropagateRequest,
    session: AsyncSession = Depends(get_project_session),
    project_path: Path = Depends(get_project_folder),
):
    """
    Propagate tracking forward from the given frame.
    
    Requires an active SAM2 session with a tracked object (add a point prompt first).
    Saves masks to HDF5 as it processes each frame.
    """
    try:
        video = await video_service.get_video_by_id(session, video_id)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    
    try:
        frames_processed = segmentation_service.propagate_forward_and_save(
            project_path=project_path,
            video=video,
            start_frame_idx=request.start_frame_idx,
            max_frames=request.max_frames,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    return PropagateResponse(frames_processed=frames_processed)


@router.post(
    "/projects/{project_id}/videos/{video_id}/propagate-backward",
    response_model=PropagateResponse,
)
async def propagate_backward(
    video_id: int,
    request: PropagateRequest,
    session: AsyncSession = Depends(get_project_session),
    project_path: Path = Depends(get_project_folder),
):
    """
    Propagate tracking backward from the given frame.
    
    Requires an active SAM2 session with a tracked object (add a point prompt first).
    Saves masks to HDF5 as it processes each frame.
    """
    try:
        video = await video_service.get_video_by_id(session, video_id)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    
    try:
        frames_processed = segmentation_service.propagate_backward_and_save(
            project_path=project_path,
            video=video,
            start_frame_idx=request.start_frame_idx,
            max_frames=request.max_frames,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    return PropagateResponse(frames_processed=frames_processed)


@router.get(
    "/projects/{project_id}/videos/{video_id}/mask/{frame_idx}",
)
async def get_mask(
    video_id: int,
    frame_idx: int,
    session: AsyncSession = Depends(get_project_session),
    project_path: Path = Depends(get_project_folder),
):
    """
    Get the segmentation mask for a specific frame.
    
    Returns PNG binary. If no mask exists, returns a transparent (all zeros) mask.
    """
    try:
        video = await video_service.get_video_by_id(session, video_id)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    
    mask_png = segmentation_service.get_mask_png(
        project_path=project_path,
        video=video,
        frame_idx=frame_idx,
    )
    return Response(content=mask_png, media_type="image/png")


@router.get(
    "/projects/{project_id}/videos/{video_id}/masks-batch",
)
async def get_masks_batch(
    video_id: int,
    start_frame: int,
    count: int = 100,
    session: AsyncSession = Depends(get_project_session),
    project_path: Path = Depends(get_project_folder),
):
    """
    Get multiple segmentation masks in a single request.
    
    Returns JSON with base64-encoded PNG masks for efficient batch transfer.
    """
    try:
        video = await video_service.get_video_by_id(session, video_id)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    
    masks = segmentation_service.get_masks_batch_json(
        project_path=project_path,
        video=video,
        start_frame=start_frame,
        count=count,
    )
    return {"masks": masks}


@router.get(
    "/projects/{project_id}/videos/{video_id}/prompts/{frame_idx}",
)
async def get_prompts_for_frame(
    video_id: int,
    frame_idx: int,
    session: AsyncSession = Depends(get_project_session),
):
    """Get all prompts for a specific frame."""
    try:
        await video_service.get_video_by_id(session, video_id)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    
    prompts = sam2_service.get_prompts_for_frame(frame_idx)
    return prompts


@router.get(
    "/projects/{project_id}/videos/{video_id}/prompts",
)
async def get_all_prompts(
    video_id: int,
    session: AsyncSession = Depends(get_project_session),
):
    """Get all prompts for all frames."""
    try:
        await video_service.get_video_by_id(session, video_id)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    
    prompts = sam2_service.get_all_prompts()
    return prompts
