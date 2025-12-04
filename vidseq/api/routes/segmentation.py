"""Segmentation API routes."""

import asyncio
import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response, StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from vidseq.api.dependencies import get_project_folder, get_project_session
from vidseq.schemas.segmentation import SegmentRequest, PropagateRequest, PropagateResponse
from vidseq.services import conditioning_service, sam3_service, segmentation_service, video_service

router = APIRouter()


@router.get("/sam3/status")
async def get_sam3_status():
    """Get the current SAM3 model loading status."""
    return sam3_service.get_status()


@router.get("/sam3/status/stream")
async def stream_sam3_status():
    """SSE endpoint for real-time SAM3 status updates."""
    print("[SSE] Client connected to /sam3/status/stream")
    async def event_generator():
        last_status_str = None
        poll_count = 0
        while True:
            current_status = sam3_service.get_status()
            current_status_str = json.dumps(current_status)
            poll_count += 1
            
            if current_status_str != last_status_str:
                print(f"[SSE] Status changed: {last_status_str} -> {current_status_str} (poll #{poll_count})")
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


@router.post("/sam3/preload")
async def preload_sam3():
    """Start loading SAM3 model in background."""
    sam3_service.start_loading_in_background()
    return {"message": "Loading started"}


@router.post("/projects/{project_id}/videos/{video_id}/session")
async def init_video_session(
    video_id: int,
    session: AsyncSession = Depends(get_project_session),
    project_path: Path = Depends(get_project_folder),
):
    """
    Initialize a SAM3 session for a video.
    
    Creates the LazyVideoFrameLoader and inference state.
    Also restores any existing conditioning frames from HDF5.
    Call this when entering the video detail view.
    Returns 503 if SAM3 model isn't loaded yet.
    """
    try:
        video = await video_service.get_video_by_id(session, video_id)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    video_path = Path(video.path)
    
    try:
        session_info = sam3_service.init_session(video_id, video_path)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    
    conditioning_frames = await conditioning_service.get_conditioning_frames(session, video_id)
    restored_count = 0
    if conditioning_frames:
        try:
            restored_count = segmentation_service.restore_conditioning_frames(
                project_path=project_path,
                video=video,
                frame_indices=conditioning_frames,
            )
        except Exception as e:
            print(f"Warning: Failed to restore some conditioning frames: {e}")
    
    return {
        "video_id": video_id,
        "num_frames": session_info.num_frames,
        "height": session_info.height,
        "width": session_info.width,
        "conditioning_frames_restored": restored_count,
    }


@router.delete("/projects/{project_id}/videos/{video_id}/session")
async def close_video_session(
    video_id: int,
):
    """
    Close a SAM3 session for a video.
    
    Frees GPU memory. Call this when leaving the video detail view.
    """
    closed = sam3_service.close_session(video_id)
    return {"closed": closed}


@router.post("/projects/{project_id}/videos/{video_id}/segment")
async def run_segmentation(
    video_id: int,
    segment_request: SegmentRequest,
    session: AsyncSession = Depends(get_project_session),
    project_path: Path = Depends(get_project_folder),
):
    """
    Run segmentation with the given prompt.
    
    Bbox and point coords should be normalized [0,1].
    Marks the frame as a conditioning frame.
    Returns the mask as PNG.
    """
    try:
        video = await video_service.get_video_by_id(session, video_id)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    
    video_path = Path(video.path)
    meta = video_service.get_video_metadata(video_path)
    
    sam3_session = sam3_service.get_session(video_id)
    has_active_object = sam3_session is not None and sam3_session.active_obj_id is not None
    
    mask = None
    
    if segment_request.type == "bbox" and not has_active_object:
        mask = sam3_service.add_bbox_prompt(
            video_id=video_id,
            video_path=video_path,
            frame_idx=segment_request.frame_idx,
            bbox=segment_request.details,
            text=segment_request.text,
        )
    elif segment_request.type in ("positive_point", "negative_point"):
        if not has_active_object:
            raise HTTPException(
                status_code=400,
                detail="No active object. Add a bounding box prompt first."
            )
        label = 1 if segment_request.type == "positive_point" else 0
        mask = sam3_service.add_point_prompt(
            video_id=video_id,
            video_path=video_path,
            frame_idx=segment_request.frame_idx,
            points=[[segment_request.details["x"], segment_request.details["y"]]],
            labels=[label],
        )
    elif segment_request.type == "bbox" and has_active_object:
        raise HTTPException(
            status_code=400,
            detail="Object already exists. Use point prompts to refine or reset frame first."
        )
    
    if mask is not None:
        from vidseq.services import mask_service
        mask_service.save_mask(
            project_path=project_path,
            video_id=video_id,
            frame_idx=segment_request.frame_idx,
            mask=mask,
            num_frames=meta.num_frames,
            height=meta.height,
            width=meta.width,
        )
        
        await conditioning_service.add_conditioning_frame(
            session=session,
            video_id=video_id,
            frame_idx=segment_request.frame_idx,
        )
        
        mask_png = segmentation_service.mask_to_png(mask)
        return Response(content=mask_png, media_type="image/png")
    
    mask_png = segmentation_service.get_mask_png(
        project_path=project_path,
        video=video,
        frame_idx=segment_request.frame_idx,
    )
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
    """
    try:
        await video_service.get_video_by_id(session, video_id)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    
    segmentation_service.clear_mask(
        project_path=project_path,
        video_id=video_id,
        frame_idx=frame_idx,
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
    Reset entire video: clear all masks and conditioning frames.
    """
    print(f"[reset_video] Called for video_id={video_id}, project_path={project_path}")
    try:
        video = await video_service.get_video_by_id(session, video_id)
        print(f"[reset_video] Video found: {video.name}")
    except LookupError as e:
        print(f"[reset_video] Video not found: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    
    print(f"[reset_video] Calling segmentation_service.clear_video...")
    segmentation_service.clear_video(
        project_path=project_path,
        video_id=video_id,
    )
    print(f"[reset_video] segmentation_service.clear_video completed")
    
    print(f"[reset_video] Calling conditioning_service.clear_conditioning_frames...")
    deleted_count = await conditioning_service.clear_conditioning_frames(
        session=session,
        video_id=video_id,
    )
    print(f"[reset_video] Cleared {deleted_count} conditioning frames")
    
    print(f"[reset_video] Done, returning success")
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
    "/projects/{project_id}/videos/{video_id}/propagate",
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
    
    Requires an active SAM3 session with an object (add a bbox prompt first).
    Saves masks to HDF5 as it processes each frame.
    """
    try:
        video = await video_service.get_video_by_id(session, video_id)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    
    try:
        frames_processed = segmentation_service.propagate_and_save(
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
