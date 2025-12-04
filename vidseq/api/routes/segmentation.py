"""Segmentation API routes."""

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from sqlalchemy.ext.asyncio import AsyncSession

from vidseq.api.dependencies import get_project_folder, get_project_session
from vidseq.schemas.segmentation import PromptCreate, PromptResponse, PropagateRequest, PropagateResponse
from vidseq.services import prompt_service, sam3_service, segmentation_service, video_service

router = APIRouter()


@router.get("/sam3/status")
async def get_sam3_status():
    """Get the current SAM3 model loading status."""
    return sam3_service.get_status()


@router.post("/sam3/preload")
async def preload_sam3():
    """Start loading SAM3 model in background."""
    sam3_service.start_loading_in_background()
    return {"message": "Loading started"}


@router.post("/projects/{project_id}/videos/{video_id}/session")
async def init_video_session(
    video_id: int,
    session: AsyncSession = Depends(get_project_session),
):
    """
    Initialize a SAM3 session for a video.
    
    Creates the LazyVideoFrameLoader and inference state.
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
    Close a SAM3 session for a video.
    
    Frees GPU memory. Call this when leaving the video detail view.
    """
    closed = sam3_service.close_session(video_id)
    return {"closed": closed}


@router.post(
    "/projects/{project_id}/videos/{video_id}/prompts",
    response_model=PromptResponse,
)
async def add_prompt(
    video_id: int,
    prompt_data: PromptCreate,
    session: AsyncSession = Depends(get_project_session),
    project_path: Path = Depends(get_project_folder),
):
    """
    Add a prompt and run segmentation.
    
    Bbox coords should be normalized [0,1].
    Returns the created prompt. The updated mask can be fetched via GET /mask/{frame_idx}.
    """
    try:
        video = await video_service.get_video_by_id(session, video_id)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    
    prompt = await prompt_service.save_prompt(
        session=session,
        video_id=video_id,
        frame_idx=prompt_data.frame_idx,
        prompt_type=prompt_data.type,
        details=prompt_data.details,
    )
    
    all_prompts = await prompt_service.get_prompts(
        session=session,
        video_id=video_id,
        frame_idx=prompt_data.frame_idx,
    )
    
    segmentation_service.run_segmentation_and_save(
        project_path=project_path,
        video=video,
        frame_idx=prompt_data.frame_idx,
        prompts=all_prompts,
    )
    
    return prompt


@router.get(
    "/projects/{project_id}/videos/{video_id}/prompts",
    response_model=list[PromptResponse],
)
async def get_prompts(
    video_id: int,
    frame_idx: int = Query(...),
    session: AsyncSession = Depends(get_project_session),
):
    """Get all prompts for a specific frame."""
    try:
        await video_service.get_video_by_id(session, video_id)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    
    prompts = await prompt_service.get_prompts(
        session=session,
        video_id=video_id,
        frame_idx=frame_idx,
    )
    
    return prompts


@router.delete(
    "/projects/{project_id}/videos/{video_id}/prompts/{prompt_id}",
)
async def delete_prompt(
    video_id: int,
    prompt_id: int,
    session: AsyncSession = Depends(get_project_session),
    project_path: Path = Depends(get_project_folder),
):
    """
    Delete a prompt and re-run segmentation with remaining prompts.
    
    Returns the updated mask as PNG.
    """
    try:
        video = await video_service.get_video_by_id(session, video_id)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    
    prompt = await prompt_service.get_prompt_by_id(session, prompt_id)
    if not prompt:
        raise HTTPException(status_code=404, detail=f"Prompt {prompt_id} not found")
    
    frame_idx = prompt.frame_idx
    
    deleted = await prompt_service.delete_prompt(session, prompt_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Prompt {prompt_id} not found")
    
    remaining_prompts = await prompt_service.get_prompts(
        session=session,
        video_id=video_id,
        frame_idx=frame_idx,
    )
    
    mask_png = segmentation_service.run_segmentation_and_save(
        project_path=project_path,
        video=video,
        frame_idx=frame_idx,
        prompts=remaining_prompts,
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
    Reset a frame: delete all prompts and clear the mask.
    """
    try:
        await video_service.get_video_by_id(session, video_id)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    
    await prompt_service.delete_prompts_for_frame(
        session=session,
        video_id=video_id,
        frame_idx=frame_idx,
    )
    
    segmentation_service.clear_mask(
        project_path=project_path,
        video_id=video_id,
        frame_idx=frame_idx,
    )
    
    return {"message": "Frame reset"}


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
