"""Segmentation API routes."""

import io
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from PIL import Image
from sqlalchemy.ext.asyncio import AsyncSession

from vidseq.api.dependencies import get_project_folder, get_project_session
from vidseq.models.video import Video
from vidseq.models.prompt import Prompt
from vidseq.schemas.segmentation import PromptCreate, PromptResponse
from vidseq.services import mask_service, prompt_service, sam3_service, video_service

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


def _mask_to_png(mask) -> bytes:
    """Convert a numpy mask to PNG bytes."""
    img = Image.fromarray(mask)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


async def _run_segmentation_and_save(
    project_path: Path,
    video: Video,
    frame_idx: int,
    prompts: list[Prompt],
) -> bytes:
    """Run SAM3 segmentation with prompts and save the mask."""
    video_path = Path(video.path)
    meta = video_service.get_video_metadata(video_path)
    
    if not prompts:
        mask = mask_service.load_mask(
            project_path=project_path,
            video_id=video.id,
            frame_idx=frame_idx,
            num_frames=meta.num_frames,
            height=meta.height,
            width=meta.width,
        )
        return _mask_to_png(mask)
    
    bbox_prompt = next((p for p in prompts if p.type == "bbox"), None)
    if bbox_prompt is None:
        mask = mask_service.load_mask(
            project_path=project_path,
            video_id=video.id,
            frame_idx=frame_idx,
            num_frames=meta.num_frames,
            height=meta.height,
            width=meta.width,
        )
        return _mask_to_png(mask)
    
    mask = sam3_service.add_bbox_prompt(
        video_id=video.id,
        video_path=video_path,
        frame_idx=frame_idx,
        bbox=bbox_prompt.details,
    )
    
    mask_service.save_mask(
        project_path=project_path,
        video_id=video.id,
        frame_idx=frame_idx,
        mask=mask,
        num_frames=meta.num_frames,
        height=meta.height,
        width=meta.width,
    )
    
    return _mask_to_png(mask)


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
    video = await video_service.get_video_by_id(session, video_id)
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
    video = await video_service.get_video_by_id(session, video_id)
    
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
    
    await _run_segmentation_and_save(
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
    await video_service.get_video_by_id(session, video_id)
    
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
    video = await video_service.get_video_by_id(session, video_id)
    
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
    
    mask_png = await _run_segmentation_and_save(
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
    await video_service.get_video_by_id(session, video_id)
    
    await prompt_service.delete_prompts_for_frame(
        session=session,
        video_id=video_id,
        frame_idx=frame_idx,
    )
    
    mask_service.clear_mask(
        project_path=project_path,
        video_id=video_id,
        frame_idx=frame_idx,
    )
    
    return {"message": "Frame reset"}


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
    video = await video_service.get_video_by_id(session, video_id)
    video_path = Path(video.path)
    meta = video_service.get_video_metadata(video_path)
    
    mask = mask_service.load_mask(
        project_path=project_path,
        video_id=video_id,
        frame_idx=frame_idx,
        num_frames=meta.num_frames,
        height=meta.height,
        width=meta.width,
    )
    
    mask_png = _mask_to_png(mask)
    return Response(content=mask_png, media_type="image/png")
