import mimetypes
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse, Response
from sqlalchemy.ext.asyncio import AsyncSession

from vidseq.api.dependencies import get_project_folder, get_project_session, get_video
from vidseq.api.schemas import VideoSelectionRequest
from vidseq.models.video import Video
from vidseq.schemas.video import VideoCreate, VideoResponse
from vidseq.services import video_service

router = APIRouter()


@router.get("/projects/{project_id}/videos", response_model=list[VideoResponse])
async def get_videos(
    session: AsyncSession = Depends(get_project_session),
):
    return await video_service.get_all_videos(session)


@router.post("/projects/{project_id}/videos", response_model=list[VideoResponse], status_code=201)
async def add_videos(
    video_data: VideoCreate,
    project_path: Path = Depends(get_project_folder),
    session: AsyncSession = Depends(get_project_session),
):
    return await video_service.add_videos(
        session, project_path, [Path(p) for p in video_data.paths]
    )


@router.get("/projects/{project_id}/videos/{video_id}")
async def get_video_route(
    video: Video = Depends(get_video),
) -> VideoResponse:
    return video


@router.get("/projects/{project_id}/videos/{video_id}/stream")
async def stream_video(
    request: Request,
    video: Video = Depends(get_video),
):
    video_path = Path(video.path)
    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"Video file not found: {video.path}")
    
    file_size = video_path.stat().st_size
    content_type = mimetypes.guess_type(str(video_path))[0] or "video/mp4"
    
    range_header = request.headers.get("range")
    if range_header:
        range_match = range_header.replace("bytes=", "").split("-")
        start = int(range_match[0])
        end = int(range_match[1]) if range_match[1] else file_size - 1
        
        chunk_size = end - start + 1
        
        def iter_file():
            with open(video_path, "rb") as f:
                f.seek(start)
                remaining = chunk_size
                while remaining > 0:
                    read_size = min(8192, remaining)
                    data = f.read(read_size)
                    if not data:
                        break
                    remaining -= len(data)
                    yield data
        
        return StreamingResponse(
            iter_file(),
            status_code=206,
            media_type=content_type,
            headers={
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(chunk_size),
            },
        )
    
    return FileResponse(
        video_path,
        media_type=content_type,
        headers={"Accept-Ranges": "bytes"},
    )


@router.get("/projects/{project_id}/videos/{video_id}/frame/{frame_idx}")
async def get_frame(
    frame_idx: int,
    video: Video = Depends(get_video),
):
    """
    Extract a specific frame from a video and return it as a JPEG image.

    Args:
        video_id: ID of the video
        frame_idx: Frame index to extract (0-based)

    Returns:
        JPEG image bytes
    """
    video_path = Path(video.path)
    jpeg_bytes = video_service.extract_frame_as_jpeg(
        video_path=video_path,
        frame_idx=frame_idx,
        num_frames=video.num_frames,
    )
    return Response(content=jpeg_bytes, media_type="image/jpeg")


@router.delete("/projects/{project_id}/videos/{video_id}/segmentation/{frame_idx}", status_code=204)
async def delete_segmentation(
    project_id: int,
    frame_idx: int,
    video: Video = Depends(get_video),
    project_path: Path = Depends(get_project_folder),
    session: AsyncSession = Depends(get_project_session),
):
    """
    Delete all segmentation data for a frame.

    Clears tracker masks, tracker logits, detector masks, final masks,
    database entries (conditioning_frame, frame_data), and SAM memory state.

    Args:
        project_id: ID of the project
        video_id: ID of the video (from path, resolved via get_video)
        frame_idx: Frame index (0-based)

    Returns:
        Status confirmation
    """
    await video_service.delete_frame_data(
        project_id=project_id,
        project_path=project_path,
        video_id=video.id,
        frame_idx=frame_idx,
        session=session,
    )
    return None


@router.delete("/projects/{project_id}/videos/{video_id}/segmentation", status_code=204)
async def delete_video_segmentation(
    project_id: int,
    video: Video = Depends(get_video),
    project_path: Path = Depends(get_project_folder),
    session: AsyncSession = Depends(get_project_session),
):
    """
    Delete all segmentation data for a video.

    Clears all masks (tracker, detector, final), logits, database entries
    (conditioning_frames, frame_data), and SAM memory state.

    If SAM session is active, closes and re-opens it with fresh state.

    Args:
        project_id: ID of the project
        video_id: ID of the video (from path, resolved via get_video)

    Returns:
        Status confirmation
    """
    await video_service.reset_video(
        project_id=project_id,
        project_path=project_path,
        video=video,
        session=session,
    )
    return None


@router.delete("/projects/{project_id}/videos", status_code=204)
async def delete_videos(
    project_id: int,
    body: VideoSelectionRequest,
    project_path: Path = Depends(get_project_folder),
    session: AsyncSession = Depends(get_project_session),
):
    """Delete videos and all associated data from a project.

    Removes DB records, H5 files, and generated videos.
    Does not touch original source video files.
    """
    await video_service.delete_videos(
        project_id=project_id,
        project_path=project_path,
        video_ids=body.video_ids,
        session=session,
    )
    return None
