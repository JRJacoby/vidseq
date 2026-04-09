import json
import mimetypes
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from vidseq.api.dependencies import get_project_folder, get_project_session, get_video
from vidseq.api.schemas import VideoSelectionRequest
from vidseq.models.frame_data import FrameData
from vidseq.models.video import Video
from vidseq.schemas.video import VideoCreate, VideoResponse
from vidseq.services import video_service

router = APIRouter()


@router.get("/projects/{project_id}/videos", response_model=list[VideoResponse])
async def get_videos(
    session: AsyncSession = Depends(get_project_session),
):
    videos = await video_service.get_all_videos(session)

    # Build associated_video_id lookup in a single query (avoids N+1)
    assoc_result = await session.execute(
        select(Video.associated_with_id, Video.id)
        .where(Video.is_associated == True)
    )
    assoc_lookup = {row[0]: row[1] for row in assoc_result.all()}

    # Bulk-fetch training frame counts (single GROUP BY query, no N+1)
    training_count_result = await session.execute(
        select(FrameData.video_id, func.count())
        .where(FrameData.frame_type == "train")
        .group_by(FrameData.video_id)
    )
    training_counts = {row[0]: row[1] for row in training_count_result.all()}

    # Enrich VideoResponse with the reverse-lookup field
    responses = []
    for video in videos:
        resp = VideoResponse.model_validate(video)
        resp.associated_video_id = assoc_lookup.get(video.id)
        count = training_counts.get(video.id)
        if count:
            resp.training_frame_count = count
        responses.append(resp)

    return responses


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
        try:
            range_spec = range_header.replace("bytes=", "").strip()
            parts = range_spec.split("-", 1)
            if parts[0] == "":
                # Suffix range: bytes=-500 means last 500 bytes
                suffix_len = int(parts[1])
                start = max(0, file_size - suffix_len)
                end = file_size - 1
            else:
                start = int(parts[0])
                end = int(parts[1]) if parts[1] else file_size - 1
        except (ValueError, IndexError):
            raise HTTPException(status_code=416, detail="Invalid range")

        # Clamp end to file bounds
        end = min(end, file_size - 1)
        if start > end or start >= file_size:
            return Response(
                status_code=416,
                headers={"Content-Range": f"bytes */{file_size}"},
            )

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


@router.delete("/projects/{project_id}/videos/{video_id}/segmentation/range", status_code=204)
async def delete_segmentation_range(
    project_id: int,
    start_frame: int,
    end_frame: int,
    video: Video = Depends(get_video),
    project_path: Path = Depends(get_project_folder),
    session: AsyncSession = Depends(get_project_session),
):
    """
    Delete all segmentation data for a range of frames (inclusive).

    Clears tracker masks, logits, detector masks, final masks,
    conditioning frames, frame_data, and SAM memory for frames
    [start_frame, end_frame].
    """
    await video_service.delete_frame_data_range(
        project_id=project_id,
        project_path=project_path,
        video_id=video.id,
        start_frame=start_frame,
        end_frame=end_frame,
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


@router.delete("/projects/{project_id}/videos/segmentation", status_code=204)
async def delete_videos_segmentation(
    project_id: int,
    body: VideoSelectionRequest,
    project_path: Path = Depends(get_project_folder),
    session: AsyncSession = Depends(get_project_session),
):
    """Delete all segmentation data for selected videos.

    Resets masks, clears DB records, and removes segmented status.
    """
    await video_service.delete_videos_segmentation(
        project_id=project_id,
        project_path=project_path,
        video_ids=body.video_ids,
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


@router.delete(
    "/projects/{project_id}/videos/{video_id}/associated-segmentation",
    status_code=204,
)
async def reset_associated_segmentation(
    project_id: int,
    video: Video = Depends(get_video),
    project_path: Path = Depends(get_project_folder),
    session: AsyncSession = Depends(get_project_session),
):
    """Reset co-segmentation results for an associated video."""
    if not video.is_associated:
        raise HTTPException(status_code=400, detail="Video is not an associated video")

    from vidseq.services.array_storage import reset_video_segmentation_arrays
    from vidseq.models.frame_data import FrameData
    from sqlalchemy import delete

    reset_video_segmentation_arrays(
        project_path=project_path,
        video_id=video.id,
        num_frames=video.num_frames,
        height=video.height,
        width=video.width,
        is_associated=True,
    )

    await session.execute(
        delete(FrameData).where(FrameData.video_id == video.id)
    )
    video.segmentation_status = None
    await session.commit()
    return None


@router.get(
    "/projects/{project_id}/videos/{video_id}/associated",
    response_model=VideoResponse,
)
async def get_associated_video(
    project_id: int,
    video_id: int,
    session: AsyncSession = Depends(get_project_session),
):
    """Get the associated video for a main video, or 404 if none."""
    result = await session.execute(
        select(Video).where(Video.associated_with_id == video_id)
    )
    assoc = result.scalar_one_or_none()
    if assoc is None:
        raise HTTPException(status_code=404, detail="No associated video found")
    return VideoResponse.model_validate(assoc)


class AssociatedVideoRequest(BaseModel):
    json_path: str


@router.post(
    "/projects/{project_id}/videos/associated",
    response_model=list[VideoResponse],
    status_code=201,
)
async def add_associated_videos(
    project_id: int,
    body: AssociatedVideoRequest,
    session: AsyncSession = Depends(get_project_session),
    project_path: Path = Depends(get_project_folder),
):
    """Add associated videos from a JSON mapping file on disk."""
    json_file = Path(body.json_path)
    if not json_file.exists():
        raise HTTPException(status_code=400, detail=f"File not found: {body.json_path}")

    try:
        mapping = json.loads(json_file.read_text())
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

    if not isinstance(mapping, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in mapping.items()
    ):
        raise HTTPException(
            status_code=400,
            detail="Expected flat {string: string} mapping",
        )

    try:
        videos = await video_service.add_associated_videos(
            session=session,
            project_path=project_path,
            mapping=mapping,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return [VideoResponse.model_validate(v) for v in videos]
