"""API routes for cropped video extraction and streaming."""

import mimetypes
from io import BytesIO
from pathlib import Path

import cv2
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from PIL import Image
from sqlalchemy.ext.asyncio import AsyncSession

from vidseq.api.dependencies import get_project_folder, get_project_session, get_video
from vidseq.models.video import Video
from vidseq.services import cropped_video_service, video_service

router = APIRouter()


@router.post("/projects/{project_id}/extract-cropped-videos")
async def extract_cropped_videos(
    project_id: int,
    session: AsyncSession = Depends(get_project_session),
    project_path: Path = Depends(get_project_folder),
):
    """
    Start cropped video extraction for all videos.

    Validates that ALL videos have segmentation_status='segmented'.
    Returns 400 if any videos are not fully segmented.
    Returns: { "job_ids": [1, 2, 3, ...] }
    """
    # Get all videos
    videos = await video_service.get_all_videos(session)
    if not videos:
        raise HTTPException(status_code=400, detail="No videos found in project")

    # Validate all videos are segmented
    unsegmented = []
    for video in videos:
        if video.segmentation_status != "segmented":
            unsegmented.append({"id": video.id, "name": video.name, "status": video.segmentation_status})

    if unsegmented:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "All videos must be segmented before extracting cropped videos.",
                "unsegmented_videos": unsegmented,
            },
        )

    # Start extraction
    service = cropped_video_service.CroppedVideoService.get_instance()

    if service.is_extracting():
        raise HTTPException(status_code=400, detail="Extraction already in progress")

    job_ids = service.extract_all_cropped_videos(
        project_id=project_id,
        project_path=project_path,
        videos=videos,
    )

    return {"job_ids": job_ids}


@router.get("/projects/{project_id}/videos/{video_id}/cropped-video/exists")
async def get_cropped_video_exists(
    video: Video = Depends(get_video),
    project_path: Path = Depends(get_project_folder),
):
    """Check if cropped video exists for a video."""
    exists = cropped_video_service.cropped_video_exists(project_path, video.name)
    path = None
    if exists:
        path = str(cropped_video_service.get_cropped_video_path(project_path, video.name))

    return {"exists": exists, "path": path}


@router.get("/projects/{project_id}/videos/{video_id}/cropped-video/stream")
async def stream_cropped_video(
    request: Request,
    video: Video = Depends(get_video),
    project_path: Path = Depends(get_project_folder),
):
    """Stream cropped video with HTTP range support."""
    cropped_path = cropped_video_service.get_cropped_video_path(project_path, video.name)

    if not cropped_path.exists():
        raise HTTPException(status_code=404, detail="Cropped video not found")

    file_size = cropped_path.stat().st_size
    content_type = mimetypes.guess_type(str(cropped_path))[0] or "video/mp4"

    range_header = request.headers.get("range")
    if range_header:
        range_match = range_header.replace("bytes=", "").split("-")
        start = int(range_match[0])
        end = int(range_match[1]) if range_match[1] else file_size - 1

        chunk_size = end - start + 1

        def iter_file():
            with open(cropped_path, "rb") as f:
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
        cropped_path,
        media_type=content_type,
        headers={"Accept-Ranges": "bytes"},
    )


@router.get("/projects/{project_id}/videos/{video_id}/cropped-video/frame/{frame_idx}")
async def get_cropped_video_frame(
    frame_idx: int,
    video: Video = Depends(get_video),
    project_path: Path = Depends(get_project_folder),
):
    """Extract a specific frame from a cropped video and return it as a JPEG image."""
    cropped_path = cropped_video_service.get_cropped_video_path(project_path, video.name)

    if not cropped_path.exists():
        raise HTTPException(status_code=404, detail="Cropped video not found")

    # Extract frame using OpenCV
    cap = cv2.VideoCapture(str(cropped_path))
    if not cap.isOpened():
        raise HTTPException(status_code=500, detail="Failed to open cropped video")

    # Get frame count for validation
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_idx < 0 or frame_idx >= frame_count:
        cap.release()
        raise HTTPException(
            status_code=400, detail=f"Frame index {frame_idx} out of range [0, {frame_count})"
        )

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise HTTPException(status_code=500, detail=f"Failed to read frame {frame_idx}")

    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert to PIL Image and then to JPEG bytes
    pil_image = Image.fromarray(frame_rgb)
    img_bytes = BytesIO()
    pil_image.save(img_bytes, format="JPEG", quality=95)
    img_bytes.seek(0)

    return Response(content=img_bytes.read(), media_type="image/jpeg")
