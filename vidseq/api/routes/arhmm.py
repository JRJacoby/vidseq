"""API routes for ARHMM model fitting."""

import asyncio
import json
import logging
import mimetypes
from pathlib import Path

import joblib
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from vidseq.api.dependencies import get_project_folder, get_project_session
from vidseq.services import video_service
from vidseq.services.arhmm_service import ARHMMService
from vidseq.services.crowd_movie_service import CrowdMovieService

logger = logging.getLogger(__name__)

router = APIRouter()


def _run_arhmm_in_background(
    service: ARHMMService,
    project_path: Path,
    fps: float,
):
    """Run ARHMM fitting synchronously (called from background task)."""
    service.run_arhmm_sync(project_path, fps)


@router.post("/projects/{project_id}/arhmm/run")
async def run_arhmm(
    project_id: int,
    session: AsyncSession = Depends(get_project_session),
    project_path: Path = Depends(get_project_folder),
):
    """Start ARHMM model fitting with binary kappa search.

    Validates that PCA scores exist, then starts fitting in background.
    Use the SSE stream endpoint (/arhmm/stream) to monitor progress.
    """
    service = ARHMMService.get_instance()

    if service.is_running():
        raise HTTPException(status_code=400, detail="ARHMM fitting already in progress")

    # Validate PCA scores exist in array_data/{video_id}/pca_scores.h5
    array_data_dir = project_path / "array_data"
    has_pca_scores = array_data_dir.exists() and any(
        (video_dir / "pca_scores.h5").exists()
        for video_dir in array_data_dir.iterdir()
        if video_dir.is_dir()
    )
    if not has_pca_scores:
        raise HTTPException(
            status_code=400,
            detail="PCA scores not found. Run PCA first before fitting ARHMM.",
        )

    # Get FPS from first video
    videos = await video_service.get_all_videos(session)
    if not videos:
        raise HTTPException(status_code=400, detail="No videos found in project")
    fps = videos[0].fps

    logger.info(
        f"POST /arhmm/run: project_id={project_id}, fps={fps}, "
        f"array_data_dir={array_data_dir}"
    )

    # Start fitting in background
    asyncio.create_task(
        asyncio.to_thread(
            _run_arhmm_in_background,
            service,
            project_path,
            fps,
        )
    )

    return {"status": "started"}


@router.post("/projects/{project_id}/arhmm/stop")
async def stop_arhmm(project_id: int):
    """Request graceful stop of ARHMM fitting."""
    service = ARHMMService.get_instance()
    if not service.is_running():
        raise HTTPException(status_code=400, detail="ARHMM fitting is not running")
    service.stop()
    return {"status": "stopping"}


@router.get("/projects/{project_id}/arhmm/status")
async def get_arhmm_status(
    project_id: int,
    project_path: Path = Depends(get_project_folder),
):
    """Get ARHMM fitting status.

    Returns saved results if available, otherwise current progress.
    """
    service = ARHMMService.get_instance()

    # If running, return live progress
    if service.is_running():
        return service.get_progress().to_dict()

    # Check for saved results on disk
    results = ARHMMService.get_results(project_path)
    if results is not None:
        return {
            "is_running": False,
            "status": "completed",
            "phase": "final_fit",
            "search_history": results.get("search_history", []),
            "log_low": 3.0,
            "log_high": 18.0,
            "current_kappa": results.get("kappa", 0),
            "current_iteration": 0,
            "total_iterations": 0,
            "chosen_kappa": results.get("kappa", 0),
            "final_median_duration_ms": results.get("median_duration_ms", 0),
            "fps": results.get("fps", 0),
            "num_videos": results.get("num_videos", 0),
            "total_frames": results.get("total_frames", 0),
            "started_at": None,
            "error": None,
            "elapsed_seconds": 0,
            "iterations_per_second": 0,
            "eta_seconds": 0,
        }

    # Nothing running, nothing saved
    return ARHMMService.get_instance().get_progress().to_dict()


@router.get("/projects/{project_id}/arhmm/stream")
async def stream_arhmm_progress(project_id: int):
    """Stream ARHMM fitting progress via Server-Sent Events."""
    service = ARHMMService.get_instance()

    async def event_generator():
        last_iteration = -1
        last_status = None
        last_history_len = -1

        while True:
            progress = service.get_progress()

            # Emit on any change
            changed = (
                progress.current_iteration != last_iteration
                or progress.status != last_status
                or len(progress.search_history) != last_history_len
            )

            if changed:
                last_iteration = progress.current_iteration
                last_status = progress.status
                last_history_len = len(progress.search_history)

                yield f"data: {json.dumps(progress.to_dict())}\n\n"

                if progress.status in ("completed", "failed"):
                    break

            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/projects/{project_id}/arhmm/results")
async def get_arhmm_results(
    project_id: int,
    project_path: Path = Depends(get_project_folder),
):
    """Get saved ARHMM results summary."""
    results = ARHMMService.get_results(project_path)
    if results is None:
        raise HTTPException(
            status_code=404,
            detail="ARHMM results not found. Run ARHMM first.",
        )
    return results


def _compute_analysis(model_path: Path, fps: float) -> dict:
    """Compute syllable analysis data from a fitted ARHMM model."""
    model = joblib.load(model_path)
    states_dict = model["states"]

    all_durations_ms: list[float] = []
    syllable_run_counts: dict[int, int] = {}

    for states in states_dict.values():
        states = np.asarray(states).ravel()
        if len(states) == 0:
            continue

        changes = np.where(np.diff(states) != 0)[0] + 1
        starts = np.concatenate([[0], changes])
        ends = np.concatenate([changes, [len(states)]])

        for s_idx, e_idx in zip(starts, ends):
            syllable = int(states[s_idx])
            duration_ms = float(e_idx - s_idx) / fps * 1000
            all_durations_ms.append(duration_ms)
            syllable_run_counts[syllable] = syllable_run_counts.get(syllable, 0) + 1

    durations = np.array(all_durations_ms) if all_durations_ms else np.array([])

    # Duration histogram
    if len(durations) > 0:
        counts, bin_edges = np.histogram(durations, bins="auto")
        bin_centers = [
            (bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(counts))
        ]
        duration_histogram = {
            "bin_edges": [round(float(x), 1) for x in bin_edges],
            "bin_centers": [round(x, 1) for x in bin_centers],
            "counts": [int(c) for c in counts],
        }
    else:
        duration_histogram = {"bin_edges": [], "bin_centers": [], "counts": []}

    # Syllable frequencies sorted by count descending
    sorted_syllables = sorted(
        syllable_run_counts.items(), key=lambda x: x[1], reverse=True
    )
    syllable_frequencies = {
        "syllables": [s for s, _ in sorted_syllables],
        "counts": [c for _, c in sorted_syllables],
    }

    return {
        "duration_histogram": duration_histogram,
        "syllable_frequencies": syllable_frequencies,
        "total_runs": len(all_durations_ms),
        "median_duration_ms": round(float(np.median(durations)), 1)
        if len(durations) > 0
        else 0,
    }


@router.get("/projects/{project_id}/arhmm/analysis")
async def get_arhmm_analysis(
    project_id: int,
    project_path: Path = Depends(get_project_folder),
):
    """Compute syllable analysis data from fitted ARHMM model."""
    model_path = project_path / "arhmm" / "model.joblib"
    if not model_path.exists():
        raise HTTPException(
            status_code=404, detail="ARHMM model not found. Run ARHMM first."
        )

    summary_path = project_path / "arhmm" / "summary.json"
    if not summary_path.exists():
        raise HTTPException(status_code=404, detail="ARHMM summary not found.")

    summary = json.loads(summary_path.read_text())
    fps = summary.get("fps", 30)

    return await asyncio.to_thread(_compute_analysis, model_path, fps)


# ── Crowd Movie Endpoints ─────────────────────────────────────────────


def _run_crowd_movies_in_background(
    service: CrowdMovieService,
    project_path: Path,
    fps: float,
):
    """Run crowd movie generation synchronously (called from background task)."""
    service.generate_crowd_movies_sync(project_path, fps)


@router.post("/projects/{project_id}/arhmm/crowd-movies/generate")
async def generate_crowd_movies(
    project_id: int,
    session: AsyncSession = Depends(get_project_session),
    project_path: Path = Depends(get_project_folder),
):
    """Start crowd movie generation for all qualifying syllables.

    Validates that ARHMM model exists, then starts generation in background.
    Use the SSE stream endpoint to monitor progress.
    """
    service = CrowdMovieService.get_instance()

    if service.is_running():
        raise HTTPException(
            status_code=400, detail="Crowd movie generation already in progress"
        )

    model_path = project_path / "arhmm" / "model.joblib"
    if not model_path.exists():
        raise HTTPException(
            status_code=400,
            detail="ARHMM model not found. Run ARHMM first before generating crowd movies.",
        )

    # Get FPS from first video
    videos = await video_service.get_all_videos(session)
    if not videos:
        raise HTTPException(status_code=400, detail="No videos found in project")
    fps = videos[0].fps

    logger.info(f"POST /arhmm/crowd-movies/generate: project_id={project_id}, fps={fps}")

    asyncio.create_task(
        asyncio.to_thread(
            _run_crowd_movies_in_background,
            service,
            project_path,
            fps,
        )
    )

    return {"status": "started"}


@router.post("/projects/{project_id}/arhmm/crowd-movies/stop")
async def stop_crowd_movies(project_id: int):
    """Request graceful stop of crowd movie generation."""
    service = CrowdMovieService.get_instance()
    if not service.is_running():
        raise HTTPException(
            status_code=400, detail="Crowd movie generation is not running"
        )
    service.stop()
    return {"status": "stopping"}


@router.get("/projects/{project_id}/arhmm/crowd-movies/status")
async def get_crowd_movie_status(project_id: int):
    """Get crowd movie generation status."""
    service = CrowdMovieService.get_instance()
    return service.get_progress().to_dict()


@router.get("/projects/{project_id}/arhmm/crowd-movies/stream")
async def stream_crowd_movie_progress(project_id: int):
    """Stream crowd movie generation progress via Server-Sent Events."""
    service = CrowdMovieService.get_instance()

    async def event_generator():
        last_completed = -1
        last_status = None

        while True:
            progress = service.get_progress()

            changed = (
                progress.completed_syllables != last_completed
                or progress.status != last_status
            )

            if changed:
                last_completed = progress.completed_syllables
                last_status = progress.status

                yield f"data: {json.dumps(progress.to_dict())}\n\n"

                if progress.status in ("completed", "failed"):
                    break

            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/projects/{project_id}/arhmm/crowd-movies/list")
async def list_crowd_movies(
    project_id: int,
    project_path: Path = Depends(get_project_folder),
):
    """List available crowd movie files with metadata."""
    return CrowdMovieService.get_available_movies(project_path)


@router.get("/projects/{project_id}/arhmm/crowd-movies/{syllable}/video")
async def stream_crowd_movie_video(
    project_id: int,
    syllable: int,
    request: Request,
    project_path: Path = Depends(get_project_folder),
):
    """Stream a crowd movie video with HTTP range support."""
    video_path = project_path / "arhmm" / "crowd_movies" / f"syllable_{syllable}.mp4"

    if not video_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Crowd movie for syllable {syllable} not found",
        )

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
