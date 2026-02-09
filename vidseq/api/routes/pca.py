"""PCA API routes for aligned mask analysis."""

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from vidseq.api.dependencies import get_project_folder
from vidseq.api.schemas import PCARequest
from vidseq.services import pca_service

router = APIRouter()


@router.post("/projects/{project_id}/pca")
async def create_pca(
    project_id: int,
    request: PCARequest,
    project_path: Path = Depends(get_project_folder),
):
    """Run PCA on aligned masks for selected videos."""
    try:
        result = pca_service.run_pca(project_path, request.n_components, request.video_ids)
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/projects/{project_id}/pca/status")
async def get_pca_status(
    project_id: int,
    project_path: Path = Depends(get_project_folder),
):
    """
    Get PCA status and summary statistics.

    Returns:
        dict with has_pca, n_components, explained_variance_ratio, total_frames
    """
    return pca_service.get_pca_status(project_path)


@router.get("/projects/{project_id}/pca/scree-plot")
async def get_scree_plot(
    project_id: int,
    project_path: Path = Depends(get_project_folder),
):
    """Return scree plot image."""
    plot_path = project_path / "pca_scree.png"
    if not plot_path.exists():
        raise HTTPException(status_code=404, detail="Scree plot not found. Run PCA first.")
    return FileResponse(plot_path, media_type="image/png")


@router.get("/projects/{project_id}/pca/components-plot")
async def get_components_plot(
    project_id: int,
    project_path: Path = Depends(get_project_folder),
):
    """Return components plot image."""
    plot_path = project_path / "pca_components.png"
    if not plot_path.exists():
        raise HTTPException(status_code=404, detail="Components plot not found. Run PCA first.")
    return FileResponse(plot_path, media_type="image/png")


@router.get("/projects/{project_id}/videos/{video_id}/pca-scores")
async def get_pca_scores(
    project_id: int,
    video_id: int,
    pc_indices: str = "0,1,2",
    max_samples: int = 800,
    start_frame: int = 0,
    end_frame: int | None = None,
    project_path: Path = Depends(get_project_folder),
):
    """
    Get LTTB-downsampled PCA scores for visualization.

    Args:
        project_id: Project ID
        video_id: Video ID
        pc_indices: Comma-separated PC indices (e.g., "0,1,2")
        max_samples: Max points per PC after downsampling
        start_frame: Start frame (inclusive)
        end_frame: End frame (inclusive), None for last frame

    Returns:
        {
            "n_components": 20,
            "scores": {
                "0": [{"frame_idx": 0, "score": 1.23}, ...],
                "1": [{"frame_idx": 0, "score": -0.45}, ...],
            }
        }
    """
    # Parse comma-separated PC indices
    try:
        pc_idx_list = [int(x.strip()) for x in pc_indices.split(",") if x.strip()]
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid pc_indices format. Expected comma-separated integers.")

    try:
        return pca_service.get_pca_scores_downsampled(
            project_path,
            video_id,
            pc_idx_list,
            max_samples,
            start_frame,
            end_frame,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/projects/{project_id}/videos/{video_id}/pca-scores/exists")
async def check_pca_scores_exist(
    project_id: int,
    video_id: int,
    project_path: Path = Depends(get_project_folder),
):
    """Check if PCA scores exist for a video."""
    exists = pca_service.check_pca_scores_exist(project_path, video_id)
    return {"exists": exists}
