"""PCA API routes for aligned mask analysis."""

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from vidseq.api.dependencies import get_project_folder
from vidseq.services import pca_service

router = APIRouter()


@router.post("/projects/{project_id}/pca/run")
async def run_pca(
    project_id: int,
    n_components: int = 20,
    project_path: Path = Depends(get_project_folder),
):
    """
    Run PCA on aligned masks.

    This is a blocking operation that processes all aligned masks in the project.
    Progress is logged via tqdm.

    Args:
        project_id: Project ID
        n_components: Number of PCA components to compute (default: 20)

    Returns:
        dict with n_components, explained_variance_ratio, total_frames
    """
    try:
        result = pca_service.run_pca(project_path, n_components)
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
