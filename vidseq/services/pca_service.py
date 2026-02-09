"""PCA service for aligned mask analysis.

GPU-accelerated PCA using JAX with partial fitting for memory efficiency.
Adapted from moseq3/pca.py.
"""

import logging
import pickle
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')  # Headless backend
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from vidseq.services.array_storage import aligned_masks, pca_scores, create_pca_scores_array

logger = logging.getLogger(__name__)


# =============================================================================
# JAX JIT-compiled helper functions
# =============================================================================

@jax.jit
def _incremental_mean_and_var(
    X: jnp.ndarray,
    last_mean: jnp.ndarray,
    last_var: jnp.ndarray,
    last_count: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Chan-Golub-LeVeque numerically stable mean/variance update.

    Args:
        X: New batch of data (n_samples, n_features)
        last_mean: Previous mean (n_features,)
        last_var: Previous variance (n_features,)
        last_count: Previous sample count (scalar)

    Returns:
        Tuple of (updated_mean, updated_var, n_total)
    """
    n_samples = X.shape[0]
    n_total = last_count + n_samples

    # New batch statistics
    new_sum = jnp.sum(X, axis=0)
    new_mean = new_sum / n_samples
    new_unnorm_var = jnp.sum((X - new_mean) ** 2, axis=0)

    # Combine old and new statistics
    last_sum = last_mean * last_count
    updated_mean = (last_sum + new_sum) / n_total

    # Variance update with correction term for combining distributions
    delta = last_mean - new_mean
    updated_unnorm_var = (
        last_var * last_count
        + new_unnorm_var
        + (last_count * n_samples / n_total) * delta ** 2
    )
    updated_var = updated_unnorm_var / n_total

    return updated_mean, updated_var, n_total


@jax.jit
def _svd_flip_v(Vt: jnp.ndarray) -> jnp.ndarray:
    """
    Apply sklearn sign convention (make max abs value positive per row).

    Args:
        Vt: Right singular vectors (k, n_features)

    Returns:
        Sign-flipped Vt
    """
    max_abs_idx = jnp.argmax(jnp.abs(Vt), axis=1)
    signs = jnp.sign(Vt[jnp.arange(Vt.shape[0]), max_abs_idx])
    return Vt * signs[:, None]


@partial(jax.jit, static_argnames=['n_components'])
def _fit_first_batch(
    X: jnp.ndarray,
    n_components: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    JIT-compiled first batch fitting: center, SVD, extract top k components.

    Args:
        X: Batch of data (n_samples, n_features)
        n_components: Number of components to keep

    Returns:
        Tuple of (components, singular_values, mean, var)
    """
    n_samples = X.shape[0]
    batch_mean = jnp.mean(X, axis=0)
    X_centered = X - batch_mean

    # SVD on centered data
    U, S, Vt = jnp.linalg.svd(X_centered, full_matrices=False)
    Vt = _svd_flip_v(Vt)

    # Keep top k components
    components = Vt[:n_components]
    singular_values = S[:n_components]

    # Variance: sum of squared deviations / n
    var = jnp.sum((X - batch_mean) ** 2, axis=0) / n_samples

    return components, singular_values, batch_mean, var


@partial(jax.jit, static_argnames=['n_components'])
def _fit_subsequent_batch(
    X: jnp.ndarray,
    old_components: jnp.ndarray,
    old_singular_values: jnp.ndarray,
    old_mean: jnp.ndarray,
    old_var: jnp.ndarray,
    n_old: jnp.ndarray,
    n_components: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    JIT-compiled subsequent batch fitting: augmented matrix SVD.

    Args:
        X: Batch of data (n_samples, n_features)
        old_components: Previous components (k, n_features)
        old_singular_values: Previous singular values (k,)
        old_mean: Previous mean (n_features,)
        old_var: Previous variance (n_features,)
        n_old: Previous sample count (scalar)
        n_components: Number of components to keep

    Returns:
        Tuple of (components, singular_values, mean, var)
    """
    n_samples = X.shape[0]
    n_total = n_old + n_samples

    # Center to batch mean
    batch_mean = jnp.mean(X, axis=0)
    X_centered = X - batch_mean

    # Update mean and variance (Chan-Golub-LeVeque)
    new_mean, new_var, _ = _incremental_mean_and_var(X, old_mean, old_var, n_old)

    # Mean correction vector
    mean_correction = jnp.sqrt((n_old * n_samples) / n_total) * (old_mean - batch_mean)

    # Construct augmented matrix
    augmented = jnp.vstack([
        old_singular_values[:, None] * old_components,  # (k, n_features)
        X_centered,                                      # (batch, n_features)
        mean_correction[None, :],                        # (1, n_features)
    ])

    # SVD on augmented matrix
    U, S, Vt = jnp.linalg.svd(augmented, full_matrices=False)
    Vt = _svd_flip_v(Vt)

    # Keep top k components
    components = Vt[:n_components]
    singular_values = S[:n_components]

    return components, singular_values, new_mean, new_var


# =============================================================================
# GPUPCA Class
# =============================================================================

class GPUPCA:
    """
    GPU-optimized Incremental PCA using JAX.

    Uses the incremental SVD algorithm from Ross et al. (2008) to maintain
    O(n_features * n_components) memory instead of O(n_features^2).
    Supports partial fitting for out-of-memory datasets.
    """

    def __init__(self, n_components: int, batch_size: int = 2000, mini_batch_size: int = 500):
        """
        Initialize IncrementalPCA.

        Args:
            n_components: Number of principal components to keep
            batch_size: Number of samples to process per SVD update
            mini_batch_size: Kept for API compatibility, unused in new algorithm
        """
        self.n_components = n_components
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size  # Kept for API compatibility

        # Public attributes (set after finalize_fit)
        self.components_ = None
        self.singular_values_ = None
        self.mean_ = None
        self.explained_variance_ratio_ = None
        self.n_samples_seen_ = None

        # Private state for incremental fitting
        self._components = None      # (k, n_features) JAX array
        self._singular_values = None  # (k,) JAX array
        self._mean = None            # (n_features,) JAX array
        self._var = None             # (n_features,) JAX array
        self._n_samples_seen = 0     # int
        self._n_features = None
        self._fitted = False

    def _partial_fit_batch(self, X: jnp.ndarray) -> None:
        """
        Process a single batch through incremental SVD.

        Args:
            X: Batch of data as JAX array (n_samples, n_features)
        """
        n_samples = X.shape[0]

        if self._n_samples_seen == 0:
            # First batch: JIT-compiled centered SVD
            components, singular_values, mean, var = _fit_first_batch(
                X, self.n_components
            )
            self._components = components
            self._singular_values = singular_values
            self._mean = mean
            self._var = var
            self._n_samples_seen = n_samples
        else:
            # Subsequent batches: JIT-compiled augmented SVD
            components, singular_values, mean, var = _fit_subsequent_batch(
                X,
                self._components,
                self._singular_values,
                self._mean,
                self._var,
                jnp.array(self._n_samples_seen, dtype=jnp.float32),
                self.n_components,
            )
            self._components = components
            self._singular_values = singular_values
            self._mean = mean
            self._var = var
            self._n_samples_seen = self._n_samples_seen + n_samples

    def partial_fit(self, X, show_progress: bool = False):
        """
        Incrementally fit PCA using batched SVD updates.

        Args:
            X: Input data array of shape (n_samples, n_features)
            show_progress: Whether to show tqdm progress bar

        Returns:
            self
        """
        n_samples, n_features = X.shape

        # Validate dimensions
        if self._n_features is None:
            self._n_features = n_features
        elif n_features != self._n_features:
            raise ValueError(f"Inconsistent feature count: expected {self._n_features}, got {n_features}")

        # Process in chunks of batch_size
        num_chunks = int(np.ceil(n_samples / self.batch_size))
        iterator = range(num_chunks)
        if show_progress:
            iterator = tqdm(iterator, desc='Fitting PCA', unit='batch')

        for chunk_idx in iterator:
            start = chunk_idx * self.batch_size
            end = min(start + self.batch_size, n_samples)
            chunk = jax.device_put(jnp.array(X[start:end], dtype=jnp.float32))
            self._partial_fit_batch(chunk)

        return self

    def finalize_fit(self):
        """
        Finalize PCA fitting and convert to numpy arrays for pickling.

        Returns:
            self
        """
        if self._n_samples_seen == 0:
            raise ValueError("No data has been fit. Call partial_fit first.")

        # Compute explained variance ratio
        total_var = jnp.sum(self._var) * self._n_samples_seen
        explained_var_ratio = self._singular_values ** 2 / total_var

        # Convert JAX arrays to numpy for pickling
        self.components_ = np.array(self._components)
        self.singular_values_ = np.array(self._singular_values)
        self.mean_ = np.array(self._mean)
        self.explained_variance_ratio_ = np.array(explained_var_ratio)
        self.n_samples_seen_ = self._n_samples_seen

        # Clear JAX state to free GPU memory
        self._components = None
        self._singular_values = None
        self._mean = None
        self._var = None

        self._fitted = True
        return self

    def fit(self, X):
        """Fit PCA on the input data X."""
        # Reset all state
        self._components = None
        self._singular_values = None
        self._mean = None
        self._var = None
        self._n_samples_seen = 0
        self._n_features = None
        self._fitted = False

        self.partial_fit(X, show_progress=True)
        self.finalize_fit()
        return self

    def transform(self, X):
        """Apply PCA transformation to X."""
        if not self._fitted:
            raise ValueError("Model has not been fitted.")
        return (X - self.mean_) @ self.components_.T


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_scree(pca) -> plt.Figure:
    """
    Create a scree plot showing cumulative explained variance.

    Args:
        pca: Fitted GPUPCA object

    Returns:
        matplotlib.figure.Figure
    """
    fig = plt.figure(figsize=(10, 6))
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    components = range(1, len(cumulative_variance) + 1)

    plt.plot(components, cumulative_variance, 'bo-')
    plt.xlabel('Component')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('Cumulative Scree Plot')
    plt.grid(True)
    plt.xticks(components)

    return fig


def plot_components(pca, height: int, width: int) -> plt.Figure:
    """
    Plot PCA components as grayscale images.

    Args:
        pca: Fitted GPUPCA object
        height: Original image height
        width: Original image width

    Returns:
        matplotlib.figure.Figure
    """
    n_components = pca.components_.shape[0]

    n_cols = int(np.ceil(np.sqrt(n_components)))
    n_rows = int(np.ceil(n_components / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 15))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    for i in range(n_components):
        row = i // n_cols
        col = i % n_cols

        component_img = pca.components_[i].reshape(height, width)

        axes[row, col].imshow(component_img, cmap='gray')
        axes[row, col].set_title(f'Component {i+1}')
        axes[row, col].axis('off')

    for i in range(n_components, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].axis('off')

    plt.tight_layout()
    return fig


# =============================================================================
# High-Level Service Functions
# =============================================================================

CHUNK_SIZE = 1000  # Frames per chunk when loading from HDF5


def _compute_and_store_scores(
    project_path: Path,
    pca: GPUPCA,
    video_ids: list[int],
) -> None:
    """
    Compute PCA scores for all videos and store in HDF5 files.

    After PCA fitting, this function transforms each video's aligned masks
    into PCA scores and stores them in array_data/{video_id}/pca_scores.h5.

    Args:
        project_path: Path to the project folder
        pca: Fitted GPUPCA object
        video_ids: List of video IDs to process
    """
    for video_id in tqdm(video_ids, desc="Computing scores"):
        with aligned_masks(project_path, video_id, "r") as masks_data:
            n_frames = masks_data.shape[0]

            # Create the PCA scores H5 file with pre-allocated dataset
            create_pca_scores_array(project_path, video_id, n_frames, pca.n_components)

            with pca_scores(project_path, video_id, "a") as scores_data:
                # Process in chunks to avoid memory issues
                for start in range(0, n_frames, CHUNK_SIZE):
                    end = min(start + CHUNK_SIZE, n_frames)
                    chunk = masks_data[start:end]
                    flat = chunk.reshape(chunk.shape[0], -1).astype(np.float32)
                    chunk_scores = pca.transform(flat)
                    scores_data[start:end] = chunk_scores

        logger.info(f"[PCA] Saved scores for video {video_id} ({n_frames} frames)")


def run_pca(project_path: Path, n_components: int, video_ids: list[int]) -> dict:
    """
    Run PCA on all aligned masks in the project.

    Args:
        project_path: Path to the project folder
        n_components: Number of PCA components to compute

    Returns:
        dict with keys: n_components, explained_variance_ratio, total_frames
    """
    logger.info(f"[PCA] Starting PCA with {n_components} components")
    logger.info(f"[PCA] JAX devices: {jax.devices()}")

    array_data_dir = project_path / "array_data"
    if not array_data_dir.exists():
        raise FileNotFoundError(f"No array_data directory found at {array_data_dir}")

    # Filter to selected videos that have aligned masks
    discovered_ids = []
    for vid in video_ids:
        mask_path = array_data_dir / str(vid) / "aligned_masks.h5"
        if mask_path.exists():
            discovered_ids.append(vid)
    video_ids = discovered_ids

    if not video_ids:
        raise FileNotFoundError("No aligned mask files found")

    logger.info(f"[PCA] Found {len(video_ids)} aligned mask files")

    # Get mask dimensions from first file
    first_video_id = video_ids[0]
    with aligned_masks(project_path, first_video_id, "r") as masks_data:
        mask_shape = masks_data.shape
        height, width = mask_shape[1], mask_shape[2]
        logger.info(f"[PCA] Mask dimensions: {height}x{width} ({height * width} features)")

    # Initialize PCA (batch_size=2000 keeps vmap parallelism while limiting memory)
    pca = GPUPCA(n_components=n_components, batch_size=2000)

    # Process each video file
    total_frames = 0
    for video_id in video_ids:
        logger.info(f"[PCA] Processing video {video_id}")

        with aligned_masks(project_path, video_id, "r") as masks_data:
            n_frames = masks_data.shape[0]
            total_frames += n_frames

            # Process in chunks
            for start in tqdm(range(0, n_frames, CHUNK_SIZE), desc=f"Video {video_id}"):
                end = min(start + CHUNK_SIZE, n_frames)
                chunk = masks_data[start:end]  # (chunk_size, H, W)

                # Flatten to (chunk_size, H*W) and convert to float
                flat = chunk.reshape(chunk.shape[0], -1).astype(np.float32)

                # Partial fit
                pca.partial_fit(flat)

    logger.info(f"[PCA] Total frames processed: {total_frames}")
    logger.info("[PCA] Finalizing fit...")

    # Finalize PCA
    pca.finalize_fit()

    logger.info("[PCA] Saving model and plots...")

    # Save PCA model
    model_path = project_path / "pca_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(pca, f)
    logger.info(f"[PCA] Model saved to {model_path}")

    # Generate and save plots
    scree_fig = plot_scree(pca)
    scree_path = project_path / "pca_scree.png"
    scree_fig.savefig(scree_path, dpi=150, bbox_inches='tight')
    plt.close(scree_fig)
    logger.info(f"[PCA] Scree plot saved to {scree_path}")

    components_fig = plot_components(pca, height, width)
    components_path = project_path / "pca_components.png"
    components_fig.savefig(components_path, dpi=150, bbox_inches='tight')
    plt.close(components_fig)
    logger.info(f"[PCA] Components plot saved to {components_path}")

    # Compute and store scores for all videos
    logger.info("[PCA] Computing and storing scores for all videos...")
    _compute_and_store_scores(project_path, pca, video_ids)

    logger.info("[PCA] Done!")

    return {
        "n_components": n_components,
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "total_frames": total_frames,
    }


def get_pca_status(project_path: Path) -> dict:
    """
    Check if PCA has been run and return status.

    Args:
        project_path: Path to the project folder

    Returns:
        dict with keys: has_pca, n_components, explained_variance_ratio, total_frames
    """
    model_path = project_path / "pca_model.pkl"

    if not model_path.exists():
        return {
            "has_pca": False,
            "n_components": None,
            "explained_variance_ratio": None,
            "total_frames": None,
        }

    try:
        with open(model_path, "rb") as f:
            pca = pickle.load(f)

        return {
            "has_pca": True,
            "n_components": pca.n_components,
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "total_frames": pca.n_samples_seen_,
        }
    except Exception as e:
        logger.error(f"[PCA] Error loading PCA model: {e}")
        return {
            "has_pca": False,
            "n_components": None,
            "explained_variance_ratio": None,
            "total_frames": None,
        }


def get_pca_scores_downsampled(
    project_path: Path,
    video_id: int,
    pc_indices: list[int],
    max_samples: int = 800,
    start_frame: int = 0,
    end_frame: int | None = None,
) -> dict:
    """
    Load LTTB-downsampled PCA scores for specified principal components.

    Args:
        project_path: Path to the project folder
        video_id: Video ID
        pc_indices: List of PC indices to fetch (0-indexed)
        max_samples: Maximum samples per PC after downsampling
        start_frame: Start frame (inclusive)
        end_frame: End frame (inclusive), None for last frame

    Returns:
        dict with keys:
            - n_components: int (total available)
            - scores: dict mapping pc_index (str) -> list of {frame_idx, score}
    """
    from vidseq.services.lttb import downsample_scores

    with pca_scores(project_path, video_id, "r") as scores_data:
        n_frames, n_components = scores_data.shape

        # Determine frame range
        if end_frame is None:
            end_frame = n_frames - 1
        end_frame = min(end_frame, n_frames - 1)
        start_frame = max(0, start_frame)

        # Validate PC indices
        valid_pc_indices = [i for i in pc_indices if 0 <= i < n_components]

        # Load the slice of data we need
        frame_slice = scores_data[start_frame:end_frame + 1, :]

        result_scores = {}
        for pc_idx in valid_pc_indices:
            # Build score points for this PC
            pc_scores = frame_slice[:, pc_idx]
            score_points = [
                {"frame_idx": start_frame + i, "score": float(pc_scores[i])}
                for i in range(len(pc_scores))
            ]

            # Downsample using LTTB
            downsampled = downsample_scores(score_points, max_samples)
            result_scores[str(pc_idx)] = downsampled

    return {
        "n_components": n_components,
        "scores": result_scores,
    }


def check_pca_scores_exist(project_path: Path, video_id: int) -> bool:
    """Check if PCA scores exist for a video."""
    scores_path = project_path / "array_data" / str(video_id) / "pca_scores.h5"
    return scores_path.exists()
