"""PCA service for aligned mask analysis.

GPU-accelerated PCA using JAX with partial fitting for memory efficiency.
Adapted from moseq3/pca.py.
"""

import logging
import pickle
from functools import partial
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')  # Headless backend
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


# =============================================================================
# JAX JIT-compiled helper functions
# =============================================================================

@jax.jit
def _compute_mini_batch_stats(mini_batch):
    """Compute sufficient statistics for a single mini-batch."""
    n_samples = mini_batch.shape[0]
    sum_x = jnp.sum(mini_batch, axis=0)
    sum_xx = mini_batch.T @ mini_batch
    return n_samples, sum_x, sum_xx


@partial(jax.jit, static_argnames=['padding_size'])
def _pad_chunk(chunk, padding_size):
    """Add zero padding to chunk."""
    n_features = chunk.shape[1]
    padding = jnp.zeros((padding_size, n_features), dtype=chunk.dtype)
    return jnp.vstack([chunk, padding])


@partial(jax.jit, static_argnames=['mini_batch_size'])
def _reshape_to_mini_batches(chunk_padded, mini_batch_size):
    """Reshape chunk into mini-batches."""
    num_mini_batches = chunk_padded.shape[0] // mini_batch_size
    return chunk_padded.reshape(num_mini_batches, mini_batch_size, chunk_padded.shape[1])


@jax.jit
def _fix_last_count(counts, remainder):
    """Fix the count of the last mini-batch to account for padding."""
    return counts.at[-1].set(remainder)


@jax.jit
def _sum_mini_batch_results(counts, sum_xs, sum_xxs):
    """Sum results across mini-batches."""
    chunk_sum_x = jnp.sum(sum_xs, axis=0)
    chunk_sum_xx = jnp.sum(sum_xxs, axis=0)
    chunk_count = jnp.sum(counts)
    return chunk_count, chunk_sum_x, chunk_sum_xx


@jax.jit
def _accumulate_statistics(total_count, sum_x, sum_xx, chunk_count, chunk_sum_x, chunk_sum_xx):
    """Accumulate chunk statistics into global totals."""
    new_total_count = total_count + chunk_count
    new_sum_x = sum_x + chunk_sum_x
    new_sum_xx = sum_xx + chunk_sum_xx
    return new_total_count, new_sum_x, new_sum_xx


@partial(jax.jit, static_argnames=['n_components', 'oversampling', 'n_iter'])
def _randomized_eigh(matrix, n_components, oversampling=10, n_iter=2, key=None):
    """
    Randomized eigendecomposition for symmetric matrices.

    Args:
        matrix: Symmetric matrix (n x n)
        n_components: Number of top eigenvectors to compute
        oversampling: Extra samples for accuracy (default 10)
        n_iter: Number of power iterations (default 2)
        key: Random key for JAX (if None, uses a fixed seed)

    Returns:
        eigenvalues: Top n_components eigenvalues (sorted descending)
        eigenvectors: Corresponding eigenvectors (n x n_components)
    """
    n = matrix.shape[0]
    k = n_components + oversampling

    if key is None:
        key = jax.random.PRNGKey(0)

    omega = jax.random.normal(key, (n, k), dtype=matrix.dtype)
    Y = matrix @ omega

    for _ in range(n_iter):
        Q, _ = jnp.linalg.qr(Y)
        Y = matrix @ Q

    Q, _ = jnp.linalg.qr(Y)
    B = Q.T @ matrix @ Q

    eigenvalues_small, eigenvectors_small = jnp.linalg.eigh(B)

    idx = jnp.argsort(eigenvalues_small)[::-1]
    eigenvalues_small = eigenvalues_small[idx]
    eigenvectors_small = eigenvectors_small[:, idx]

    eigenvalues = eigenvalues_small[:n_components]
    eigenvectors = Q @ eigenvectors_small[:, :n_components]

    return eigenvalues, eigenvectors


@partial(jax.jit, static_argnames=['n_components'])
def _compute_final_pca(sum_x, sum_xx, total_count, n_components):
    """Compute final PCA from accumulated statistics using randomized eigendecomposition."""
    mean = sum_x / total_count
    mean_outer = jnp.outer(mean, mean)

    cov_matrix = (sum_xx - total_count * mean_outer) / (total_count - 1)

    eigenvalues, eigenvectors = _randomized_eigh(cov_matrix, n_components)

    components = eigenvectors.T
    singular_values = jnp.sqrt(eigenvalues * (total_count - 1))
    explained_variance = eigenvalues

    total_variance = jnp.trace(cov_matrix)
    explained_variance_ratio = explained_variance / total_variance

    return mean, components, singular_values, explained_variance, explained_variance_ratio


# =============================================================================
# GPUPCA Class
# =============================================================================

class GPUPCA:
    """
    Parallel GPU-optimized PCA using sufficient statistics.

    Uses vmap to process mini-batches in parallel for better GPU utilization.
    Supports partial fitting for out-of-memory datasets.
    """

    def __init__(self, n_components: int, batch_size: int = 10_000, mini_batch_size: int = 500):
        self.n_components = n_components
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size

        self.components_ = None
        self.singular_values_ = None
        self.mean_ = None
        self.explained_variance_ratio_ = None
        self.n_samples_seen_ = None

        # Accumulated statistics for partial fitting
        self._total_count = None
        self._sum_x = None
        self._sum_xx = None
        self._n_features = None
        self._fitted = False

    def partial_fit(self, X, show_progress: bool = False):
        """
        Incrementally fit PCA by updating accumulated statistics.

        Args:
            X: Input data array of shape (n_samples, n_features)
            show_progress: Whether to show tqdm progress bar

        Returns:
            self
        """
        N = X.shape[0]
        n_features = X.shape[1]

        # Initialize accumulators on first call
        if self._total_count is None:
            self._n_features = n_features
            self._total_count = jnp.array(0, dtype=jnp.int32)
            self._sum_x = jnp.zeros(n_features, dtype=jnp.float32)
            self._sum_xx = jnp.zeros((n_features, n_features), dtype=jnp.float32)
        else:
            if n_features != self._n_features:
                raise ValueError(f"Inconsistent feature count: expected {self._n_features}, got {n_features}")

        num_chunks = int(np.ceil(N / self.batch_size))
        iterator = range(num_chunks)
        if show_progress:
            iterator = tqdm(iterator, desc='Fitting PCA', unit='Batch')

        for chunk_idx in iterator:
            start_idx = chunk_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, N)
            chunk_np = X[start_idx:end_idx]

            chunk = jax.device_put(jnp.array(chunk_np, dtype=jnp.float32))

            chunk_size = chunk.shape[0]
            remainder = chunk_size % self.mini_batch_size

            if remainder != 0:
                padding_size = self.mini_batch_size - remainder
                chunk_padded = _pad_chunk(chunk, padding_size)
            else:
                chunk_padded = chunk

            mini_batches = _reshape_to_mini_batches(chunk_padded, self.mini_batch_size)
            counts, sum_xs, sum_xxs = jax.vmap(_compute_mini_batch_stats)(mini_batches)

            if remainder != 0:
                counts = _fix_last_count(counts, remainder)

            chunk_count, chunk_sum_x, chunk_sum_xx = _sum_mini_batch_results(counts, sum_xs, sum_xxs)

            self._total_count, self._sum_x, self._sum_xx = _accumulate_statistics(
                self._total_count, self._sum_x, self._sum_xx, chunk_count, chunk_sum_x, chunk_sum_xx
            )

        return self

    def finalize_fit(self):
        """
        Compute final PCA components from accumulated statistics.

        Returns:
            self
        """
        if self._total_count is None:
            raise ValueError("No data has been fit. Call partial_fit first.")

        mean, components, singular_values, explained_variance, explained_variance_ratio = _compute_final_pca(
            self._sum_x, self._sum_xx, self._total_count, self.n_components
        )

        self.n_samples_seen_ = int(self._total_count)
        self.mean_ = np.array(mean)
        self.components_ = np.array(components)
        self.singular_values_ = np.array(singular_values)
        self.explained_variance_ratio_ = np.array(explained_variance_ratio)

        self.components_ = self._svd_flip_v_based(self.components_)

        self._fitted = True
        return self

    def fit(self, X):
        """Fit PCA on the input data X."""
        self._total_count = None
        self._sum_x = None
        self._sum_xx = None
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

    def _svd_flip_v_based(self, components):
        """Apply scikit-learn's sign convention."""
        max_abs_indices = np.argmax(np.abs(components), axis=1)
        signs = np.sign(components[np.arange(len(components)), max_abs_indices])
        return components * signs[:, np.newaxis]


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
    h5_files: list[Path],
) -> None:
    """
    Compute PCA scores for all videos and store in HDF5 files.

    After PCA fitting, this function transforms each video's aligned masks
    into PCA scores and stores them in pca_scores/{video_id}.h5.

    Args:
        project_path: Path to the project folder
        pca: Fitted GPUPCA object
        h5_files: List of aligned mask HDF5 file paths
    """
    scores_dir = project_path / "pca_scores"
    scores_dir.mkdir(exist_ok=True)

    for h5_path in tqdm(h5_files, desc="Computing scores"):
        video_id = h5_path.stem
        scores_path = scores_dir / f"{video_id}.h5"

        with h5py.File(h5_path, "r") as f_in:
            masks = f_in["masks"]
            n_frames = masks.shape[0]

            with h5py.File(scores_path, "w") as f_out:
                # Pre-allocate scores dataset
                scores_ds = f_out.create_dataset(
                    "scores",
                    shape=(n_frames, pca.n_components),
                    dtype=np.float32,
                )

                # Process in chunks to avoid memory issues
                for start in range(0, n_frames, CHUNK_SIZE):
                    end = min(start + CHUNK_SIZE, n_frames)
                    chunk = masks[start:end]
                    flat = chunk.reshape(chunk.shape[0], -1).astype(np.float32)
                    chunk_scores = pca.transform(flat)
                    scores_ds[start:end] = chunk_scores

        logger.info(f"[PCA] Saved scores for video {video_id} ({n_frames} frames)")


def run_pca(project_path: Path, n_components: int = 20) -> dict:
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

    aligned_masks_dir = project_path / "aligned_masks"
    if not aligned_masks_dir.exists():
        raise FileNotFoundError(f"No aligned_masks directory found at {aligned_masks_dir}")

    h5_files = sorted(aligned_masks_dir.glob("*.h5"))
    if not h5_files:
        raise FileNotFoundError("No aligned mask files found")

    logger.info(f"[PCA] Found {len(h5_files)} aligned mask files")

    # Get mask dimensions from first file
    with h5py.File(h5_files[0], "r") as f:
        mask_shape = f["masks"].shape
        height, width = mask_shape[1], mask_shape[2]
        logger.info(f"[PCA] Mask dimensions: {height}x{width} ({height * width} features)")

    # Initialize PCA
    pca = GPUPCA(n_components=n_components)

    # Process each video file
    total_frames = 0
    for h5_path in h5_files:
        logger.info(f"[PCA] Processing {h5_path.name}")

        with h5py.File(h5_path, "r") as f:
            masks = f["masks"]
            n_frames = masks.shape[0]
            total_frames += n_frames

            # Process in chunks
            for start in tqdm(range(0, n_frames, CHUNK_SIZE), desc=f"Video {h5_path.stem}"):
                end = min(start + CHUNK_SIZE, n_frames)
                chunk = masks[start:end]  # (chunk_size, H, W)

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
    _compute_and_store_scores(project_path, pca, h5_files)

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

    scores_path = project_path / "pca_scores" / f"{video_id}.h5"

    if not scores_path.exists():
        raise FileNotFoundError(f"PCA scores not found for video {video_id}")

    with h5py.File(scores_path, "r") as f:
        scores_ds = f["scores"]
        n_frames, n_components = scores_ds.shape

        # Determine frame range
        if end_frame is None:
            end_frame = n_frames - 1
        end_frame = min(end_frame, n_frames - 1)
        start_frame = max(0, start_frame)

        # Validate PC indices
        valid_pc_indices = [i for i in pc_indices if 0 <= i < n_components]

        # Load the slice of data we need
        frame_slice = scores_ds[start_frame:end_frame + 1, :]

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
    scores_path = project_path / "pca_scores" / f"{video_id}.h5"
    return scores_path.exists()
