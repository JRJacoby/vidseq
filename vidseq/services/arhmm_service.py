"""ARHMM model fitting service with binary kappa search.

Adapted from moseq3/model.py. Uses jax-moseq's ARHMM implementation
for Gibbs sampling with an HDP-AR-HMM model.
"""

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import joblib
import numpy as np
from jax_moseq.models import arhmm
from jax_moseq.utils.utils import batch, get_durations, unbatch

from vidseq.services.array_storage import pca_scores

jax.config.update("jax_enable_x64", True)

logger = logging.getLogger(__name__)

FIT_CHECKPOINT_FREQUENCY = 25  # Save every 25 iterations during final fit


def _atomic_json_write(path: Path, data: dict) -> None:
    """Write JSON atomically: write to tmp, then rename."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.rename(path)


def _atomic_joblib_write(path: Path, obj: object) -> None:
    """Write joblib atomically: write to tmp, then rename."""
    tmp = path.with_name(path.name + ".tmp")
    joblib.dump(obj, tmp)
    tmp.rename(path)


@dataclass
class ARHMMProgress:
    """Real-time ARHMM fitting progress state for SSE streaming."""

    is_running: bool = False
    status: str = "idle"  # idle, searching, fitting, completed, failed

    # Binary search state
    phase: str = "search"  # search, final_fit
    search_history: list[dict] = field(default_factory=list)
    log_low: float = 3.0  # log10(1e3)
    log_high: float = 18.0  # log10(1e18)
    current_kappa: float = 0.0

    # Gibbs iteration progress (within current kappa attempt or final fit)
    current_iteration: int = 0
    total_iterations: int = 0

    # Timing
    elapsed_seconds: float = 0.0
    iterations_per_second: float = 0.0
    eta_seconds: float = 0.0

    # Final fit info
    chosen_kappa: float = 0.0
    final_median_duration_ms: float = 0.0

    # Metadata
    fps: float = 0.0
    num_videos: int = 0
    total_frames: int = 0
    started_at: float | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class ARHMMService:
    """Singleton service for ARHMM model fitting with binary kappa search."""

    _instance = None

    @classmethod
    def get_instance(cls) -> "ARHMMService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self._progress = ARHMMProgress()
        self._is_running = False
        self._stop_requested = False
        self._iteration_timestamps: list[float] = []

    def is_running(self) -> bool:
        return self._is_running

    def get_progress(self) -> ARHMMProgress:
        return self._progress

    def stop(self):
        """Request graceful stop of the current fitting process."""
        if self._is_running:
            self._stop_requested = True
            logger.info("[ARHMM] Stop requested")

    def _update_timing(self) -> None:
        """Update iterations_per_second and eta_seconds from recent iteration timestamps."""
        now = time.time()
        self._iteration_timestamps.append(now)
        # Keep last 20 timestamps for rolling average
        if len(self._iteration_timestamps) > 20:
            self._iteration_timestamps = self._iteration_timestamps[-20:]

        if len(self._iteration_timestamps) >= 2:
            elapsed = self._iteration_timestamps[-1] - self._iteration_timestamps[0]
            n_iters = len(self._iteration_timestamps) - 1
            ips = n_iters / elapsed if elapsed > 0 else 0.0
            self._progress.iterations_per_second = round(ips, 2)

            remaining = self._progress.total_iterations - self._progress.current_iteration
            self._progress.eta_seconds = round(remaining / ips, 1) if ips > 0 else 0.0

        if self._progress.started_at:
            self._progress.elapsed_seconds = round(now - self._progress.started_at, 1)

    def _reset_timing(self) -> None:
        """Reset iteration timestamps for a new phase."""
        self._iteration_timestamps.clear()

    def run_arhmm_sync(
        self,
        project_path: Path,
        fps: float,
        search_iterations: int = 25,
        final_iterations: int = 200,
    ) -> None:
        """Run the full ARHMM pipeline synchronously (called from background thread).

        Supports resuming from checkpoints:
        - If search_checkpoint.json exists, resumes kappa search.
        - If fit_checkpoint exists, resumes final fit.
        """
        try:
            self._is_running = True
            self._stop_requested = False
            self._progress = ARHMMProgress(
                is_running=True,
                status="searching",
                phase="search",
                fps=fps,
                started_at=time.time(),
            )

            results_dir = project_path / "arhmm"
            results_dir.mkdir(exist_ok=True)

            # Load PCA scores
            sessions_dict = self._load_pca_scores(project_path)
            if not sessions_dict:
                raise ValueError("No PCA score files found")

            first_scores = next(iter(sessions_dict.values()))
            latent_dim = first_scores.shape[1]
            total_frames = sum(s.shape[0] for s in sessions_dict.values())

            self._progress.num_videos = len(sessions_dict)
            self._progress.total_frames = total_frames

            logger.info(
                f"[ARHMM] Loaded {len(sessions_dict)} sessions, "
                f"{total_frames} total frames, latent_dim={latent_dim}, fps={fps}"
            )

            if self._stop_requested:
                self._progress.status = "failed"
                self._progress.error = "Stopped before search began"
                return

            # --- Phase 1: Binary search for kappa (with checkpoint resume) ---
            chosen_kappa = self._binary_search_kappa(
                sessions_dict, latent_dim, fps, search_iterations, project_path
            )

            if self._stop_requested:
                self._progress.status = "failed"
                self._progress.error = "Stopped during kappa search"
                return

            if chosen_kappa is None:
                raise ValueError("Binary search failed to find any valid kappa")

            logger.info(f"[ARHMM] Chosen kappa: {chosen_kappa:.2e}")

            # --- Phase 2: Final fit (with checkpoint resume) ---
            self._progress.phase = "final_fit"
            self._progress.status = "fitting"
            self._progress.chosen_kappa = chosen_kappa
            self._progress.current_iteration = 0
            self._progress.total_iterations = final_iterations
            self._reset_timing()

            model = self._run_final_fit(
                sessions_dict, chosen_kappa, latent_dim, final_iterations, project_path
            )

            if self._stop_requested:
                self._progress.status = "failed"
                self._progress.error = "Stopped during final fit"
                return

            # Compute final median duration
            durations = get_durations(model["states"])
            median_frames = float(np.median(durations))
            median_ms = median_frames / fps * 1000
            self._progress.final_median_duration_ms = round(median_ms, 1)

            logger.info(f"[ARHMM] Final median duration: {median_ms:.1f}ms")

            # Save final results
            self._save_results(project_path, model, latent_dim)

            # Clean up checkpoint files (no longer needed)
            self._cleanup_checkpoints(project_path)

            # Generate crowd movies
            from vidseq.services.crowd_movie_service import CrowdMovieService

            crowd_service = CrowdMovieService.get_instance()
            if not crowd_service.is_running():
                logger.info("[ARHMM] Starting crowd movie generation")
                crowd_service.generate_crowd_movies_sync(project_path, fps)

            self._progress.status = "completed"
            logger.info("[ARHMM] Pipeline completed successfully")

        except Exception as e:
            logger.exception(f"[ARHMM] Pipeline failed: {e}")
            self._progress.status = "failed"
            self._progress.error = str(e)
        finally:
            self._progress.is_running = False
            self._is_running = False
            self._stop_requested = False

    # ── PCA score loading ──────────────────────────────────────────────

    def _load_pca_scores(self, project_path: Path) -> dict[str, np.ndarray]:
        """Load PCA scores from h5 files into a sessions dict."""
        array_data_dir = project_path / "array_data"
        video_ids = []
        for video_dir in sorted(array_data_dir.iterdir()):
            if video_dir.is_dir() and (video_dir / "pca_scores.h5").exists():
                try:
                    video_ids.append(int(video_dir.name))
                except ValueError:
                    continue

        sessions_dict = {}
        for video_id in video_ids:
            with pca_scores(project_path, video_id, "r") as scores:
                sessions_dict[str(video_id)] = scores[:]
        return sessions_dict

    # ── Binary search with checkpointing ───────────────────────────────

    def _binary_search_kappa(
        self,
        sessions_dict: dict[str, np.ndarray],
        latent_dim: int,
        fps: float,
        search_iterations: int,
        project_path: Path,
    ) -> float | None:
        """Binary search for kappa in log space to achieve 300-400ms median syllable duration.

        Resumes from search_checkpoint.json if it exists.
        """
        log_low = 3.0  # log10(1e3)
        log_high = 18.0  # log10(1e18)
        max_search_steps = 30
        start_step = 0
        best_kappa = None
        best_distance = float("inf")

        # Check for search checkpoint
        checkpoint_path = project_path / "arhmm" / "search_checkpoint.json"
        if checkpoint_path.exists():
            try:
                ckpt = json.loads(checkpoint_path.read_text())
                log_low = ckpt["log_low"]
                log_high = ckpt["log_high"]
                best_kappa = ckpt.get("best_kappa")
                best_distance = ckpt.get("best_distance", float("inf"))
                start_step = ckpt["next_step"]
                self._progress.search_history = ckpt.get("search_history", [])
                logger.info(
                    f"[ARHMM] Resuming search from step {start_step}, "
                    f"bounds=[{log_low:.2f}, {log_high:.2f}]"
                )

                # Check if search was already completed
                if ckpt.get("search_complete"):
                    logger.info(f"[ARHMM] Search already complete, kappa={best_kappa:.2e}")
                    self._progress.chosen_kappa = best_kappa or 0.0
                    return best_kappa
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"[ARHMM] Invalid search checkpoint, starting fresh: {e}")
                start_step = 0

        self._progress.log_low = log_low
        self._progress.log_high = log_high

        search_complete = False

        for step in range(start_step, max_search_steps):
            if self._stop_requested:
                break

            log_mid = (log_low + log_high) / 2
            kappa = 10**log_mid

            self._progress.current_kappa = kappa
            self._progress.current_iteration = 0
            self._progress.total_iterations = search_iterations
            self._reset_timing()

            logger.info(
                f"[ARHMM] Search step {step + 1}: "
                f"kappa={kappa:.2e} (log={log_mid:.2f}), "
                f"bounds=[{log_low:.2f}, {log_high:.2f}]"
            )

            # Run short ARHMM fit
            model, _keys, _bounds = self._run_arhmm_model(
                sessions_dict, kappa, latent_dim, search_iterations
            )

            if self._stop_requested:
                break

            # Compute median syllable duration
            durations = get_durations(model["states"])
            if len(durations) == 0:
                median_ms = 0.0
            else:
                median_frames = float(np.median(durations))
                median_ms = median_frames / fps * 1000

            # Classify result
            if median_ms < 300:
                result = "too_low"
                log_low = log_mid
            elif median_ms > 400:
                result = "too_high"
                log_high = log_mid
            else:
                result = "found"

            logger.info(
                f"[ARHMM] Search step {step + 1}: "
                f"median_duration={median_ms:.1f}ms -> {result}"
            )

            # Track search history
            self._progress.search_history.append(
                {
                    "step": step + 1,
                    "kappa": kappa,
                    "log_kappa": round(log_mid, 2),
                    "median_duration_ms": round(median_ms, 1),
                    "result": result,
                }
            )
            self._progress.log_low = log_low
            self._progress.log_high = log_high

            # Track best kappa (closest to 350ms target)
            distance = abs(median_ms - 350)
            if distance < best_distance:
                best_distance = distance
                best_kappa = kappa

            found = result == "found"
            converged = log_high - log_low < 0.1

            if found or converged:
                search_complete = True
                if converged and not found:
                    logger.info(
                        f"[ARHMM] Search converged (range {log_high - log_low:.3f}), "
                        f"using best kappa={best_kappa:.2e}"
                    )

            # Atomically save search checkpoint AFTER step is fully complete
            _atomic_json_write(
                checkpoint_path,
                {
                    "log_low": log_low,
                    "log_high": log_high,
                    "best_kappa": best_kappa,
                    "best_distance": best_distance,
                    "next_step": step + 1,
                    "search_history": self._progress.search_history,
                    "search_complete": search_complete,
                },
            )

            if search_complete:
                break

        return best_kappa

    # ── Short ARHMM fit (for search, no checkpointing) ────────────────

    def _run_arhmm_model(
        self,
        sessions_dict: dict[str, np.ndarray],
        kappa: float,
        latent_dim: int,
        num_iterations: int,
    ) -> tuple[dict, np.ndarray, np.ndarray]:
        """Run ARHMM pipeline: batch, init, fit, unbatch.

        Used for search iterations (short, no checkpointing).
        """
        data, mask, (keys, bounds) = batch(sessions_dict)
        data = jax.device_put(jnp.array(data, dtype=jnp.float64))
        mask = jax.device_put(jnp.array(mask, dtype=jnp.float64))
        batched_data = {"x": data, "mask": mask}

        num_states = 100
        nlags = 3
        trans_hypparams = {
            "gamma": 1e3,
            "alpha": 5.7,
            "kappa": kappa,
            "num_states": num_states,
        }
        ar_hypparams = {
            "S_0_scale": 0.01,
            "K_0_scale": 10,
            "num_states": num_states,
            "nlags": nlags,
            "latent_dim": latent_dim,
        }

        model = arhmm.init_model(
            data=batched_data,
            ar_hypparams=ar_hypparams,
            trans_hypparams=trans_hypparams,
            seed=jr.PRNGKey(42),
        )

        for i in range(num_iterations):
            if self._stop_requested:
                break
            model = arhmm.resample_model(batched_data, **model)
            self._progress.current_iteration = i + 1
            self._update_timing()

        states_dict = unbatch(model["states"]["z"], keys, bounds)
        model["states"] = states_dict
        return model, keys, bounds

    # ── Final fit with checkpointing ──────────────────────────────────

    def _run_final_fit(
        self,
        sessions_dict: dict[str, np.ndarray],
        kappa: float,
        latent_dim: int,
        num_iterations: int,
        project_path: Path,
    ) -> dict:
        """Run the final ARHMM fit with periodic checkpointing.

        Resumes from fit_checkpoint if it exists and matches the current kappa.
        Returns model dict with unbatched states.
        """
        data, mask, (keys, bounds) = batch(sessions_dict)
        data = jax.device_put(jnp.array(data, dtype=jnp.float64))
        mask = jax.device_put(jnp.array(mask, dtype=jnp.float64))
        batched_data = {"x": data, "mask": mask}

        num_states = 100
        nlags = 3
        trans_hypparams = {
            "gamma": 1e3,
            "alpha": 5.7,
            "kappa": kappa,
            "num_states": num_states,
        }
        ar_hypparams = {
            "S_0_scale": 0.01,
            "K_0_scale": 10,
            "num_states": num_states,
            "nlags": nlags,
            "latent_dim": latent_dim,
        }

        # Check for fit checkpoint
        start_iteration = 0
        model = None
        meta_path = project_path / "arhmm" / "fit_checkpoint_meta.json"
        model_path = project_path / "arhmm" / "fit_checkpoint.joblib"

        if meta_path.exists() and model_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                if meta.get("kappa") == kappa and meta.get("latent_dim") == latent_dim:
                    model = joblib.load(model_path)
                    start_iteration = meta["iteration"]
                    self._progress.current_iteration = start_iteration
                    logger.info(
                        f"[ARHMM] Resuming final fit from iteration {start_iteration}/{num_iterations}"
                    )
                else:
                    logger.info("[ARHMM] Fit checkpoint kappa mismatch, starting fresh")
            except Exception as e:
                logger.warning(f"[ARHMM] Invalid fit checkpoint, starting fresh: {e}")

        if model is None:
            model = arhmm.init_model(
                data=batched_data,
                ar_hypparams=ar_hypparams,
                trans_hypparams=trans_hypparams,
                seed=jr.PRNGKey(42),
            )

        # Gibbs sampling with checkpointing
        for i in range(start_iteration, num_iterations):
            if self._stop_requested:
                break
            model = arhmm.resample_model(batched_data, **model)
            self._progress.current_iteration = i + 1
            self._update_timing()

            # Save checkpoint every N iterations (atomically, after resample completes)
            if (i + 1) % FIT_CHECKPOINT_FREQUENCY == 0:
                self._save_fit_checkpoint(project_path, model, i + 1, kappa, latent_dim)

        # Unbatch states
        states_dict = unbatch(model["states"]["z"], keys, bounds)
        model["states"] = states_dict
        return model

    def _save_fit_checkpoint(
        self,
        project_path: Path,
        model: dict,
        iteration: int,
        kappa: float,
        latent_dim: int,
    ) -> None:
        """Save a fit checkpoint atomically."""
        results_dir = project_path / "arhmm"

        # Save model state first, then metadata
        # If crash between the two, metadata won't exist → won't resume from stale model
        _atomic_joblib_write(results_dir / "fit_checkpoint.joblib", model)
        _atomic_json_write(
            results_dir / "fit_checkpoint_meta.json",
            {"iteration": iteration, "kappa": kappa, "latent_dim": latent_dim},
        )
        logger.info(f"[ARHMM] Saved fit checkpoint at iteration {iteration}")

    # ── Results saving and cleanup ─────────────────────────────────────

    def _save_results(self, project_path: Path, model: dict, latent_dim: int) -> None:
        """Save fitted model and summary to disk atomically."""
        results_dir = project_path / "arhmm"
        results_dir.mkdir(exist_ok=True)

        _atomic_joblib_write(results_dir / "model.joblib", model)
        logger.info(f"[ARHMM] Saved model to {results_dir / 'model.joblib'}")

        summary = {
            "kappa": self._progress.chosen_kappa,
            "median_duration_ms": self._progress.final_median_duration_ms,
            "num_videos": self._progress.num_videos,
            "total_frames": self._progress.total_frames,
            "fps": self._progress.fps,
            "search_history": self._progress.search_history,
            "n_components": latent_dim,
        }
        _atomic_json_write(results_dir / "summary.json", summary)
        logger.info(f"[ARHMM] Saved summary to {results_dir / 'summary.json'}")

    @staticmethod
    def _cleanup_checkpoints(project_path: Path) -> None:
        """Remove checkpoint files after successful completion."""
        arhmm_dir = project_path / "arhmm"
        for name in [
            "search_checkpoint.json",
            "fit_checkpoint.joblib",
            "fit_checkpoint_meta.json",
        ]:
            p = arhmm_dir / name
            if p.exists():
                p.unlink()
        # Also clean up any leftover tmp files
        for tmp in arhmm_dir.glob("*.tmp"):
            tmp.unlink(missing_ok=True)
        logger.info("[ARHMM] Cleaned up checkpoint files")

    @staticmethod
    def get_results(project_path: Path) -> dict | None:
        """Load saved ARHMM results summary from disk, or None if not found."""
        summary_path = project_path / "arhmm" / "summary.json"
        if summary_path.exists():
            return json.loads(summary_path.read_text())
        return None
