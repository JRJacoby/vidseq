"""Crowd movie generation for ARHMM syllables.

Generates one MP4 video per syllable showing up to 50 instances of that
behavior overlaid as semi-transparent blue masks on a black background,
time-aligned to syllable onset.
"""

import json
import logging
import random
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import h5py
import imageio_ffmpeg
import joblib
import numpy as np

logger = logging.getLogger(__name__)

# --- Constants ---
TIME_WINDOW_START = -30  # frames before syllable onset
TIME_WINDOW_END = 60  # frames after syllable onset
CLIP_LENGTH = TIME_WINDOW_END - TIME_WINDOW_START  # 90 frames
MAX_INSTANCES = 50
MIN_INSTANCES = 5
OPACITY = 0.2
BLUE_COLOR = np.array([70, 130, 255], dtype=np.float32)  # RGB


@dataclass
class CrowdMovieProgress:
    """Progress state for crowd movie generation."""

    is_running: bool = False
    status: str = "idle"  # idle, generating, completed, failed
    total_syllables: int = 0
    completed_syllables: int = 0
    skipped_syllables: int = 0
    current_syllable: int = -1
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class CrowdMovieService:
    """Singleton service for generating crowd movies from ARHMM results."""

    _instance = None

    @classmethod
    def get_instance(cls) -> "CrowdMovieService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self._progress = CrowdMovieProgress()
        self._is_running = False
        self._stop_requested = False

    def is_running(self) -> bool:
        return self._is_running

    def get_progress(self) -> CrowdMovieProgress:
        return self._progress

    def stop(self):
        if self._is_running:
            self._stop_requested = True
            logger.info("[CrowdMovie] Stop requested")

    # ── Main entry point ──────────────────────────────────────────────

    def generate_crowd_movies_sync(self, project_path: Path, fps: float) -> None:
        """Generate crowd movies for all qualifying syllables.

        Called from a background thread. Resumable: skips syllables
        that already have a completed .mp4 on disk.
        """
        try:
            self._is_running = True
            self._stop_requested = False
            self._progress = CrowdMovieProgress(
                is_running=True,
                status="generating",
            )

            # Load ARHMM states
            model_path = project_path / "arhmm" / "model.joblib"
            if not model_path.exists():
                raise FileNotFoundError("ARHMM model not found")

            model = joblib.load(model_path)
            states_dict = model["states"]

            # Get video dimensions from first mask HDF5
            video_height, video_width = self._get_video_dimensions(
                project_path, states_dict
            )

            # Build syllable frequency table
            syllable_counts = self._count_syllable_runs(states_dict)
            # Filter and sort by frequency
            qualifying = [
                (syl, count)
                for syl, count in syllable_counts.items()
                if count >= MIN_INSTANCES
            ]
            qualifying.sort(key=lambda x: x[1], reverse=True)

            if not qualifying:
                logger.info("[CrowdMovie] No syllables with enough instances")
                self._progress.status = "completed"
                return

            output_dir = project_path / "arhmm" / "crowd_movies"
            output_dir.mkdir(parents=True, exist_ok=True)

            self._progress.total_syllables = len(qualifying)
            manifest_syllables = []

            logger.info(
                f"[CrowdMovie] Generating crowd movies for {len(qualifying)} syllables "
                f"(canvas {video_width}x{video_height}, fps={fps})"
            )

            # Get num_frames per video for boundary checks
            video_num_frames = self._get_video_num_frames(project_path, states_dict)

            for syllable, total_count in qualifying:
                if self._stop_requested:
                    break

                self._progress.current_syllable = int(syllable)
                final_path = output_dir / f"syllable_{syllable}.mp4"

                # Resumability: skip if already exists
                if final_path.exists():
                    self._progress.skipped_syllables += 1
                    self._progress.completed_syllables += 1
                    # Still include in manifest
                    manifest_syllables.append(
                        {
                            "syllable": int(syllable),
                            "instance_count": total_count,
                            "sampled": min(total_count, MAX_INSTANCES),
                        }
                    )
                    logger.info(
                        f"[CrowdMovie] Skipping syllable {syllable} (already exists)"
                    )
                    continue

                # Generate crowd movie for this syllable
                sampled = self._generate_single_movie(
                    project_path,
                    states_dict,
                    syllable,
                    fps,
                    video_height,
                    video_width,
                    video_num_frames,
                    final_path,
                )

                if sampled is not None:
                    manifest_syllables.append(
                        {
                            "syllable": int(syllable),
                            "instance_count": total_count,
                            "sampled": sampled,
                        }
                    )

                self._progress.completed_syllables += 1

            # Save manifest
            manifest = {
                "fps": fps,
                "video_width": video_width,
                "video_height": video_height,
                "time_window": [TIME_WINDOW_START, TIME_WINDOW_END],
                "max_instances": MAX_INSTANCES,
                "opacity": OPACITY,
                "syllables": manifest_syllables,
            }
            manifest_path = output_dir / "manifest.json"
            tmp_manifest = manifest_path.with_name(manifest_path.name + ".tmp")
            tmp_manifest.write_text(json.dumps(manifest, indent=2))
            tmp_manifest.rename(manifest_path)

            if self._stop_requested:
                self._progress.status = "failed"
                self._progress.error = "Stopped by user"
            else:
                self._progress.status = "completed"
                logger.info(
                    f"[CrowdMovie] Done. Generated {self._progress.completed_syllables - self._progress.skipped_syllables} new, "
                    f"skipped {self._progress.skipped_syllables}"
                )

        except Exception as e:
            logger.exception(f"[CrowdMovie] Failed: {e}")
            self._progress.status = "failed"
            self._progress.error = str(e)
        finally:
            self._progress.is_running = False
            self._is_running = False
            self._stop_requested = False

    # ── Single movie generation ───────────────────────────────────────

    def _generate_single_movie(
        self,
        project_path: Path,
        states_dict: dict,
        syllable: int,
        fps: float,
        video_height: int,
        video_width: int,
        video_num_frames: dict[str, int],
        final_path: Path,
    ) -> int | None:
        """Generate a crowd movie for one syllable. Returns sampled count or None on failure."""
        # Find all instances
        instances = _find_syllable_instances(states_dict, syllable)

        # Expand time window and filter invalid
        expanded = []
        for video_id, onset in instances:
            start = onset + TIME_WINDOW_START
            end = onset + TIME_WINDOW_END
            num_frames = video_num_frames.get(video_id, 0)
            if start >= 0 and end <= num_frames:
                expanded.append((video_id, start))

        if len(expanded) < MIN_INSTANCES:
            logger.info(
                f"[CrowdMovie] Syllable {syllable}: only {len(expanded)} valid instances, skipping"
            )
            return None

        # Sample
        if len(expanded) > MAX_INSTANCES:
            sampled = random.sample(expanded, MAX_INSTANCES)
        else:
            sampled = expanded

        logger.info(
            f"[CrowdMovie] Syllable {syllable}: {len(sampled)} instances "
            f"(from {len(expanded)} valid / {len(instances)} total)"
        )

        # Open mask HDF5 files for involved videos
        involved_video_ids = set(vid for vid, _ in sampled)
        h5_files: dict[str, h5py.File] = {}

        try:
            for vid in involved_video_ids:
                h5_path = project_path / "masks" / f"{vid}.h5"
                if h5_path.exists():
                    h5_files[vid] = h5py.File(h5_path, "r")
                else:
                    logger.warning(f"[CrowdMovie] Mask file not found: {h5_path}")

            # Write video
            self._write_crowd_video(
                sampled, h5_files, fps, video_height, video_width, final_path
            )

        finally:
            for f in h5_files.values():
                f.close()

        return len(sampled)

    def _write_crowd_video(
        self,
        instances: list[tuple[str, int]],
        h5_files: dict[str, h5py.File],
        fps: float,
        height: int,
        width: int,
        final_path: Path,
    ) -> None:
        """Composite and write the crowd movie video."""
        output_dir = final_path.parent
        raw_tmp = output_dir / f"{final_path.stem}_raw.tmp.mp4"
        h264_tmp = output_dir / f"{final_path.stem}_h264.tmp.mp4"

        # Ensure dimensions are even (required by H.264)
        write_height = height if height % 2 == 0 else height + 1
        write_width = width if width % 2 == 0 else width + 1

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(raw_tmp), fourcc, fps, (write_width, write_height)
        )

        try:
            blue_bgr = BLUE_COLOR[::-1]  # RGB → BGR for OpenCV

            for frame_t in range(CLIP_LENGTH):
                if self._stop_requested:
                    break

                canvas = np.zeros((height, width, 3), dtype=np.float32)

                for video_id, clip_start in instances:
                    frame_idx = clip_start + frame_t
                    h5 = h5_files.get(video_id)
                    if h5 is None or "masks" not in h5:
                        continue

                    mask = h5["masks"][frame_idx]  # (H, W) uint8
                    mask_bool = mask > 0
                    canvas[mask_bool] += blue_bgr * OPACITY

                canvas = np.clip(canvas, 0, 255).astype(np.uint8)

                # Pad to even dimensions if needed
                if write_height != height or write_width != width:
                    padded = np.zeros(
                        (write_height, write_width, 3), dtype=np.uint8
                    )
                    padded[:height, :width] = canvas
                    canvas = padded

                writer.write(canvas)

            writer.release()

            if self._stop_requested:
                raw_tmp.unlink(missing_ok=True)
                return

            # Re-encode to H.264 for browser
            success = _reencode_to_h264(raw_tmp, h264_tmp)
            if not success:
                raise RuntimeError(f"FFmpeg re-encode failed for syllable {final_path.stem}")

            # Atomic rename
            h264_tmp.rename(final_path)

        finally:
            # Clean up temp files
            raw_tmp.unlink(missing_ok=True)
            if h264_tmp.exists():
                h264_tmp.unlink(missing_ok=True)

    # ── Helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _get_video_dimensions(
        project_path: Path, states_dict: dict
    ) -> tuple[int, int]:
        """Get video (height, width) from the first mask HDF5 file."""
        for video_id in states_dict:
            h5_path = project_path / "masks" / f"{video_id}.h5"
            if h5_path.exists():
                with h5py.File(h5_path, "r") as f:
                    if "masks" in f:
                        _, h, w = f["masks"].shape
                        return int(h), int(w)
        raise ValueError("No mask HDF5 files found to determine video dimensions")

    @staticmethod
    def _get_video_num_frames(
        project_path: Path, states_dict: dict
    ) -> dict[str, int]:
        """Get num_frames for each video from mask HDF5 files."""
        result = {}
        for video_id in states_dict:
            h5_path = project_path / "masks" / f"{video_id}.h5"
            if h5_path.exists():
                with h5py.File(h5_path, "r") as f:
                    if "masks" in f:
                        result[video_id] = f["masks"].shape[0]
        return result

    @staticmethod
    def _count_syllable_runs(states_dict: dict) -> dict[int, int]:
        """Count total number of runs per syllable across all videos."""
        counts: dict[int, int] = {}
        for states in states_dict.values():
            states = np.asarray(states).ravel()
            if len(states) == 0:
                continue
            # Count run onsets for each syllable
            current = states[0]
            counts[int(current)] = counts.get(int(current), 0) + 1
            for i in range(1, len(states)):
                if states[i] != states[i - 1]:
                    syl = int(states[i])
                    counts[syl] = counts.get(syl, 0) + 1
        return counts

    @staticmethod
    def get_available_movies(project_path: Path) -> list[dict]:
        """List available crowd movie files with metadata from manifest."""
        movies_dir = project_path / "arhmm" / "crowd_movies"
        if not movies_dir.exists():
            return []

        # Load manifest for instance counts
        manifest_data = {}
        manifest_path = movies_dir / "manifest.json"
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text())
                for entry in manifest.get("syllables", []):
                    manifest_data[entry["syllable"]] = entry
            except (json.JSONDecodeError, KeyError):
                pass

        # Scan for mp4 files
        result = []
        for mp4_path in sorted(movies_dir.glob("syllable_*.mp4")):
            stem = mp4_path.stem  # e.g., "syllable_3"
            try:
                syllable = int(stem.split("_")[1])
            except (IndexError, ValueError):
                continue

            entry = {"syllable": syllable}
            if syllable in manifest_data:
                entry["instance_count"] = manifest_data[syllable].get(
                    "instance_count", 0
                )
                entry["sampled"] = manifest_data[syllable].get("sampled", 0)
            else:
                entry["instance_count"] = 0
                entry["sampled"] = 0

            result.append(entry)

        # Sort by instance count descending (most frequent first)
        result.sort(key=lambda x: x["instance_count"], reverse=True)
        return result


# ── Module-level helpers ──────────────────────────────────────────────


def _find_syllable_instances(
    states_dict: dict, syllable: int
) -> list[tuple[str, int]]:
    """Find all onset frames for a syllable across all videos.

    Returns list of (video_id_str, onset_frame) tuples.
    """
    instances = []
    for video_id, states in states_dict.items():
        states = np.asarray(states).ravel()
        if len(states) == 0:
            continue
        is_syllable = states == syllable
        diffs = np.diff(is_syllable.astype(np.int8))
        onsets = np.where(diffs == 1)[0] + 1
        if is_syllable[0]:
            onsets = np.concatenate([[0], onsets])
        for onset in onsets:
            instances.append((video_id, int(onset)))
    return instances


def _reencode_to_h264(input_path: Path, output_path: Path) -> bool:
    """Re-encode video to H.264 for browser compatibility."""
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_path,
        "-y",
        "-i", str(input_path),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(output_path),
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"[CrowdMovie] FFmpeg error: {e.stderr}")
        return False
