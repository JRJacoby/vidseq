"""Compute per-frame jitter metrics for masks and plot time traces.

Metrics:
  1. Frame-to-frame mask IoU
  2. Centroid displacement (Euclidean distance between consecutive frame centroids)
  3. Delta area (absolute change in mask pixel count)

Outputs three figures to an output directory, each with 5 panels (one per video).

Usage:
    uv run python scripts/mask_jitter_analysis.py                  # cropped (default)
    uv run python scripts/mask_jitter_analysis.py --mask-type aligned
"""

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import sqlite3


PROJECT_DIR = Path("test_projects/Test")
OUTPUT_DIR = Path("scripts/jitter_figures")


def load_video_names() -> dict[int, str]:
    conn = sqlite3.connect(PROJECT_DIR / "vidseq.db")
    rows = conn.execute("SELECT id, name FROM videos ORDER BY id").fetchall()
    conn.close()
    return {vid: name.replace(".top.ir.mp4", "") for vid, name in rows}


MASK_FILES = {
    "cropped": "cropped_masks.h5",
    "aligned": "aligned_masks.h5",
}


def load_masks(video_id: int, mask_type: str) -> np.ndarray:
    path = PROJECT_DIR / "array_data" / str(video_id) / MASK_FILES[mask_type]
    with h5py.File(path, "r") as f:
        return f["data"][:]  # (T, H, W) uint8


def compute_metrics(masks: np.ndarray) -> dict[str, np.ndarray]:
    T = masks.shape[0]
    binary = masks > 127  # (T, H, W) bool

    # --- Areas ---
    areas = binary.sum(axis=(1, 2)).astype(np.float64)  # (T,)

    # --- Centroids ---
    # weighted centroid via meshgrid
    H, W = masks.shape[1], masks.shape[2]
    ys, xs = np.mgrid[0:H, 0:W]  # (H, W) each
    centroids = np.zeros((T, 2), dtype=np.float64)
    for t in range(T):
        m = binary[t]
        a = areas[t]
        if a > 0:
            centroids[t, 0] = (xs * m).sum() / a  # cx
            centroids[t, 1] = (ys * m).sum() / a  # cy
        else:
            centroids[t] = np.nan

    # --- Frame-to-frame IoU (T-1 values) ---
    intersection = np.logical_and(binary[:-1], binary[1:]).sum(axis=(1, 2)).astype(np.float64)
    union = np.logical_or(binary[:-1], binary[1:]).sum(axis=(1, 2)).astype(np.float64)
    iou = np.where(union > 0, intersection / union, 1.0)  # both empty = agreement

    # --- Centroid displacement (T-1 values) ---
    centroid_disp = np.sqrt(np.sum(np.diff(centroids, axis=0) ** 2, axis=1))

    # --- Delta area (T-1 values) ---
    delta_area = np.abs(np.diff(areas))

    return {
        "iou": iou,
        "centroid_disp": centroid_disp,
        "delta_area": delta_area,
    }


def plot_metric(
    all_metrics: dict[int, dict[str, np.ndarray]],
    video_names: dict[int, str],
    key: str,
    ylabel: str,
    title: str,
    output_path: Path,
):
    video_ids = sorted(all_metrics.keys())
    fig, axes = plt.subplots(5, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(title, fontsize=14)

    for ax, vid in zip(axes, video_ids):
        data = all_metrics[vid][key]
        frames = np.arange(len(data))
        ax.plot(frames, data, linewidth=0.5, color="steelblue")
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(video_names[vid], fontsize=9, loc="left")
        ax.tick_params(labelsize=7)

        # Add median line
        med = np.nanmedian(data)
        ax.axhline(med, color="tomato", linewidth=0.8, linestyle="--", alpha=0.7, label=f"median={med:.3f}")
        ax.legend(fontsize=7, loc="lower right")

    axes[-1].set_xlabel("Frame pair (t → t+1)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Mask jitter analysis")
    parser.add_argument("--mask-type", choices=list(MASK_FILES), default="cropped")
    args = parser.parse_args()
    mask_type: str = args.mask_type
    label = mask_type.capitalize()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    video_names = load_video_names()
    video_ids = sorted(video_names.keys())

    all_metrics: dict[int, dict[str, np.ndarray]] = {}
    for vid in video_ids:
        print(f"Loading video {vid}: {video_names[vid]}")
        masks = load_masks(vid, mask_type)
        all_metrics[vid] = compute_metrics(masks)

    # Print summary statistics
    print(f"\n--- Summary ({label} Masks) ---")
    for vid in video_ids:
        m = all_metrics[vid]
        print(f"Video {vid} ({video_names[vid]}):")
        print(f"  IoU:       median={np.nanmedian(m['iou']):.4f}  min={np.nanmin(m['iou']):.4f}  std={np.nanstd(m['iou']):.4f}")
        print(f"  Centroid:  median={np.nanmedian(m['centroid_disp']):.2f}px  max={np.nanmax(m['centroid_disp']):.2f}px  std={np.nanstd(m['centroid_disp']):.2f}px")
        print(f"  Δ area:    median={np.nanmedian(m['delta_area']):.0f}px  max={np.nanmax(m['delta_area']):.0f}px  std={np.nanstd(m['delta_area']):.0f}px")

    # Plot figures
    print("\nGenerating figures...")
    prefix = mask_type
    plot_metric(all_metrics, video_names, "iou", "IoU", f"Frame-to-Frame Mask IoU ({label} Masks)", OUTPUT_DIR / f"{prefix}_iou.png")
    plot_metric(all_metrics, video_names, "centroid_disp", "px", f"Frame-to-Frame Centroid Displacement ({label} Masks)", OUTPUT_DIR / f"{prefix}_centroid_disp.png")
    plot_metric(all_metrics, video_names, "delta_area", "px²", f"Frame-to-Frame Δ Area ({label} Masks)", OUTPUT_DIR / f"{prefix}_delta_area.png")

    print(f"\nDone. Figures saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
