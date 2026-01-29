#!/usr/bin/env python3
"""
Test SAM2++ point tracking on mouse video.

Usage:
    python test_point_tracking.py --point 320,288
    python test_point_tracking.py --point 320,288 --point 400,300  # Multiple points

The point coordinates should be in (x, y) format, matching the pixel coordinates
from frame_0_with_grid.png.
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

# Add SAM2++ repo to path
SAM2_PLUS_REPO = Path(__file__).parent.parent.parent / "sam2_plus_repo"
sys.path.insert(0, str(SAM2_PLUS_REPO))

# SAM2++ imports
from sam2_plus.build_sam import build_sam2_video_predictor_plus


def extract_frames_to_dir(video_path: Path, output_dir: Path) -> list[str]:
    """Extract video frames as JPEGs (SAM2++ expects image directory)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    frame_names = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_name = f"{frame_idx:05d}.jpg"
        cv2.imwrite(str(output_dir / frame_name), frame)
        frame_names.append(frame_name.replace(".jpg", ""))
        frame_idx += 1

    cap.release()
    print(f"Extracted {frame_idx} frames to {output_dir}")
    return frame_names


def run_point_tracking(
    video_path: Path,
    points: list[tuple[float, float]],
    output_dir: Path,
    radius: int = 5,
    sigma: int = 2,
):
    """
    Run SAM2++ point tracking on a video.

    Args:
        video_path: Path to input video
        points: List of (x, y) coordinates to track
        output_dir: Where to save results
        radius: Gaussian radius for point prompt
        sigma: Gaussian sigma for point prompt
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract frames to temp directory (SAM2++ requires image directory)
    frames_dir = output_dir / "frames"
    frame_names = extract_frames_to_dir(video_path, frames_dir)
    num_frames = len(frame_names)

    print(f"\nLoading SAM2++ model...")
    # Change to SAM2++ repo directory (hydra expects relative config paths)
    import os
    original_cwd = os.getcwd()
    os.chdir(SAM2_PLUS_REPO)

    predictor = build_sam2_video_predictor_plus(
        config_file="configs/sam2.1/sam2.1_hiera_b+_predmasks_decoupled_MAME.yaml",
        ckpt_path="./checkpoints/SAM2-Plus/checkpoint_phase123.pt",
        apply_postprocessing=False,
        hydra_overrides_extra=["++model.non_overlap_masks=false"],
        vos_optimized=False,
        task='point'
    )

    os.chdir(original_cwd)

    print(f"Initializing inference state...")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        inference_state = predictor.init_state(video_path=str(frames_dir))
        height = inference_state["video_height"]
        width = inference_state["video_width"]
        print(f"Video size: {width}x{height}, {num_frames} frames")

        # Add each point as a separate object
        for obj_id, (x, y) in enumerate(points):
            print(f"Adding point {obj_id}: ({x}, {y})")
            predictor.add_new_points_and_generate_gaussian_mask(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=obj_id,
                points=np.array([[x, y]], dtype=np.float32),
                labels=np.array([1]),
                radius=radius,
                sigma=sigma,
            )

        # Track through video
        print(f"\nTracking {len(points)} point(s) through {num_frames} frames...")
        num_points = len(points)
        point_array = np.full((num_frames, num_points, 2), -1, dtype=np.float32)
        visible_array = np.zeros((num_frames, num_points), dtype=bool)
        score_thresh = 0

        for out_frame_idx, out_obj_ids, out_mask_logits, out_box_xyxys, out_obj_score_logits in tqdm(
            predictor.propagate_in_video(inference_state),
            total=num_frames,
            desc="Propagating"
        ):
            for out_obj_id, out_mask_logit, out_obj_score_logit in zip(
                out_obj_ids, out_mask_logits, out_obj_score_logits
            ):
                out_mask_logit = out_mask_logit.squeeze(0)
                out_obj_score_logit = out_obj_score_logit.squeeze(0)

                # Extract point as argmax of mask logits
                max_index = torch.argmax(out_mask_logit)
                max_y, max_x = torch.unravel_index(max_index, out_mask_logit.shape)

                point_array[out_frame_idx, out_obj_id] = np.array([max_x.cpu(), max_y.cpu()])
                visible_array[out_frame_idx, out_obj_id] = (out_obj_score_logit > score_thresh).cpu().numpy()

    # Save tracking results
    results_path = output_dir / "tracking_results.npz"
    np.savez(
        results_path,
        trajs_2d=point_array,
        visibs=visible_array,
        size=(width, height),
        initial_points=np.array(points),
    )
    print(f"\nSaved tracking results to: {results_path}")

    # Visualize results
    print(f"\nGenerating visualization...")
    vis_dir = output_dir / "visualization"
    vis_dir.mkdir(exist_ok=True)

    colors = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue (BGR)
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
    ]

    for frame_idx in tqdm(range(num_frames), desc="Visualizing"):
        frame = cv2.imread(str(frames_dir / f"{frame_idx:05d}.jpg"))

        for pt_idx in range(num_points):
            x, y = point_array[frame_idx, pt_idx]
            visible = visible_array[frame_idx, pt_idx]
            init_x, init_y = points[pt_idx]

            # Draw initial prompt location (small hollow circle, always visible)
            cv2.circle(frame, (int(init_x), int(init_y)), 8, (255, 255, 255), 1)
            cv2.circle(frame, (int(init_x), int(init_y)), 2, (255, 255, 255), -1)

            if x >= 0 and y >= 0:
                color = colors[pt_idx % len(colors)]
                thickness = 2 if visible else 1

                # Draw tracked point (filled crosshair)
                x, y = int(x), int(y)
                cv2.circle(frame, (x, y), 5, color, thickness)
                cv2.line(frame, (x - 10, y), (x + 10, y), color, thickness)
                cv2.line(frame, (x, y - 10), (x, y + 10), color, thickness)

                # Draw line from initial to tracked (shows drift)
                cv2.line(frame, (int(init_x), int(init_y)), (x, y), (128, 128, 128), 1)

                # Label
                cv2.putText(frame, f"P{pt_idx}", (x + 8, y - 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        cv2.imwrite(str(vis_dir / f"{frame_idx:05d}.png"), frame)

    # Create video from visualization
    print(f"Creating output video...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(
        str(output_dir / "tracking_result.mp4"),
        fourcc, 30.0, (width, height)
    )

    for frame_idx in range(num_frames):
        frame = cv2.imread(str(vis_dir / f"{frame_idx:05d}.png"))
        out_video.write(frame)
    out_video.release()

    print(f"\nResults saved to: {output_dir}")
    print(f"  - tracking_results.npz: Raw coordinates")
    print(f"  - tracking_result.mp4: Visualization video")
    print(f"  - visualization/: Individual frames")

    # Print tracking summary
    print(f"\nTracking Summary:")
    for pt_idx in range(num_points):
        visible_count = visible_array[:, pt_idx].sum()
        print(f"  Point {pt_idx}: visible in {visible_count}/{num_frames} frames ({100*visible_count/num_frames:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Test SAM2++ point tracking")
    parser.add_argument(
        "--video",
        type=str,
        default="/n/groups/datta/john/projects/vidseq/test_videos/open_field_2D_10sec/21_11_8_one_mouse.top.ir.mp4",
        help="Path to input video"
    )
    parser.add_argument(
        "--point",
        type=str,
        action="append",
        required=True,
        help="Point to track as 'x,y'. Can specify multiple times."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/n/groups/datta/john/projects/vidseq/experiments/sam2_plus/output",
        help="Output directory"
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=5,
        help="Gaussian radius for point prompt (default: 5)"
    )
    parser.add_argument(
        "--sigma",
        type=int,
        default=2,
        help="Gaussian sigma for point prompt (default: 2)"
    )

    args = parser.parse_args()

    # Parse points
    points = []
    for pt_str in args.point:
        x, y = map(float, pt_str.split(","))
        points.append((x, y))

    print(f"Video: {args.video}")
    print(f"Points to track: {points}")
    print(f"Output: {args.output}")

    run_point_tracking(
        video_path=Path(args.video),
        points=points,
        output_dir=Path(args.output),
        radius=args.radius,
        sigma=args.sigma,
    )


if __name__ == "__main__":
    main()
