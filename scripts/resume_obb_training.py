"""Resume OBB training from best checkpoint until early stopping.

Re-exports the training dataset (since the original /tmp dir is gone),
then trains from the best.pt checkpoint with epochs=1000 and patience=20.

Usage: uv run python scripts/resume_obb_training.py /path/to/project [video_ids...]

Example:
    uv run python scripts/resume_obb_training.py /path/to/project 1 2 18 41
"""
import shutil
import sys
from pathlib import Path

from ultralytics import YOLO

# Add the project root to sys.path so we can import vidseq services
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} /path/to/project video_id1 video_id2 ...")
        sys.exit(1)

    project_path = Path(sys.argv[1])
    video_ids = [int(v) for v in sys.argv[2:]]
    model_dir = project_path / "models"
    best_pt = model_dir / "obb_train" / "weights" / "best.pt"

    if not best_pt.exists():
        print(f"Error: No checkpoint found at {best_pt}")
        sys.exit(1)

    # Re-export the training dataset
    from vidseq.services.detector_service import DetectorService

    service = DetectorService.get_instance()
    print(f"Gathering training frames for videos {video_ids}...")
    all_frames = service._gather_training_frames(project_path, video_ids)
    print(f"Found {len(all_frames)} training frames")

    if not all_frames:
        print("Error: No training frames found")
        sys.exit(1)

    # 80/20 split
    import random
    random.seed(42)
    random.shuffle(all_frames)
    split = int(len(all_frames) * 0.8)
    train_frames = all_frames[:split]
    val_frames = all_frames[split:]

    import tempfile
    tmp_dir = Path(tempfile.mkdtemp(prefix="vidseq_obb_resume_"))
    print(f"Writing dataset to {tmp_dir}")

    # Create directory structure expected by _write_yolo_dataset
    for split in ("train", "val"):
        (tmp_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (tmp_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    service._write_yolo_dataset(train_frames, "train", tmp_dir, project_path, "obb")
    service._write_yolo_dataset(val_frames, "val", tmp_dir, project_path, "obb")

    # Write dataset YAML
    yaml_path = tmp_dir / "dataset.yaml"
    yaml_path.write_text(
        f"path: {tmp_dir}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n"
        f"  0: animal\n"
    )

    # Load from best checkpoint and train
    print(f"Loading model from {best_pt}")
    model = YOLO(str(best_pt))

    print("Starting training (epochs=1000, patience=20)...")
    model.train(
        data=str(yaml_path),
        epochs=1000,
        imgsz=640,
        batch=8,
        lr0=1e-4,
        optimizer="AdamW",
        patience=20,
        single_cls=True,
        device=0,
        workers=4,
        plots=False,
        project=str(model_dir),
        name="obb_train_resumed",
        exist_ok=True,
        verbose=True,
        task="obb",
    )

    # Copy best weights
    resumed_best = model_dir / "obb_train_resumed" / "weights" / "best.pt"
    if resumed_best.exists():
        dest = model_dir / "obb_detector.pt"
        shutil.copy2(resumed_best, dest)
        print(f"Best weights saved to {dest}")
    else:
        print("Warning: best.pt not found after training")

    # Cleanup tmp
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print("Done.")


if __name__ == "__main__":
    main()
