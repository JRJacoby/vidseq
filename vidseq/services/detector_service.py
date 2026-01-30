"""Detector Service for SegFormer-based segmentation training and inference."""

import logging
import os
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from PIL import Image

if TYPE_CHECKING:
    from vidseq.services.detector_model import SegFormerDetector

import cv2
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from sqlalchemy import select
from sqlalchemy.orm import Session

from vidseq.models.video import Video
from vidseq.models.frame_data import FrameData
from vidseq.schemas.detector import DetectorTrainingProgress
from vidseq.services.database_manager import DatabaseManager

# Disable HDF5's internal file locking
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

logger = logging.getLogger("vidseq.detector")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "[%(asctime)s] [Detector] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

MIN_FRAMES_FOR_VALIDATION = 10


class DetectorDataset(Dataset):
    """Dataset for detector training from video frames and tracker masks."""

    def __init__(
        self,
        frames: list[tuple[Path, int, int]],  # (video_path, video_id, frame_idx)
        project_path: Path,
        processor,  # SegformerImageProcessor
    ):
        """Initialize dataset.

        Args:
            frames: List of (video_path, video_id, frame_idx) tuples.
            project_path: Path to project folder (for HDF5 mask files).
            processor: SegformerImageProcessor instance for preprocessing.
        """
        self.frames = frames
        self.project_path = project_path
        self.processor = processor
        self._h5_cache: dict[int, h5py.File] = {}

    def __len__(self) -> int:
        return len(self.frames)

    def _get_h5_file(self, video_id: int) -> h5py.File:
        """Get or open HDF5 file for video."""
        if video_id not in self._h5_cache:
            h5_path = self.project_path / "masks" / f"{video_id}.h5"
            self._h5_cache[video_id] = h5py.File(h5_path, "r")
        return self._h5_cache[video_id]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        video_path, video_id, frame_idx = self.frames[idx]

        # Load frame from video
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise RuntimeError(f"Failed to read frame {frame_idx} from {video_path}")

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Load mask from HDF5 (values are 0 or 255)
        h5_file = self._get_h5_file(video_id)
        mask = np.array(h5_file["masks"][frame_idx], dtype=np.uint8)

        # Convert 0/255 to 0/1 class labels
        mask_labels = (mask > 127).astype(np.uint8)

        # Use processor for preprocessing
        inputs = self.processor(
            images=Image.fromarray(frame_rgb),
            segmentation_maps=Image.fromarray(mask_labels),
            return_tensors="pt",
        )

        # Remove batch dimension (DataLoader will add it back)
        pixel_values = inputs["pixel_values"].squeeze(0)
        labels = inputs["labels"].squeeze(0)

        return pixel_values, labels

    def close(self):
        """Close all HDF5 files."""
        for f in self._h5_cache.values():
            f.close()
        self._h5_cache.clear()


class DetectorService:
    """Singleton service for detector training and inference."""

    _instance: Optional["DetectorService"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "DetectorService":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._is_training = False
        self._stop_requested = False
        self._training_thread: Optional[threading.Thread] = None
        self._training_progress = DetectorTrainingProgress()

        self._initialized = True

    @classmethod
    def get_instance(cls) -> "DetectorService":
        """Get the singleton instance."""
        return cls()

    def is_training(self) -> bool:
        """Check if training is in progress."""
        return self._is_training

    def get_training_progress(self) -> DetectorTrainingProgress:
        """Get current training progress."""
        return self._training_progress

    def stop_training(self) -> bool:
        """Request training to stop."""
        if self._is_training:
            self._stop_requested = True
            return True
        return False

    def model_exists(self, project_path: Path) -> bool:
        """Check if trained model exists."""
        return (project_path / "models" / "detector.pt").exists()

    def train(
        self,
        project_path: Path,
        max_epochs: int = 1000,
        batch_size: int = 4,
        lr: float = 1e-4,
        lr_patience: int = 10,
        lr_factor: float = 0.25,
        early_stop_patience: int = 20,
    ) -> bool:
        """Start training in background thread.

        Returns immediately. Use get_training_progress() to monitor.
        """
        if self._is_training:
            raise RuntimeError("Training already in progress")

        self._is_training = True
        self._stop_requested = False

        def _train_thread():
            try:
                self._train_sync(
                    project_path,
                    max_epochs,
                    batch_size,
                    lr,
                    lr_patience,
                    lr_factor,
                    early_stop_patience,
                )
            except Exception as e:
                logger.exception("Training failed")
                self._training_progress.status = "failed"
                self._training_progress.error_message = str(e)
            finally:
                self._is_training = False

        self._training_thread = threading.Thread(target=_train_thread, daemon=True)
        self._training_thread.start()
        return True

    def _train_sync(
        self,
        project_path: Path,
        max_epochs: int,
        batch_size: int,
        lr: float,
        lr_patience: int,
        lr_factor: float,
        early_stop_patience: int,
    ) -> None:
        """Synchronous training implementation."""
        from torch.utils.data import DataLoader
        from vidseq.services.detector_model import SegFormerDetector, get_processor

        logger.info(f"Starting detector training: max_epochs={max_epochs}, batch_size={batch_size}")

        # Gather training data
        all_frames = self._gather_training_frames(project_path)
        if len(all_frames) == 0:
            raise RuntimeError("No training frames found. Mark training ranges first.")

        # Train/val split
        import random
        random.shuffle(all_frames)
        use_validation = len(all_frames) >= MIN_FRAMES_FOR_VALIDATION

        if use_validation:
            split_idx = int(len(all_frames) * 0.8)
            train_frames = all_frames[:split_idx]
            val_frames = all_frames[split_idx:]
        else:
            train_frames = all_frames
            val_frames = []

        # Initialize progress
        self._training_progress = DetectorTrainingProgress(
            is_training=True,
            current_epoch=0,
            max_epochs=max_epochs,
            current_lr=lr,
            lr_patience=lr_patience,
            early_stop_patience=early_stop_patience,
            status="training",
            started_at=time.time(),
            num_train_frames=len(train_frames),
            num_val_frames=len(val_frames),
        )

        # Get processor for preprocessing
        processor = get_processor()

        # Create datasets and loaders
        train_dataset = DetectorDataset(train_frames, project_path, processor)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )

        val_loader = None
        val_dataset = None
        if use_validation:
            val_dataset = DetectorDataset(val_frames, project_path, processor)
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
            )

        # Initialize model
        model = SegFormerDetector(device="cuda")
        model.train()

        # Optimizer (only decoder parameters)
        optimizer = torch.optim.AdamW(
            model.decoder.parameters(), lr=lr, weight_decay=1e-4
        )

        def dice_loss(probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            """Dice loss for binary segmentation.

            Args:
                probs: Foreground probabilities, shape (B, H, W), values in [0, 1]
                targets: Binary targets, shape (B, H, W), values 0 or 1
            """
            probs_flat = probs.contiguous().view(-1)
            targets_flat = targets.contiguous().view(-1).float()
            intersection = (probs_flat * targets_flat).sum()
            union = probs_flat.sum() + targets_flat.sum()
            dice = (2.0 * intersection + 1.0) / (union + 1.0)
            return 1.0 - dice

        # Training state
        best_val_loss = float("inf")
        best_epoch = 0
        epochs_without_improvement = 0
        current_lr = lr
        lr_reduced_this_plateau = False
        min_lr = 1e-7

        # Use bfloat16 autocast for faster training on Ampere+ GPUs
        use_amp = True
        amp_dtype = torch.bfloat16

        # Ensure model save directory exists
        (project_path / "models").mkdir(parents=True, exist_ok=True)
        model_path = project_path / "models" / "detector.pt"

        # Training loop
        for epoch in range(max_epochs):
            if self._stop_requested:
                logger.info("Training stopped by user")
                self._training_progress.status = "stopped"
                break

            # Training phase
            model.train()
            train_losses = []
            total_batches = len(train_loader)
            self._training_progress.total_batches = total_batches
            self._training_progress.current_batch = 0

            from tqdm import tqdm
            pbar = tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{max_epochs}",
                leave=False,
                ncols=100,
            )
            for batch_idx, (pixel_values, labels) in enumerate(pbar):
                pixel_values = pixel_values.to("cuda")
                labels = labels.to("cuda")

                optimizer.zero_grad()

                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                    logits = model(pixel_values)  # (B, 2, H/4, W/4)

                    # Upsample logits to match label size
                    logits_upsampled = F.interpolate(
                        logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
                    )

                    # Cross-entropy loss
                    ce_loss = F.cross_entropy(logits_upsampled, labels)

                    # Dice loss on foreground probabilities
                    probs = F.softmax(logits_upsampled, dim=1)[:, 1]  # (B, H, W)
                    d_loss = dice_loss(probs, labels.float())

                    loss = 0.5 * ce_loss + 0.5 * d_loss

                # Backward pass
                loss.backward()
                optimizer.step()

                batch_loss = loss.item()
                train_losses.append(batch_loss)

                # Update batch progress
                self._training_progress.current_batch = batch_idx + 1
                self._training_progress.batch_loss = batch_loss
                pbar.set_postfix(loss=f"{batch_loss:.4f}")

            pbar.close()
            avg_train_loss = sum(train_losses) / len(train_losses)

            # Validation phase
            avg_val_loss = avg_train_loss  # Default if no validation
            if val_loader is not None:
                model.eval()
                val_losses = []
                val_pbar = tqdm(
                    val_loader,
                    desc=f"Epoch {epoch + 1} Val",
                    leave=False,
                    ncols=100,
                )
                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                    for pixel_values, labels in val_pbar:
                        pixel_values = pixel_values.to("cuda")
                        labels = labels.to("cuda")

                        logits = model(pixel_values)
                        logits_upsampled = F.interpolate(
                            logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
                        )

                        ce_loss = F.cross_entropy(logits_upsampled, labels)
                        probs = F.softmax(logits_upsampled, dim=1)[:, 1]
                        d_loss = dice_loss(probs, labels.float())

                        loss = 0.5 * ce_loss + 0.5 * d_loss
                        val_losses.append(loss.item())
                val_pbar.close()
                avg_val_loss = sum(val_losses) / len(val_losses)

            # Update progress
            self._training_progress.current_epoch = epoch + 1
            self._training_progress.current_train_loss = avg_train_loss
            self._training_progress.current_val_loss = avg_val_loss
            self._training_progress.train_loss_history.append(avg_train_loss)
            self._training_progress.val_loss_history.append(avg_val_loss)

            # Check for improvement
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch + 1
                epochs_without_improvement = 0
                lr_reduced_this_plateau = False
                model.save_decoder(str(model_path))
                logger.info(
                    f"Epoch {epoch + 1}/{max_epochs}: train={avg_train_loss:.6f}, "
                    f"val={avg_val_loss:.6f} (new best)"
                )
            else:
                epochs_without_improvement += 1
                logger.info(
                    f"Epoch {epoch + 1}/{max_epochs}: train={avg_train_loss:.6f}, "
                    f"val={avg_val_loss:.6f}, no improvement x{epochs_without_improvement}"
                )

            self._training_progress.best_val_loss = best_val_loss
            self._training_progress.best_epoch = best_epoch
            self._training_progress.epochs_without_improvement = epochs_without_improvement

            # Early stopping
            if epochs_without_improvement >= early_stop_patience:
                logger.info(f"Early stopping after {epoch + 1} epochs")
                break

            # LR reduction
            if (
                epochs_without_improvement >= lr_patience
                and not lr_reduced_this_plateau
                and current_lr > min_lr
            ):
                current_lr *= lr_factor
                for param_group in optimizer.param_groups:
                    param_group["lr"] = current_lr
                lr_reduced_this_plateau = True
                logger.info(f"Reduced learning rate to {current_lr:.2e}")

            self._training_progress.current_lr = current_lr
            self._training_progress.lr_reduced_this_plateau = lr_reduced_this_plateau

        # Cleanup
        train_dataset.close()
        if val_dataset:
            val_dataset.close()

        # Mark completed and apply to training data
        if self._training_progress.status == "training":
            self._training_progress.status = "applying"
            self._apply_to_training_data(project_path, model)
            self._training_progress.status = "completed"

        self._training_progress.is_training = False
        logger.info(f"Training completed. Best epoch: {best_epoch}, best val loss: {best_val_loss:.6f}")

    def _gather_training_frames(
        self,
        project_path: Path,
    ) -> list[tuple[Path, int, int]]:
        """Gather all training frames from all videos in project.

        Training frames are identified by FrameData.frame_type == 'train'.

        Returns:
            List of (video_path, video_id, frame_idx) tuples.
        """
        db_manager = DatabaseManager.get_instance()
        engine = db_manager.get_project_engine(project_path)

        frames = []
        with Session(engine) as session:
            # Get all videos
            videos = session.execute(select(Video)).scalars().all()
            video_map = {v.id: v for v in videos}

            # Get all training frames
            training_frames = session.execute(
                select(FrameData).where(FrameData.frame_type == "train")
            ).scalars().all()

            for frame_data in training_frames:
                video = video_map.get(frame_data.video_id)
                if video is None:
                    continue

                video_path = Path(video.path)
                h5_path = project_path / "masks" / f"{video.id}.h5"

                if not h5_path.exists():
                    continue

                frames.append((video_path, video.id, frame_data.frame_idx))

        logger.info(f"Gathered {len(frames)} training frames from {project_path}")
        return frames

    def _apply_to_training_data(
        self,
        project_path: Path,
        model: "SegFormerDetector",
    ) -> None:
        """Apply trained detector to all training frames and save masks."""
        from vidseq.services.detector_model import get_processor

        logger.info("Applying detector to training data...")

        all_frames = self._gather_training_frames(project_path)
        self._training_progress.apply_total = len(all_frames)
        self._training_progress.apply_current = 0

        model.eval()
        processor = get_processor()

        # Group frames by video for efficient HDF5 access
        frames_by_video: dict[int, list[tuple[Path, int]]] = {}
        for video_path, video_id, frame_idx in all_frames:
            if video_id not in frames_by_video:
                frames_by_video[video_id] = []
            frames_by_video[video_id].append((video_path, frame_idx))

        # Use bfloat16 for faster inference
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for video_id, frame_list in frames_by_video.items():
                tracker_h5_path = project_path / "masks" / f"{video_id}.h5"
                detector_h5_path = project_path / "masks" / f"{video_id}_detector.h5"

                # Read original mask shape from tracker h5 file
                with h5py.File(tracker_h5_path, "r") as tracker_h5:
                    mask_shape = tracker_h5["masks"].shape  # (N, H, W)
                    _, orig_h, orig_w = mask_shape

                with h5py.File(detector_h5_path, "a") as h5_file:
                    # Create or get masks dataset
                    if "masks" not in h5_file:
                        h5_file.create_dataset(
                            "masks",
                            shape=mask_shape,
                            dtype=np.uint8,
                            chunks=(1, orig_h, orig_w),
                            compression="gzip",
                        )
                    detector_masks = h5_file["masks"]

                    for video_path, frame_idx in frame_list:
                        # Load frame
                        cap = cv2.VideoCapture(str(video_path))
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                        ret, frame = cap.read()
                        cap.release()

                        if not ret:
                            self._training_progress.apply_current += 1
                            continue

                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                        # Preprocess with processor
                        inputs = processor(
                            images=Image.fromarray(frame_rgb),
                            return_tensors="pt",
                        )
                        pixel_values = inputs["pixel_values"].to("cuda")

                        # Inference
                        logits = model(pixel_values)  # (1, 2, H/4, W/4)

                        # Upsample to original size and get binary mask
                        logits_upsampled = F.interpolate(
                            logits, size=(orig_h, orig_w), mode="bilinear", align_corners=False
                        )
                        mask_pred = logits_upsampled.argmax(dim=1)  # (1, H, W)
                        mask_np = (mask_pred[0].cpu().numpy() * 255).astype(np.uint8)

                        # Save to HDF5
                        detector_masks[frame_idx] = mask_np

                        self._training_progress.apply_current += 1

        logger.info(f"Applied detector to {len(all_frames)} training frames")

    def apply_to_all(self, project_path: Path) -> bool:
        """Apply detector to all frames of all videos in the project.

        Runs in background thread. Skips frames that already have detector masks.
        Use get_training_progress() to monitor (reuses apply_current/apply_total).

        Returns:
            True if started successfully.
        """
        if self._is_training:
            raise RuntimeError("Training or apply already in progress")

        model_path = project_path / "models" / "detector.pt"
        if not model_path.exists():
            raise RuntimeError("No trained detector model found. Train first.")

        self._is_training = True
        self._stop_requested = False

        def _apply_thread():
            try:
                self._apply_to_all_sync(project_path)
            except Exception as e:
                logger.exception("Apply to all failed")
                self._training_progress.status = "failed"
                self._training_progress.error_message = str(e)
            finally:
                self._is_training = False

        self._training_thread = threading.Thread(target=_apply_thread, daemon=True)
        self._training_thread.start()
        return True

    def _apply_to_all_sync(self, project_path: Path) -> None:
        """Synchronous implementation of apply to all frames."""
        from vidseq.services.detector_model import SegFormerDetector, get_processor

        logger.info("Applying detector to all frames in project...")

        # Load model
        model = SegFormerDetector(device="cuda")
        model.load_decoder(str(project_path / "models" / "detector.pt"))
        model.eval()

        processor = get_processor()

        # Get all videos
        db_manager = DatabaseManager.get_instance()
        engine = db_manager.get_project_engine(project_path)

        with Session(engine) as session:
            videos = session.execute(select(Video)).scalars().all()
            video_list = [(v.id, v.path, v.num_frames, v.height, v.width) for v in videos]

        # Count total frames to process
        total_frames = sum(nf for _, _, nf, _, _ in video_list if nf)

        # Initialize progress
        self._training_progress = DetectorTrainingProgress(
            is_training=True,
            status="applying",
            started_at=time.time(),
            apply_current=0,
            apply_total=total_frames,
        )

        processed = 0
        skipped = 0

        from tqdm import tqdm

        # Use bfloat16 for faster inference
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for video_idx, (video_id, video_path, frame_count, orig_h, orig_w) in enumerate(video_list):
                if self._stop_requested:
                    logger.info("Apply stopped by user")
                    self._training_progress.status = "stopped"
                    break

                if not frame_count:
                    logger.info(f"Video {video_idx + 1}/{len(video_list)}: skipping (no frames)")
                    continue

                detector_h5_path = project_path / "masks" / f"{video_id}_detector.h5"

                # Open video
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    logger.warning(f"Video {video_idx + 1}/{len(video_list)}: could not open {video_path}")
                    self._training_progress.apply_current += frame_count
                    continue

                video_name = Path(video_path).name
                logger.info(f"Video {video_idx + 1}/{len(video_list)}: {video_name}")

                # Use video dimensions for mask shape
                num_frames = frame_count
                mask_shape = (num_frames, orig_h, orig_w)

                # Ensure masks directory exists
                detector_h5_path.parent.mkdir(parents=True, exist_ok=True)

                try:
                    h5_file = h5py.File(str(detector_h5_path), "a")
                except OSError as e:
                    raise RuntimeError(
                        f"Failed to open/create detector HDF5 file: {detector_h5_path}. Error: {e}"
                    )

                try:
                    # Create masks dataset if needed
                    if "masks" not in h5_file:
                        h5_file.create_dataset(
                            "masks",
                            shape=mask_shape,
                            dtype=np.uint8,
                            chunks=(1, orig_h, orig_w),
                            compression="gzip",
                        )
                    detector_masks = h5_file["masks"]

                    video_processed = 0
                    video_skipped = 0

                    pbar = tqdm(
                        range(num_frames),
                        desc=f"Video {video_idx + 1}/{len(video_list)}",
                        leave=False,
                        ncols=100,
                    )
                    for frame_idx in pbar:
                        if self._stop_requested:
                            break

                        # Check if mask already exists (non-zero)
                        try:
                            existing = detector_masks[frame_idx]
                            if np.any(existing):
                                video_skipped += 1
                                skipped += 1
                                self._training_progress.apply_current += 1
                                continue
                        except OSError:
                            pass

                        # Read frame
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                        ret, frame = cap.read()
                        if not ret:
                            self._training_progress.apply_current += 1
                            continue

                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                        # Preprocess with processor
                        inputs = processor(
                            images=Image.fromarray(frame_rgb),
                            return_tensors="pt",
                        )
                        pixel_values = inputs["pixel_values"].to("cuda")

                        # Inference
                        logits = model(pixel_values)  # (1, 2, H/4, W/4)

                        # Upsample to original size and get binary mask
                        logits_upsampled = F.interpolate(
                            logits, size=(orig_h, orig_w), mode="bilinear", align_corners=False
                        )
                        mask_pred = logits_upsampled.argmax(dim=1)  # (1, H, W)
                        mask_np = (mask_pred[0].cpu().numpy() * 255).astype(np.uint8)

                        # Save to HDF5
                        detector_masks[frame_idx] = mask_np
                        processed += 1
                        video_processed += 1

                        self._training_progress.apply_current += 1

                    pbar.close()
                finally:
                    h5_file.close()

                cap.release()
                logger.info(
                    f"  Processed {video_processed} frames, skipped {video_skipped} existing"
                )

        if self._training_progress.status != "stopped":
            self._training_progress.status = "completed"

        self._training_progress.is_training = False
        logger.info(f"Applied detector to {processed} frames, skipped {skipped} existing")
