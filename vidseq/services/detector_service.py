"""Detector Service for DINOv2-based segmentation training and inference."""

import logging
import os
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from vidseq.services.detector_model import DINOv2Detector

import cv2
import h5py
import numpy as np
import torch
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
        target_size: int = 518,
    ):
        """Initialize dataset.

        Args:
            frames: List of (video_path, video_id, frame_idx) tuples.
            project_path: Path to project folder (for HDF5 mask files).
            target_size: Target size for longest edge.
        """
        self.frames = frames
        self.project_path = project_path
        self.target_size = target_size
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

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]

        # Load mask from HDF5
        h5_file = self._get_h5_file(video_id)
        mask = h5_file["masks"][frame_idx]
        mask = np.array(mask, dtype=np.float32) / 255.0

        # Resize preserving aspect ratio, then trim to multiple of 14
        scale = self.target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        # Trim to multiple of 14
        new_h = (frame.shape[0] // 14) * 14
        new_w = (frame.shape[1] // 14) * 14
        frame = frame[:new_h, :new_w]
        mask = mask[:new_h, :new_w]

        # Convert to tensors
        frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask).unsqueeze(0).float()

        return frame, mask

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
        import torch.nn as nn
        from torch.utils.data import DataLoader
        from vidseq.services.detector_model import DINOv2Detector

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

        # Create datasets and loaders
        train_dataset = DetectorDataset(train_frames, project_path)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )

        val_loader = None
        val_dataset = None
        if use_validation:
            val_dataset = DetectorDataset(val_frames, project_path)
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
            )

        # Initialize model
        model = DINOv2Detector(device="cuda")
        model.train()

        # Loss and optimizer
        bce_loss = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(
            model.decoder.parameters(), lr=lr, weight_decay=1e-4
        )

        def dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            probs = torch.sigmoid(logits)
            intersection = (probs * targets).sum(dim=(2, 3))
            union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
            dice = (2 * intersection + 1) / (union + 1)
            return 1 - dice.mean()

        # Training state
        best_val_loss = float("inf")
        best_epoch = 0
        epochs_without_improvement = 0
        current_lr = lr
        lr_reduced_this_plateau = False
        min_lr = 1e-7

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
            for batch_idx, (images, masks) in enumerate(pbar):
                images = images.to("cuda")
                masks = masks.to("cuda")

                optimizer.zero_grad()
                logits = model(images)
                loss = 0.5 * bce_loss(logits, masks) + 0.5 * dice_loss(logits, masks)
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
                with torch.no_grad():
                    for images, masks in val_pbar:
                        images = images.to("cuda")
                        masks = masks.to("cuda")
                        logits = model(images)
                        loss = 0.5 * bce_loss(logits, masks) + 0.5 * dice_loss(logits, masks)
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
        model: "DINOv2Detector",
    ) -> None:
        """Apply trained detector to all training frames and save masks."""
        logger.info("Applying detector to training data...")

        all_frames = self._gather_training_frames(project_path)
        self._training_progress.apply_total = len(all_frames)
        self._training_progress.apply_current = 0

        model.eval()

        # Group frames by video for efficient HDF5 access
        frames_by_video: dict[int, list[tuple[Path, int]]] = {}
        for video_path, video_id, frame_idx in all_frames:
            if video_id not in frames_by_video:
                frames_by_video[video_id] = []
            frames_by_video[video_id].append((video_path, frame_idx))

        with torch.no_grad():
            for video_id, frame_list in frames_by_video.items():
                h5_path = project_path / "masks" / f"{video_id}.h5"

                with h5py.File(h5_path, "a") as h5_file:
                    # Get original mask shape
                    mask_shape = h5_file["masks"].shape  # (N, H, W)
                    _, orig_h, orig_w = mask_shape

                    # Create or get detector_masks dataset
                    if "detector_masks" not in h5_file:
                        h5_file.create_dataset(
                            "detector_masks",
                            shape=mask_shape,
                            dtype=np.uint8,
                            chunks=(1, orig_h, orig_w),
                            compression="gzip",
                        )
                    detector_masks = h5_file["detector_masks"]

                    for video_path, frame_idx in frame_list:
                        # Load and preprocess frame
                        cap = cv2.VideoCapture(str(video_path))
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                        ret, frame = cap.read()
                        cap.release()

                        if not ret:
                            continue

                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        h, w = frame.shape[:2]

                        # Resize preserving aspect ratio
                        target_size = 518
                        scale = target_size / max(h, w)
                        new_h, new_w = int(h * scale), int(w * scale)
                        frame_resized = cv2.resize(frame, (new_w, new_h))

                        # Trim to multiple of 14
                        trim_h = (frame_resized.shape[0] // 14) * 14
                        trim_w = (frame_resized.shape[1] // 14) * 14
                        frame_resized = frame_resized[:trim_h, :trim_w]

                        # To tensor
                        frame_tensor = (
                            torch.from_numpy(frame_resized)
                            .permute(2, 0, 1)
                            .float()
                            / 255.0
                        )
                        frame_tensor = frame_tensor.unsqueeze(0).to("cuda")

                        # Inference
                        logits = model(frame_tensor)
                        mask_pred = (torch.sigmoid(logits) > 0.5).float()

                        # Resize back to original size
                        mask_pred = torch.nn.functional.interpolate(
                            mask_pred,
                            size=(orig_h, orig_w),
                            mode="nearest",
                        )
                        mask_np = (mask_pred[0, 0].cpu().numpy() * 255).astype(np.uint8)

                        # Save to HDF5
                        detector_masks[frame_idx] = mask_np

                        self._training_progress.apply_current += 1

        logger.info(f"Applied detector to {len(all_frames)} training frames")
