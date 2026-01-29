# DINOv2 Detector Training Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement DINOv2-based detector training, application to training data, and verification UI.

**Architecture:** DINOv2-giant backbone (frozen) with lightweight convolutional decoder. Training orchestrated by DetectorService, exposed via FastAPI endpoints, with Vue frontend for training progress visualization and mask view switching.

**Tech Stack:** PyTorch, DINOv2 (torch.hub), FastAPI, Vue 3, Chart.js

---

## Task 1: Detector Model Architecture

**Files:**
- Create: `vidseq/services/detector_model.py`

**Step 1: Create the decoder module**

```python
"""DINOv2-based detector model for segmentation.

Uses DINOv2-giant as frozen backbone with a lightweight convolutional decoder
that outputs per-pixel segmentation logits.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentationDecoder(nn.Module):
    """Convolutional decoder that upsamples DINOv2 patch features to full resolution."""

    def __init__(self, in_channels: int = 1536):
        """Initialize decoder.

        Args:
            in_channels: Number of input channels from DINOv2 (1536 for giant).
        """
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Patch features from DINOv2, shape (B, C, H/14, W/14).
            output_size: Target output size (H, W) for final mask.

        Returns:
            Logits tensor of shape (B, 1, H, W).
        """
        x = self.conv1(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

        x = self.conv2(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

        x = self.conv3(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

        x = self.conv4(x)
        x = F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)

        x = self.head(x)
        return x


class DINOv2Detector(nn.Module):
    """DINOv2-giant backbone with segmentation decoder."""

    # ImageNet normalization
    MEAN = torch.tensor([0.485, 0.456, 0.406])
    STD = torch.tensor([0.229, 0.224, 0.225])

    def __init__(self, device: str = "cuda"):
        """Initialize model.

        Args:
            device: Device to load model on.
        """
        super().__init__()
        self.device = device

        # Load DINOv2-giant backbone (frozen)
        self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitg14")
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Trainable decoder
        self.decoder = SegmentationDecoder(in_channels=1536)

        self.to(device)

        # Move normalization tensors to device
        self.register_buffer("mean", self.MEAN.view(1, 3, 1, 1))
        self.register_buffer("std", self.STD.view(1, 3, 1, 1))

    def preprocess(self, images: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        """Preprocess images for DINOv2.

        Args:
            images: Input images, shape (B, C, H, W), values in [0, 1].

        Returns:
            Tuple of (preprocessed images, original size).
        """
        original_size = (images.shape[2], images.shape[3])

        # Trim to multiple of 14
        h, w = images.shape[2], images.shape[3]
        new_h = (h // 14) * 14
        new_w = (w // 14) * 14
        images = images[:, :, :new_h, :new_w]

        # Normalize
        images = (images - self.mean) / self.std

        return images, original_size

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            images: Input images, shape (B, C, H, W), values in [0, 1].

        Returns:
            Logits tensor of shape (B, 1, H, W) at original resolution.
        """
        images, original_size = self.preprocess(images)
        trimmed_size = (images.shape[2], images.shape[3])

        # Get patch features from DINOv2
        with torch.no_grad():
            features = self.backbone.forward_features(images)
            patch_tokens = features["x_norm_patchtokens"]  # (B, N, C)

        # Reshape to spatial grid
        B, N, C = patch_tokens.shape
        h = images.shape[2] // 14
        w = images.shape[3] // 14
        patch_tokens = patch_tokens.permute(0, 2, 1).reshape(B, C, h, w)

        # Decode to mask logits
        logits = self.decoder(patch_tokens, trimmed_size)

        return logits

    def save_decoder(self, path: str) -> None:
        """Save only the decoder weights."""
        torch.save(self.decoder.state_dict(), path)

    def load_decoder(self, path: str) -> None:
        """Load decoder weights."""
        self.decoder.load_state_dict(torch.load(path, map_location=self.device))
```

**Step 2: Commit**

```bash
git add vidseq/services/detector_model.py
git commit -m "feat: add DINOv2 detector model architecture"
```

---

## Task 2: Detector Training Progress Schema

**Files:**
- Create: `vidseq/schemas/detector.py`

**Step 1: Create schema file**

```python
"""Schemas for detector training."""

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class DetectorTrainingProgress:
    """Real-time training progress state for SSE streaming."""

    is_training: bool = False
    current_epoch: int = 0
    max_epochs: int = 1000

    # Training loss
    current_train_loss: float = 0.0
    train_loss_history: list[float] = field(default_factory=list)

    # Validation loss
    current_val_loss: float = 0.0
    val_loss_history: list[float] = field(default_factory=list)

    # Best model tracking (based on validation loss)
    best_val_loss: float = float("inf")
    best_epoch: int = 0

    # Learning rate
    current_lr: float = 1e-4

    # Patience counters (based on val loss)
    epochs_without_improvement: int = 0
    lr_patience: int = 10
    early_stop_patience: int = 20
    lr_reduced_this_plateau: bool = False

    # Status
    status: str = "idle"  # idle, training, applying, completed, stopped, failed
    started_at: Optional[float] = None
    error_message: Optional[str] = None

    # Dataset info
    num_train_frames: int = 0
    num_val_frames: int = 0

    # Apply progress
    apply_current: int = 0
    apply_total: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        # Handle infinity
        if d["best_val_loss"] == float("inf"):
            d["best_val_loss"] = None
        return d
```

**Step 2: Commit**

```bash
git add vidseq/schemas/detector.py
git commit -m "feat: add detector training progress schema"
```

---

## Task 3: Detector Service - Core Structure

**Files:**
- Create: `vidseq/services/detector_service.py`

**Step 1: Create service skeleton with training data gathering**

```python
"""Detector Service for DINOv2-based segmentation training and inference."""

import logging
import os
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sqlalchemy import select
from sqlalchemy.orm import Session

from vidseq.models.video import Video
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

    def _gather_training_frames(
        self,
        project_path: Path,
    ) -> list[tuple[Path, int, int]]:
        """Gather all training frames from all videos in project.

        Returns:
            List of (video_path, video_id, frame_idx) tuples.
        """
        db_manager = DatabaseManager.get_instance()
        engine = db_manager.get_project_engine(project_path)

        frames = []
        with Session(engine) as session:
            videos = session.execute(select(Video)).scalars().all()

            for video in videos:
                # Get training ranges from video
                if not video.training_ranges:
                    continue

                video_path = Path(video.path)
                h5_path = project_path / "masks" / f"{video.id}.h5"

                if not h5_path.exists():
                    continue

                # Parse training ranges and add frames
                for range_str in video.training_ranges.split(","):
                    if "-" in range_str:
                        start, end = map(int, range_str.split("-"))
                        for frame_idx in range(start, end + 1):
                            frames.append((video_path, video.id, frame_idx))

        logger.info(f"Gathered {len(frames)} training frames from {project_path}")
        return frames
```

**Step 2: Commit**

```bash
git add vidseq/services/detector_service.py
git commit -m "feat: add detector service skeleton with data gathering"
```

---

## Task 4: Detector Service - Training Loop

**Files:**
- Modify: `vidseq/services/detector_service.py`

**Step 1: Add training method**

Add this method to the `DetectorService` class:

```python
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
            for images, masks in train_loader:
                images = images.to("cuda")
                masks = masks.to("cuda")

                optimizer.zero_grad()
                logits = model(images)
                loss = 0.5 * bce_loss(logits, masks) + 0.5 * dice_loss(logits, masks)
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())

            avg_train_loss = sum(train_losses) / len(train_losses)

            # Validation phase
            avg_val_loss = avg_train_loss  # Default if no validation
            if val_loader is not None:
                model.eval()
                val_losses = []
                with torch.no_grad():
                    for images, masks in val_loader:
                        images = images.to("cuda")
                        masks = masks.to("cuda")
                        logits = model(images)
                        loss = 0.5 * bce_loss(logits, masks) + 0.5 * dice_loss(logits, masks)
                        val_losses.append(loss.item())
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
```

**Step 2: Commit**

```bash
git add vidseq/services/detector_service.py
git commit -m "feat: add detector training loop with early stopping"
```

---

## Task 5: Detector Service - Apply to Training Data

**Files:**
- Modify: `vidseq/services/detector_service.py`

**Step 1: Add apply method**

Add this method to the `DetectorService` class:

```python
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
                    num_frames, orig_h, orig_w = mask_shape

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
```

**Step 2: Commit**

```bash
git add vidseq/services/detector_service.py
git commit -m "feat: add detector apply to training data"
```

---

## Task 6: Detector API Routes

**Files:**
- Create: `vidseq/api/routes/detector.py`

**Step 1: Create routes**

```python
"""API routes for detector training and inference."""

import asyncio
import json
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

from vidseq.api.dependencies import get_project_folder
from vidseq.services.detector_service import DetectorService

router = APIRouter()

logger = logging.getLogger("vidseq.detector.api")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "[%(asctime)s] [Detector API] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class TrainRequest(BaseModel):
    """Request body for starting training."""
    max_epochs: int = 1000
    lr_patience: int = 10
    early_stop_patience: int = 20


class DetectorStatusResponse(BaseModel):
    """Response for detector status."""
    model_exists: bool
    is_training: bool


@router.get("/projects/{project_id}/detector/status")
async def get_detector_status(
    project_id: int,
    project_folder: Path = Depends(get_project_folder),
) -> DetectorStatusResponse:
    """Get detector model status."""
    service = DetectorService.get_instance()
    return DetectorStatusResponse(
        model_exists=service.model_exists(project_folder),
        is_training=service.is_training(),
    )


@router.post("/projects/{project_id}/detector/train")
async def train_detector(
    project_id: int,
    request: TrainRequest,
    project_folder: Path = Depends(get_project_folder),
):
    """Start detector training.

    Returns immediately. Use SSE stream to monitor progress.
    """
    logger.info(f"POST /detector/train: project_id={project_id}")

    service = DetectorService.get_instance()

    if service.is_training():
        raise HTTPException(status_code=400, detail="Training already in progress")

    try:
        service.train(
            project_folder,
            max_epochs=request.max_epochs,
            lr_patience=request.lr_patience,
            early_stop_patience=request.early_stop_patience,
        )
        return {"status": "started"}
    except Exception as e:
        logger.exception("Failed to start training")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/projects/{project_id}/detector/stop")
async def stop_detector_training(
    project_id: int,
):
    """Stop detector training."""
    service = DetectorService.get_instance()
    if service.stop_training():
        return {"status": "stopping"}
    raise HTTPException(status_code=400, detail="No training in progress")


@router.get("/projects/{project_id}/detector/training/status")
async def get_training_status(
    project_id: int,
):
    """Get current training progress."""
    service = DetectorService.get_instance()
    progress = service.get_training_progress()
    return progress.to_dict()


@router.get("/projects/{project_id}/detector/training/stream")
async def stream_training_progress(
    project_id: int,
):
    """Stream training progress via Server-Sent Events."""
    logger.info(f"GET /detector/training/stream: SSE connection started")

    service = DetectorService.get_instance()

    async def event_generator():
        last_epoch = -1
        last_status = None

        while True:
            progress = service.get_training_progress()

            # Send update if epoch changed or status changed
            if progress.current_epoch != last_epoch or progress.status != last_status:
                last_epoch = progress.current_epoch
                last_status = progress.status
                yield f"data: {json.dumps(progress.to_dict())}\n\n"

                # Stop streaming if finished
                if progress.status in ("completed", "stopped", "failed"):
                    logger.info(f"SSE closing - status={progress.status}")
                    break

            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/projects/{project_id}/videos/{video_id}/detector-mask/{frame_idx}")
async def get_detector_mask(
    project_id: int,
    video_id: int,
    frame_idx: int,
    project_folder: Path = Depends(get_project_folder),
):
    """Get detector mask for a frame."""
    import h5py
    import numpy as np
    import cv2

    h5_path = project_folder / "masks" / f"{video_id}.h5"

    if not h5_path.exists():
        raise HTTPException(status_code=404, detail="Mask file not found")

    with h5py.File(h5_path, "r") as f:
        if "detector_masks" not in f:
            raise HTTPException(status_code=404, detail="Detector masks not found")

        mask = f["detector_masks"][frame_idx]
        mask = np.array(mask, dtype=np.uint8)

    # Encode as PNG
    _, png_data = cv2.imencode(".png", mask)

    return Response(content=png_data.tobytes(), media_type="image/png")


@router.get("/projects/{project_id}/videos/{video_id}/detector-masks/exists")
async def detector_masks_exist(
    project_id: int,
    video_id: int,
    project_folder: Path = Depends(get_project_folder),
):
    """Check if detector masks exist for a video."""
    import h5py

    h5_path = project_folder / "masks" / f"{video_id}.h5"

    if not h5_path.exists():
        return {"exists": False}

    with h5py.File(h5_path, "r") as f:
        exists = "detector_masks" in f

    return {"exists": exists}
```

**Step 2: Register routes in main app**

Add to `vidseq/api/main.py` or wherever routes are registered:

```python
from vidseq.api.routes.detector import router as detector_router
app.include_router(detector_router, tags=["detector"])
```

**Step 3: Commit**

```bash
git add vidseq/api/routes/detector.py
git commit -m "feat: add detector API routes"
```

---

## Task 7: Frontend API Functions

**Files:**
- Modify: `frontend/src/services/api.ts`

**Step 1: Add detector API functions**

Add these to `api.ts`:

```typescript
// Detector types
export interface DetectorStatus {
    model_exists: boolean
    is_training: boolean
}

export interface DetectorTrainingProgress {
    is_training: boolean
    status: string
    current_epoch: number
    max_epochs: number
    current_train_loss: number
    current_val_loss: number
    train_loss_history: number[]
    val_loss_history: number[]
    best_val_loss: number | null
    best_epoch: number
    current_lr: number
    epochs_without_improvement: number
    lr_patience: number
    early_stop_patience: number
    num_train_frames: number
    num_val_frames: number
    apply_current: number
    apply_total: number
    error_message: string | null
}

export async function getDetectorStatus(projectId: number): Promise<DetectorStatus> {
    const response = await fetch(`${API_BASE}/projects/${projectId}/detector/status`)
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to get detector status'))
    }
    return response.json()
}

export async function trainDetector(
    projectId: number,
    maxEpochs: number = 1000,
    lrPatience: number = 10,
    earlyStopPatience: number = 20,
): Promise<void> {
    const response = await fetch(`${API_BASE}/projects/${projectId}/detector/train`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            max_epochs: maxEpochs,
            lr_patience: lrPatience,
            early_stop_patience: earlyStopPatience,
        }),
    })
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to start detector training'))
    }
}

export async function stopDetectorTraining(projectId: number): Promise<void> {
    const response = await fetch(`${API_BASE}/projects/${projectId}/detector/stop`, {
        method: 'POST',
    })
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to stop training'))
    }
}

export async function getDetectorTrainingStatus(projectId: number): Promise<DetectorTrainingProgress> {
    const response = await fetch(`${API_BASE}/projects/${projectId}/detector/training/status`)
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to get training status'))
    }
    return response.json()
}

export function getDetectorTrainingStreamUrl(projectId: number): string {
    return `${API_BASE}/projects/${projectId}/detector/training/stream`
}

export async function getDetectorMask(
    projectId: number,
    videoId: number,
    frameIdx: number,
): Promise<Blob> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/${videoId}/detector-mask/${frameIdx}`
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to fetch detector mask'))
    }
    return response.blob()
}

export async function detectorMasksExist(
    projectId: number,
    videoId: number,
): Promise<{ exists: boolean }> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/${videoId}/detector-masks/exists`
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to check detector masks'))
    }
    return response.json()
}
```

**Step 2: Commit**

```bash
git add frontend/src/services/api.ts
git commit -m "feat: add detector API functions to frontend"
```

---

## Task 8: Frontend useDetector Composable

**Files:**
- Create: `frontend/src/composables/useDetector.ts`

**Step 1: Create composable**

```typescript
import { ref, watch, type Ref } from 'vue'
import {
    getDetectorStatus,
    trainDetector,
    stopDetectorTraining,
    type DetectorStatus,
} from '@/services/api'

export interface UseDetectorReturn {
    isTraining: Ref<boolean>
    modelExists: Ref<boolean>
    startTraining: (maxEpochs?: number) => Promise<void>
    stopTraining: () => Promise<void>
    checkStatus: () => Promise<void>
}

export function useDetector(projectId: Ref<number | null>): UseDetectorReturn {
    const isTraining = ref(false)
    const modelExists = ref(false)

    const checkStatus = async () => {
        if (!projectId.value) return
        try {
            const status = await getDetectorStatus(projectId.value)
            modelExists.value = status.model_exists
            isTraining.value = status.is_training
        } catch (e) {
            console.error('Failed to check detector status:', e)
        }
    }

    const startTraining = async (maxEpochs: number = 1000) => {
        if (!projectId.value || isTraining.value) return
        isTraining.value = true
        try {
            await trainDetector(projectId.value, maxEpochs)
        } catch (e) {
            console.error('Failed to start detector training:', e)
            isTraining.value = false
            throw e
        }
    }

    const stopTraining = async () => {
        if (!projectId.value) return
        try {
            await stopDetectorTraining(projectId.value)
        } catch (e) {
            console.error('Failed to stop training:', e)
            throw e
        }
    }

    // Check status on mount and when projectId changes
    watch(projectId, async (newId) => {
        if (newId !== null) {
            await checkStatus()
        }
    }, { immediate: true })

    // Poll while training
    let pollInterval: number | null = null

    watch(isTraining, (training) => {
        if (training) {
            if (pollInterval === null && projectId.value !== null) {
                pollInterval = window.setInterval(async () => {
                    await checkStatus()
                    if (!isTraining.value && pollInterval !== null) {
                        clearInterval(pollInterval)
                        pollInterval = null
                    }
                }, 2000)
            }
        } else {
            if (pollInterval !== null) {
                clearInterval(pollInterval)
                pollInterval = null
            }
        }
    })

    return {
        isTraining,
        modelExists,
        startTraining,
        stopTraining,
        checkStatus,
    }
}
```

**Step 2: Commit**

```bash
git add frontend/src/composables/useDetector.ts
git commit -m "feat: add useDetector composable"
```

---

## Task 9: DetectorTraining.vue Component

**Files:**
- Create: `frontend/src/components/DetectorTraining.vue`

**Step 1: Create component (copy Alignment.vue pattern)**

```vue
<script setup lang="ts">
import { ref, onMounted, onUnmounted, computed, watch } from 'vue'
import { useProjectStore } from '@/stores/project'
import {
    getDetectorTrainingStreamUrl,
    getDetectorTrainingStatus,
    type DetectorTrainingProgress,
} from '@/services/api'
import { Chart, registerables } from 'chart.js'

Chart.register(...registerables)

const projectStore = useProjectStore()
const projectId = computed(() => projectStore.currentProjectId)

const progress = ref<DetectorTrainingProgress | null>(null)
const chartCanvas = ref<HTMLCanvasElement | null>(null)
let chart: Chart | null = null
let eventSource: EventSource | null = null

function initChart() {
    if (!chartCanvas.value) return

    chart = new Chart(chartCanvas.value, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Training Loss',
                    data: [],
                    borderColor: 'rgb(59, 130, 246)',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    fill: false,
                    tension: 0.1,
                },
                {
                    label: 'Validation Loss',
                    data: [],
                    borderColor: 'rgb(239, 68, 68)',
                    backgroundColor: 'rgba(239, 68, 68, 0.1)',
                    fill: false,
                    tension: 0.1,
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: true, position: 'top' }
            },
            scales: {
                x: { title: { display: true, text: 'Epoch' } },
                y: {
                    type: 'logarithmic',
                    title: { display: true, text: 'Loss (log scale)' },
                }
            },
            animation: { duration: 0 }
        }
    })
}

function updateChart(trainHistory: number[], valHistory: number[]) {
    if (!chart?.data.datasets[0] || !chart?.data.datasets[1]) return

    const maxLen = Math.max(trainHistory.length, valHistory.length)
    chart.data.labels = Array.from({ length: maxLen }, (_, i) => i + 1)
    chart.data.datasets[0].data = trainHistory
    chart.data.datasets[1].data = valHistory
    chart.update('none')
}

function connectToStream() {
    if (!projectId.value) return

    if (eventSource) {
        eventSource.close()
        eventSource = null
    }

    const url = getDetectorTrainingStreamUrl(projectId.value)
    eventSource = new EventSource(url)

    eventSource.onmessage = (event) => {
        const data = JSON.parse(event.data) as DetectorTrainingProgress
        progress.value = data
        updateChart(data.train_loss_history, data.val_loss_history)

        if (['completed', 'stopped', 'failed'].includes(data.status)) {
            eventSource?.close()
            eventSource = null
        }
    }

    eventSource.onerror = () => {
        eventSource?.close()
        eventSource = null
    }
}

async function loadStatus() {
    if (!projectId.value) return

    try {
        progress.value = await getDetectorTrainingStatus(projectId.value)
        if (progress.value.train_loss_history.length > 0) {
            updateChart(progress.value.train_loss_history, progress.value.val_loss_history)
        }

        if (progress.value.is_training || progress.value.status === 'applying') {
            connectToStream()
        }
    } catch (e) {
        console.error('Failed to load training status:', e)
    }
}

onMounted(async () => {
    initChart()
    await loadStatus()
})

onUnmounted(() => {
    if (eventSource) {
        eventSource.close()
        eventSource = null
    }
    if (chart) {
        chart.destroy()
        chart = null
    }
})

watch(projectId, async () => {
    await loadStatus()
})
</script>

<template>
    <div class="detector-training">
        <h2>Detector Training</h2>

        <div v-if="progress" class="status-panel">
            <div class="status-row">
                <span class="label">Status:</span>
                <span :class="['status', progress.status]">{{ progress.status }}</span>
            </div>

            <div v-if="progress.is_training || progress.status === 'applying'" class="status-row">
                <span class="label">Epoch:</span>
                <span>{{ progress.current_epoch }} / {{ progress.max_epochs }}</span>
            </div>

            <div v-if="progress.status === 'applying'" class="status-row">
                <span class="label">Applying:</span>
                <span>{{ progress.apply_current }} / {{ progress.apply_total }} frames</span>
            </div>

            <div v-if="progress.best_epoch > 0" class="status-row">
                <span class="label">Best Epoch:</span>
                <span>{{ progress.best_epoch }} (loss: {{ progress.best_val_loss?.toFixed(6) ?? 'N/A' }})</span>
            </div>

            <div v-if="progress.is_training" class="status-row">
                <span class="label">Learning Rate:</span>
                <span>{{ progress.current_lr.toExponential(2) }}</span>
            </div>

            <div v-if="progress.is_training" class="status-row">
                <span class="label">No Improvement:</span>
                <span>{{ progress.epochs_without_improvement }} / {{ progress.early_stop_patience }} (early stop)</span>
            </div>

            <div class="status-row">
                <span class="label">Training Frames:</span>
                <span>{{ progress.num_train_frames }} train, {{ progress.num_val_frames }} val</span>
            </div>

            <div v-if="progress.error_message" class="error">
                {{ progress.error_message }}
            </div>
        </div>

        <div class="chart-container">
            <canvas ref="chartCanvas"></canvas>
        </div>
    </div>
</template>

<style scoped>
.detector-training {
    padding: 1rem;
}

h2 {
    margin-bottom: 1rem;
}

.status-panel {
    background: #f5f5f5;
    border-radius: 8px;
    padding: 1rem;
    margin-bottom: 1rem;
}

.status-row {
    display: flex;
    gap: 0.5rem;
    margin-bottom: 0.5rem;
}

.label {
    font-weight: 600;
    min-width: 140px;
}

.status {
    font-weight: 600;
    text-transform: capitalize;
}

.status.training { color: #3b82f6; }
.status.applying { color: #8b5cf6; }
.status.completed { color: #10b981; }
.status.stopped { color: #f59e0b; }
.status.failed { color: #ef4444; }
.status.idle { color: #6b7280; }

.error {
    color: #ef4444;
    margin-top: 0.5rem;
    padding: 0.5rem;
    background: #fef2f2;
    border-radius: 4px;
}

.chart-container {
    height: 400px;
    background: white;
    border-radius: 8px;
    padding: 1rem;
}
</style>
```

**Step 2: Commit**

```bash
git add frontend/src/components/DetectorTraining.vue
git commit -m "feat: add DetectorTraining component with loss chart"
```

---

## Task 10: Add Route for DetectorTraining

**Files:**
- Modify: `frontend/src/router/index.ts`

**Step 1: Add route**

Add import at top:
```typescript
import DetectorTraining from '../components/DetectorTraining.vue'
```

Add route in the project children array (after alignment route):
```typescript
        {
          path: 'detector',
          name: 'detector',
          component: DetectorTraining
        },
```

**Step 2: Commit**

```bash
git add frontend/src/router/index.ts
git commit -m "feat: add detector training route"
```

---

## Task 11: Add Train Detector Button to VideoPipeline

**Files:**
- Modify: `frontend/src/components/VideoPipeline.vue`

**Step 1: Import useDetector and add to template**

Add import:
```typescript
import { useDetector } from '@/composables/useDetector'
```

Add after useYOLO:
```typescript
const {
  isTraining: isDetectorTraining,
  modelExists: detectorModelExists,
  startTraining: startDetectorTraining,
  checkStatus: checkDetectorStatus,
} = useDetector(projectId)
```

Add handler:
```typescript
const handleTrainDetector = async () => {
  if (!projectId.value || isDetectorTraining.value) return
  try {
    await startDetectorTraining(1000)
    // Navigate to detector training page to see progress
    router.push(`/project/${projectId.value}/detector`)
  } catch (e: any) {
    console.error('Failed to start detector training:', e)
    alert(e.message || 'Failed to start detector training')
  }
}
```

**Step 2: Add button to template**

Add in the operations section (find where alignment train button is, add nearby):
```vue
<button
  @click="handleTrainDetector"
  :disabled="isDetectorTraining"
  class="action-btn"
>
  {{ isDetectorTraining ? 'Training Detector...' : 'Train Detector' }}
</button>

<button
  v-if="detectorModelExists || isDetectorTraining"
  @click="router.push(`/project/${projectId}/detector`)"
  class="action-btn secondary"
>
  View Detector Training
</button>
```

**Step 3: Commit**

```bash
git add frontend/src/components/VideoPipeline.vue
git commit -m "feat: add Train Detector button to pipeline"
```

---

## Task 12: Add Mask View Dropdown to VideoDetail

**Files:**
- Modify: `frontend/src/components/VideoDetail.vue`

**Step 1: Add mask view mode state**

Add ref:
```typescript
type MaskViewMode = 'tracker' | 'detector'
const maskViewMode = ref<MaskViewMode>('tracker')
```

Add import for detectorMasksExist:
```typescript
import { detectorMasksExist } from '@/services/api'
```

Add state for detector masks availability:
```typescript
const hasDetectorMasks = ref(false)

const checkDetectorMasks = async () => {
  if (!projectId.value || !videoId.value) return
  try {
    const result = await detectorMasksExist(projectId.value, videoId.value)
    hasDetectorMasks.value = result.exists
  } catch {
    hasDetectorMasks.value = false
  }
}
```

Call in onMounted and when video changes.

**Step 2: Add dropdown to template**

Add in toolbar area:
```vue
<select v-model="maskViewMode" class="mask-view-select">
  <option value="tracker">Tracker</option>
  <option value="detector" :disabled="!hasDetectorMasks">
    Detector {{ hasDetectorMasks ? '' : '(not available)' }}
  </option>
</select>
```

**Step 3: Commit**

```bash
git add frontend/src/components/VideoDetail.vue
git commit -m "feat: add mask view mode dropdown to VideoDetail"
```

---

## Task 13: Update useSegmentation for Mask View Mode

**Files:**
- Modify: `frontend/src/composables/useSegmentation.ts`

**Step 1: Add maskViewMode parameter**

Update function signature to accept maskViewMode:
```typescript
export function useSegmentation(
    projectId: Ref<number | null>,
    videoId: Ref<number | null>,
    currentFrameIdx: Ref<number>,
    isPlaying: Ref<boolean> = ref(false),
    videoRef: Ref<HTMLVideoElement | null> = ref(null),
    fps: Ref<number> = ref(30),
    maskViewMode: Ref<'tracker' | 'detector'> = ref('tracker'),
): UseSegmentationReturn {
```

**Step 2: Update mask fetching to use correct endpoint**

Import getDetectorMask:
```typescript
import { getDetectorMask } from '@/services/api'
```

Update the mask fetching logic to check maskViewMode and call appropriate endpoint:
```typescript
const fetchMask = async (frameIdx: number): Promise<Blob | null> => {
    if (!projectId.value || !videoId.value) return null

    try {
        if (maskViewMode.value === 'detector') {
            return await getDetectorMask(projectId.value, videoId.value, frameIdx)
        } else {
            return await getMask(projectId.value, videoId.value, frameIdx)
        }
    } catch {
        return null
    }
}
```

**Step 3: Clear cache when view mode changes**

Add watcher:
```typescript
watch(maskViewMode, () => {
    maskCache.clear()
    prefetchedUpTo = -1
    loadFrameData(currentFrameIdx.value)
})
```

**Step 4: Commit**

```bash
git add frontend/src/composables/useSegmentation.ts
git commit -m "feat: support mask view mode in useSegmentation"
```

---

## Task 14: Wire Up Mask View Mode in VideoDetail

**Files:**
- Modify: `frontend/src/components/VideoDetail.vue`

**Step 1: Pass maskViewMode to useSegmentation**

Update the useSegmentation call to pass maskViewMode:
```typescript
const {
  activeTool,
  currentMask,
  // ... other destructured values
} = useSegmentation(
  projectId,
  videoId,
  currentFrameIdx,
  isPlaying,
  videoRef,
  fps,
  maskViewMode,  // Add this parameter
)
```

**Step 2: Commit**

```bash
git add frontend/src/components/VideoDetail.vue
git commit -m "feat: wire mask view mode to useSegmentation"
```

---

## Task 15: Register Detector Routes in Backend

**Files:**
- Modify: `vidseq/api/main.py` (or equivalent)

**Step 1: Find where routes are registered and add detector router**

Find the file and add:
```python
from vidseq.api.routes.detector import router as detector_router

# In the route registration section:
app.include_router(detector_router, tags=["detector"])
```

**Step 2: Commit**

```bash
git add vidseq/api/main.py
git commit -m "feat: register detector routes in main app"
```

---

## Task 16: Final Integration Test

**Step 1: Start backend and frontend**

```bash
# Terminal 1: Backend
vidseq

# Terminal 2: Frontend
cd frontend && npm run dev
```

**Step 2: Test workflow**

1. Open a project with videos
2. Add point prompts and propagate to generate tracker masks
3. Mark training ranges via DataTrack
4. Click "Train Detector" in VideoPipeline
5. Verify navigation to DetectorTraining page
6. Watch loss chart update in real-time
7. Wait for training to complete (or early stop)
8. Verify "applying" status shows progress
9. Navigate to VideoDetail
10. Verify "Detector" option appears in dropdown
11. Switch to "Detector" view
12. Verify detector masks display on training frames

**Step 3: Commit any fixes**

```bash
git add -A
git commit -m "fix: integration fixes for detector workflow"
```

---

## Summary

This plan implements:

1. **Backend:** DINOv2Detector model, DetectorService with training/apply, API routes
2. **Frontend:** useDetector composable, DetectorTraining.vue, mask view dropdown
3. **Integration:** Wire everything together, test end-to-end

Total: 16 tasks, approximately 2-3 hours of implementation time.
