"""
UNet Worker Process.

Runs in a separate process to avoid CUDA/signal conflicts with FastAPI.
Communicates with the main process via multiprocessing queues.

Handles UNet training and inference.
"""

import sys
import random
from pathlib import Path
from typing import Any, Dict
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

import segmentation_models_pytorch as smp


def worker_loop(command_queue, result_queue):
    """
    Main loop for UNet worker process.
    
    Receives commands from command_queue, sends results to result_queue.
    """
    print("[UNet Worker] Starting worker process...")
    
    while True:
        try:
            cmd = command_queue.get()
        except KeyboardInterrupt:
            print("[UNet Worker] Interrupted, shutting down...")
            break
        
        cmd_type = cmd.get("type")
        request_id = cmd.get("request_id")
        
        if cmd_type == "shutdown":
            print("[UNet Worker] Shutting down...")
            break
        
        elif cmd_type == "train_model":
            project_path = Path(cmd["project_path"])
            try:
                _train_model(project_path)
                result_queue.put({
                    "type": "train_model_result",
                    "request_id": request_id,
                    "status": "ok",
                })
            except Exception as e:
                print(f"[UNet Worker] Training failed: {e}")
                import traceback
                traceback.print_exc()
                result_queue.put({
                    "type": "train_model_result",
                    "request_id": request_id,
                    "status": "error",
                    "error": str(e),
                })
        
        elif cmd_type == "apply_model":
            project_path = Path(cmd["project_path"])
            video_id = cmd["video_id"]
            try:
                _apply_model(project_path, video_id)
                result_queue.put({
                    "type": "apply_model_result",
                    "request_id": request_id,
                    "status": "ok",
                })
            except Exception as e:
                print(f"[UNet Worker] Application failed: {e}")
                import traceback
                traceback.print_exc()
                result_queue.put({
                    "type": "apply_model_result",
                    "request_id": request_id,
                    "status": "error",
                    "error": str(e),
                })
        
        elif cmd_type == "test_apply_model":
            project_path = Path(cmd["project_path"])
            video_id = cmd["video_id"]
            start_frame = cmd["start_frame"]
            try:
                _test_apply_model(project_path, video_id, start_frame)
                result_queue.put({
                    "type": "test_apply_model_result",
                    "request_id": request_id,
                    "status": "ok",
                })
            except Exception as e:
                print(f"[UNet Worker] Test application failed: {e}")
                import traceback
                traceback.print_exc()
                result_queue.put({
                    "type": "test_apply_model_result",
                    "request_id": request_id,
                    "status": "error",
                    "error": str(e),
                })


_video_cache: Dict[Path, cv2.VideoCapture] = {}


def _get_video_capture(video_path: Path) -> cv2.VideoCapture:
    """Get or create a cached VideoCapture object for a video."""
    if video_path not in _video_cache:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        _video_cache[video_path] = cap
    return _video_cache[video_path]


def _load_video_frame(video_path: Path, frame_idx: int, current_pos: int = None) -> np.ndarray:
    """
    Load a raw RGB frame from video.
    
    Args:
        video_path: Path to video file
        frame_idx: Frame index to load
        current_pos: Current position of VideoCapture (for sequential reading optimization)
    
    Returns:
        RGB frame as numpy array
    """
    cap = _get_video_capture(video_path)
    
    if current_pos is None or current_pos != frame_idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    
    ret, frame = cap.read()
    
    if not ret:
        raise RuntimeError(f"Failed to read frame {frame_idx} from {video_path}")
    
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _clear_video_cache():
    """Close all cached video files."""
    for cap in _video_cache.values():
        cap.release()
    _video_cache.clear()


def _build_training_dataset(project_path: Path):
    """
    Build training dataset from all videos in project.
    
    Returns list of (video_id, frame_idx, rgb_path, prev_mask_path, target_mask_path) tuples.
    """
    from vidseq.services import mask_service
    from vidseq.services.database_manager import DatabaseManager
    
    db = DatabaseManager.get_instance()
    db_path = project_path / "vidseq.db"
    
    if not db_path.exists():
        raise RuntimeError(f"Project database not found at {db_path}")
    
    import sqlalchemy
    from sqlalchemy import create_engine, select
    from sqlalchemy.orm import sessionmaker
    from vidseq.models.video import Video
    
    engine = create_engine(f"sqlite:///{db_path}")
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        result = session.execute(select(Video).order_by(Video.id))
        videos = result.scalars().all()
    finally:
        session.close()
    
    training_samples = []
    
    for video in videos:
        video_path = Path(video.path)
        training_frames = mask_service.get_training_frames(
            project_path=project_path,
            video_id=video.id,
            num_frames=video.num_frames,
        )
        
        for frame_idx in training_frames:
            prev_frame_idx = frame_idx - 1 if frame_idx > 0 else None
            training_samples.append({
                "video_id": video.id,
                "video_path": video_path,
                "frame_idx": frame_idx,
                "prev_frame_idx": prev_frame_idx,
                "num_frames": video.num_frames,
                "height": video.height,
                "width": video.width,
            })
    
    return training_samples


def _apply_horizontal_flip(rgb, prev_mask, target_mask):
    """Flip all tensors horizontally (left-right)."""
    rgb_flipped = torch.flip(rgb, dims=[2])
    prev_mask_flipped = torch.flip(prev_mask, dims=[2])
    target_mask_flipped = torch.flip(target_mask, dims=[2])
    return rgb_flipped, prev_mask_flipped, target_mask_flipped


def _apply_rotation_180(rgb, prev_mask, target_mask):
    """Rotate all tensors 180 degrees."""
    rgb_rotated = torch.rot90(rgb, k=2, dims=[1, 2])
    prev_mask_rotated = torch.rot90(prev_mask, k=2, dims=[1, 2])
    target_mask_rotated = torch.rot90(target_mask, k=2, dims=[1, 2])
    return rgb_rotated, prev_mask_rotated, target_mask_rotated


def _apply_color_augmentation(rgb):
    """
    Apply aggressive color/brightness/contrast augmentations.
    Brightness: ±40% (multiply by 0.6-1.4)
    Contrast: ±40% (adjust contrast)
    Hue/Saturation: Random HSV shifts
    """
    rgb_aug = rgb.clone()
    
    brightness_factor = random.uniform(0.6, 1.4)
    rgb_aug = rgb_aug * brightness_factor
    rgb_aug = torch.clamp(rgb_aug, 0.0, 1.0)
    
    contrast_factor = random.uniform(0.6, 1.4)
    mean = rgb_aug.mean(dim=(1, 2), keepdim=True)
    rgb_aug = (rgb_aug - mean) * contrast_factor + mean
    rgb_aug = torch.clamp(rgb_aug, 0.0, 1.0)
    
    hue_shift = random.uniform(-0.1, 0.1)
    saturation_factor = random.uniform(0.6, 1.4)
    
    rgb_aug_np = rgb_aug.permute(1, 2, 0).cpu().numpy()
    rgb_aug_np_uint8 = (rgb_aug_np * 255.0).astype(np.uint8)
    hsv = cv2.cvtColor(rgb_aug_np_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)
    
    hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift * 180) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_factor, 0, 255)
    
    rgb_aug_np_uint8 = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    rgb_aug = torch.from_numpy(rgb_aug_np_uint8).permute(2, 0, 1).float() / 255.0
    
    return rgb_aug


def _apply_scale_augmentation(rgb, prev_mask, target_mask, scale_range=(0.8, 1.2)):
    """
    Apply random crop/zoom augmentation.
    Scale range: 0.8-1.2 (zoom out to zoom in)
    """
    _, h, w = rgb.shape
    scale = random.uniform(scale_range[0], scale_range[1])
    
    new_h = int(h * scale)
    new_w = int(w * scale)
    
    if scale < 1.0:
        rgb_scaled = F.interpolate(rgb.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False).squeeze(0)
        prev_mask_scaled = F.interpolate(prev_mask.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False).squeeze(0)
        target_mask_scaled = F.interpolate(target_mask.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False).squeeze(0)
        
        top = random.randint(0, h - new_h)
        left = random.randint(0, w - new_w)
        
        rgb_padded = torch.zeros_like(rgb)
        rgb_padded[:, top:top+new_h, left:left+new_w] = rgb_scaled
        
        prev_mask_padded = torch.zeros_like(prev_mask)
        prev_mask_padded[:, top:top+new_h, left:left+new_w] = prev_mask_scaled
        
        target_mask_padded = torch.zeros_like(target_mask)
        target_mask_padded[:, top:top+new_h, left:left+new_w] = target_mask_scaled
        
        return rgb_padded, prev_mask_padded, target_mask_padded
    else:
        rgb_scaled = F.interpolate(rgb.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False).squeeze(0)
        prev_mask_scaled = F.interpolate(prev_mask.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False).squeeze(0)
        target_mask_scaled = F.interpolate(target_mask.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False).squeeze(0)
        
        top = random.randint(0, new_h - h)
        left = random.randint(0, new_w - w)
        
        rgb_cropped = rgb_scaled[:, top:top+h, left:left+w]
        prev_mask_cropped = prev_mask_scaled[:, top:top+h, left:left+w]
        target_mask_cropped = target_mask_scaled[:, top:top+h, left:left+w]
        
        return rgb_cropped, prev_mask_cropped, target_mask_cropped


def _random_empty_prev_mask():
    """Return True with 50% probability for empty prev_mask augmentation."""
    return random.random() < 0.5


def _apply_prev_mask_perturbation(prev_mask, scale_range=(0.9, 1.1), max_shift=0.05, max_rotation=10):
    """
    Apply small random transformations to previous mask to teach model to correct mistakes.
    Rotation is performed around the center of mass of the mask, not the image center.
    
    Args:
        prev_mask: Previous mask tensor (1, H, W)
        scale_range: Range for random scaling (e.g., 0.9-1.1 means ±10%)
        max_shift: Maximum shift as fraction of image size (e.g., 0.05 = 5%)
        max_rotation: Maximum rotation in degrees
    
    Returns:
        Perturbed previous mask tensor
    """
    _, h, w = prev_mask.shape
    
    scale = random.uniform(scale_range[0], scale_range[1])
    shift_x = random.uniform(-max_shift, max_shift) * w
    shift_y = random.uniform(-max_shift, max_shift) * h
    rotation = random.uniform(-max_rotation, max_rotation)
    
    angle_rad = rotation * np.pi / 180.0
    
    mask_sum = prev_mask.sum()
    if mask_sum > 0:
        y_coords, x_coords = torch.meshgrid(
            torch.arange(h, dtype=torch.float32, device=prev_mask.device),
            torch.arange(w, dtype=torch.float32, device=prev_mask.device),
            indexing='ij'
        )
        center_x = (x_coords * prev_mask.squeeze(0)).sum() / mask_sum
        center_y = (y_coords * prev_mask.squeeze(0)).sum() / mask_sum
    else:
        center_x, center_y = w / 2.0, h / 2.0
    
    cos_a = scale * np.cos(angle_rad)
    sin_a = scale * np.sin(angle_rad)
    
    tx = center_x + shift_x - center_x * cos_a + center_y * sin_a
    ty = center_y + shift_y - center_x * sin_a - center_y * cos_a
    
    theta = torch.tensor([
        [cos_a, -sin_a, tx],
        [sin_a, cos_a, ty]
    ], dtype=torch.float32)
    
    grid = F.affine_grid(theta.unsqueeze(0), prev_mask.unsqueeze(0).shape, align_corners=False)
    prev_mask_perturbed = F.grid_sample(
        prev_mask.unsqueeze(0),
        grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=False
    ).squeeze(0)
    
    return prev_mask_perturbed


class VideoMaskDataset(torch.utils.data.Dataset):
    """Dataset for training UNet on video frames + previous masks."""
    
    def __init__(self, samples, project_path: Path, apply_augmentation: bool = True):
        self.samples = samples
        self.project_path = project_path
        self.apply_augmentation = apply_augmentation
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        from vidseq.services import mask_service
        
        sample = self.samples[idx]
        video_path = sample["video_path"]
        frame_idx = sample["frame_idx"]
        prev_frame_idx = sample["prev_frame_idx"]
        height = sample["height"]
        width = sample["width"]
        
        rgb_frame = _load_video_frame(video_path, frame_idx)
        rgb_tensor = torch.from_numpy(rgb_frame).permute(2, 0, 1).float() / 255.0
        
        if prev_frame_idx is not None:
            prev_mask = mask_service.load_mask(
                project_path=self.project_path,
                video_id=sample["video_id"],
                frame_idx=prev_frame_idx,
                num_frames=sample["num_frames"],
                height=height,
                width=width,
            )
            prev_mask_tensor = torch.from_numpy(prev_mask).float().unsqueeze(0) / 255.0
        else:
            prev_mask_tensor = torch.zeros((1, height, width), dtype=torch.float32)
        
        target_mask = mask_service.load_mask(
            project_path=self.project_path,
            video_id=sample["video_id"],
            frame_idx=frame_idx,
            num_frames=sample["num_frames"],
            height=height,
            width=width,
        )
        target_mask_tensor = torch.from_numpy(target_mask).float().unsqueeze(0) / 255.0
        
        if self.apply_augmentation:
            if prev_frame_idx is not None:
                if _random_empty_prev_mask():
                    prev_mask_tensor = torch.zeros_like(prev_mask_tensor)
                elif random.random() < 0.5:
                    prev_mask_tensor = _apply_prev_mask_perturbation(prev_mask_tensor)
            
            geometric_aug = random.choice(['none', 'flip', 'rotate', 'scale'])
            
            if geometric_aug == 'flip':
                rgb_tensor, prev_mask_tensor, target_mask_tensor = _apply_horizontal_flip(
                    rgb_tensor, prev_mask_tensor, target_mask_tensor
                )
            elif geometric_aug == 'rotate':
                rgb_tensor, prev_mask_tensor, target_mask_tensor = _apply_rotation_180(
                    rgb_tensor, prev_mask_tensor, target_mask_tensor
                )
            elif geometric_aug == 'scale':
                rgb_tensor, prev_mask_tensor, target_mask_tensor = _apply_scale_augmentation(
                    rgb_tensor, prev_mask_tensor, target_mask_tensor
                )
            
            if random.random() < 0.8:
                rgb_tensor = _apply_color_augmentation(rgb_tensor)
        
        input_tensor = torch.cat([rgb_tensor, prev_mask_tensor], dim=0)
        
        return input_tensor, target_mask_tensor


def dice_loss(pred, target, smooth=1e-6):
    """
    Compute Dice loss for binary segmentation.
    
    Args:
        pred: Predicted logits (before sigmoid)
        target: Ground truth masks (0 or 1)
        smooth: Smoothing factor to avoid division by zero
    
    Returns:
        Dice loss (1 - Dice coefficient)
    """
    pred = torch.sigmoid(pred)
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum()
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice


def combined_bce_dice_loss(pred_logits, target):
    """
    Combined BCE + Dice loss with 1:1 weighting.
    
    Args:
        pred_logits: Predicted logits (before sigmoid)
        target: Ground truth masks (0 or 1)
    
    Returns:
        Combined loss value
    """
    bce_loss = nn.functional.binary_cross_entropy_with_logits(pred_logits, target)
    dice = dice_loss(pred_logits, target)
    return bce_loss + dice


def _evaluate_model(model, dataloader, criterion, device):
    """Evaluate model on a dataset."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_inputs, batch_targets in dataloader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_targets)
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    model.train()
    return avg_loss


def _train_model(project_path: Path):
    """Train UNet model on project data."""
    print("[UNet Worker] Building training dataset...")
    samples = _build_training_dataset(project_path)
    
    if len(samples) == 0:
        raise RuntimeError("No training frames found. Segment some frames first.")
    
    print(f"[UNet Worker] Found {len(samples)} training samples")
    
    train_size = int(0.8 * len(samples))
    test_size = len(samples) - train_size
    
    indices = list(range(len(samples)))
    generator = torch.Generator().manual_seed(42)
    train_indices = torch.randperm(len(samples), generator=generator)[:train_size].tolist()
    test_indices = [i for i in indices if i not in train_indices]
    
    train_samples = [samples[i] for i in train_indices]
    test_samples = [samples[i] for i in test_indices]
    
    print(f"[UNet Worker] Split: {len(train_samples)} train, {len(test_samples)} test samples")
    
    train_dataset = VideoMaskDataset(train_samples, project_path, apply_augmentation=True)
    test_dataset = VideoMaskDataset(test_samples, project_path, apply_augmentation=False)
    
    train_sampler = torch.utils.data.WeightedRandomSampler(
        weights=torch.ones(len(train_dataset)),
        num_samples=len(train_dataset) * 5,
        replacement=True
    )
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=64,
        sampler=train_sampler,
        num_workers=0,
        pin_memory=True,
    )
    
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[UNet Worker] Using device: {device}")
    
    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights="imagenet",
        in_channels=4,
        classes=1,
    ).to(device)
    
    criterion = combined_bce_dice_loss
    initial_lr = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
    )
    
    max_epochs = 100
    early_stopping_patience = 5
    best_test_loss = float('inf')
    no_improve_count = 0
    best_model_state = None
    
    model.train()
    
    print(f"[UNet Worker] Starting training with adaptive LR (initial: {initial_lr})...")
    for epoch in range(max_epochs):
        epoch_train_loss = 0.0
        num_batches = 0
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{max_epochs}")
        
        for batch_inputs, batch_targets in pbar:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"train_loss": loss.item()})
        
        avg_train_loss = epoch_train_loss / num_batches if num_batches > 0 else 0.0
        
        test_loss = _evaluate_model(model, test_dataloader, criterion, device)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(
            f"[UNet Worker] Epoch {epoch+1}/{max_epochs}, "
            f"Train Loss: {avg_train_loss:.4f}, Test Loss: {test_loss:.4f}, "
            f"LR: {current_lr:.2e}"
        )
        
        scheduler.step(test_loss)
        
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            no_improve_count = 0
            best_model_state = model.state_dict().copy()
            print(f"[UNet Worker] New best test loss: {best_test_loss:.4f}")
        else:
            no_improve_count += 1
            print(f"[UNet Worker] No improvement for {no_improve_count}/{early_stopping_patience} epochs")
        
        if no_improve_count >= early_stopping_patience:
            print(
                f"[UNet Worker] Early stopping triggered after {epoch+1} epochs. "
                f"Best test loss: {best_test_loss:.4f}"
            )
            break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("[UNet Worker] Loaded best model checkpoint")
    
    model_path = project_path / "segmentation_student_unet.pth"
    torch.save(model.state_dict(), model_path)
    print(f"[UNet Worker] Model saved to {model_path}")
    
    _clear_video_cache()


def _apply_model(project_path: Path, video_id: int):
    """Apply trained UNet model to a video."""
    model_path = project_path / "segmentation_student_unet.pth"
    
    if not model_path.exists():
        raise RuntimeError(f"Model not found at {model_path}. Train model first.")
    
    from vidseq.services import mask_service
    from vidseq.services.database_manager import DatabaseManager
    
    db = DatabaseManager.get_instance()
    db_path = project_path / "vidseq.db"
    
    import sqlalchemy
    from sqlalchemy import create_engine, select
    from sqlalchemy.orm import sessionmaker
    from vidseq.models.video import Video
    
    engine = create_engine(f"sqlite:///{db_path}")
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        result = session.execute(select(Video).where(Video.id == video_id))
        video = result.scalar_one_or_none()
        if not video:
            raise RuntimeError(f"Video {video_id} not found")
    finally:
        session.close()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[UNet Worker] Using device: {device}")
    
    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights="imagenet",
        in_channels=4,
        classes=1,
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    video_path = Path(video.path)
    
    print(f"[UNet Worker] Applying model to video {video_id}...")
    
    prev_mask = None
    
    with torch.no_grad():
        for frame_idx in tqdm(range(video.num_frames), desc="Applying model"):
            current_mask = mask_service.load_mask(
                project_path=project_path,
                video_id=video_id,
                frame_idx=frame_idx,
                num_frames=video.num_frames,
                height=video.height,
                width=video.width,
            )
            
            if current_mask.sum() > 0:
                prev_mask = current_mask
                continue
            
            rgb_frame = _load_video_frame(video_path, frame_idx)
            rgb_tensor = torch.from_numpy(rgb_frame).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            
            if prev_mask is not None:
                prev_mask_tensor = torch.from_numpy(prev_mask).float().unsqueeze(0).unsqueeze(0) / 255.0
            else:
                prev_mask_tensor = torch.zeros((1, 1, video.height, video.width), dtype=torch.float32)
            
            input_tensor = torch.cat([rgb_tensor, prev_mask_tensor], dim=1).to(device)
            
            output = model(input_tensor)
            predicted_mask = torch.sigmoid(output).squeeze().cpu().numpy()
            predicted_mask_binary = (predicted_mask > 0.5).astype(np.uint8) * 255
            
            mask_service.save_mask(
                project_path=project_path,
                video_id=video_id,
                frame_idx=frame_idx,
                mask=predicted_mask_binary,
                num_frames=video.num_frames,
                height=video.height,
                width=video.width,
            )
            
            prev_mask = predicted_mask_binary
    
    print(f"[UNet Worker] Finished applying model to video {video_id}")


def _test_apply_model(project_path: Path, video_id: int, start_frame: int):
    """Apply trained UNet model to a limited range of frames (for testing)."""
    model_path = project_path / "segmentation_student_unet.pth"
    
    if not model_path.exists():
        raise RuntimeError(f"Model not found at {model_path}. Train model first.")
    
    from vidseq.services import mask_service
    from vidseq.services.database_manager import DatabaseManager
    
    db = DatabaseManager.get_instance()
    db_path = project_path / "vidseq.db"
    
    import sqlalchemy
    from sqlalchemy import create_engine, select
    from sqlalchemy.orm import sessionmaker
    from vidseq.models.video import Video
    
    engine = create_engine(f"sqlite:///{db_path}")
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        result = session.execute(select(Video).where(Video.id == video_id))
        video = result.scalar_one_or_none()
        if not video:
            raise RuntimeError(f"Video {video_id} not found")
    finally:
        session.close()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[UNet Worker] Using device: {device}")
    
    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights="imagenet",
        in_channels=4,
        classes=1,
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    video_path = Path(video.path)
    end_frame = min(start_frame + 1000, video.num_frames)
    
    print(f"[UNet Worker] Test applying model to video {video_id}, frames {start_frame} to {end_frame-1}...")
    
    prev_mask = None
    
    frames_processed = 0
    frames_skipped_train = 0
    frames_skipped_existing = 0
    frames_with_predictions = 0
    
    with torch.no_grad():
        for frame_idx in tqdm(range(start_frame, end_frame), desc="Test applying model"):
            frame_type = mask_service.get_frame_type(
                project_path=project_path,
                video_id=video_id,
                frame_idx=frame_idx,
                num_frames=video.num_frames,
            )
            
            if frame_type == "train":
                frames_skipped_train += 1
                continue
            
            current_mask = mask_service.load_mask(
                project_path=project_path,
                video_id=video_id,
                frame_idx=frame_idx,
                num_frames=video.num_frames,
                height=video.height,
                width=video.width,
            )
            
            if current_mask.sum() > 0:
                prev_mask = current_mask
                frames_skipped_existing += 1
                continue
            
            rgb_frame = _load_video_frame(video_path, frame_idx)
            rgb_tensor = torch.from_numpy(rgb_frame).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            
            if prev_mask is not None:
                prev_mask_tensor = torch.from_numpy(prev_mask).float().unsqueeze(0).unsqueeze(0) / 255.0
            else:
                prev_mask_tensor = torch.zeros((1, 1, video.height, video.width), dtype=torch.float32)
            
            input_tensor = torch.cat([rgb_tensor, prev_mask_tensor], dim=1).to(device)
            
            output = model(input_tensor)
            predicted_mask = torch.sigmoid(output).squeeze().cpu().numpy()
            predicted_mask_binary = (predicted_mask > 0.5).astype(np.uint8) * 255
            
            frames_processed += 1
            
            if frame_idx == start_frame or frame_idx % 100 == 0 or frames_processed <= 10:
                pixels_above_threshold = (predicted_mask > 0.5).sum()
                print(
                    f"[UNet Worker] Frame {frame_idx}: "
                    f"pred_max={predicted_mask.max():.4f}, "
                    f"pred_mean={predicted_mask.mean():.4f}, "
                    f"pred_min={predicted_mask.min():.4f}, "
                    f"pixels_above_0.5={pixels_above_threshold}, "
                    f"binary_sum={predicted_mask_binary.sum()}"
                )
            
            if predicted_mask_binary.sum() > 0:
                frames_with_predictions += 1
            
            mask_service.save_mask(
                project_path=project_path,
                video_id=video_id,
                frame_idx=frame_idx,
                mask=predicted_mask_binary,
                num_frames=video.num_frames,
                height=video.height,
                width=video.width,
            )
            
            prev_mask = predicted_mask_binary
    
    print(
        f"[UNet Worker] Test apply summary: "
        f"processed={frames_processed}, "
        f"skipped_train={frames_skipped_train}, "
        f"skipped_existing={frames_skipped_existing}, "
        f"frames_with_predictions={frames_with_predictions}"
    )
    
    print(f"[UNet Worker] Finished test applying model to video {video_id}")

