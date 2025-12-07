"""YOLO dataset preparation utilities."""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional

from vidseq.services import mask_service


def pixel_to_yolo(x1: float, y1: float, x2: float, y2: float, img_width: int, img_height: int) -> tuple[float, float, float, float]:
    """
    Convert bounding box from pixel coordinates [x1, y1, x2, y2] to YOLO format.
    
    YOLO format: (center_x, center_y, width, height) normalized to [0, 1]
    
    Args:
        x1, y1, x2, y2: Pixel coordinates (top-left and bottom-right corners)
        img_width, img_height: Image dimensions
        
    Returns:
        Tuple of (center_x, center_y, width, height) normalized to [0, 1]
    """
    center_x = ((x1 + x2) / 2) / img_width
    center_y = ((y1 + y2) / 2) / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    
    # Clamp to [0, 1] range
    center_x = max(0.0, min(1.0, center_x))
    center_y = max(0.0, min(1.0, center_y))
    width = max(0.0, min(1.0, width))
    height = max(0.0, min(1.0, height))
    
    return center_x, center_y, width, height


def yolo_to_pixel(center_x: float, center_y: float, width: float, height: float, img_width: int, img_height: int) -> tuple[float, float, float, float]:
    """
    Convert bounding box from YOLO format to pixel coordinates [x1, y1, x2, y2].
    
    Args:
        center_x, center_y, width, height: YOLO format normalized to [0, 1]
        img_width, img_height: Image dimensions
        
    Returns:
        Tuple of (x1, y1, x2, y2) in pixel coordinates
    """
    x_center = center_x * img_width
    y_center = center_y * img_height
    w = width * img_width
    h = height * img_height
    
    x1 = x_center - w / 2
    y1 = y_center - h / 2
    x2 = x_center + w / 2
    y2 = y_center + h / 2
    
    return x1, y1, x2, y2


def extract_frame(video_path: Path, frame_idx: int) -> Optional[np.ndarray]:
    """
    Extract a single frame from a video file.
    
    Args:
        video_path: Path to video file
        frame_idx: Frame index to extract
        
    Returns:
        RGB image as numpy array (H, W, 3) or None if extraction fails
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return None
    
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame_rgb


TrainingSample = tuple[int, int, int, Path, int, int]
"""
Training sample tuple: (video_id, frame_idx, img_width, img_height, video_path, bbox_x1, bbox_y1, bbox_x2, bbox_y2)
Actually: (video_id, frame_idx, img_width, img_height, video_path, x1, y1, x2, y2)
"""


def collect_training_samples(
    project_path: Path,
    videos: list["Video"],
) -> list[TrainingSample]:
    """
    Collect all training samples from videos in the project.
    
    Args:
        project_path: Path to the project folder
        videos: List of Video model instances
        
    Returns:
        List of training samples, each as (video_id, frame_idx, img_width, img_height, video_path, x1, y1, x2, y2)
    """
    samples = []
    
    with mask_service.open_h5(project_path, 'r') as h5_file:
        for video in videos:
            video_path = Path(video.path)
            if not video_path.exists():
                continue
            
            # Get all training frames for this video
            training_frames = mask_service.get_training_frames(
                project_path=project_path,
                video_id=video.id,
                num_frames=video.num_frames,
            )
            
            # For each training frame, get the bbox
            for frame_idx in training_frames:
                bbox = mask_service.load_bbox(
                    project_path=project_path,
                    video_id=video.id,
                    frame_idx=frame_idx,
                    num_frames=video.num_frames,
                    h5_file=h5_file,
                )
                
                if bbox is None:
                    continue
                
                # Validate bbox
                x1, y1, x2, y2 = bbox
                if x1 >= x2 or y1 >= y2:
                    continue
                if x1 < 0 or y1 < 0 or x2 > video.width or y2 > video.height:
                    continue
                
                samples.append((
                    video.id,
                    frame_idx,
                    video.width,
                    video.height,
                    video_path,
                    float(x1),
                    float(y1),
                    float(x2),
                    float(y2),
                ))
    
    return samples


def create_yolo_dataset(
    project_path: Path,
    videos: list["Video"],
    output_dir: Optional[Path] = None,
) -> Path:
    """
    Create YOLO dataset structure from training samples.
    
    Args:
        project_path: Path to the project folder
        videos: List of Video model instances
        output_dir: Optional output directory (defaults to {project_path}/yolo_dataset)
        
    Returns:
        Path to the created dataset directory
        
    Raises:
        RuntimeError: If no training samples found
    """
    if output_dir is None:
        output_dir = project_path / "yolo_dataset"
    
    output_dir = Path(output_dir)
    images_train_dir = output_dir / "images" / "train"
    labels_train_dir = output_dir / "labels" / "train"
    images_val_dir = output_dir / "images" / "val"
    labels_val_dir = output_dir / "labels" / "val"
    
    images_train_dir.mkdir(parents=True, exist_ok=True)
    labels_train_dir.mkdir(parents=True, exist_ok=True)
    images_val_dir.mkdir(parents=True, exist_ok=True)
    labels_val_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect training samples
    samples = collect_training_samples(project_path, videos)
    
    if len(samples) == 0:
        raise RuntimeError("No training samples found. Segment some frames first.")
    
    print(f"[YOLO Dataset] Found {len(samples)} training samples")
    
    # Split into train/val (80/20)
    import random
    random.seed(42)
    random.shuffle(samples)
    split_idx = int(0.8 * len(samples))
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]
    
    print(f"[YOLO Dataset] Split: {len(train_samples)} train, {len(val_samples)} val")
    
    # Process train samples
    for idx, (video_id, frame_idx, img_width, img_height, video_path, x1, y1, x2, y2) in enumerate(train_samples):
        # Extract frame
        frame = extract_frame(video_path, frame_idx)
        if frame is None:
            print(f"[YOLO Dataset] Warning: Failed to extract frame {frame_idx} from video {video_id}")
            continue
        
        # Save image
        image_filename = f"{video_id}_{frame_idx:06d}.jpg"
        image_path = images_train_dir / image_filename
        cv2.imwrite(str(image_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
        # Convert bbox to YOLO format
        center_x, center_y, width, height = pixel_to_yolo(x1, y1, x2, y2, img_width, img_height)
        
        # Save label (class_id = 0 for single object class)
        label_filename = f"{video_id}_{frame_idx:06d}.txt"
        label_path = labels_train_dir / label_filename
        with open(label_path, 'w') as f:
            f.write(f"0 {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
        
        if (idx + 1) % 100 == 0:
            print(f"[YOLO Dataset] Processed {idx + 1}/{len(train_samples)} train samples...")
    
    # Process val samples
    for idx, (video_id, frame_idx, img_width, img_height, video_path, x1, y1, x2, y2) in enumerate(val_samples):
        # Extract frame
        frame = extract_frame(video_path, frame_idx)
        if frame is None:
            print(f"[YOLO Dataset] Warning: Failed to extract frame {frame_idx} from video {video_id}")
            continue
        
        # Save image
        image_filename = f"{video_id}_{frame_idx:06d}.jpg"
        image_path = images_val_dir / image_filename
        cv2.imwrite(str(image_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
        # Convert bbox to YOLO format
        center_x, center_y, width, height = pixel_to_yolo(x1, y1, x2, y2, img_width, img_height)
        
        # Save label (class_id = 0 for single object class)
        label_filename = f"{video_id}_{frame_idx:06d}.txt"
        label_path = labels_val_dir / label_filename
        with open(label_path, 'w') as f:
            f.write(f"0 {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
        
        if (idx + 1) % 100 == 0:
            print(f"[YOLO Dataset] Processed {idx + 1}/{len(val_samples)} val samples...")
    
    print(f"[YOLO Dataset] Created dataset at {output_dir}")
    return output_dir

