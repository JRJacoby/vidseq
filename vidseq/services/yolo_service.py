"""YOLO Service for training and inference using Ultralytics YOLOv11-nano."""

import threading
from pathlib import Path
from typing import Optional

from ultralytics import YOLO


class YOLOService:
    """
    Singleton service for managing YOLO training and inference.
    
    Uses Ultralytics YOLOv11-nano for bounding box detection.
    """
    
    _instance: Optional["YOLOService"] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> "YOLOService":
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
        self._is_applying = False
        self._training_thread: Optional[threading.Thread] = None
        self._applying_thread: Optional[threading.Thread] = None
        
        self._initialized = True
    
    @classmethod
    def get_instance(cls) -> "YOLOService":
        """Get the singleton instance."""
        return cls()
    
    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton (useful for testing)."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance = None
    
    def train_model(self, project_path: Path) -> None:
        """
        Start training YOLO model for a project.
        
        Args:
            project_path: Path to the project folder
            
        Raises:
            RuntimeError: If training is already in progress
        """
        if self._is_training:
            raise RuntimeError("Training already in progress")
        
        self._is_training = True
        
        def _train():
            try:
                from vidseq.services import yolo_dataset
                from vidseq.services.database_manager import DatabaseManager
                from sqlalchemy import select
                from sqlalchemy.ext.asyncio import AsyncSession
                import asyncio
                
                # Get all videos in project
                from vidseq.models.video import Video
                db_manager = DatabaseManager.get_instance()
                session_factory = db_manager.get_project_session_factory(str(project_path))
                
                async def get_videos():
                    async with session_factory() as session:
                        result = await session.execute(select(Video).order_by(Video.id))
                        return result.scalars().all()
                
                videos = asyncio.run(get_videos())
                
                # Create YOLO dataset
                dataset_dir = yolo_dataset.create_yolo_dataset(project_path, videos)
                
                # Create dataset.yaml for Ultralytics
                dataset_yaml = dataset_dir / "dataset.yaml"
                with open(dataset_yaml, 'w') as f:
                    f.write(f"""path: {dataset_dir}
train: images/train
val: images/val

nc: 1  # number of classes
names: ['object']  # class names
""")
                
                # Train YOLO model
                print(f"[YOLO Service] Starting training...")
                model = YOLO('yolo11n.pt')  # Load nano model
                
                print(f"[YOLO Service] Model loaded, beginning training epochs...")
                model.train(
                    data=str(dataset_yaml),
                    epochs=100,
                    imgsz=640,
                    batch=16,
                    device=0 if __import__('torch').cuda.is_available() else 'cpu',
                    project=str(project_path),
                    name='yolo_training',
                    exist_ok=True,
                    save=True,
                )
                
                # Save final model to standard location
                best_model_path = project_path / "yolo_training" / "weights" / "best.pt"
                final_model_path = self.get_model_path(project_path)
                if best_model_path.exists():
                    import shutil
                    shutil.copy2(best_model_path, final_model_path)
                    print(f"[YOLO Service] Model saved to {final_model_path}")
                
            except Exception as e:
                print(f"[YOLO Service] Training failed: {e}")
                import traceback
                traceback.print_exc()
            finally:
                self._is_training = False
        
        self._training_thread = threading.Thread(target=_train, daemon=True)
        self._training_thread.start()
    
    def run_initial_detection(self, project_path: Path) -> None:
        """
        Run initial detection on all videos in the project.
        
        Applies YOLO model to frame 0 of each video in the project.
        Skips videos that already have a bounding box on frame 0.
        
        Args:
            project_path: Path to the project folder
            
        Raises:
            RuntimeError: If application is already in progress
        """
        if self._is_applying:
            raise RuntimeError("Application already in progress")
        
        if not self.model_exists(project_path):
            raise RuntimeError("Model not found. Train model first.")
        
        self._is_applying = True
        
        def _run_detection():
            try:
                from vidseq.services import mask_service
                from vidseq.services.database_manager import DatabaseManager
                from vidseq.models.video import Video
                from sqlalchemy import select
                from sqlalchemy.ext.asyncio import AsyncSession
                import asyncio
                import cv2
                import numpy as np
                
                # Get all videos in project
                db_manager = DatabaseManager.get_instance()
                session_factory = db_manager.get_project_session_factory(str(project_path))
                
                async def get_all_videos():
                    async with session_factory() as session:
                        result = await session.execute(select(Video).order_by(Video.id))
                        return result.scalars().all()
                
                videos = asyncio.run(get_all_videos())
                
                if len(videos) == 0:
                    print("[YOLO Service] No videos found in project")
                    return
                
                # Load YOLO model once
                model_path = self.get_model_path(project_path)
                model = YOLO(str(model_path))
                print(f"[YOLO Service] Loaded YOLO model, processing {len(videos)} videos...")
                
                processed_count = 0
                skipped_count = 0
                
                with mask_service.open_h5(project_path, 'a') as h5_file:
                    for video in videos:
                        try:
                            # Check if frame 0 already has a bbox
                            existing_bbox = mask_service.load_bbox(
                                project_path=project_path,
                                video_id=video.id,
                                frame_idx=0,
                                num_frames=video.num_frames,
                                h5_file=h5_file,
                            )
                            if existing_bbox is not None:
                                skipped_count += 1
                                print(f"[YOLO Service] Video {video.id} ({video.name}): Frame 0 already has bbox, skipping")
                                continue
                            
                            # Open video and read frame 0
                            video_path = Path(video.path)
                            cap = cv2.VideoCapture(str(video_path))
                            if not cap.isOpened():
                                print(f"[YOLO Service] Failed to open video {video.id}: {video_path}")
                                continue
                            
                            # Read frame 0
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            ret, frame = cap.read()
                            cap.release()
                            
                            if not ret:
                                print(f"[YOLO Service] Failed to read frame 0 from video {video.id}")
                                continue
                            
                            # Run YOLO inference on frame 0
                            results = model(frame, verbose=False)
                            
                            # Extract bbox from results
                            if len(results) > 0 and len(results[0].boxes) > 0:
                                # Get highest confidence box
                                boxes = results[0].boxes
                                best_box = boxes[0]  # Already sorted by confidence
                                
                                # Convert from YOLO format to pixel coordinates
                                x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy()
                                
                                # Save bbox
                                mask_service.save_bbox(
                                    project_path=project_path,
                                    video_id=video.id,
                                    frame_idx=0,
                                    bbox=np.array([x1, y1, x2, y2], dtype=np.float32),
                                    num_frames=video.num_frames,
                                    h5_file=h5_file,
                                )
                                
                                processed_count += 1
                                print(f"[YOLO Service] Video {video.id} ({video.name}): Detected bbox on frame 0")
                            else:
                                print(f"[YOLO Service] Video {video.id} ({video.name}): No detection on frame 0")
                        
                        except Exception as e:
                            print(f"[YOLO Service] Error processing video {video.id}: {e}")
                            import traceback
                            traceback.print_exc()
                            continue
                
                print(f"[YOLO Service] Initial detection complete: {processed_count} videos processed, {skipped_count} skipped")
                
            except Exception as e:
                print(f"[YOLO Service] Initial detection failed: {e}")
                import traceback
                traceback.print_exc()
            finally:
                self._is_applying = False
        
        self._applying_thread = threading.Thread(target=_run_detection, daemon=True)
        self._applying_thread.start()
    
    def is_training(self) -> bool:
        """Check if training is in progress."""
        return self._is_training
    
    def is_applying(self) -> bool:
        """Check if application is in progress."""
        return self._is_applying
    
    def get_model_path(self, project_path: Path) -> Path:
        """Get path to the trained model file."""
        return project_path / "yolo_model.pt"
    
    def model_exists(self, project_path: Path) -> bool:
        """Check if model file exists."""
        return self.get_model_path(project_path).exists()

