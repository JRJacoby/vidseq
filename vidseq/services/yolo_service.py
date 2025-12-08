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
    
    def apply_model(self, project_path: Path, video_id: int) -> None:
        """
        Start applying YOLO model to a video.
        
        Args:
            project_path: Path to the project folder
            video_id: ID of the video to apply model to
            
        Raises:
            RuntimeError: If application is already in progress
        """
        if self._is_applying:
            raise RuntimeError("Application already in progress")
        
        if not self.model_exists(project_path):
            raise RuntimeError("Model not found. Train model first.")
        
        self._is_applying = True
        
        def _apply():
            try:
                from vidseq.services import mask_service
                from vidseq.services.database_manager import DatabaseManager
                from vidseq.models.video import Video
                from sqlalchemy import select
                from sqlalchemy.ext.asyncio import AsyncSession
                import asyncio
                import cv2
                import numpy as np
                
                # Get video
                db_manager = DatabaseManager.get_instance()
                session_factory = db_manager.get_project_session_factory(str(project_path))
                
                async def get_video():
                    async with session_factory() as session:
                        result = await session.execute(select(Video).where(Video.id == video_id))
                        return result.scalar_one_or_none()
                
                video = asyncio.run(get_video())
                if video is None:
                    raise RuntimeError(f"Video {video_id} not found")
                
                # Load YOLO model
                model_path = self.get_model_path(project_path)
                model = YOLO(str(model_path))
                
                # Open video
                video_path = Path(video.path)
                cap = cv2.VideoCapture(str(video_path))
                if not cap.isOpened():
                    raise RuntimeError(f"Failed to open video: {video_path}")
                
                frame_idx = 0
                with mask_service.open_h5(project_path, 'a') as h5_file:
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        # Skip if frame already has bbox
                        existing_bbox = mask_service.load_bbox(
                            project_path=project_path,
                            video_id=video_id,
                            frame_idx=frame_idx,
                            num_frames=video.num_frames,
                            h5_file=h5_file,
                        )
                        if existing_bbox is not None:
                            frame_idx += 1
                            continue
                        
                        # Run YOLO inference
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
                                video_id=video_id,
                                frame_idx=frame_idx,
                                bbox=np.array([x1, y1, x2, y2], dtype=np.float32),
                                num_frames=video.num_frames,
                                h5_file=h5_file,
                            )
                        
                        frame_idx += 1
                        
                        if frame_idx % 100 == 0:
                            print(f"[YOLO Service] Processed {frame_idx}/{video.num_frames} frames...")
                
                cap.release()
                print(f"[YOLO Service] Applied model to video {video_id}")
                
            except Exception as e:
                print(f"[YOLO Service] Application failed: {e}")
                import traceback
                traceback.print_exc()
            finally:
                self._is_applying = False
        
        self._applying_thread = threading.Thread(target=_apply, daemon=True)
        self._applying_thread.start()
    
    def test_apply_model(self, project_path: Path, video_id: int, start_frame: int) -> None:
        """
        Start test applying YOLO model to a limited range of frames.
        
        Args:
            project_path: Path to the project folder
            video_id: ID of the video to apply model to
            start_frame: Starting frame index
            
        Raises:
            RuntimeError: If application is already in progress
        """
        if self._is_applying:
            raise RuntimeError("Application already in progress")
        
        if not self.model_exists(project_path):
            raise RuntimeError("Model not found. Train model first.")
        
        self._is_applying = True
        
        def _test_apply():
            try:
                from vidseq.services import mask_service
                from vidseq.services.database_manager import DatabaseManager
                from vidseq.models.video import Video
                from sqlalchemy import select
                from sqlalchemy.ext.asyncio import AsyncSession
                import asyncio
                import cv2
                import numpy as np
                
                # Get video
                db_manager = DatabaseManager.get_instance()
                session_factory = db_manager.get_project_session_factory(str(project_path))
                
                async def get_video():
                    async with session_factory() as session:
                        result = await session.execute(select(Video).where(Video.id == video_id))
                        return result.scalar_one_or_none()
                
                video = asyncio.run(get_video())
                if video is None:
                    raise RuntimeError(f"Video {video_id} not found")
                
                # Load YOLO model
                model_path = self.get_model_path(project_path)
                model = YOLO(str(model_path))
                
                # Open video
                video_path = Path(video.path)
                cap = cv2.VideoCapture(str(video_path))
                if not cap.isOpened():
                    raise RuntimeError(f"Failed to open video: {video_path}")
                
                # Seek to start frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                
                end_frame = min(start_frame + 1000, video.num_frames)
                frame_idx = start_frame
                
                with mask_service.open_h5(project_path, 'a') as h5_file:
                    while frame_idx < end_frame:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        # Skip if frame already has bbox
                        existing_bbox = mask_service.load_bbox(
                            project_path=project_path,
                            video_id=video_id,
                            frame_idx=frame_idx,
                            num_frames=video.num_frames,
                            h5_file=h5_file,
                        )
                        if existing_bbox is not None:
                            frame_idx += 1
                            continue
                        
                        # Run YOLO inference
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
                                video_id=video_id,
                                frame_idx=frame_idx,
                                bbox=np.array([x1, y1, x2, y2], dtype=np.float32),
                                num_frames=video.num_frames,
                                h5_file=h5_file,
                            )
                        
                        frame_idx += 1
                        
                        if (frame_idx - start_frame) % 100 == 0:
                            print(f"[YOLO Service] Processed {frame_idx - start_frame}/{end_frame - start_frame} frames...")
                
                cap.release()
                print(f"[YOLO Service] Test applied model to video {video_id} (frames {start_frame} to {frame_idx-1})")
                
            except Exception as e:
                print(f"[YOLO Service] Test application failed: {e}")
                import traceback
                traceback.print_exc()
            finally:
                self._is_applying = False
        
        self._applying_thread = threading.Thread(target=_test_apply, daemon=True)
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

