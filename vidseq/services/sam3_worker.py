"""
SAM3 Worker Process.

Runs in a separate process to avoid CUDA/signal conflicts with FastAPI.
Communicates with the main process via multiprocessing queues.
"""

import uuid
import time
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np


def _init_state_with_lazy_loader(model, loader) -> dict:
    """Initialize SAM3 inference state using lazy loader."""
    inference_state = {}
    inference_state["image_size"] = model.image_size
    inference_state["num_frames"] = len(loader)
    inference_state["orig_height"] = loader._video_height
    inference_state["orig_width"] = loader._video_width
    inference_state["constants"] = {}
    
    model._construct_initial_input_batch(inference_state, loader)
    
    inference_state["tracker_inference_states"] = []
    inference_state["tracker_metadata"] = {}
    inference_state["feature_cache"] = {}
    inference_state["cached_frame_outputs"] = {}
    inference_state["action_history"] = []
    inference_state["is_image_only"] = False
    
    return inference_state


def _start_session_with_lazy_loader(predictor, loader) -> str:
    """Start a SAM3 session using lazy loader. Returns session_id."""
    inference_state = _init_state_with_lazy_loader(predictor.model, loader)
    
    session_id = str(uuid.uuid4())
    
    predictor._ALL_INFERENCE_STATES[session_id] = {
        "state": inference_state,
        "session_id": session_id,
        "start_time": time.time(),
    }
    
    return session_id


def _xyxy_to_xywh(x1: float, y1: float, x2: float, y2: float) -> list:
    """Convert normalized xyxy coords to xywh format."""
    return [x1, y1, x2 - x1, y2 - y1]


def _extract_mask(response: dict, height: int, width: int) -> np.ndarray:
    """Extract and process mask from predictor response."""
    import cv2
    
    outputs = response.get("outputs", {})
    out_obj_ids = outputs.get("out_obj_ids", [])
    out_masks = outputs.get("out_binary_masks", [])
    
    # Check using obj_ids length (out_masks might be a tensor/array)
    if len(out_obj_ids) == 0:
        return np.zeros((height, width), dtype=np.uint8)
    
    combined_mask = None
    for mask in out_masks:
        if hasattr(mask, 'cpu'):
            mask_np = mask.cpu().numpy()
        else:
            mask_np = np.array(mask)
        
        if mask_np.ndim == 3:
            mask_np = mask_np[0]
        
        if mask_np.shape != (height, width):
            mask_np = cv2.resize(
                mask_np.astype(np.uint8),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
        
        mask_np = mask_np.astype(bool)
        
        if combined_mask is None:
            combined_mask = mask_np
        else:
            combined_mask = combined_mask | mask_np
    
    if combined_mask is None:
        return np.zeros((height, width), dtype=np.uint8)
    
    return (combined_mask * 255).astype(np.uint8)


def worker_loop(command_queue, result_queue):
    """
    Main loop for SAM3 worker process.
    
    Receives commands from command_queue, sends results to result_queue.
    """
    print("[SAM3 Worker] Starting worker process...")
    
    predictor = None
    sessions: Dict[int, Tuple[str, Any]] = {}  # video_id -> (session_id, loader)
    
    while True:
        try:
            cmd = command_queue.get()
        except KeyboardInterrupt:
            print("[SAM3 Worker] Interrupted, shutting down...")
            break
        
        cmd_type = cmd.get("type")
        
        if cmd_type == "load_model":
            print("[SAM3 Worker] Loading SAM3 model...")
            result_queue.put({"type": "status", "status": "loading_model"})
            
            try:
                from sam3.model.sam3_video_predictor import Sam3VideoPredictor
                predictor = Sam3VideoPredictor(apply_temporal_disambiguation=True)
                print("[SAM3 Worker] Model loaded successfully!")
                result_queue.put({"type": "status", "status": "ready"})
            except Exception as e:
                print(f"[SAM3 Worker] Failed to load model: {e}")
                result_queue.put({"type": "status", "status": "error", "error": str(e)})
        
        elif cmd_type == "init_session":
            video_id = cmd["video_id"]
            video_path = Path(cmd["video_path"])
            request_id = cmd.get("request_id")
            
            print(f"[SAM3 Worker] Initializing session for video {video_id}...")
            
            try:
                if predictor is None:
                    raise RuntimeError("Model not loaded")
                
                # Check if session already exists
                if video_id in sessions:
                    session_id, loader = sessions[video_id]
                    result_queue.put({
                        "type": "init_session_result",
                        "request_id": request_id,
                        "status": "ok",
                        "video_id": video_id,
                        "num_frames": len(loader),
                        "height": loader._video_height,
                        "width": loader._video_width,
                    })
                    continue
                
                from vidseq.sam3streaming import LazyVideoFrameLoader
                loader = LazyVideoFrameLoader(video_path, device="cuda")
                session_id = _start_session_with_lazy_loader(predictor, loader)
                sessions[video_id] = (session_id, loader)
                
                print(f"[SAM3 Worker] Session initialized: {session_id}")
                result_queue.put({
                    "type": "init_session_result",
                    "request_id": request_id,
                    "status": "ok",
                    "video_id": video_id,
                    "num_frames": len(loader),
                    "height": loader._video_height,
                    "width": loader._video_width,
                })
            except Exception as e:
                print(f"[SAM3 Worker] Failed to init session: {e}")
                result_queue.put({
                    "type": "init_session_result",
                    "request_id": request_id,
                    "status": "error",
                    "error": str(e),
                })
        
        elif cmd_type == "add_bbox_prompt":
            video_id = cmd["video_id"]
            frame_idx = cmd["frame_idx"]
            bbox = cmd["bbox"]
            request_id = cmd.get("request_id")
            
            print(f"[SAM3 Worker] Adding bbox prompt for video {video_id}, frame {frame_idx}...")
            
            try:
                if predictor is None:
                    raise RuntimeError("Model not loaded")
                
                if video_id not in sessions:
                    raise RuntimeError(f"No session for video {video_id}")
                
                session_id, loader = sessions[video_id]
                
                x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
                boxes_xywh = [_xyxy_to_xywh(x1, y1, x2, y2)]
                
                response = predictor.handle_request({
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": frame_idx,
                    "bounding_boxes": boxes_xywh,
                    "bounding_box_labels": [1],
                })
                
                height = loader._video_height
                width = loader._video_width
                mask = _extract_mask(response, height, width)
                
                print(f"[SAM3 Worker] Prompt processed, mask shape: {mask.shape}")
                result_queue.put({
                    "type": "add_bbox_prompt_result",
                    "request_id": request_id,
                    "status": "ok",
                    "mask_bytes": mask.tobytes(),
                    "mask_shape": mask.shape,
                    "mask_dtype": str(mask.dtype),
                })
            except Exception as e:
                print(f"[SAM3 Worker] Failed to add prompt: {e}")
                import traceback
                traceback.print_exc()
                result_queue.put({
                    "type": "add_bbox_prompt_result",
                    "request_id": request_id,
                    "status": "error",
                    "error": str(e),
                })
        
        elif cmd_type == "close_session":
            video_id = cmd["video_id"]
            request_id = cmd.get("request_id")
            
            print(f"[SAM3 Worker] Closing session for video {video_id}...")
            
            try:
                if video_id in sessions:
                    session_id, loader = sessions.pop(video_id)
                    if predictor and session_id in predictor._ALL_INFERENCE_STATES:
                        del predictor._ALL_INFERENCE_STATES[session_id]
                    loader.close()
                
                result_queue.put({
                    "type": "close_session_result",
                    "request_id": request_id,
                    "status": "ok",
                })
            except Exception as e:
                print(f"[SAM3 Worker] Failed to close session: {e}")
                result_queue.put({
                    "type": "close_session_result",
                    "request_id": request_id,
                    "status": "error",
                    "error": str(e),
                })
        
        elif cmd_type == "shutdown":
            print("[SAM3 Worker] Shutting down...")
            # Close all sessions
            for video_id, (session_id, loader) in sessions.items():
                loader.close()
            sessions.clear()
            
            result_queue.put({"type": "shutdown_result", "status": "ok"})
            break
        
        else:
            print(f"[SAM3 Worker] Unknown command type: {cmd_type}")
            result_queue.put({
                "type": "error",
                "error": f"Unknown command type: {cmd_type}",
            })
    
    print("[SAM3 Worker] Worker process exiting.")

