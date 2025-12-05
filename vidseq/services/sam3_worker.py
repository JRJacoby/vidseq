"""
SAM3 Worker Process.

Runs in a separate process to avoid CUDA/signal conflicts with FastAPI.
Communicates with the main process via multiprocessing queues.

Uses Sam3VideoPredictor model with lazy frame loading.
"""

import uuid
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np


def _extract_mask(outputs: dict, height: int, width: int):
    """
    Extract mask from predictor outputs.
    
    Args:
        outputs: Dict with 'video_res_masks' tensor of shape (num_objects, 1, H, W)
                 and 'obj_ids' list
        height: Target height
        width: Target width
        
    Returns:
        Binary mask as numpy array (height, width), dtype=uint8, values 0 or 255
    """
    import cv2
    
    video_res_masks = outputs.get("video_res_masks")
    obj_ids = outputs.get("obj_ids", [])
    
    if video_res_masks is None or len(obj_ids) == 0:
        return np.zeros((height, width), dtype=np.uint8)
    
    mask_tensor = video_res_masks[0]  # First object (we only track one)
    if hasattr(mask_tensor, 'cpu'):
        mask_np = (mask_tensor > 0).cpu().numpy()
    else:
        mask_np = np.array(mask_tensor) > 0
    
    if mask_np.ndim == 3:
        mask_np = mask_np[0]
    
    if mask_np.shape != (height, width):
        mask_np = cv2.resize(
            mask_np.astype(np.uint8),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )
    
    return (mask_np * 255).astype(np.uint8)


def _init_state_with_lazy_loader(model, loader):
    """
    Initialize inference state using our lazy loader instead of loading all frames.
    
    Replicates what model.init_state() does but with custom images source.
    """
    inference_state = {}
    inference_state["image_size"] = model.image_size
    inference_state["num_frames"] = len(loader)
    inference_state["orig_height"] = loader._video_height
    inference_state["orig_width"] = loader._video_width
    inference_state["constants"] = {}
    
    # Use our lazy loader as the images source
    model._construct_initial_input_batch(inference_state, loader)
    
    # Initialize extra states
    inference_state["tracker_inference_states"] = []
    inference_state["tracker_metadata"] = {}
    inference_state["feature_cache"] = {}
    inference_state["cached_frame_outputs"] = {}
    inference_state["action_history"] = []
    inference_state["is_image_only"] = False
    
    return inference_state


def worker_loop(command_queue, result_queue):
    """
    Main loop for SAM3 worker process.
    
    Receives commands from command_queue, sends results to result_queue.
    """
    print("[SAM3 Worker] Starting worker process...")
    
    predictor = None
    # video_id -> (session_id, inference_state, loader)
    sessions: Dict[int, Tuple[str, dict, Any]] = {}
    
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
                predictor = Sam3VideoPredictor(
                    apply_temporal_disambiguation=True,
                )
                print("[SAM3 Worker] SAM3 model loaded successfully!")
                result_queue.put({"type": "status", "status": "ready"})
            except Exception as e:
                print(f"[SAM3 Worker] Failed to load model: {e}")
                import traceback
                traceback.print_exc()
                result_queue.put({"type": "status", "status": "error", "error": str(e)})
        
        elif cmd_type == "init_session":
            video_id = cmd["video_id"]
            video_path = Path(cmd["video_path"])
            request_id = cmd.get("request_id")
            
            print(f"[SAM3 Worker] Initializing session for video {video_id}...")
            
            try:
                if predictor is None:
                    raise RuntimeError("Model not loaded")
                
                if video_id in sessions:
                    session_id, inference_state, loader = sessions[video_id]
                    result_queue.put({
                        "type": "init_session_result",
                        "request_id": request_id,
                        "status": "ok",
                        "video_id": video_id,
                        "num_frames": inference_state["num_frames"],
                        "height": inference_state["orig_height"],
                        "width": inference_state["orig_width"],
                    })
                    continue
                
                # Use our lazy loader instead of loading all frames
                from vidseq.services.sam3streaming import LazyVideoFrameLoader
                loader = LazyVideoFrameLoader(video_path, device="cuda")
                
                # Initialize state with lazy loader
                inference_state = _init_state_with_lazy_loader(predictor.model, loader)
                
                # Register with predictor's session management
                session_id = str(uuid.uuid4())
                predictor._ALL_INFERENCE_STATES[session_id] = {
                    "state": inference_state,
                    "session_id": session_id,
                }
                
                sessions[video_id] = (session_id, inference_state, loader)
                
                height = inference_state["orig_height"]
                width = inference_state["orig_width"]
                num_frames = inference_state["num_frames"]
                
                print(f"[SAM3 Worker] Session initialized for video {video_id} ({num_frames} frames, {width}x{height})")
                result_queue.put({
                    "type": "init_session_result",
                    "request_id": request_id,
                    "status": "ok",
                    "video_id": video_id,
                    "num_frames": num_frames,
                    "height": height,
                    "width": width,
                })
            except Exception as e:
                print(f"[SAM3 Worker] Failed to init session: {e}")
                import traceback
                traceback.print_exc()
                result_queue.put({
                    "type": "init_session_result",
                    "request_id": request_id,
                    "status": "error",
                    "error": str(e),
                })
        
        elif cmd_type == "add_point_prompt":
            video_id = cmd["video_id"]
            frame_idx = cmd["frame_idx"]
            points = cmd["points"]  # [[x, y], ...] normalized coords
            labels = cmd["labels"]  # [1, 0, ...] (1=positive, 0=negative)
            obj_id = cmd.get("obj_id", 1)
            request_id = cmd.get("request_id")
            
            print(f"[SAM3 Worker] Adding point prompt for video {video_id}, frame {frame_idx}, obj_id={obj_id}, points={len(points)}...")
            
            try:
                if predictor is None:
                    raise RuntimeError("Model not loaded")
                
                if video_id not in sessions:
                    raise RuntimeError(f"No session for video {video_id}")
                
                session_id, inference_state, loader = sessions[video_id]
                height = inference_state["orig_height"]
                width = inference_state["orig_width"]
                
                # add_prompt with points routes to tracker internally
                response = predictor.add_prompt(
                    session_id=session_id,
                    frame_idx=frame_idx,
                    points=points,  # normalized coords
                    point_labels=labels,
                    obj_id=obj_id,
                )
                
                outputs = response["outputs"]
                mask = _extract_mask(outputs, height, width)
                
                print(f"[SAM3 Worker] Point prompt processed, mask shape: {mask.shape}")
                result_queue.put({
                    "type": "add_point_prompt_result",
                    "request_id": request_id,
                    "status": "ok",
                    "mask_bytes": mask.tobytes(),
                    "mask_shape": mask.shape,
                    "mask_dtype": str(mask.dtype),
                    "obj_id": obj_id,
                })
            except Exception as e:
                print(f"[SAM3 Worker] Failed to add point prompt: {e}")
                import traceback
                traceback.print_exc()
                result_queue.put({
                    "type": "add_point_prompt_result",
                    "request_id": request_id,
                    "status": "error",
                    "error": str(e),
                })
        
        elif cmd_type == "remove_object":
            video_id = cmd["video_id"]
            obj_id = cmd["obj_id"]
            request_id = cmd.get("request_id")
            
            print(f"[SAM3 Worker] Removing object {obj_id} from video {video_id}...")
            
            try:
                if predictor is None:
                    raise RuntimeError("Model not loaded")
                
                if video_id not in sessions:
                    result_queue.put({
                        "type": "remove_object_result",
                        "request_id": request_id,
                        "status": "ok",
                    })
                    continue
                
                session_id, inference_state, loader = sessions[video_id]
                
                # Use reset_session to clear all tracking state
                predictor.reset_session(session_id)
                
                print(f"[SAM3 Worker] Object {obj_id} removed (session reset)")
                result_queue.put({
                    "type": "remove_object_result",
                    "request_id": request_id,
                    "status": "ok",
                })
            except Exception as e:
                print(f"[SAM3 Worker] Failed to remove object: {e}")
                import traceback
                traceback.print_exc()
                result_queue.put({
                    "type": "remove_object_result",
                    "request_id": request_id,
                    "status": "error",
                    "error": str(e),
                })
        
        elif cmd_type == "inject_mask":
            video_id = cmd["video_id"]
            frame_idx = cmd["frame_idx"]
            mask_bytes = cmd["mask_bytes"]
            mask_shape = cmd["mask_shape"]
            obj_id = cmd["obj_id"]
            request_id = cmd.get("request_id")
            
            print(f"[SAM3 Worker] Injecting mask for video {video_id}, frame {frame_idx}, obj_id={obj_id}...")
            
            try:
                import torch
                
                if predictor is None:
                    raise RuntimeError("Model not loaded")
                
                if video_id not in sessions:
                    raise RuntimeError(f"No session for video {video_id}")
                
                session_id, inference_state, loader = sessions[video_id]
                
                mask = np.frombuffer(mask_bytes, dtype=np.uint8).reshape(mask_shape)
                mask_tensor = torch.from_numpy(mask > 127).float().cuda()
                
                # Use the model's add_new_mask method
                predictor.model.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=frame_idx,
                    obj_id=obj_id,
                    mask=mask_tensor,
                )
                
                print(f"[SAM3 Worker] Mask injected successfully for frame {frame_idx}")
                result_queue.put({
                    "type": "inject_mask_result",
                    "request_id": request_id,
                    "status": "ok",
                })
            except Exception as e:
                print(f"[SAM3 Worker] Failed to inject mask: {e}")
                import traceback
                traceback.print_exc()
                result_queue.put({
                    "type": "inject_mask_result",
                    "request_id": request_id,
                    "status": "error",
                    "error": str(e),
                })
        
        elif cmd_type == "propagate_forward":
            video_id = cmd["video_id"]
            start_frame_idx = cmd["start_frame_idx"]
            max_frames = cmd["max_frames"]
            request_id = cmd.get("request_id")
            
            print(f"[SAM3 Worker] Propagating forward from frame {start_frame_idx} for up to {max_frames} frames...")
            
            try:
                if predictor is None:
                    raise RuntimeError("Model not loaded")
                
                if video_id not in sessions:
                    raise RuntimeError(f"No session for video {video_id}")
                
                session_id, inference_state, loader = sessions[video_id]
                height = inference_state["orig_height"]
                width = inference_state["orig_width"]
                
                masks_data = []
                for response in predictor.propagate_in_video(
                    session_id=session_id,
                    propagation_direction="forward",
                    start_frame_idx=start_frame_idx,
                    max_frame_num_to_track=max_frames,
                ):
                    frame_idx = response["frame_index"]
                    outputs = response["outputs"]
                    mask = _extract_mask(outputs, height, width)
                    masks_data.append({
                        "frame_idx": frame_idx,
                        "mask_bytes": mask.tobytes(),
                        "mask_shape": mask.shape,
                    })
                
                print(f"[SAM3 Worker] Propagation complete, processed {len(masks_data)} frames")
                result_queue.put({
                    "type": "propagate_forward_result",
                    "request_id": request_id,
                    "status": "ok",
                    "masks": masks_data,
                })
            except Exception as e:
                print(f"[SAM3 Worker] Failed to propagate: {e}")
                import traceback
                traceback.print_exc()
                result_queue.put({
                    "type": "propagate_forward_result",
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
                    session_id, inference_state, loader = sessions.pop(video_id)
                    if predictor is not None:
                        predictor.close_session(session_id)
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
            for video_id, (session_id, inference_state, loader) in list(sessions.items()):
                try:
                    if predictor is not None:
                        predictor.close_session(session_id)
                    loader.close()
                except Exception:
                    pass
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
