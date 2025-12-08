"""
SAM2 Worker Process.

Runs in a separate process to avoid CUDA/signal conflicts with FastAPI.
Communicates with the main process via multiprocessing queues.

Uses SAM2VideoPredictor with lazy frame loading and vos_optimized for torch.compile.
"""

from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np


def _extract_mask(video_res_masks, obj_ids: list, height: int, width: int) -> np.ndarray:
    """
    Extract mask from predictor outputs.
    
    Args:
        video_res_masks: Tensor of shape (num_objects, 1, H, W)
        obj_ids: List of object IDs
        height: Target height
        width: Target width
        
    Returns:
        Binary mask as numpy array (height, width), dtype=uint8, values 0 or 255
    """
    import cv2
    
    if video_res_masks is None or len(obj_ids) == 0:
        return np.zeros((height, width), dtype=np.uint8)
    
    mask_tensor = video_res_masks[0]
    mask_np = (mask_tensor > 0).cpu().numpy()
    
    if mask_np.ndim == 3:
        mask_np = mask_np[0]
    
    if mask_np.shape != (height, width):
        mask_np = cv2.resize(
            mask_np.astype(np.uint8),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )
    
    return (mask_np * 255).astype(np.uint8)


def _init_state_with_lazy_loader(predictor, loader) -> dict:
    """
    Initialize inference state using our lazy loader instead of loading all frames.
    
    Replicates what predictor.init_state() does but with custom images source.
    """
    import torch
    compute_device = predictor.device
    offload_state_to_cpu = False
    
    inference_state = {}
    inference_state["images"] = loader
    inference_state["num_frames"] = len(loader)
    inference_state["offload_video_to_cpu"] = loader.offload_to_cpu
    inference_state["offload_state_to_cpu"] = offload_state_to_cpu
    inference_state["video_height"] = loader._video_height
    inference_state["video_width"] = loader._video_width
    inference_state["device"] = compute_device
    inference_state["storage_device"] = compute_device
    
    inference_state["point_inputs_per_obj"] = {}
    inference_state["mask_inputs_per_obj"] = {}
    inference_state["cached_features"] = {}
    inference_state["constants"] = {}
    inference_state["obj_id_to_idx"] = OrderedDict()
    inference_state["obj_idx_to_id"] = OrderedDict()
    inference_state["obj_ids"] = []
    inference_state["output_dict_per_obj"] = {}
    inference_state["temp_output_dict_per_obj"] = {}
    inference_state["frames_tracked_per_obj"] = {}
    
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        predictor._get_image_feature(inference_state, frame_idx=0, batch_size=1)
    
    return inference_state


def worker_loop(command_queue, result_queue):
    """
    Main loop for SAM2 worker process.
    
    Receives commands from command_queue, sends results to result_queue.
    """
    import torch
    
    print("[SAM2 Worker] Starting worker process...")
    
    predictor = None
    sessions: dict[int, tuple[dict, Any]] = {}
    
    config_name = "configs/sam2.1/sam2.1_hiera_t.yaml"
    checkpoint_path = Path(__file__).parent.parent.parent / "sam2_models" / "sam2.1_hiera_tiny.pt"
    
    while True:
        try:
            cmd = command_queue.get()
        except KeyboardInterrupt:
            print("[SAM2 Worker] Interrupted, shutting down...")
            break
        
        cmd_type = cmd.get("type")
        
        if cmd_type == "load_model":
            print("[SAM2 Worker] Loading SAM2 model...")
            result_queue.put({"type": "status", "status": "loading_model"})
            
            try:
                from sam2.build_sam import build_sam2_video_predictor
                
                torch.set_float32_matmul_precision('medium')
                
                predictor = build_sam2_video_predictor(
                    config_file=config_name,
                    ckpt_path=str(checkpoint_path),
                    device="cuda",
                    vos_optimized=False,
                )
                predictor.to(dtype=torch.bfloat16)
                
                print("[SAM2 Worker] Compiling image encoder...")
                predictor.image_encoder = torch.compile(predictor.image_encoder, mode="max-autotune", fullgraph=True)
                
                print("[SAM2 Worker] SAM2 model loaded and compiled!")
                result_queue.put({"type": "status", "status": "ready"})
            except Exception as e:
                print(f"[SAM2 Worker] Failed to load model: {e}")
                import traceback
                traceback.print_exc()
                result_queue.put({"type": "status", "status": "error", "error": str(e)})
        
        elif cmd_type == "init_session":
            video_id = cmd["video_id"]
            video_path = Path(cmd["video_path"])
            request_id = cmd.get("request_id")
            
            print(f"[SAM2 Worker] Initializing session for video {video_id}...")
            
            try:
                if predictor is None:
                    raise RuntimeError("Model not loaded")
                
                if video_id in sessions:
                    inference_state, loader = sessions[video_id]
                    result_queue.put({
                        "type": "init_session_result",
                        "request_id": request_id,
                        "status": "ok",
                        "video_id": video_id,
                        "num_frames": inference_state["num_frames"],
                        "height": inference_state["video_height"],
                        "width": inference_state["video_width"],
                    })
                    continue
                
                from vidseq.services.sam2streaming import LazyVideoFrameLoader
                loader = LazyVideoFrameLoader(video_path, offload_to_cpu=False, device="cuda")
                
                inference_state = _init_state_with_lazy_loader(predictor, loader)
                
                sessions[video_id] = (inference_state, loader)
                
                height = inference_state["video_height"]
                width = inference_state["video_width"]
                num_frames = inference_state["num_frames"]
                
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
                print(f"[SAM2 Worker] Failed to init session: {e}")
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
            points = cmd["points"]
            labels = cmd["labels"]
            obj_id = cmd.get("obj_id", 1)
            request_id = cmd.get("request_id")
            
            try:
                if predictor is None:
                    raise RuntimeError("Model not loaded")
                
                if video_id not in sessions:
                    raise RuntimeError(f"No session for video {video_id}")
                
                inference_state, loader = sessions[video_id]
                height = inference_state["video_height"]
                width = inference_state["video_width"]
                
                points_arr = np.array(points, dtype=np.float32)
                points_arr[:, 0] *= width
                points_arr[:, 1] *= height
                labels_arr = np.array(labels, dtype=np.int32)
                
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    _, out_obj_ids, video_res_masks = predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=frame_idx,
                        obj_id=obj_id,
                        points=points_arr,
                        labels=labels_arr,
                        clear_old_points=False,
                    )
                    mask = _extract_mask(video_res_masks, out_obj_ids, height, width)
                
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
                print(f"[SAM2 Worker] Failed to add point prompt: {e}")
                import traceback
                traceback.print_exc()
                result_queue.put({
                    "type": "add_point_prompt_result",
                    "request_id": request_id,
                    "status": "error",
                    "error": str(e),
                })
        
        elif cmd_type == "propagate_forward":
            video_id = cmd["video_id"]
            start_frame_idx = cmd["start_frame_idx"]
            max_frames = cmd["max_frames"]
            request_id = cmd.get("request_id")
            
            try:
                if predictor is None:
                    raise RuntimeError("Model not loaded")
                
                if video_id not in sessions:
                    raise RuntimeError(f"No session for video {video_id}")
                
                inference_state, loader = sessions[video_id]
                height = inference_state["video_height"]
                width = inference_state["video_width"]
                
                masks_data = []
                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                    from sam2.utils.misc import mask_to_box
                    
                    for frame_idx, out_obj_ids, video_res_masks in predictor.propagate_in_video(
                        inference_state=inference_state,
                        start_frame_idx=start_frame_idx,
                        max_frame_num_to_track=max_frames,
                        reverse=False,
                    ):
                        mask = _extract_mask(video_res_masks, out_obj_ids, height, width)
                        
                        # Compute bounding box from mask
                        mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)
                        # Convert to boolean (True/False) - mask_to_box expects boolean tensor
                        mask_binary = (mask_tensor > 0).bool()
                        bbox_tensor = mask_to_box(mask_binary)
                        bbox_np = bbox_tensor[0, 0].cpu().numpy()  # Shape: [4] with [x1, y1, x2, y2]
                        
                        # Check if bbox is valid (not all zeros)
                        if np.all(bbox_np == 0):
                            bbox_np = None
                        
                        masks_data.append({
                            "frame_idx": frame_idx,
                            "mask_bytes": mask.tobytes(),
                            "mask_shape": mask.shape,
                            "bbox": bbox_np.tolist() if bbox_np is not None else None,
                        })
                
                result_queue.put({
                    "type": "propagate_forward_result",
                    "request_id": request_id,
                    "status": "ok",
                    "masks": masks_data,
                })
            except Exception as e:
                print(f"[SAM2 Worker] Failed to propagate forward: {e}")
                import traceback
                traceback.print_exc()
                result_queue.put({
                    "type": "propagate_forward_result",
                    "request_id": request_id,
                    "status": "error",
                    "error": str(e),
                })
        
        elif cmd_type == "reset_state":
            video_id = cmd["video_id"]
            request_id = cmd.get("request_id")
            
            print(f"[SAM2 Worker] Resetting state for video {video_id}...")
            
            try:
                if predictor is None:
                    raise RuntimeError("Model not loaded")
                
                if video_id not in sessions:
                    result_queue.put({
                        "type": "reset_state_result",
                        "request_id": request_id,
                        "status": "ok",
                    })
                    continue
                
                inference_state, loader = sessions[video_id]
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    predictor.reset_state(inference_state)
                
                print(f"[SAM2 Worker] State reset for video {video_id}")
                result_queue.put({
                    "type": "reset_state_result",
                    "request_id": request_id,
                    "status": "ok",
                })
            except Exception as e:
                print(f"[SAM2 Worker] Failed to reset state: {e}")
                import traceback
                traceback.print_exc()
                result_queue.put({
                    "type": "reset_state_result",
                    "request_id": request_id,
                    "status": "error",
                    "error": str(e),
                })
        
        elif cmd_type == "close_session":
            video_id = cmd["video_id"]
            request_id = cmd.get("request_id")
            
            print(f"[SAM2 Worker] Closing session for video {video_id}...")
            
            try:
                if video_id in sessions:
                    inference_state, loader = sessions.pop(video_id)
                    loader.close()
                
                result_queue.put({
                    "type": "close_session_result",
                    "request_id": request_id,
                    "status": "ok",
                })
            except Exception as e:
                print(f"[SAM2 Worker] Failed to close session: {e}")
                result_queue.put({
                    "type": "close_session_result",
                    "request_id": request_id,
                    "status": "error",
                    "error": str(e),
                })
        
        elif cmd_type == "shutdown":
            print("[SAM2 Worker] Shutting down...")
            for video_id, (inference_state, loader) in list(sessions.items()):
                try:
                    loader.close()
                except Exception:
                    pass
            sessions.clear()
            
            result_queue.put({"type": "shutdown_result", "status": "ok"})
            break
        
        else:
            print(f"[SAM2 Worker] Unknown command type: {cmd_type}")
            result_queue.put({
                "type": "error",
                "error": f"Unknown command type: {cmd_type}",
            })
    
    print("[SAM2 Worker] Worker process exiting.")


