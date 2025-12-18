"""
SAM2 TCP Server Worker.

Runs as standalone TCP server process. Accepts multiple connections,
queues all commands, processes them sequentially for GPU safety.
"""

import base64
import json
import os
import queue
import signal
import socket
import struct
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional

import h5py
import numpy as np

from vidseq.services.sam2_config import (
    cleanup_port_files,
    find_free_port,
    write_pid_file,
    write_port_file,
)
from vidseq.services.mask_service import open_h5


def _encode_mask_rle(mask: np.ndarray) -> str:
    """
    Encode binary mask as binary RLE + base64.
    
    Format: Each run is (value: 1 byte, length: 4 bytes big-endian uint32)
    Then base64 encoded for JSON transport.
    
    Args:
        mask: Binary mask array (height, width), dtype=uint8, values 0 or 255
        
    Returns:
        Base64-encoded RLE string
    """
    flat = mask.flatten()
    binary_data = bytearray()
    i = 0
    
    while i < len(flat):
        value = flat[i]
        length = 1
        
        # Count consecutive identical values
        while i + length < len(flat) and flat[i + length] == value:
            length += 1
        
        # Pack as: 1 byte value, 4 bytes length (big-endian uint32)
        binary_data.extend(struct.pack('>BI', int(value), length))
        
        i += length
    
    # Base64 encode for JSON transport
    return base64.b64encode(bytes(binary_data)).decode('utf-8')


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


class SAM2TCPServer:
    """TCP server for SAM2 worker with sequential command processing."""
    
    def __init__(self, port: Optional[int] = None):
        """
        Initialize TCP server.
        
        Args:
            port: Port to bind to. If None, finds free port automatically.
        """
        self.port = port or find_free_port()
        self.server_socket: Optional[socket.socket] = None
        self.command_queue: queue.Queue = queue.Queue()
        self.active_connections: set[socket.socket] = set()
        self.connection_lock = threading.Lock()
        self.shutdown_timer: Optional[threading.Timer] = None
        self.shutdown_timeout = 10.0
        self.running = False
        self._is_processing = False
        
        # SAM2 state
        self.predictor = None
        self.sessions: dict[int, tuple[dict, Any]] = {}
        
        # Config paths
        self.config_name = "configs/sam2.1/sam2.1_hiera_t.yaml"
        self.checkpoint_path = Path(__file__).parent.parent.parent / "sam2_models" / "sam2.1_hiera_tiny.pt"
    
    def start(self) -> None:
        """Start the TCP server."""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind(("localhost", self.port))
        self.server_socket.listen(5)
        self.server_socket.settimeout(1.0)  # Set timeout so accept() can be interrupted
        self.running = True
        
        print(f"[SAM2 Worker] TCP server started on localhost:{self.port}")
        
        # Write port and PID files
        write_port_file(self.port)
        write_pid_file(os.getpid())
        
        # Start command processing thread
        processing_thread = threading.Thread(target=self._process_commands, daemon=True)
        processing_thread.start()
        
        # Accept connections
        while self.running:
            try:
                conn, addr = self.server_socket.accept()
                # Only log on first connection or when connection count changes significantly
                with self.connection_lock:
                    conn_count = len(self.active_connections)
                if conn_count == 0:
                    print(f"[SAM2 Worker] First client connected from {addr}")
                client_thread = threading.Thread(
                    target=self._handle_client,
                    args=(conn, addr),
                    daemon=True
                )
                client_thread.start()
            except socket.timeout:
                # Timeout - check if we should still be running
                continue
            except OSError:
                # Server socket closed
                break
    
    def _handle_client(self, conn: socket.socket, addr: tuple) -> None:
        """Handle a client connection."""
        with self.connection_lock:
            self.active_connections.add(conn)
            conn_count = len(self.active_connections)
            # Cancel any pending shutdown
            if self.shutdown_timer:
                self.shutdown_timer.cancel()
                self.shutdown_timer = None
        
        # Only log if this is the first connection
        if conn_count == 1:
            print(f"[SAM2 Worker] Client connected from {addr}")
        
        try:
            # Set a short timeout for initial read to detect quick disconnects
            conn.settimeout(1.0)
            
            while self.running:
                # Receive command length
                length_bytes = self._recv_exact(conn, 4)
                if len(length_bytes) != 4:
                    break
                
                # Once we receive a command, remove timeout (command processing may take time)
                conn.settimeout(None)
                
                cmd_length = struct.unpack('>I', length_bytes)[0]
                
                # Receive command data
                cmd_bytes = self._recv_exact(conn, cmd_length)
                cmd_json = cmd_bytes.decode('utf-8')
                cmd = json.loads(cmd_json)
                
                # Create response callback
                response_callback = lambda result: self._send_response(conn, result)
                
                # Add to command queue
                self.command_queue.put((cmd, response_callback))
        except socket.timeout:
            # Connection opened but closed quickly (likely health check)
            # Don't log - this is expected behavior
            pass
        except Exception as e:
            # Only log actual errors, not quick disconnects
            if "timed out" not in str(e).lower():
                print(f"[SAM2 Worker] Error handling client {addr}: {e}")
        finally:
            # Connection closed
            with self.connection_lock:
                was_last = len(self.active_connections) == 1
                self.active_connections.discard(conn)
                # If no connections left, schedule shutdown
                if len(self.active_connections) == 0:
                    if was_last:
                        print(f"[SAM2 Worker] Last client disconnected from {addr}")
                    self._schedule_shutdown()
            try:
                conn.close()
            except Exception:
                pass
    
    def _recv_exact(self, conn: socket.socket, n: int) -> bytes:
        """Receive exactly n bytes from socket."""
        data = b''
        while len(data) < n:
            chunk = conn.recv(n - len(data))
            if not chunk:
                return data
            data += chunk
        return data
    
    def _send_response(self, conn: socket.socket, result: dict) -> None:
        """Send response to client."""
        # Check if connection is still active
        with self.connection_lock:
            if conn not in self.active_connections:
                return  # Connection closed, skip sending
        
        try:
            result_json = json.dumps(result)
            result_bytes = result_json.encode('utf-8')
            
            # Send length prefix
            length_prefix = struct.pack('>I', len(result_bytes))
            conn.sendall(length_prefix)
            
            # Send response data
            conn.sendall(result_bytes)
        except Exception as e:
            # Connection may have closed, remove it
            with self.connection_lock:
                self.active_connections.discard(conn)
            print(f"[SAM2 Worker] Error sending response: {e}")
    
    def _process_commands(self) -> None:
        """Process commands sequentially from queue."""
        while self.running:
            try:
                cmd, response_callback = self.command_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            
            try:
                self._is_processing = True
                self._handle_command(cmd, response_callback)
            except Exception as e:
                print(f"[SAM2 Worker] Error processing command: {e}")
                import traceback
                traceback.print_exc()
                error_result = {
                    "type": "error",
                    "error": str(e),
                    "request_id": cmd.get("request_id"),
                }
                response_callback(error_result)
            finally:
                self._is_processing = False
                # If no connections, schedule shutdown now that we're done processing
                with self.connection_lock:
                    if len(self.active_connections) == 0:
                        self._schedule_shutdown()
    
    def _handle_command(self, cmd: dict, response_callback) -> None:
        """Handle a single command and send result(s) via callback."""
        cmd_type = cmd.get("type")
        request_id = cmd.get("request_id")
        
        if cmd_type == "load_model":
            print("[SAM2 Worker] Loading SAM2 model...")
            
            try:
                import torch
                from sam2.build_sam import build_sam2_video_predictor
                
                torch.set_float32_matmul_precision('medium')
                
                self.predictor = build_sam2_video_predictor(
                    config_file=self.config_name,
                    ckpt_path=str(self.checkpoint_path),
                    device="cuda",
                    vos_optimized=False,
                )
                self.predictor.to(dtype=torch.bfloat16)
                
                print("[SAM2 Worker] Compiling image encoder...")
                print(f"[SAM2 Worker] Image encoder type before compile: {type(self.predictor.image_encoder)}")
                self.predictor.image_encoder = torch.compile(
                    self.predictor.image_encoder,
                    mode="max-autotune",
                    fullgraph=True
                )
                print(f"[SAM2 Worker] Image encoder type after compile: {type(self.predictor.image_encoder)}")
                print(f"[SAM2 Worker] Image encoder forward type: {type(self.predictor.image_encoder.forward)}")
                
                print("[SAM2 Worker] SAM2 model loaded and compiled!")
                result = {"type": "status", "status": "ready"}
                if request_id is not None:
                    result["request_id"] = request_id
                response_callback(result)
            except Exception as e:
                print(f"[SAM2 Worker] Failed to load model: {e}")
                import traceback
                traceback.print_exc()
                result = {
                    "type": "status",
                    "status": "error",
                    "error": str(e),
                }
                if request_id is not None:
                    result["request_id"] = request_id
                response_callback(result)
        
        elif cmd_type == "init_session":
            video_id = cmd["video_id"]
            video_path = Path(cmd["video_path"])
            
            print(f"[SAM2 Worker] Initializing session for video {video_id}...")
            
            try:
                if self.predictor is None:
                    raise RuntimeError("Model not loaded")
                
                if video_id in self.sessions:
                    inference_state, loader = self.sessions[video_id]
                    response_callback({
                        "type": "init_session_result",
                        "request_id": request_id,
                        "status": "ok",
                        "video_id": video_id,
                        "num_frames": inference_state["num_frames"],
                        "height": inference_state["video_height"],
                        "width": inference_state["video_width"],
                    })
                    return
                
                from vidseq.services.sam2streaming import LazyVideoFrameLoader
                loader = LazyVideoFrameLoader(video_path, offload_to_cpu=False, device="cuda")
                
                inference_state = _init_state_with_lazy_loader(self.predictor, loader)
                self.sessions[video_id] = (inference_state, loader)
                
                response_callback({
                    "type": "init_session_result",
                    "request_id": request_id,
                    "status": "ok",
                    "video_id": video_id,
                    "num_frames": inference_state["num_frames"],
                    "height": inference_state["video_height"],
                    "width": inference_state["video_width"],
                })
            except Exception as e:
                print(f"[SAM2 Worker] Failed to init session: {e}")
                import traceback
                traceback.print_exc()
                response_callback({
                    "type": "init_session_result",
                    "request_id": request_id,
                    "status": "error",
                    "error": str(e),
                })
        
        elif cmd_type == "add_point_prompt":
            video_id = cmd["video_id"]
            frame_idx = cmd["frame_idx"]
            points = cmd.get("points")
            labels = cmd.get("labels")
            box = cmd.get("box")
            obj_id = cmd.get("obj_id", 1)
            
            try:
                if self.predictor is None:
                    raise RuntimeError("Model not loaded")
                
                if video_id not in self.sessions:
                    raise RuntimeError(f"No session for video {video_id}")
                
                inference_state, loader = self.sessions[video_id]
                height = inference_state["video_height"]
                width = inference_state["video_width"]
                
                import torch
                points_arr = None
                labels_arr = None
                box_arr = None
                
                if points is not None and labels is not None:
                    points_arr = np.array(points, dtype=np.float32)
                    points_arr[:, 0] *= width
                    points_arr[:, 1] *= height
                    labels_arr = np.array(labels, dtype=np.int32)
                
                if box is not None:
                    # box format: [x1, y1, x2, y2]
                    box_arr = np.array(box, dtype=np.float32)
                
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    _, out_obj_ids, video_res_masks = self.predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=frame_idx,
                        obj_id=obj_id,
                        points=points_arr,
                        labels=labels_arr,
                        box=box_arr,
                        clear_old_points=False if box is None else True,
                    )
                    mask = _extract_mask(video_res_masks, out_obj_ids, height, width)
                
                response_callback({
                    "type": "add_point_prompt_result",
                    "request_id": request_id,
                    "status": "ok",
                    "mask_rle": _encode_mask_rle(mask),
                    "mask_shape": mask.shape,
                    "mask_dtype": str(mask.dtype),
                    "obj_id": obj_id,
                })
            except Exception as e:
                print(f"[SAM2 Worker] Failed to add point prompt: {e}")
                import traceback
                traceback.print_exc()
                response_callback({
                    "type": "add_point_prompt_result",
                    "request_id": request_id,
                    "status": "error",
                    "error": str(e),
                })
        
        elif cmd_type == "generate_training_masks":
            video_id = cmd["video_id"]
            start_frame_idx = cmd["start_frame_idx"]
            max_frames = cmd["max_frames"]
            project_path = Path(cmd["project_path"])
            num_frames = cmd["num_frames"]
            height = cmd["height"]
            width = cmd["width"]
            
            try:
                if self.predictor is None:
                    raise RuntimeError("Model not loaded")
                
                if video_id not in self.sessions:
                    raise RuntimeError(f"No session for video {video_id}")
                
                inference_state, loader = self.sessions[video_id]
                
                frame_count, frame_indices = self._propagate_video(
                    inference_state=inference_state,
                    video_id=video_id,
                    start_frame_idx=start_frame_idx,
                    max_frames=max_frames,
                    project_path=project_path,
                    num_frames=num_frames,
                    height=height,
                    width=width,
                )
                
                response_callback({
                    "type": "generate_training_masks_result",
                    "request_id": request_id,
                    "status": "ok",
                    "frames_processed": frame_count,
                    "frame_indices": frame_indices,
                })
            except Exception as e:
                print(f"[SAM2 Worker] Failed to generate training masks: {e}")
                import traceback
                traceback.print_exc()
                response_callback({
                    "type": "generate_training_masks_result",
                    "request_id": request_id,
                    "status": "error",
                    "error": str(e),
                })

        elif cmd_type == "segment_videos_batch":
            videos = cmd["videos"]
            project_path = Path(cmd["project_path"])
            log_path = Path(cmd["log_path"]) if cmd.get("log_path") else None
            
            try:
                if self.predictor is None:
                    raise RuntimeError("Model not loaded")
                
                log_file = None
                if log_path:
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    log_file = open(log_path, "a")
                
                def log(msg):
                    print(f"[SAM2 Worker] {msg}")
                    if log_file:
                        log_file.write(f"{msg}\n")
                        log_file.flush()
                
                log(f"Starting batch segmentation for {len(videos)} videos")
                
                processed_count = 0
                for v_info in videos:
                    video_id = v_info["video_id"]
                    job_id = v_info["job_id"]
                    video_path = Path(v_info["video_path"])
                    bbox = v_info["bbox"]
                    num_frames = v_info["num_frames"]
                    height = v_info["height"]
                    width = v_info["width"]
                    
                    log(f"Processing video {video_id} (Job #{job_id}): {video_path.name}")
                    
                    try:
                        # Close existing session if any
                        if video_id in self.sessions:
                            inf_state, ldr = self.sessions.pop(video_id)
                            ldr.close()
                        
                        # Init new session
                        from vidseq.services.sam2streaming import LazyVideoFrameLoader
                        loader = LazyVideoFrameLoader(video_path, offload_to_cpu=False, device="cuda")
                        inference_state = _init_state_with_lazy_loader(self.predictor, loader)
                        self.sessions[video_id] = (inference_state, loader)
                        
                        # Add bbox on frame 0
                        import torch
                        box_arr = np.array(bbox, dtype=np.float32)
                        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                            self.predictor.add_new_points_or_box(
                                inference_state=inference_state,
                                frame_idx=0,
                                obj_id=1,
                                box=box_arr,
                                clear_old_points=True,
                            )
                        
                        # Propagate through all frames
                        def progress_update(frame_idx):
                            response_callback({
                                "type": "batch_progress",
                                "request_id": request_id,
                                "job_id": job_id,
                                "video_id": video_id,
                                "current_frame": frame_idx,
                                "total_frames": num_frames,
                            })
                        
                        self._propagate_video(
                            inference_state=inference_state,
                            video_id=video_id,
                            start_frame_idx=0,
                            max_frames=num_frames,
                            project_path=project_path,
                            num_frames=num_frames,
                            height=height,
                            width=width,
                            progress_callback=progress_update
                        )
                        
                        # Close session to free memory
                        if video_id in self.sessions:
                            inf_state, ldr = self.sessions.pop(video_id)
                            ldr.close()
                            
                        log(f"Successfully processed video {video_id}")
                        response_callback({
                            "type": "batch_video_complete",
                            "request_id": request_id,
                            "job_id": job_id,
                            "video_id": video_id,
                            "status": "completed"
                        })
                        processed_count += 1
                        
                    except Exception as ve:
                        log(f"Error processing video {video_id}: {ve}")
                        import traceback
                        log(traceback.format_exc())
                        response_callback({
                            "type": "batch_video_complete",
                            "request_id": request_id,
                            "job_id": job_id,
                            "video_id": video_id,
                            "status": "failed",
                            "error": str(ve)
                        })
                
                log(f"Batch complete: {processed_count}/{len(videos)} videos processed")
                if log_file:
                    log_file.close()
                    
                response_callback({
                    "type": "segment_videos_batch_result",
                    "request_id": request_id,
                    "status": "ok",
                    "processed_count": processed_count
                })
                
            except Exception as e:
                print(f"[SAM2 Worker] Batch segmentation failed: {e}")
                import traceback
                traceback.print_exc()
                response_callback({
                    "type": "segment_videos_batch_result",
                    "request_id": request_id,
                    "status": "error",
                    "error": str(e),
                })
        
        elif cmd_type == "reset_state":
            video_id = cmd["video_id"]
            
            print(f"[SAM2 Worker] Resetting state for video {video_id}...")
            
            try:
                if self.predictor is None:
                    raise RuntimeError("Model not loaded")
                
                if video_id not in self.sessions:
                    response_callback({
                        "type": "reset_state_result",
                        "request_id": request_id,
                        "status": "ok",
                    })
                    return
                
                inference_state, loader = self.sessions[video_id]
                import torch
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    self.predictor.reset_state(inference_state)
                
                print(f"[SAM2 Worker] State reset for video {video_id}")
                response_callback({
                    "type": "reset_state_result",
                    "request_id": request_id,
                    "status": "ok",
                })
            except Exception as e:
                print(f"[SAM2 Worker] Failed to reset state: {e}")
                import traceback
                traceback.print_exc()
                response_callback({
                    "type": "reset_state_result",
                    "request_id": request_id,
                    "status": "error",
                    "error": str(e),
                })
        
        elif cmd_type == "close_session":
            video_id = cmd["video_id"]
            
            print(f"[SAM2 Worker] Closing session for video {video_id}...")
            
            try:
                if video_id in self.sessions:
                    inference_state, loader = self.sessions.pop(video_id)
                    loader.close()
                
                response_callback({
                    "type": "close_session_result",
                    "request_id": request_id,
                    "status": "ok",
                })
            except Exception as e:
                print(f"[SAM2 Worker] Failed to close session: {e}")
                response_callback({
                    "type": "close_session_result",
                    "request_id": request_id,
                    "status": "error",
                    "error": str(e),
                })
        
        elif cmd_type == "shutdown":
            print("[SAM2 Worker] Shutting down...")
            for video_id, (inference_state, loader) in list(self.sessions.items()):
                try:
                    loader.close()
                except Exception:
                    pass
            self.sessions.clear()
            self.running = False
            result = {"type": "shutdown_result", "status": "ok"}
            if request_id is not None:
                result["request_id"] = request_id
            response_callback(result)
        
        else:
            print(f"[SAM2 Worker] Unknown command type: {cmd_type}")
            response_callback({
                "type": "error",
                "error": f"Unknown command type: {cmd_type}",
                "request_id": request_id,
            })
    
    def _propagate_video(
        self,
        inference_state,
        video_id: int,
        start_frame_idx: int,
        max_frames: int,
        project_path: Path,
        num_frames: int,
        height: int,
        width: int,
        progress_callback=None
    ) -> tuple[int, list[int]]:
        """Helper to run SAM2 propagation and save to H5."""
        import torch
        frame_indices = []
        frame_count = 0
        
        with open_h5(project_path, 'a') as h5_file:
            # Pre-create all datasets
            mask_dataset_name = f"segmentation_masks/{video_id}"
            bbox_dataset_name = f"bounding_boxes/{video_id}"
            frame_type_dataset_name = f"frame_types/{video_id}"
            
            if mask_dataset_name not in h5_file:
                h5_file.create_dataset(
                    mask_dataset_name,
                    shape=(num_frames, height, width),
                    dtype=np.uint8,
                    fillvalue=0,
                    chunks=(1, height, width),
                    compression=None,
                )
            
            if bbox_dataset_name not in h5_file:
                h5_file.create_dataset(
                    bbox_dataset_name,
                    shape=(num_frames, 4),
                    dtype=np.float32,
                    fillvalue=0.0,
                    chunks=(1, 4),
                    compression=None,
                )
            
            if frame_type_dataset_name not in h5_file:
                h5_file.create_dataset(
                    frame_type_dataset_name,
                    shape=(num_frames,),
                    dtype=h5py.string_dtype(encoding='utf-8'),
                    fillvalue='',
                    chunks=(num_frames,),
                    compression=None,
                )
            
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                iterator = self.predictor.propagate_in_video(
                    inference_state=inference_state,
                    start_frame_idx=start_frame_idx,
                    max_frame_num_to_track=max_frames,
                    reverse=False,
                )
                
                for frame_idx, out_obj_ids, video_res_masks in iterator:
                    mask = _extract_mask(video_res_masks, out_obj_ids, height, width)
                    
                    mask_binary = mask > 0
                    if np.any(mask_binary):
                        rows = np.any(mask_binary, axis=1)
                        cols = np.any(mask_binary, axis=0)
                        if np.any(rows) and np.any(cols):
                            y_indices = np.where(rows)[0]
                            x_indices = np.where(cols)[0]
                            y1, y2 = y_indices[0], y_indices[-1]
                            x1, x2 = x_indices[0], x_indices[-1]
                            bbox_np = np.array([x1, y1, x2, y2], dtype=np.float32)
                        else:
                            bbox_np = None
                    else:
                        bbox_np = None
                    
                    h5_file[mask_dataset_name][frame_idx] = mask
                    if bbox_np is not None:
                        h5_file[bbox_dataset_name][frame_idx] = bbox_np
                    h5_file[frame_type_dataset_name][frame_idx] = 'train'
                    
                    frame_indices.append(frame_idx)
                    frame_count += 1
                    
                    if progress_callback and frame_count % 10 == 0:
                        progress_callback(frame_idx)
            
            h5_file.flush()
            
        return frame_count, frame_indices

    def _schedule_shutdown(self) -> None:
        """Schedule shutdown if no connections arrive within timeout."""
        def shutdown_if_still_empty():
            time.sleep(self.shutdown_timeout)
            with self.connection_lock:
                if len(self.active_connections) == 0 and not self._is_processing:
                    print("[SAM2 Worker] No connections and not processing, shutting down...")
                    self.stop()  # Clean up resources
                    print("[SAM2 Worker] Exiting process...")
                    os._exit(0)  # Forcefully exit the process (works from any thread)
        
        if self.shutdown_timer:
            self.shutdown_timer.cancel()
        self.shutdown_timer = threading.Timer(self.shutdown_timeout, shutdown_if_still_empty)
        self.shutdown_timer.start()
    
    def stop(self) -> None:
        """Stop the server."""
        self.running = False
        
        # Close all active connections
        with self.connection_lock:
            for conn in list(self.active_connections):
                try:
                    conn.close()
                except Exception:
                    pass
            self.active_connections.clear()
        
        # Close server socket
        if self.server_socket:
            try:
                self.server_socket.close()
            except Exception:
                pass
        
        # Clean up sessions
        for video_id, (inference_state, loader) in list(self.sessions.items()):
            try:
                loader.close()
            except Exception:
                pass
        self.sessions.clear()
        
        # Free GPU memory by deleting predictor and clearing CUDA cache
        if self.predictor is not None:
            import torch
            del self.predictor
            self.predictor = None
            torch.cuda.empty_cache()
            print("[SAM2 Worker] GPU memory cleared")
        
        cleanup_port_files()


def main():
    """Main entry point for standalone server."""
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="SAM2 TCP Worker Server")
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to bind to (default: auto-assign)"
    )
    args = parser.parse_args()
    
    server = SAM2TCPServer(port=args.port)
    
    def signal_handler(signum, frame):
        print("\n[SAM2 Worker] Received shutdown signal, cleaning up...")
        server.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        server.start()
    except KeyboardInterrupt:
        print("\n[SAM2 Worker] Interrupted, shutting down...")
        server.stop()
    finally:
        cleanup_port_files()


if __name__ == "__main__":
    main()
