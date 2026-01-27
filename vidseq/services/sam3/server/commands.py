"""SAM3 TCP command handlers.

Each function handles one command type. Functions take only the arguments they need.
"""

from pathlib import Path
from typing import Any, Callable

import numpy as np
from sqlalchemy import update
from sqlalchemy.orm import Session

from vidseq.models.registry import Job
from vidseq.models.video import Video
from vidseq.models.utils import utc_now
from vidseq.services.database_manager import DatabaseManager
from vidseq.services.sam3.inference.propagate import propagate_video
from vidseq.services.sam3.utils import encode_mask_rle, extract_mask_from_sam3_output, init_state_with_lazy_loader


def handle_load_model(checkpoint_path: Path) -> tuple[dict, Any]:
    """
    Load SAM3 model.

    Returns:
        Tuple of (response_dict, model)
    """
    import torch
    from sam3.model_builder import build_sam3_video_model

    print("[SAM3 Worker] Loading SAM3 model...")

    torch.set_float32_matmul_precision("medium")

    model = build_sam3_video_model(
        checkpoint_path=str(checkpoint_path),
        load_from_HF=False,
        apply_temporal_disambiguation=True,
        compile=True,
        device="cuda",
    )

    print("[SAM3 Worker] SAM3 model loaded!")
    return {"type": "status", "status": "ready"}, model


def handle_init_session(
    params: dict,
    model,
    sessions: dict,
) -> dict:
    """
    Initialize video inference session.

    Args:
        params: Command params with video_id, video_path
        model: SAM3 model instance
        sessions: Dict of video_id -> (inference_state, loader)

    Returns:
        Response dict
    """
    from vidseq.services.sam3.inference.streaming import LazyVideoFrameLoader

    video_id = params["video_id"]
    video_path = Path(params["video_path"])

    print(f"[SAM3 Worker] Initializing session for video {video_id}...")

    if model is None:
        raise RuntimeError("Model not loaded")

    # Return existing session if already initialized
    if video_id in sessions:
        inference_state, loader = sessions[video_id]
        return {
            "type": "init_session_result",
            "status": "ok",
            "video_id": video_id,
            "num_frames": inference_state["num_frames"],
            "height": inference_state["orig_height"],
            "width": inference_state["orig_width"],
        }

    loader = LazyVideoFrameLoader(video_path, offload_to_cpu=False, device="cuda")
    inference_state = init_state_with_lazy_loader(model, loader)
    sessions[video_id] = (inference_state, loader)

    return {
        "type": "init_session_result",
        "status": "ok",
        "video_id": video_id,
        "num_frames": inference_state["num_frames"],
        "height": inference_state["orig_height"],
        "width": inference_state["orig_width"],
    }


def handle_add_prompt(
    params: dict,
    model,
    sessions: dict,
) -> dict:
    """
    Add point/box prompt to a frame.

    Args:
        params: Command params with video_id, frame_idx, points, labels, box, obj_id
        model: SAM3 model instance
        sessions: Dict of video_id -> (inference_state, loader)

    Returns:
        Response dict with mask_rle
    """
    video_id = params["video_id"]
    frame_idx = params["frame_idx"]
    points = params.get("points")
    labels = params.get("labels")
    box = params.get("box")
    obj_id = params.get("obj_id", 1)

    if model is None:
        raise RuntimeError("Model not loaded")

    if video_id not in sessions:
        raise RuntimeError(f"No session for video {video_id}")

    inference_state, loader = sessions[video_id]
    height = inference_state["orig_height"]
    width = inference_state["orig_width"]

    if points is not None and labels is not None:
        # Points are already in [0,1] normalized coords from frontend
        # SAM3 add_prompt with points goes to tracker (instance-level)
        _, postprocessed_out = model.add_prompt(
            inference_state,
            frame_idx=frame_idx,
            points=np.array(points, dtype=np.float32),
            point_labels=np.array(labels, dtype=np.int32),
            obj_id=obj_id,
            rel_coordinates=True,
        )
    elif box is not None:
        # box is [x1, y1, x2, y2] in pixel coords from frontend
        # Convert to normalized [0,1] XYWH for SAM3 detector
        x1, y1, x2, y2 = box
        norm_x = x1 / width
        norm_y = y1 / height
        norm_w = (x2 - x1) / width
        norm_h = (y2 - y1) / height
        boxes_xywh = np.array([[norm_x, norm_y, norm_w, norm_h]], dtype=np.float32)
        box_labels = np.array([1], dtype=np.int64)  # positive

        _, postprocessed_out = model.add_prompt(
            inference_state,
            frame_idx=frame_idx,
            boxes_xywh=boxes_xywh,
            box_labels=box_labels,
        )
    else:
        raise RuntimeError("Either points+labels or box must be provided")

    # Extract mask from postprocessed output
    mask = extract_mask_from_sam3_output(postprocessed_out, height, width)

    return {
        "type": "add_point_prompt_result",
        "status": "ok",
        "mask_rle": encode_mask_rle(mask),
        "mask_shape": mask.shape,
        "mask_dtype": str(mask.dtype),
        "obj_id": obj_id,
    }


def handle_generate_training_masks(
    params: dict,
    model,
    sessions: dict,
) -> dict:
    """
    Generate training masks by propagating through video.

    Args:
        params: Command params with video_id, start_frame_idx, max_frames, etc.
        model: SAM3 model instance
        sessions: Dict of video_id -> (inference_state, loader)

    Returns:
        Response dict with frames_processed, frame_indices
    """
    video_id = params["video_id"]
    start_frame_idx = params["start_frame_idx"]
    max_frames = params["max_frames"]
    project_path = Path(params["project_path"])
    num_frames = params["num_frames"]
    height = params["height"]
    width = params["width"]

    if model is None:
        raise RuntimeError("Model not loaded")

    if video_id not in sessions:
        raise RuntimeError(f"No session for video {video_id}")

    inference_state, loader = sessions[video_id]

    frame_count, frame_indices, _ = propagate_video(
        model=model,
        inference_state=inference_state,
        video_id=video_id,
        start_frame_idx=start_frame_idx,
        max_frames=max_frames,
        project_path=project_path,
        num_frames=num_frames,
        height=height,
        width=width,
    )

    return {
        "type": "generate_training_masks_result",
        "status": "ok",
        "frames_processed": frame_count,
        "frame_indices": frame_indices,
    }


def handle_segment_videos_batch(
    params: dict,
    model,
    sessions: dict,
    response_callback: Callable[[dict], None],
) -> dict:
    """
    Segment multiple videos in batch.

    Sends initial response with job IDs, then processes videos. Progress is tracked
    via job status updates in the database (UI polls for updates). TCP progress
    callbacks are avoided to prevent blocking when client disconnects.

    Args:
        params: Command params with videos, project_id, project_path
        model: SAM3 model instance
        sessions: Dict of video_id -> (inference_state, loader)
        response_callback: Callback for initial response only

    Returns:
        Final response dict
    """
    from vidseq.services.sam3.inference.streaming import LazyVideoFrameLoader

    videos = params["videos"]
    project_id = params["project_id"]
    project_path = Path(params["project_path"])
    request_id = params["request_id"]

    if model is None:
        raise RuntimeError("Model not loaded")

    # Create jobs for all videos in registry DB
    db_manager = DatabaseManager.get_instance()
    registry_engine = db_manager.get_registry_engine()
    project_engine = db_manager.get_project_engine(project_path)

    job_ids = []
    video_configs = []

    with Session(registry_engine) as registry_session:
        for v_info in videos:
            video_id = v_info["video_id"]
            log_path = project_path / "logs" / f"segmentation_video_{video_id}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)

            job = Job(
                type="video_segmentation",
                status="pending",
                project_id=project_id,
                details={
                    "video_id": video_id,
                    "current_frame": 0,
                    "total_frames": v_info["num_frames"],
                },
                log_path=str(log_path),
            )
            registry_session.add(job)
            registry_session.flush()

            job_ids.append(job.id)
            video_configs.append({
                "job_id": job.id,
                "video_id": video_id,
                "video_path": v_info["video_path"],
                "bbox": v_info["bbox"],
                "num_frames": v_info["num_frames"],
                "height": v_info["height"],
                "width": v_info["width"],
            })

        registry_session.commit()

    # Send initial response with job IDs
    response_callback({
        "type": "segment_videos_batch_started",
        "request_id": request_id,
        "job_ids": job_ids,
    })

    # Process each video
    processed_count = 0
    for v_config in video_configs:
        video_id = v_config["video_id"]
        job_id = v_config["job_id"]
        video_path = Path(v_config["video_path"])
        bbox = v_config["bbox"]
        num_frames = v_config["num_frames"]
        height = v_config["height"]
        width = v_config["width"]
        log_path = project_path / "logs" / f"segmentation_video_{video_id}.log"

        log_file = None
        try:
            log_file = open(log_path, "a")

            def log(msg):
                print(f"[SAM3 Worker] {msg}")
                if log_file:
                    log_file.write(f"{msg}\n")
                    log_file.flush()

            log(f"Processing video {video_id} (Job #{job_id}): {video_path.name}")

            # Update Job.status = 'running' and Video.segmentation_status = 'in_progress'
            with Session(registry_engine) as session:
                session.execute(
                    update(Job)
                    .where(Job.id == job_id)
                    .values(status="running", updated_at=utc_now())
                )
                session.commit()

            with Session(project_engine) as session:
                session.execute(
                    update(Video)
                    .where(Video.id == video_id)
                    .values(segmentation_status="in_progress")
                )
                session.commit()

            # Close existing session if any
            if video_id in sessions:
                inf_state, ldr = sessions.pop(video_id)
                ldr.close()

            # Init new session
            loader = LazyVideoFrameLoader(video_path, offload_to_cpu=False, device="cuda")
            inference_state = init_state_with_lazy_loader(model, loader)
            sessions[video_id] = (inference_state, loader)

            # Add bbox on frame 0
            # Convert XYXY pixel bbox to normalized [0,1] XYWH
            x1, y1, x2, y2 = bbox
            norm_x = x1 / width
            norm_y = y1 / height
            norm_w = (x2 - x1) / width
            norm_h = (y2 - y1) / height
            boxes_xywh = np.array([[norm_x, norm_y, norm_w, norm_h]], dtype=np.float32)
            box_labels = np.array([1], dtype=np.int64)

            model.add_prompt(
                inference_state,
                frame_idx=0,
                boxes_xywh=boxes_xywh,
                box_labels=box_labels,
            )

            # Progress callback for this video
            # NOTE: Only updates DB, not TCP. The client disconnects after receiving
            # the initial response, so TCP progress updates would block on full buffer.
            # UI polls job status from DB instead.
            def progress_update(frame_idx):
                with Session(registry_engine) as session:
                    session.execute(
                        update(Job)
                        .where(Job.id == job_id)
                        .values(
                            details={
                                "video_id": video_id,
                                "current_frame": frame_idx,
                                "total_frames": num_frames,
                            },
                            updated_at=utc_now(),
                        )
                    )
                    session.commit()

            # Propagate through all frames
            _, _, stats = propagate_video(
                model=model,
                inference_state=inference_state,
                video_id=video_id,
                start_frame_idx=0,
                max_frames=num_frames,
                project_path=project_path,
                num_frames=num_frames,
                height=height,
                width=width,
                progress_callback=progress_update,
            )

            # Close session to free memory
            if video_id in sessions:
                inf_state, ldr = sessions.pop(video_id)
                ldr.close()

            # Update Video.segmentation_status = 'segmented' and Job.status = 'completed'
            update_values = {
                "segmentation_status": "segmented",
                "min_confidence": stats.get("min"),
                "p50_confidence": stats.get("p50"),
                "p95_confidence": stats.get("p95"),
            }

            with Session(project_engine) as session:
                session.execute(
                    update(Video).where(Video.id == video_id).values(**update_values)
                )
                session.commit()

            with Session(registry_engine) as session:
                session.execute(
                    update(Job)
                    .where(Job.id == job_id)
                    .values(
                        status="completed",
                        details={
                            "video_id": video_id,
                            "current_frame": num_frames,
                            "total_frames": num_frames,
                        },
                        updated_at=utc_now(),
                    )
                )
                session.commit()

            log(f"Successfully processed video {video_id}")
            # NOTE: Don't send TCP response here - client may have disconnected.
            # Job status in DB is sufficient for UI polling.
            processed_count += 1

        except Exception as ve:
            error_msg = str(ve)
            if log_file:
                log_file.write(f"Error processing video {video_id}: {ve}\n")
                import traceback

                log_file.write(traceback.format_exc())
                log_file.flush()

            print(f"[SAM3 Worker] Error processing video {video_id}: {ve}")
            import traceback

            traceback.print_exc()

            # Update Video.segmentation_status = None and Job.status = 'failed'
            with Session(project_engine) as session:
                session.execute(
                    update(Video).where(Video.id == video_id).values(segmentation_status=None)
                )
                session.commit()

            with Session(registry_engine) as session:
                session.execute(
                    update(Job)
                    .where(Job.id == job_id)
                    .values(
                        status="failed",
                        details={
                            "video_id": video_id,
                            "error": error_msg,
                        },
                        updated_at=utc_now(),
                    )
                )
                session.commit()

            # NOTE: Don't send TCP response here - client may have disconnected.
            # Job status in DB is sufficient for UI polling.
        finally:
            if log_file:
                log_file.close()

    print(f"[SAM3 Worker] Batch complete: {processed_count}/{len(video_configs)} videos processed")

    return {
        "type": "segment_videos_batch_result",
        "status": "ok",
        "processed_count": processed_count,
        "job_ids": job_ids,
    }


def handle_reset_state(
    params: dict,
    model,
    sessions: dict,
) -> dict:
    """
    Reset inference state for a video.

    Args:
        params: Command params with video_id
        model: SAM3 model instance
        sessions: Dict of video_id -> (inference_state, loader)

    Returns:
        Response dict
    """
    video_id = params["video_id"]

    print(f"[SAM3 Worker] Resetting state for video {video_id}...")

    if model is None:
        raise RuntimeError("Model not loaded")

    if video_id not in sessions:
        return {"type": "reset_state_result", "status": "ok"}

    inference_state, loader = sessions[video_id]
    model.reset_state(inference_state)

    print(f"[SAM3 Worker] State reset for video {video_id}")
    return {"type": "reset_state_result", "status": "ok"}


def handle_clear_frame_prompts(
    params: dict,
    model,
    sessions: dict,
) -> dict:
    """
    Clear prompts for a specific frame.

    SAM3 doesn't expose clear_all_prompts_in_frame through the video model API.
    Use model.reset_state() as a simpler approach (clears ALL prompts).

    Args:
        params: Command params with video_id, frame_idx, obj_id
        model: SAM3 model instance
        sessions: Dict of video_id -> (inference_state, loader)

    Returns:
        Response dict
    """
    video_id = params["video_id"]
    frame_idx = params["frame_idx"]
    obj_id = params.get("obj_id", 1)

    print(f"[SAM3 Worker] Clearing prompts for video {video_id}, frame {frame_idx}, obj {obj_id}...")

    if model is None:
        raise RuntimeError("Model not loaded")

    if video_id not in sessions:
        return {"type": "clear_frame_prompts_result", "status": "ok"}

    inference_state, loader = sessions[video_id]
    # SAM3 doesn't have per-frame prompt clearing via the video model API.
    # Reset the entire state as a fallback.
    model.reset_state(inference_state)

    print(f"[SAM3 Worker] Cleared prompts for video {video_id}, frame {frame_idx}")
    return {"type": "clear_frame_prompts_result", "status": "ok"}


def handle_close_session(
    params: dict,
    sessions: dict,
) -> dict:
    """
    Close video session and free resources.

    Args:
        params: Command params with video_id
        sessions: Dict of video_id -> (inference_state, loader)

    Returns:
        Response dict
    """
    video_id = params["video_id"]

    print(f"[SAM3 Worker] Closing session for video {video_id}...")

    if video_id in sessions:
        inference_state, loader = sessions.pop(video_id)
        loader.close()

    return {"type": "close_session_result", "status": "ok"}


def handle_shutdown(sessions: dict) -> dict:
    """
    Shutdown server and clean up all sessions.

    Args:
        sessions: Dict of video_id -> (inference_state, loader)

    Returns:
        Response dict
    """
    print("[SAM3 Worker] Shutting down...")

    for video_id, (inference_state, loader) in list(sessions.items()):
        try:
            loader.close()
        except Exception:
            pass
    sessions.clear()

    return {"type": "shutdown_result", "status": "ok"}
