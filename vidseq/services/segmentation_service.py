"""Segmentation service - orchestrates SAM3 inference with mask storage."""

import base64
import io
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from vidseq.models.video import Video
from vidseq.services import frame_data_service, segmentation_tcp_client
from vidseq.services.array_storage import final_masks, tracker_masks
from vidseq.services.exceptions import MultiPointWithoutMaskError

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


def mask_to_png(mask: np.ndarray) -> bytes:
    """Convert a numpy mask to PNG bytes.

    Creates an RGBA PNG with the mask rendered as a semi-transparent blue overlay.
    Masked pixels (value > 0) become light blue, non-masked pixels are transparent.
    """
    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    masked = mask > 0
    rgba[masked, 0] = 102   # R
    rgba[masked, 1] = 179   # G
    rgba[masked, 2] = 255   # B
    rgba[masked, 3] = 102   # A (semi-transparent)
    # Non-masked pixels stay (0, 0, 0, 0) = fully transparent
    img = Image.fromarray(rgba, mode='RGBA')
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def get_mask_png(
    project_path: Path,
    video: Video,
    frame_idx: int,
) -> bytes:
    """
    Load and return mask as PNG bytes.

    Args:
        project_path: Path to the project folder
        video: Video model instance
        frame_idx: Frame index

    Returns:
        PNG bytes of the mask (zeros if no mask exists)
    """
    with tracker_masks(project_path, video.id, "r") as masks:
        mask = masks[frame_idx]

    return mask_to_png(mask)


def get_masks_batch_json(
    project_path: Path,
    video: Video,
    start_frame: int,
    count: int,
) -> list[dict]:
    """
    Load multiple masks and return as list of dicts with base64-encoded PNGs.

    Args:
        project_path: Path to the project folder
        video: Video model instance
        start_frame: Starting frame index
        count: Number of frames to load

    Returns:
        List of {"frame_idx": int, "png_base64": str}
    """
    with tracker_masks(project_path, video.id, "r") as masks_arr:
        end_frame = min(start_frame + count, video.num_frames)
        masks = np.array(masks_arr[start_frame:end_frame])

    result = []
    for i, mask in enumerate(masks):
        png_bytes = mask_to_png(mask)
        png_base64 = base64.b64encode(png_bytes).decode('ascii')
        result.append({
            "frame_idx": start_frame + i,
            "png_base64": png_base64,
        })

    return result


def get_final_mask_png(
    project_path: Path,
    video: Video,
    frame_idx: int,
) -> bytes:
    """
    Load and return final (corrected) mask as PNG bytes.

    Args:
        project_path: Path to the project folder
        video: Video model instance
        frame_idx: Frame index

    Returns:
        PNG bytes of the mask (zeros if no mask exists)
    """
    with final_masks(project_path, video.id, "r") as masks:
        mask = masks[frame_idx]
    return mask_to_png(mask)


def get_final_masks_batch_json(
    project_path: Path,
    video: Video,
    start_frame: int,
    count: int,
) -> list[dict]:
    """
    Load multiple final (corrected) masks and return as list of dicts with base64-encoded PNGs.

    Args:
        project_path: Path to the project folder
        video: Video model instance
        start_frame: Starting frame index
        count: Number of frames to load

    Returns:
        List of {"frame_idx": int, "png_base64": str}
    """
    with final_masks(project_path, video.id, "r") as masks_arr:
        end_frame = min(start_frame + count, video.num_frames)
        masks = np.array(masks_arr[start_frame:end_frame])

    result = []
    for i, mask in enumerate(masks):
        png_bytes = mask_to_png(mask)
        png_base64 = base64.b64encode(png_bytes).decode('ascii')
        result.append({
            "frame_idx": start_frame + i,
            "png_base64": png_base64,
        })

    return result


async def init_session(
    session: "AsyncSession",
    project_id: int,
    video: "Video",
    project_path: Path,
) -> "segmentation_tcp_client.VideoSessionInfo":
    """Initialize a SAM session for a video.

    Queries conditioning frames from the database and initializes
    the session on the GPU worker.

    Args:
        session: Async database session
        project_id: ID of the project
        video: Video model instance
        project_path: Path to the project folder

    Returns:
        VideoSessionInfo with video dimensions and frame count
    """
    from sqlalchemy import select
    from vidseq.models.conditioning_frame import ConditioningFrame

    # Query conditioning frames for this video
    result = await session.execute(
        select(ConditioningFrame.frame_idx)
        .where(ConditioningFrame.video_id == video.id)
    )
    cond_frame_indices = list(result.scalars().all())

    return segmentation_tcp_client.init_session(
        project_id=project_id,
        video_id=video.id,
        video_path=Path(video.path),
        project_path=project_path,
        num_frames=video.num_frames,
        height=video.height,
        width=video.width,
        cond_frame_indices=cond_frame_indices,
    )


def reset_frame_memory(project_id: int, video_id: int, frame_idx: int) -> None:
    """Clear SAM memory for a single frame.

    Only clears the in-memory state on the GPU worker. Does not touch
    H5 files or database. Does nothing if no active session exists.

    Args:
        project_id: ID of the project
        video_id: ID of the video
        frame_idx: Frame index to clear from memory
    """
    segmentation_tcp_client.reset_frame_memory(project_id, video_id, frame_idx)


async def submit_prompt(
    session: "AsyncSession",
    project_id: int,
    video_id: int,
    frame_idx: int,
    points: list[dict],
    labels: list[int],
    use_cond_memory: bool = True,
    use_non_cond_memory: bool = True,
) -> tuple[np.ndarray, int, int]:
    """Submit point prompt(s) for segmentation.

    Workflow determined by existing state:
    - 1 point, no existing mask: creates new mask (add_point_prompt)
    - 1 point, existing mask: refines mask with single point
    - 2+ points, existing mask: refines mask with all points
    - 2+ points, no existing mask: ERROR (can't refine without mask)

    Args:
        session: Async database session
        project_id: ID of the project
        video_id: ID of the video
        frame_idx: Frame index (0-based)
        points: List of {"x": float, "y": float} normalized coords
        labels: List of labels (1=positive, 0=negative)

    Returns:
        Tuple of (mask, n_cond_used, n_non_cond_used)

    Raises:
        MultiPointWithoutMaskError: If multi-point submitted without existing mask
        RuntimeError: If TCP client fails
    """
    # Check if this frame already has a mask
    has_existing_mask = await frame_data_service.get_has_tracker_mask(
        session, video_id, frame_idx
    )

    # Validate: multi-point requires existing mask
    if len(points) > 1 and not has_existing_mask:
        raise MultiPointWithoutMaskError()

    if has_existing_mask:
        # Refine existing mask using previous logits as dense prompt
        mask, score, n_cond_used, n_non_cond_used = segmentation_tcp_client.refine_mask(
            project_id=project_id,
            video_id=video_id,
            frame_idx=frame_idx,
            points=points,
            labels=labels,
            use_cond_memory=use_cond_memory,
            use_non_cond_memory=use_non_cond_memory,
        )
    else:
        # Create new mask on blank frame (single point only, validated above)
        p = points[0]
        label = labels[0]
        mask, score, n_cond_used, n_non_cond_used = segmentation_tcp_client.add_point_prompt(
            project_id=project_id,
            video_id=video_id,
            frame_idx=frame_idx,
            x=p["x"],
            y=p["y"],
            label=label,
            use_cond_memory=use_cond_memory,
            use_non_cond_memory=use_non_cond_memory,
        )

    # Ensure conditioning frame DB record exists — both new prompts and
    # refinements create conditioning frames in SAM2's in-memory state,
    # so the DB needs to match for session reconstruction on reopen.
    from sqlalchemy import select
    from vidseq.models.conditioning_frame import ConditioningFrame

    existing = await session.execute(
        select(ConditioningFrame)
        .where(ConditioningFrame.video_id == video_id)
        .where(ConditioningFrame.frame_idx == frame_idx)
    )
    if existing.scalar_one_or_none() is None:
        session.add(ConditioningFrame(video_id=video_id, frame_idx=frame_idx))
        await session.commit()

    # Update mask presence index and save confidence score
    has_content = bool(np.any(mask > 0))
    await frame_data_service.set_has_tracker_mask(
        session, video_id, frame_idx, has_content
    )
    await frame_data_service.save_score(session, video_id, frame_idx, score)

    # If refinement resulted in an empty mask, reset the frame's SAM state
    if has_existing_mask and not has_content:
        # Remove conditioning frame record from database
        from sqlalchemy import delete
        from vidseq.models.conditioning_frame import ConditioningFrame

        await session.execute(
            delete(ConditioningFrame)
            .where(ConditioningFrame.video_id == video_id)
            .where(ConditioningFrame.frame_idx == frame_idx)
        )
        await session.commit()

        # Clear SAM memory
        segmentation_tcp_client.reset_frame(
            project_id=project_id,
            video_id=video_id,
            frame_idx=frame_idx,
        )

    return mask, n_cond_used, n_non_cond_used


async def submit_box_prompt(
    session: "AsyncSession",
    project_id: int,
    video_id: int,
    frame_idx: int,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    use_cond_memory: bool = True,
    use_non_cond_memory: bool = True,
) -> tuple[np.ndarray, int, int]:
    """Submit a bounding box prompt for segmentation.

    Box is always an initial prompt — creates a new mask. Any existing
    conditioning state for this frame is cleared by the segmentor.

    Args:
        session: Async database session
        project_id: ID of the project
        video_id: ID of the video
        frame_idx: Frame index (0-based)
        x1, y1: Top-left corner in normalized [0, 1] coords
        x2, y2: Bottom-right corner in normalized [0, 1] coords

    Returns:
        Tuple of (mask, n_cond_used, n_non_cond_used)
    """
    mask, score, n_cond_used, n_non_cond_used = segmentation_tcp_client.add_box_prompt(
        project_id=project_id,
        video_id=video_id,
        frame_idx=frame_idx,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        use_cond_memory=use_cond_memory,
        use_non_cond_memory=use_non_cond_memory,
    )

    # Ensure conditioning frame DB record exists (upsert pattern)
    from sqlalchemy import select
    from vidseq.models.conditioning_frame import ConditioningFrame

    existing = await session.execute(
        select(ConditioningFrame)
        .where(ConditioningFrame.video_id == video_id)
        .where(ConditioningFrame.frame_idx == frame_idx)
    )
    if existing.scalar_one_or_none() is None:
        session.add(ConditioningFrame(video_id=video_id, frame_idx=frame_idx))
        await session.commit()

    # Update mask presence index and save confidence score
    has_content = bool(np.any(mask > 0))
    await frame_data_service.set_has_tracker_mask(
        session, video_id, frame_idx, has_content
    )
    await frame_data_service.save_score(session, video_id, frame_idx, score)

    return mask, n_cond_used, n_non_cond_used


async def propagate(
    session: "AsyncSession",
    project_id: int,
    video_id: int,
    project_path: Path,
    start_frame_idx: int,
    max_frames: int,
    num_frames: int,
    height: int,
    width: int,
) -> int:
    """Propagate segmentation masks forward from a frame.

    Requires an active SAM session with a tracked object.

    Args:
        session: Async database session
        project_id: ID of the project
        video_id: ID of the video
        project_path: Path to the project folder
        start_frame_idx: Frame to start propagation from
        max_frames: Maximum number of frames to propagate
        num_frames: Total frames in video
        height: Video height
        width: Video width

    Returns:
        Number of frames processed

    Raises:
        RuntimeError: If propagation fails (no active session, etc.)
    """
    frame_indices, scores = segmentation_tcp_client.generate_training_masks(
        project_id=project_id,
        video_id=video_id,
        start_frame_idx=start_frame_idx,
        max_frames=max_frames,
        project_path=project_path,
        num_frames=num_frames,
        height=height,
        width=width,
    )

    # Update has_tracker_mask for all propagated frames
    await frame_data_service.set_has_tracker_mask(session, video_id, frame_indices, True)

    # Save confidence scores
    await frame_data_service.save_scores_batch(session, video_id, scores)

    return len(frame_indices)


async def propagate_without_memory(
    session: "AsyncSession",
    project_id: int,
    video_id: int,
    project_path: Path,
    start_frame_idx: int,
    max_frames: int,
    num_frames: int,
    height: int,
    width: int,
) -> int:
    """Propagate using only conditioning frame memories (no temporal window).

    Same as propagate() but uses cond-only mode to prevent drift.
    """
    frame_indices, scores = segmentation_tcp_client.propagate_without_memory(
        project_id=project_id,
        video_id=video_id,
        start_frame_idx=start_frame_idx,
        max_frames=max_frames,
        project_path=project_path,
        num_frames=num_frames,
        height=height,
        width=width,
    )

    await frame_data_service.set_has_tracker_mask(session, video_id, frame_indices, True)
    await frame_data_service.save_scores_batch(session, video_id, scores)

    return len(frame_indices)


async def apply_detector(
    session: "AsyncSession",
    project_id: int,
    project_path: Path,
    video_ids: list[int],
) -> int:
    """Run trained detector on all frames of selected videos.

    Returns the number of videos processed.
    """
    from sqlalchemy import select
    from vidseq.models.video import Video

    # Fetch video objects
    result = await session.execute(
        select(Video).where(Video.id.in_(video_ids))
    )
    videos = list(result.scalars().all())
    if not videos:
        raise RuntimeError("No videos found")

    # Validate detector model exists
    model_path = project_path / "models" / "detector.pt"
    if not model_path.exists():
        raise RuntimeError("No trained detector model found. Train first.")

    # Run detector via TCP (blocking call to GPU worker)
    scores_by_video, bboxes_by_video = segmentation_tcp_client.apply_detector(
        project_path=project_path,
        videos=videos,
    )

    # Save results to database
    for video in videos:
        vid_scores = scores_by_video.get(video.id, [])
        vid_bboxes = bboxes_by_video.get(video.id, [])
        if vid_scores:
            await frame_data_service.save_detector_scores_batch(
                session, video.id, vid_scores
            )
        if vid_bboxes:
            await frame_data_service.save_detector_bboxes_batch(
                session, video.id, vid_bboxes
            )

    await session.commit()
    return len(videos)


async def apply_obb_detector(
    session: "AsyncSession",
    project_id: int,
    project_path: Path,
    video_ids: list[int],
) -> int:
    """Run OBB detector on all frames of selected videos.

    Returns the number of videos processed.
    """
    from sqlalchemy import select
    from vidseq.models.video import Video

    result = await session.execute(
        select(Video).where(Video.id.in_(video_ids))
    )
    videos = list(result.scalars().all())
    if not videos:
        raise RuntimeError("No videos found")

    model_path = project_path / "models" / "obb_detector.pt"
    if not model_path.exists():
        raise RuntimeError("No trained OBB detector model found. Train first.")

    scores_by_video, bboxes_by_video = segmentation_tcp_client.apply_obb_detector(
        project_path=project_path,
        videos=videos,
    )

    for video in videos:
        vid_scores = scores_by_video.get(video.id, [])
        vid_bboxes = bboxes_by_video.get(video.id, [])
        if vid_scores:
            await frame_data_service.save_obb_scores_batch(
                session, video.id, vid_scores
            )
        if vid_bboxes:
            await frame_data_service.save_obb_bboxes_batch(
                session, video.id, vid_bboxes
            )

    await session.commit()
    return len(videos)


async def apply_seg_detector(
    session: "AsyncSession",
    project_id: int,
    project_path: Path,
    video_ids: list[int],
) -> int:
    """Run seg detector on all frames of selected videos.

    Writes masks to detector_masks.h5 (via GPU worker) and scores/bboxes to DB.
    Returns the number of videos processed.
    """
    from sqlalchemy import select
    from vidseq.models.video import Video

    result = await session.execute(
        select(Video).where(Video.id.in_(video_ids))
    )
    videos = list(result.scalars().all())
    if not videos:
        raise RuntimeError("No videos found")

    model_path = project_path / "models" / "seg_detector.pt"
    if not model_path.exists():
        raise RuntimeError("No trained seg detector model found. Train first.")

    scores_by_video, bboxes_by_video = segmentation_tcp_client.apply_seg_detector(
        project_path=project_path,
        videos=videos,
    )

    for video in videos:
        vid_scores = scores_by_video.get(video.id, [])
        vid_bboxes = bboxes_by_video.get(video.id, [])
        if vid_scores:
            await frame_data_service.save_detector_scores_batch(
                session, video.id, vid_scores
            )
        if vid_bboxes:
            await frame_data_service.save_detector_bboxes_batch(
                session, video.id, vid_bboxes
            )
        # Set has_detector_mask flags for all frames with detections
        all_frame_indices = [int(s[0]) for s in vid_scores if len(s) >= 2 and s[1] > 0]
        if all_frame_indices:
            await frame_data_service.set_has_detector_mask_batch(
                session, video.id, all_frame_indices, True
            )

    await session.commit()
    return len(videos)


async def segment_all_videos(
    session: "AsyncSession",
    project_id: int,
    project_path: Path,
    video_ids: list[int],
    mode: str = "full",
) -> list[int]:
    """Segment selected videos using detector-tracker approach.

    Fetches videos by ID, validates detector model exists,
    runs propagate_with_detector on each video, and updates database flags.

    Args:
        session: Async database session
        project_id: ID of the project
        project_path: Path to the project folder
        video_ids: List of video IDs to segment
        mode: "full" for full propagate_with_detector, "simple" for simple forward-only mode.

    Returns:
        List of job IDs (empty for now - synchronous execution)

    Raises:
        ValueError: If no videos found or detector model missing
    """
    from sqlalchemy import select
    from vidseq.models.conditioning_frame import ConditioningFrame
    from vidseq.models.video import Video

    # Fetch videos by ID
    result = await session.execute(
        select(Video).where(Video.id.in_(video_ids)).order_by(Video.id)
    )
    videos = list(result.scalars().all())
    if not videos:
        raise ValueError("No videos found for the given IDs")

    # Check that detector model exists
    detector_model_path = project_path / "models" / "detector.pt"
    if not detector_model_path.exists():
        raise ValueError("Detector model not found. Please train the detector first.")

    # Query conditioning frames for all videos
    cond_frames_by_video: dict[int, list[int]] = {}
    training_frames_by_video: dict[int, list[int]] = {}
    for video in videos:
        result = await session.execute(
            select(ConditioningFrame.frame_idx)
            .where(ConditioningFrame.video_id == video.id)
        )
        cond_frames_by_video[video.id] = list(result.scalars().all())
        # Training frames are only needed in full mode
        if mode != "simple":
            training_frames_by_video[video.id] = await frame_data_service.get_training_frames(
                session, video.id
            )

    job_ids, scores_by_video, detector_scores_by_video, detector_bboxes_by_video = await segmentation_tcp_client.segment_all_videos(
        project_id=project_id,
        project_path=project_path,
        videos=videos,
        cond_frames_by_video=cond_frames_by_video,
        training_frames_by_video=training_frames_by_video,
        mode=mode,
    )

    # Update database flags for all frames in all videos
    for video in videos:
        all_frames = list(range(video.num_frames))
        await frame_data_service.set_has_tracker_mask(session, video.id, all_frames, True)
        await frame_data_service.set_has_final_mask(session, video.id, all_frames, True)

        # Save confidence scores
        video_scores = scores_by_video.get(video.id, [])
        if video_scores:
            await frame_data_service.save_scores_batch(session, video.id, video_scores)

        # Save detector confidence scores
        video_detector_scores = detector_scores_by_video.get(video.id, [])
        if video_detector_scores:
            await frame_data_service.save_detector_scores_batch(session, video.id, video_detector_scores)

        # Save detector bounding boxes
        video_detector_bboxes = detector_bboxes_by_video.get(video.id, [])
        if video_detector_bboxes:
            await frame_data_service.save_detector_bboxes_batch(
                session, video.id, video_detector_bboxes
            )

        video.segmentation_status = "segmented"

    await session.commit()
    return job_ids


async def co_segment_videos(
    session: "AsyncSession",
    project_id: int,
    project_path: Path,
    video_ids: list[int],
    confidence_threshold: float = 0.9,
) -> None:
    """Batch co-segmentation of associated videos.

    For each main video:
    1. Load its scores from DB
    2. Find its associated video
    3. Init session for the associated video
    4. Run propagate_with_associated
    5. Save results to DB
    6. Close session

    Continues on per-video errors.
    """
    from sqlalchemy import select

    # Fetch main videos
    result = await session.execute(
        select(Video).where(Video.id.in_(video_ids)).order_by(Video.id)
    )
    main_videos = list(result.scalars().all())
    if not main_videos:
        raise ValueError("No videos found for the given IDs")

    # Fetch associated videos
    result = await session.execute(
        select(Video).where(
            Video.associated_with_id.in_(video_ids),
            Video.is_associated == True,
        )
    )
    assoc_by_main = {v.associated_with_id: v for v in result.scalars().all()}

    for main_video in main_videos:
        assoc_video = assoc_by_main.get(main_video.id)
        if assoc_video is None:
            print(f"[Co-Segment] Video {main_video.id} has no associated video, skipping")
            continue

        try:
            # Load main video scores
            main_scores = await frame_data_service.load_all_scores(
                session, main_video.id
            )

            # Init session for associated video
            segmentation_tcp_client.init_session(
                project_id=project_id,
                video_id=assoc_video.id,
                video_path=Path(assoc_video.path),
                project_path=project_path,
                num_frames=assoc_video.num_frames,
                height=assoc_video.height,
                width=assoc_video.width,
                cond_frame_indices=[],
            )

            # Run propagation
            result = segmentation_tcp_client.propagate_with_associated(
                video_id=assoc_video.id,
                main_video_id=main_video.id,
                project_path=project_path,
                num_frames=assoc_video.num_frames,
                confidence_threshold=confidence_threshold,
                main_height=main_video.height,
                main_width=main_video.width,
                main_video_scores=main_scores,
            )

            if result.get("status") == "ok":
                # Save scores to DB
                scores = result.get("scores", [])
                if scores:
                    await frame_data_service.save_scores_batch(
                        session, assoc_video.id, scores
                    )

                # Update mask flags
                processed_frames = [s[0] for s in scores]
                await frame_data_service.set_has_tracker_mask(
                    session, assoc_video.id, processed_frames, True
                )
                await frame_data_service.set_has_final_mask(
                    session, assoc_video.id, processed_frames, True
                )

                assoc_video.segmentation_status = "segmented"
                print(f"[Co-Segment] Video {main_video.id} -> {assoc_video.id} complete")
            else:
                print(f"[Co-Segment] Failed for video {main_video.id}: {result.get('error')}")

            # Close session
            segmentation_tcp_client.close_session(project_id, assoc_video.id)

        except Exception as e:
            import traceback
            print(f"[Co-Segment] Error for video {main_video.id}: {e}")
            traceback.print_exc()
            try:
                segmentation_tcp_client.close_session(project_id, assoc_video.id)
            except Exception:
                pass

    await session.commit()


def get_tracker_mask_bbox(
    project_path: Path,
    video_id: int,
    frame_idx: int,
) -> dict | None:
    """Get bounding box of tracker mask for a single frame.

    Returns {x1, y1, x2, y2} dict or None if mask is empty.
    Coordinates are in native video pixel space.
    """
    from vidseq.services.array_storage import compute_bbox_from_mask

    try:
        with tracker_masks(project_path, video_id, "r") as masks:
            mask = np.asarray(masks[frame_idx])
            bbox = compute_bbox_from_mask(mask)
    except FileNotFoundError:
        return None

    if bbox is None:
        return None
    return {"x1": float(bbox[0]), "y1": float(bbox[1]), "x2": float(bbox[2]), "y2": float(bbox[3])}


def get_tracker_mask_bboxes_batch(
    project_path: Path,
    video_id: int,
    start_frame: int,
    count: int,
    num_frames: int,
) -> list[dict]:
    """Get bounding boxes for a range of tracker mask frames.

    Returns list of {frame_idx, x1, y1, x2, y2} dicts, only for non-empty masks.
    Uses batch H5 read for performance.
    """
    from vidseq.services.array_storage import compute_bbox_from_mask

    end_frame = min(start_frame + count, num_frames)
    bboxes = []

    try:
        with tracker_masks(project_path, video_id, "r") as masks:
            masks_arr = np.asarray(masks[start_frame:end_frame])
            for i, mask in enumerate(masks_arr):
                bbox = compute_bbox_from_mask(mask)
                if bbox is not None:
                    bboxes.append({
                        "frame_idx": start_frame + i,
                        "x1": float(bbox[0]),
                        "y1": float(bbox[1]),
                        "x2": float(bbox[2]),
                        "y2": float(bbox[3]),
                    })
    except FileNotFoundError:
        pass

    return bboxes


def shutdown() -> None:
    """Shutdown the SAM2 worker, freeing GPU memory.

    Raises RuntimeError if sessions are active.
    """
    segmentation_tcp_client.shutdown_worker()

