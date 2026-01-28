"""SAM3 service package."""

from vidseq.services.sam3.frame_source import VideoFrameSource
from vidseq.services.sam3.service import (
    SAM3Service,
    close_session,
    get_status,
    init_session,
    get_session,
    segment_all_videos,
    start_loading_in_background,
    add_point_prompt,
    propagate,
    reset_frame,
    reset_video,
    generate_training_masks,
    shutdown_worker,
)
from vidseq.services.sam3.streaming_segmentor import StreamingSegmentor

__all__ = [
    "SAM3Service",
    "StreamingSegmentor",
    "VideoFrameSource",
    "close_session",
    "get_status",
    "init_session",
    "get_session",
    "segment_all_videos",
    "start_loading_in_background",
    "add_point_prompt",
    "propagate",
    "reset_frame",
    "reset_video",
    "generate_training_masks",
    "shutdown_worker",
]
