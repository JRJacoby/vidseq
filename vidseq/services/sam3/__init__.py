"""SAM3 service package."""

from vidseq.services.sam3.frame_source import VideoFrameSource
from vidseq.services.sam3.service import (
    SAM3Service,
    close_session,
    get_all_prompts,
    get_prompts_for_frame,
    get_status,
    init_session,
    reset_state,
    segment_all_videos,
    start_loading_in_background,
    add_point_prompt,
    clear_frame_prompts,
    clear_prompts_for_frame,
    generate_training_masks,
)
from vidseq.services.sam3.streaming_segmentor import StreamingSegmentor

__all__ = [
    "SAM3Service",
    "StreamingSegmentor",
    "VideoFrameSource",
    "close_session",
    "get_all_prompts",
    "get_prompts_for_frame",
    "get_status",
    "init_session",
    "reset_state",
    "segment_all_videos",
    "start_loading_in_background",
    "add_point_prompt",
    "clear_frame_prompts",
    "clear_prompts_for_frame",
    "generate_training_masks",
]
