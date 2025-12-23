"""SAM2 service package."""

from vidseq.services.sam2.service import (
    SAM2Service,
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

__all__ = [
    "SAM2Service",
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
