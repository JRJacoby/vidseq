"""Backwards-compatible import shim for sam2.service.

This module re-exports everything from the new location for backwards compatibility.
New code should import directly from vidseq.services.sam2.
"""

from vidseq.services.sam2.service import *
from vidseq.services.sam2.service import (
    SAM2Service,
    SAM2Status,
    VideoSessionInfo,
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
