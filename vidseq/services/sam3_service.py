"""Backwards-compatible import shim for sam3.service.

This module re-exports everything from the new location for backwards compatibility.
New code should import directly from vidseq.services.sam3.
"""

from vidseq.services.sam3.service import *
from vidseq.services.sam3.service import (
    SAM3Service,
    SAM3Status,
    VideoSessionInfo,
    add_point_prompt,
    close_session,
    generate_training_masks,
    get_session,
    get_status,
    init_session,
    propagate,
    reset_frame,
    reset_video,
    segment_all_videos,
    shutdown_worker,
    start_loading_in_background,
)
