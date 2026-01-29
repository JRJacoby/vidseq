"""SAM2-based streaming video segmentor for interactive annotation workflows.

================================================================================
PURPOSE
================================================================================

This module provides a SAM2StreamingSegmentor class that mirrors the API of the
SAM3 StreamingSegmentor (vidseq/services/sam3/streaming_segmentor.py). The goal
is to enable comparison testing between SAM2 and SAM3 backends for video object
segmentation.

SAM2 (Segment Anything Model 2) is Meta's video-capable segmentation model that
uses memory attention to propagate object masks across video frames.

================================================================================
ARCHITECTURE
================================================================================

SAM2 uses a different approach than SAM3:
- SAM2: build_sam2_video_predictor returns a SAM2VideoPredictor that manages
  inference state internally via init_state() and propagate_in_video()
- SAM3: separate backbone + tracker with explicit memory management

This wrapper adapts SAM2's API to match the StreamingSegmentor interface used
by the rest of the vidseq application.

================================================================================
USAGE
================================================================================

    # Initialize once (loads SAM2 model)
    segmentor = SAM2StreamingSegmentor()

    # Open a video session
    segmentor.open_video(
        video_id="video_001",
        frames=my_frame_source,
        masks=my_mask_storage,
        frame_dims=(1080, 1920),
        cond_frame_indices={0, 50},
    )

    # Add a point prompt
    segmentor.add_point_prompt(
        video_id="video_001",
        frame_idx=150,
        location=(320, 240),
        label=1,
    )

    # Propagate tracking
    segmentor.propagate(video_id="video_001", frame_idx=151)

    # Close when done
    segmentor.close_video("video_001")

"""

from __future__ import annotations

import sys

# Add SAM2 repo to path for imports
sys.path.insert(0, "/n/groups/datta/john/repos/sam2")

import cv2
import numpy as np
import torch

from sam2.build_sam import build_sam2_video_predictor


class SAM2StreamingSegmentor:
    """Stateful manager for SAM2 video segmentation sessions.

    This class provides an interface compatible with the SAM3 StreamingSegmentor,
    allowing the vidseq application to use either backend interchangeably.

    Attributes:
        device: Torch device for model inference.
        predictor: SAM2VideoPredictor instance.
        sessions: Dict mapping video_id to session state.
    """

    # Model input size (SAM2 uses 1024x1024)
    INPUT_SIZE = 1024

    def __init__(self, device: str | None = None):
        """Initialize the segmentor and load the SAM2 model.

        Args:
            device: Torch device string ("cuda", "cpu", etc.).
                    Defaults to "cuda" if available, else "cpu".
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading SAM2 model on {self.device}...")
        self.predictor = build_sam2_video_predictor(
            config_file="configs/sam2.1/sam2.1_hiera_l.yaml",
            ckpt_path="/n/groups/datta/john/repos/sam2/checkpoints/sam2.1_hiera_large.pt",
            device=self.device,
        )
        print("SAM2 model loaded.")

        # Sessions dict: video_id -> session state
        self.sessions: dict[str, dict] = {}

    def _scale_point(
        self, x: float, y: float, orig_w: int, orig_h: int
    ) -> tuple[float, float]:
        """Normalize point coordinates to [0, 1] range.

        SAM2's add_new_points_or_box expects coordinates normalized by video
        dimensions when normalize_coords=True (the default).

        Args:
            x, y: Point coordinates in original frame space.
            orig_w, orig_h: Original frame dimensions.

        Returns:
            (norm_x, norm_y) normalized to [0, 1] range.
        """
        norm_x = x / orig_w
        norm_y = y / orig_h
        return norm_x, norm_y
