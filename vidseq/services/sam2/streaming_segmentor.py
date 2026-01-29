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

    def _mask_exists(self, masks, idx: int) -> bool:
        """Check if a valid mask exists at index.

        Args:
            masks: Indexable mask storage (array or h5 dataset).
            idx: Frame index to check.

        Returns:
            True if a non-empty mask exists at the index.
        """
        try:
            mask = masks[idx]
            if isinstance(mask, np.ndarray):
                return mask.any()
            return mask is not None
        except (IndexError, KeyError):
            return False

    def _get_image_features(
        self, video_id: str, frame_idx: int
    ) -> tuple[torch.Tensor, dict]:
        """Get image features for a frame, with caching.

        Loads the frame, preprocesses it, and runs through the SAM2 image encoder.
        Caches only the most recent frame's features to avoid memory bloat.

        Args:
            video_id: The video identifier.
            frame_idx: Frame index to encode.

        Returns:
            (image_tensor, backbone_out) where:
            - image_tensor is the preprocessed image tensor (1, 3, 1024, 1024)
            - backbone_out is the dict from forward_image with backbone_fpn, vision_pos_enc
        """
        session = self.sessions[video_id]
        cached = session.get("cached_features", {})

        # Check cache
        if cached.get("frame_idx") == frame_idx:
            return cached["image"], cached["backbone_out"]

        # Load frame from source
        frames = session["frames"]
        frame_bgr = frames[frame_idx]

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Resize to model input size (1024x1024)
        frame_resized = cv2.resize(
            frame_rgb, (self.INPUT_SIZE, self.INPUT_SIZE), interpolation=cv2.INTER_LINEAR
        )

        # Convert to tensor: HWC uint8 -> CHW float [0, 1]
        image_tensor = torch.from_numpy(frame_resized).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0).to(self.device)  # (1, 3, 1024, 1024)

        # Run through image encoder
        with torch.inference_mode():
            backbone_out = self.predictor.forward_image(image_tensor)

        # Cache (only most recent frame)
        session["cached_features"] = {
            "frame_idx": frame_idx,
            "image": image_tensor,
            "backbone_out": backbone_out,
        }

        return image_tensor, backbone_out

    def _prepare_backbone_features(
        self, backbone_out: dict
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[tuple[int, int]]]:
        """Prepare backbone features for track_step.

        Extracts and reshapes the FPN features and position encodings from
        the backbone output to the format expected by track_step.

        Args:
            backbone_out: Dict from forward_image with backbone_fpn, vision_pos_enc.

        Returns:
            (current_vision_feats, current_vision_pos_embeds, feat_sizes) where:
            - current_vision_feats: list of tensors in (HW, B, C) format
            - current_vision_pos_embeds: list of tensors in (HW, B, C) format
            - feat_sizes: list of (H, W) tuples for each feature level
        """
        # Get the number of feature levels from predictor
        num_feature_levels = self.predictor.num_feature_levels

        # Extract relevant feature levels (last num_feature_levels levels)
        feature_maps = backbone_out["backbone_fpn"][-num_feature_levels:]
        vision_pos_enc = backbone_out["vision_pos_enc"][-num_feature_levels:]

        # Get feature sizes from position encodings
        feat_sizes = [(x.shape[-2], x.shape[-1]) for x in vision_pos_enc]

        # Reshape from (B, C, H, W) to (HW, B, C)
        current_vision_feats = [x.flatten(2).permute(2, 0, 1) for x in feature_maps]
        current_vision_pos_embeds = [x.flatten(2).permute(2, 0, 1) for x in vision_pos_enc]

        return current_vision_feats, current_vision_pos_embeds, feat_sizes

    def _make_compact_output(self, current_out: dict) -> dict:
        """Extract essential fields from track_step output for memory storage.

        Args:
            current_out: Full output dict from track_step.

        Returns:
            Dict with only the fields needed for memory:
            maskmem_features, maskmem_pos_enc, pred_masks, obj_ptr, object_score_logits
        """
        return {
            "maskmem_features": current_out.get("maskmem_features"),
            "maskmem_pos_enc": current_out.get("maskmem_pos_enc"),
            "pred_masks": current_out.get("pred_masks"),
            "obj_ptr": current_out.get("obj_ptr"),
            "object_score_logits": current_out.get("object_score_logits"),
        }

    def _encode_stored_mask(self, video_id: str, frame_idx: int) -> None:
        """Encode a stored mask into the memory bank.

        Loads a mask from storage, converts it to a tensor, and runs through
        track_step with mask_inputs to encode it into the output_dict memory.

        This is used to reconstruct conditioning frame memories when opening
        a video that has existing masks.

        Args:
            video_id: The video identifier.
            frame_idx: Frame index whose mask should be encoded.

        Side Effects:
            - Stores result in output_dict["cond_frame_outputs"][frame_idx]
        """
        session = self.sessions[video_id]
        masks = session["masks"]
        output_dict = session["output_dict"]

        # Load mask from storage
        stored_mask = masks[frame_idx]

        # Resize mask to model input size and convert to tensor
        # Mask should be float in range [0, 1] with shape (1, 1, H, W)
        mask_resized = cv2.resize(
            stored_mask.astype(np.float32),
            (self.INPUT_SIZE, self.INPUT_SIZE),
            interpolation=cv2.INTER_NEAREST,
        )
        # Convert to binary float (0.0 or 1.0) and add batch/channel dims
        mask_tensor = torch.from_numpy((mask_resized > 127).astype(np.float32))
        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0).to(self.device)  # (1, 1, H, W)

        # Get image features
        image_tensor, backbone_out = self._get_image_features(video_id, frame_idx)

        # Prepare features for track_step
        current_vision_feats, current_vision_pos_embeds, feat_sizes = (
            self._prepare_backbone_features(backbone_out)
        )

        # Run track_step with mask_inputs to encode the mask into memory
        with torch.inference_mode():
            current_out = self.predictor.track_step(
                frame_idx=frame_idx,
                is_init_cond_frame=True,  # Treat as conditioning frame
                current_vision_feats=current_vision_feats,
                current_vision_pos_embeds=current_vision_pos_embeds,
                feat_sizes=feat_sizes,
                point_inputs=None,
                mask_inputs=mask_tensor,
                output_dict=output_dict,
                num_frames=session["num_frames"],
            )

        # Store in conditioning frame outputs
        output_dict["cond_frame_outputs"][frame_idx] = self._make_compact_output(current_out)

    def open_video(
        self,
        video_id: str,
        frames,  # Indexable returning BGR uint8 (H, W, 3)
        masks,  # Indexable/assignable for binary mask storage
        logits,  # Indexable/assignable for logits storage
        frame_dims: tuple[int, int],  # (height, width)
        cond_frame_indices: set[int] | list[int] | None = None,
    ) -> None:
        """Create a session with external frame, mask, and logits sources.

        This initializes the session and reconstructs conditioning frame memories
        from stored masks. Call this before using add_point_prompt or refine_mask.

        The caller is responsible for managing the lifecycle of frames, masks,
        and logits (e.g., opening/closing file handles).

        Args:
            video_id: Unique identifier for this video session.
            frames: Indexable frame source returning BGR uint8 arrays (H, W, 3).
                    Must support frames[frame_idx] access.
            masks: Indexable storage for binary masks (array or h5 dataset).
                   Will be used for both reading existing masks and writing new ones.
            logits: Indexable storage for low-res logits.
                    Used for iterative refinement with refine_mask().
            frame_dims: Tuple of (height, width) for the video frames.
            cond_frame_indices: Frame indices that have existing user prompts.
                   Their memories will be reconstructed from stored masks.
                   Pass None or empty for a fresh session.

        Raises:
            ValueError: If video_id already exists.
        """
        if video_id in self.sessions:
            raise ValueError(f"Video '{video_id}' is already open. Close it first.")

        # Normalize cond_frame_indices to a set
        if cond_frame_indices is None:
            cond_frame_indices = set()
        else:
            cond_frame_indices = set(cond_frame_indices)

        # Create session with SAM2-style memory structure
        session = {
            "frames": frames,
            "masks": masks,
            "logits": logits,
            "cond_frame_indices": cond_frame_indices,
            "frame_dims": frame_dims,
            "num_frames": 10000,  # Large default (SAM2 uses this for temporal position encoding)
            "output_dict": {
                "cond_frame_outputs": {},
                "non_cond_frame_outputs": {},
            },
            "cached_features": {},  # Cache for most recent frame's features
        }

        self.sessions[video_id] = session

        # Reconstruct conditioning frame memories from stored masks
        cond_to_reconstruct = [
            idx for idx in sorted(cond_frame_indices) if self._mask_exists(masks, idx)
        ]
        if cond_to_reconstruct:
            import sys

            print(f"  Reconstructing {len(cond_to_reconstruct)} cond frame memories...")
            sys.stdout.flush()
            for i, idx in enumerate(cond_to_reconstruct):
                print(
                    f"    [{i + 1}/{len(cond_to_reconstruct)}] Encoding frame {idx}...",
                    end="",
                )
                sys.stdout.flush()
                self._encode_stored_mask(video_id, idx)
                print(" done")
                sys.stdout.flush()

        print(
            f"Opened video '{video_id}' with "
            f"{len(session['output_dict']['cond_frame_outputs'])} cond frames"
        )

    def close_video(self, video_id: str) -> bool:
        """Close a video session and free resources.

        Clears the cached features and output_dict memory, then removes
        the session from tracking.

        Note: The caller is responsible for closing the frames and masks sources
        that were passed to open_video(). This method only removes the session
        from SAM2StreamingSegmentor's internal tracking.

        Args:
            video_id: The video identifier to close.

        Returns:
            True if the session was closed, False if video_id not found.
        """
        if video_id not in self.sessions:
            return False

        session = self.sessions[video_id]

        # Clear cached features to free GPU memory
        session["cached_features"].clear()

        # Clear output_dict memory
        session["output_dict"]["cond_frame_outputs"].clear()
        session["output_dict"]["non_cond_frame_outputs"].clear()

        # Remove session
        del self.sessions[video_id]

        return True
