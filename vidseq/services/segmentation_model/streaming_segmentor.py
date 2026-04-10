"""SAM2-based streaming video segmentor for interactive annotation workflows.

================================================================================
PURPOSE
================================================================================

This module provides a SAM2StreamingSegmentor class for video object segmentation.
SAM2 (Segment Anything Model 2) is Meta's video-capable segmentation model that
uses memory attention to propagate object masks across video frames.

================================================================================
ARCHITECTURE
================================================================================

SAM2's architecture:
- build_sam2_video_predictor returns a SAM2VideoPredictor that manages
  inference state internally via init_state() and propagate_in_video()
- Memory attention allows tracking objects across frames
- Supports point prompts, box prompts, and mask prompts

This wrapper provides a StreamingSegmentor interface used by the vidseq application.

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

from typing import Callable

import cv2
import numpy as np
import torch

from sam2.build_sam import build_sam2_hq_video_predictor


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

    # Low-res logits size (SAM2 outputs 256x256)
    LOGITS_SIZE = 256

    # Memory window size for non-conditioning frames.
    # SAM2's num_maskmem defaults to 7 (1 current + 6 previous frames).
    # We use 6 for loading previous frames, 7 for eviction threshold.
    MEM_WINDOW = 6

    def __init__(self, device: str | None = None, compile_model: bool = True):
        """Initialize the segmentor and load the SAM2 model.

        Args:
            device: Torch device string ("cuda", "cpu", etc.).
                    Defaults to "cuda" if available, else "cpu".
            compile_model: Whether to use torch.compile for faster inference.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading SAM2-HQ model on {self.device}...")
        self.predictor = build_sam2_hq_video_predictor(
            config_file="configs/sam2.1/sam2.1_hiera_l.yaml",
            ckpt_path="/n/groups/datta/john/repos/sam-hq/sam-hq2/checkpoints/sam2.1_hq_hiera_large.pt",
            device=self.device,
        )
        print("SAM2-HQ model loaded.")

        # Compile model components for faster inference
        if compile_model and self.device == "cuda":
            print("Compiling SAM2 image encoder with torch.compile...")
            # Compile image encoder (the main bottleneck - 55% of inference time)
            self.predictor.image_encoder = torch.compile(
                self.predictor.image_encoder,
            )
            # Run warmup inference to trigger actual compilation (torch.compile is lazy)
            print("Running warmup inference (this may take a minute)...")
            dummy_input = torch.zeros(1, 3, self.INPUT_SIZE, self.INPUT_SIZE, device=self.device)
            with torch.inference_mode(), torch.autocast("cuda", torch.bfloat16):
                _ = self.predictor.forward_image(dummy_input)
            print("SAM2 image encoder ready.")

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
        self, video_id: str, frame_idx: int, frame: np.ndarray
    ) -> tuple[torch.Tensor, dict]:
        """Get image features for a frame.

        Preprocesses the frame and runs through the SAM2 image encoder.

        Note: No caching - with torch.compile and CUDA graphs, the backbone is fast,
        and caching causes issues with stale tensor references to CUDA graph buffers.

        Args:
            video_id: The video identifier.
            frame_idx: Frame index (unused, kept for API compatibility).
            frame: BGR uint8 frame data (H, W, 3).

        Returns:
            (image_tensor, backbone_out) where:
            - image_tensor is the preprocessed image tensor (1, 3, 1024, 1024)
            - backbone_out is the dict from forward_image with backbone_fpn, vision_pos_enc
        """
        # GPU-accelerated preprocessing: transfer raw BGR uint8, then process on GPU
        # 1. Transfer raw uint8 to GPU (smaller transfer than float32)
        frame_gpu = torch.from_numpy(frame).to(self.device)  # (H, W, 3) uint8

        # 2. BGR→RGB by indexing channels, HWC→CHW, normalize to [0,1] - all fused on GPU
        #    frame_gpu[..., [2,1,0]] does BGR→RGB reorder
        #    .permute(2,0,1) does HWC→CHW
        #    .float() / 255.0 normalizes
        image_tensor = frame_gpu[..., [2, 1, 0]].permute(2, 0, 1).float().div_(255.0)

        # 3. Resize on GPU to model input size (1024x1024)
        image_tensor = torch.nn.functional.interpolate(
            image_tensor.unsqueeze(0),  # Add batch dim: (1, 3, H, W)
            size=(self.INPUT_SIZE, self.INPUT_SIZE),
            mode="bilinear",
            align_corners=False,
        )  # (1, 3, 1024, 1024)

        # Run through image encoder with autocast for automatic mixed precision
        with torch.inference_mode(), torch.autocast("cuda", torch.bfloat16):
            backbone_out = self.predictor.forward_image(image_tensor)

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

    def _extract_score(self, current_out: dict) -> float:
        """Extract predicted IoU score from track_step output.

        Args:
            current_out: Output dict from track_step containing 'ious'.

        Returns:
            Float confidence score in [0, 1].
        """
        return current_out["ious"][0, 0].item()

    def _encode_stored_mask(
        self, video_id: str, frame_idx: int, frame: np.ndarray, mask: np.ndarray,
        store_as_cond: bool = True,
    ) -> dict | None:
        """Encode a stored mask into the memory bank.

        Converts the mask to a tensor and runs through track_step with
        mask_inputs to encode it into the output_dict memory.

        This is used to reconstruct conditioning frame memories when opening
        a video that has existing masks, or to encode non-conditioning frame
        memories for arbitrary frame access.

        Args:
            video_id: The video identifier.
            frame_idx: Frame index whose mask should be encoded.
            frame: BGR uint8 frame data (H, W, 3).
            mask: Binary mask data (H, W) uint8.
            store_as_cond: If True, store to cond_frame_outputs and return None.
                If False, return the compact output dict without storing.

        Returns:
            None if store_as_cond=True, otherwise the compact output dict.

        Side Effects:
            - If store_as_cond=True, stores result in output_dict["cond_frame_outputs"][frame_idx]
        """
        session = self.sessions[video_id]
        output_dict = session["output_dict"]

        # Use provided mask
        stored_mask = mask

        # Resize mask to model input size and convert to tensor
        # Mask should be bfloat16 in range [0, 1] with shape (1, 1, H, W)
        mask_resized = cv2.resize(
            stored_mask.astype(np.float32),
            (self.INPUT_SIZE, self.INPUT_SIZE),
            interpolation=cv2.INTER_NEAREST,
        )
        # Convert to binary float (0.0 or 1.0) and add batch/channel dims
        mask_tensor = torch.from_numpy((mask_resized > 127).astype(np.float32))
        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0).to(self.device)  # (1, 1, H, W)

        # Get image features
        image_tensor, backbone_out = self._get_image_features(video_id, frame_idx, frame)

        # Prepare features for track_step
        current_vision_feats, current_vision_pos_embeds, feat_sizes = (
            self._prepare_backbone_features(backbone_out)
        )

        # Run track_step with mask_inputs to encode the mask into memory
        with torch.inference_mode(), torch.autocast("cuda", torch.bfloat16):
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

        # Store or return based on store_as_cond
        compact = self._make_compact_output(current_out)
        if store_as_cond:
            output_dict["cond_frame_outputs"][frame_idx] = compact
            return None
        else:
            return compact

    def _set_memory_frame(
        self,
        video_id: str,
        frame_idx: int,
        frames,  # Indexable frame source: frames[idx] -> np.ndarray (H, W, 3)
        masks,  # Indexable mask source: masks[idx] -> np.ndarray (H, W)
    ) -> None:
        """Prepare non_cond_frame_outputs for tracking at frame_idx.

        Clears all existing non-cond memory, then walks backward from frame_idx-1,
        loading masks and encoding them into memory. Stops after MEM_WINDOW frames
        or when hitting a gap (empty mask).

        Args:
            video_id: The video session ID.
            frame_idx: Target frame we're about to track/prompt.
            frames: Indexable frame source returning BGR uint8 (H, W, 3).
            masks: Indexable mask source returning uint8 (H, W).
        """

        session = self.sessions[video_id]
        output_dict = session["output_dict"]
        cond_frame_indices = session["cond_frame_indices"]

        # 1. Clear all existing non-cond memory
        output_dict["non_cond_frame_outputs"].clear()

        # 2. Early return if no previous frames exist
        if frame_idx <= 0:
            return

        # 3. Walk backward, encode into memory
        frames_added = 0
        for prev_idx in range(frame_idx - 1, -1, -1):
            if frames_added >= self.MEM_WINDOW:
                break

            # Skip conditioning frames (handled separately by SAM2)
            if prev_idx in cond_frame_indices:
                continue

            # Load mask - stop at gaps
            mask = np.asarray(masks[prev_idx])
            if mask.sum() == 0:
                break

            # Encode into memory (store_as_cond=False returns the dict)
            frame = frames[prev_idx]
            memory_out = self._encode_stored_mask(
                video_id, prev_idx, frame, mask, store_as_cond=False
            )
            output_dict["non_cond_frame_outputs"][prev_idx] = memory_out
            frames_added += 1

    def open_video(
        self,
        video_id: str,
        frame_dims: tuple[int, int],  # (height, width)
        cond_frame_indices: set[int] | list[int] | None = None,
        frames=None,  # Optional: for memory reconstruction only
        masks=None,  # Optional: for memory reconstruction only
    ) -> None:
        """Create a session for video segmentation.

        This initializes the session with modeling state only. If frames and masks
        are provided along with cond_frame_indices, conditioning frame memories
        will be reconstructed from the stored masks.

        The caller is responsible for managing the lifecycle of any file handles.
        This method does NOT store references to frames, masks, or other handles.

        Args:
            video_id: Unique identifier for this video session.
            frame_dims: Tuple of (height, width) for the video frames.
            cond_frame_indices: Frame indices that have existing user prompts.
                   Their memories will be reconstructed from stored masks.
                   Pass None or empty for a fresh session.
            frames: Optional indexable frame source (only needed if reconstructing).
            masks: Optional indexable mask source (only needed if reconstructing).

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

        # Create session with ONLY modeling state (no file handles)
        session = {
            "cond_frame_indices": cond_frame_indices,
            "frame_dims": frame_dims,
            "num_frames": 10000,  # Large default (SAM2 uses this for temporal position encoding)
            "output_dict": {
                "cond_frame_outputs": {},
                "non_cond_frame_outputs": {},
            },
        }

        self.sessions[video_id] = session

        # Reconstruct conditioning frame memories from stored masks (if provided)
        if frames is not None and masks is not None:
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
                    frame = frames[idx]
                    mask = masks[idx]
                    self._encode_stored_mask(video_id, idx, frame, mask)
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

        # Clear output_dict memory
        session["output_dict"]["cond_frame_outputs"].clear()
        session["output_dict"]["non_cond_frame_outputs"].clear()

        # Remove session
        del self.sessions[video_id]

        return True

    def reset_frame(self, video_id: str, frame_idx: int) -> None:
        """Reset a frame: clear its memory state.

        This only clears the SAM modeling state (conditioning frames, memory).
        The caller is responsible for clearing any stored masks/logits in H5.

        Args:
            video_id: The video identifier.
            frame_idx: Frame index to reset.
        """
        session = self.sessions[video_id]
        output_dict = session["output_dict"]
        cond_frame_indices = session["cond_frame_indices"]

        # Remove from cond_frame_indices
        cond_frame_indices.discard(frame_idx)

        # Remove from output_dict
        output_dict["cond_frame_outputs"].pop(frame_idx, None)
        output_dict["non_cond_frame_outputs"].pop(frame_idx, None)

    def reset_video(self, video_id: str) -> None:
        """Reset entire video: clear all masks and memory.

        Args:
            video_id: The video identifier.
        """
        session = self.sessions[video_id]
        output_dict = session["output_dict"]
        cond_frame_indices = session["cond_frame_indices"]

        # Clear output_dict (memory bank)
        output_dict["cond_frame_outputs"].clear()
        output_dict["non_cond_frame_outputs"].clear()

        # Clear cond_frame_indices
        cond_frame_indices.clear()

    def add_point_prompt(
        self,
        video_id: str,
        frame_idx: int,
        location: tuple[float, float] | list[tuple[float, float]],
        label: int | list[int],
        frames,  # Indexable frame source
        masks,  # Indexable mask source
        use_cond_memory: bool = True,
        use_non_cond_memory: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, float, int, int]:
        """Add point prompt(s) to a BLANK frame and generate initial mask.

        This method is for adding prompts to frames that don't have existing masks.
        For refining existing masks with additional points, use refine_mask() instead.

        Args:
            video_id: The video identifier.
            frame_idx: Index of the frame to annotate.
            location: (x, y) point or list of points in original frame coords.
            label: Label(s) for each point. 1=positive, 0=negative.
            frames: Indexable frame source returning BGR uint8 (H, W, 3).
            masks: Indexable mask source returning uint8 (H, W).

        Returns:
            Tuple of (mask, logits, score) where:
            - mask: Binary mask array (height, width) with dtype uint8, values 0 or 255.
            - logits: Low-res logits array (256, 256) for potential refinement.
            - score: Predicted IoU confidence in [0, 1].

        Raises:
            KeyError: If video_id is not open.
        """
        # 1. Get session state
        session = self.sessions[video_id]
        frame_dims = session["frame_dims"]  # (height, width)
        output_dict = session["output_dict"]
        cond_frame_indices = session["cond_frame_indices"]

        # 2. Prepare memory for arbitrary frame access
        self._set_memory_frame(video_id, frame_idx, frames, masks)

        # 3. Get frame from source
        frame = frames[frame_idx]

        # 4. Normalize location/label to lists
        if isinstance(location, tuple) and len(location) == 2 and not isinstance(location[0], tuple):
            # Single point: (x, y)
            locations = [location]
        else:
            locations = list(location)

        if isinstance(label, int):
            labels = [label]
        else:
            labels = list(label)

        # 5. Scale points to INPUT_SIZE (1024) space
        # Original coords are in frame_dims (height, width), need to scale to 1024x1024
        orig_h, orig_w = frame_dims
        scaled_points = []
        for x, y in locations:
            # Scale from original frame space to 1024x1024 space
            scaled_x = x * self.INPUT_SIZE / orig_w
            scaled_y = y * self.INPUT_SIZE / orig_h
            scaled_points.append([scaled_x, scaled_y])

        # 6. Create point_inputs dict with tensors on device
        point_coords = torch.tensor(scaled_points, dtype=torch.float32, device=self.device)
        point_coords = point_coords.unsqueeze(0)  # (1, N, 2) - batch dim
        point_labels = torch.tensor(labels, dtype=torch.int32, device=self.device)
        point_labels = point_labels.unsqueeze(0)  # (1, N)

        point_inputs = {
            "point_coords": point_coords,
            "point_labels": point_labels,
        }

        # 7. Get image features and prepare backbone features
        _, backbone_out = self._get_image_features(video_id, frame_idx, frame)
        current_vision_feats, current_vision_pos_embeds, feat_sizes = (
            self._prepare_backbone_features(backbone_out)
        )

        # 8. Build filtered output_dict based on memory flags
        filtered_output_dict = {
            "cond_frame_outputs": output_dict["cond_frame_outputs"] if use_cond_memory else {},
            "non_cond_frame_outputs": output_dict["non_cond_frame_outputs"] if use_non_cond_memory else {},
        }

        # 9. Determine is_init_cond_frame from *filtered* cond outputs
        is_init_cond_frame = len(filtered_output_dict["cond_frame_outputs"]) == 0

        # 10. Call track_step with filtered memory
        with torch.inference_mode(), torch.autocast("cuda", torch.bfloat16):
            current_out = self.predictor.track_step(
                frame_idx=frame_idx,
                is_init_cond_frame=is_init_cond_frame,
                current_vision_feats=current_vision_feats,
                current_vision_pos_embeds=current_vision_pos_embeds,
                feat_sizes=feat_sizes,
                point_inputs=point_inputs,
                mask_inputs=None,
                output_dict=filtered_output_dict,
                num_frames=session["num_frames"],
            )

        # 11. Extract mask from pred_masks_high_res
        # pred_masks_high_res has shape (1, num_objects, H, W), we want first object
        pred_mask_high_res = current_out["pred_masks_high_res"][0, 0]  # (H, W)

        # 11. Threshold at 0, convert to uint8 * 255, resize to original dims
        mask_binary = (pred_mask_high_res > 0).to(torch.uint8).mul(255).cpu().numpy()
        mask_resized = cv2.resize(
            mask_binary,
            (orig_w, orig_h),  # (width, height) for cv2.resize
            interpolation=cv2.INTER_NEAREST,
        )

        # 12. Get low-res logits for potential refinement
        # pred_masks has shape (1, num_objects, H, W) - low res version
        pred_masks_low_res = current_out["pred_masks"][0, 0].cpu().numpy()

        # 13. Add frame_idx to cond_frame_indices
        cond_frame_indices.add(frame_idx)

        # 14. Store compact output in output_dict["cond_frame_outputs"]
        output_dict["cond_frame_outputs"][frame_idx] = self._make_compact_output(current_out)

        # 15. Return mask, logits, and score (caller saves to storage)
        score = self._extract_score(current_out)
        n_cond_used = len(filtered_output_dict["cond_frame_outputs"])
        n_non_cond_used = len(filtered_output_dict["non_cond_frame_outputs"])
        return mask_resized, pred_masks_low_res, score, n_cond_used, n_non_cond_used

    def add_box_prompt(
        self,
        video_id: str,
        frame_idx: int,
        box: tuple[float, float, float, float],  # (x1, y1, x2, y2) in pixel coords
        frames,  # Indexable frame source: frames[idx] -> np.ndarray (H, W, 3)
        masks,   # Indexable mask source: masks[idx] -> np.ndarray (H, W)
        use_cond_memory: bool = True,
        use_non_cond_memory: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, float, int, int]:
        """Add a bounding box prompt to a frame and generate initial mask.

        Creates a new mask from the box prompt. Any existing conditioning state
        for this frame is cleared first. For refining an existing mask with
        additional points, use refine_mask() instead.

        Args:
            video_id: The video identifier.
            frame_idx: Index of the frame to annotate.
            box: (x1, y1, x2, y2) bounding box in original frame pixel coords.
            frames: Indexable frame source returning BGR uint8 (H, W, 3).
            masks: Indexable mask source returning uint8 (H, W).

        Returns:
            Tuple of (mask, logits, score) where:
            - mask: Binary mask array (height, width) with dtype uint8, values 0 or 255.
            - logits: Low-res logits array (256, 256) for potential refinement.
            - score: Predicted IoU confidence in [0, 1].
        """
        # 1. Get session state
        session = self.sessions[video_id]
        frame_dims = session["frame_dims"]  # (height, width)
        output_dict = session["output_dict"]
        cond_frame_indices = session["cond_frame_indices"]

        # 2. Pop existing conditioning output for this frame to avoid self-bias
        output_dict["cond_frame_outputs"].pop(frame_idx, None)

        # 3. Prepare memory for arbitrary frame access
        self._set_memory_frame(video_id, frame_idx, frames, masks)

        # 4. Get frame from source
        frame = frames[frame_idx]

        # 5. Scale box coords to INPUT_SIZE (1024) space
        orig_h, orig_w = frame_dims
        x1, y1, x2, y2 = box
        scaled_x1 = x1 * self.INPUT_SIZE / orig_w
        scaled_y1 = y1 * self.INPUT_SIZE / orig_h
        scaled_x2 = x2 * self.INPUT_SIZE / orig_w
        scaled_y2 = y2 * self.INPUT_SIZE / orig_h

        # 6. Create point_inputs with SAM2 box convention (labels 2=TL, 3=BR)
        point_coords = torch.tensor(
            [[[scaled_x1, scaled_y1], [scaled_x2, scaled_y2]]],
            dtype=torch.float32,
            device=self.device,
        )
        point_labels = torch.tensor(
            [[2, 3]], dtype=torch.int32, device=self.device
        )
        point_inputs = {
            "point_coords": point_coords,
            "point_labels": point_labels,
        }

        # 7. Get image features and prepare backbone features
        _, backbone_out = self._get_image_features(video_id, frame_idx, frame)
        current_vision_feats, current_vision_pos_embeds, feat_sizes = (
            self._prepare_backbone_features(backbone_out)
        )

        # 8. Build filtered output_dict based on memory flags
        filtered_output_dict = {
            "cond_frame_outputs": output_dict["cond_frame_outputs"] if use_cond_memory else {},
            "non_cond_frame_outputs": output_dict["non_cond_frame_outputs"] if use_non_cond_memory else {},
        }

        # 9. Determine is_init_cond_frame from *filtered* cond outputs
        is_init_cond_frame = len(filtered_output_dict["cond_frame_outputs"]) == 0

        # 10. Call track_step with filtered memory
        with torch.inference_mode(), torch.autocast("cuda", torch.bfloat16):
            current_out = self.predictor.track_step(
                frame_idx=frame_idx,
                is_init_cond_frame=is_init_cond_frame,
                current_vision_feats=current_vision_feats,
                current_vision_pos_embeds=current_vision_pos_embeds,
                feat_sizes=feat_sizes,
                point_inputs=point_inputs,
                mask_inputs=None,
                output_dict=filtered_output_dict,
                num_frames=session["num_frames"],
            )

        # 10. Extract high-res mask, threshold, resize to original dims
        pred_mask_high_res = current_out["pred_masks_high_res"][0, 0]
        mask_binary = (pred_mask_high_res > 0).to(torch.uint8).mul(255).cpu().numpy()
        mask_resized = cv2.resize(
            mask_binary,
            (orig_w, orig_h),
            interpolation=cv2.INTER_NEAREST,
        )

        # 11. Get low-res logits for potential refinement
        pred_masks_low_res = current_out["pred_masks"][0, 0].cpu().numpy()

        # 12. Add frame_idx to cond_frame_indices
        cond_frame_indices.add(frame_idx)

        # 13. Store compact output in cond_frame_outputs
        output_dict["cond_frame_outputs"][frame_idx] = self._make_compact_output(current_out)

        # 14. Return mask, logits, and score
        score = self._extract_score(current_out)
        n_cond_used = len(filtered_output_dict["cond_frame_outputs"])
        n_non_cond_used = len(filtered_output_dict["non_cond_frame_outputs"])
        return mask_resized, pred_masks_low_res, score, n_cond_used, n_non_cond_used

    def refine_mask(
        self,
        video_id: str,
        frame_idx: int,
        location: tuple[float, float] | list[tuple[float, float]],
        label: int | list[int],
        frames,  # Indexable frame source
        masks,  # Indexable mask source
        prev_logits: np.ndarray,
        use_cond_memory: bool = True,
        use_non_cond_memory: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, float, int, int]:
        """Refine an existing mask with point prompt(s).

        Uses the previous mask logits as context for refinement.

        Args:
            video_id: The video identifier.
            frame_idx: Index of the frame to refine.
            location: (x, y) point or list of points in original frame coords.
            label: Label(s) for each point. 1=positive, 0=negative.
            frames: Indexable frame source returning BGR uint8 (H, W, 3).
            masks: Indexable mask source returning uint8 (H, W).
            prev_logits: Previous low-res logits (256, 256) from prior call.

        Returns:
            Tuple of (mask, logits, score) where:
            - mask: Refined binary mask array (height, width) with dtype uint8.
            - logits: Updated low-res logits array (256, 256).
            - score: Predicted IoU confidence in [0, 1].

        Raises:
            KeyError: If video_id is not open.
        """
        # 1. Get session state
        session = self.sessions[video_id]
        frame_dims = session["frame_dims"]  # (height, width)
        output_dict = session["output_dict"]

        # 2. Remove current frame from cond memory to avoid self-bias during refinement.
        #    We don't want the OLD mask memory influencing the NEW refinement.
        old_cond_output = output_dict["cond_frame_outputs"].pop(frame_idx, None)

        # 3. Prepare memory for arbitrary frame access (loads non-cond frames)
        self._set_memory_frame(video_id, frame_idx, frames, masks)

        # 4. Build filtered output_dict based on memory flags
        filtered_output_dict = {
            "cond_frame_outputs": output_dict["cond_frame_outputs"] if use_cond_memory else {},
            "non_cond_frame_outputs": output_dict["non_cond_frame_outputs"] if use_non_cond_memory else {},
        }

        # 5. Determine is_init_cond_frame from filtered cond outputs.
        # When filtered cond is empty (no cond memory available or disabled),
        # use init path — SAM2 asserts cond frames exist on the non-init path.
        # prev_logits still provide refinement context regardless.
        is_init_cond_frame = len(filtered_output_dict["cond_frame_outputs"]) == 0

        # 5. Get frame from source
        frame = frames[frame_idx]

        # 6. Convert prev_logits to tensor: shape (1, 1, 256, 256), float32, on device
        # SAM2 low-res logits are 256x256. Clamp to [-32, 32] to avoid numerical issues
        # (matching SAM2's add_new_points_or_box behavior)
        prev_logits_tensor = torch.from_numpy(prev_logits.astype(np.float32))
        prev_logits_tensor = prev_logits_tensor.unsqueeze(0).unsqueeze(0).to(self.device)
        prev_logits_tensor = torch.clamp(prev_logits_tensor, -32.0, 32.0)

        # 7. Normalize location/label to lists
        if isinstance(location, tuple) and len(location) == 2 and not isinstance(location[0], tuple):
            # Single point: (x, y)
            locations = [location]
        else:
            locations = list(location)

        if isinstance(label, int):
            labels = [label]
        else:
            labels = list(label)

        # 8. Scale points to INPUT_SIZE (1024) space
        orig_h, orig_w = frame_dims
        scaled_points = []
        for x, y in locations:
            scaled_x = x * self.INPUT_SIZE / orig_w
            scaled_y = y * self.INPUT_SIZE / orig_h
            scaled_points.append([scaled_x, scaled_y])

        # 9. Create point_inputs dict with tensors on device
        point_coords = torch.tensor(scaled_points, dtype=torch.float32, device=self.device)
        point_coords = point_coords.unsqueeze(0)  # (1, N, 2) - batch dim
        point_labels = torch.tensor(labels, dtype=torch.int32, device=self.device)
        point_labels = point_labels.unsqueeze(0)  # (1, N)

        point_inputs = {
            "point_coords": point_coords,
            "point_labels": point_labels,
        }

        # 10. Get image features and prepare backbone features
        _, backbone_out = self._get_image_features(video_id, frame_idx, frame)
        current_vision_feats, current_vision_pos_embeds, feat_sizes = (
            self._prepare_backbone_features(backbone_out)
        )

        # 11. Log exactly what's being passed to SAM2
        cond_frames = sorted(output_dict["cond_frame_outputs"].keys())
        non_cond_frames = sorted(output_dict["non_cond_frame_outputs"].keys())
        print(
            f"[refine_mask] frame={frame_idx} "
            f"is_init_cond={is_init_cond_frame} "
            f"points={len(locations)} labels={labels} "
            f"prev_logits_range=[{prev_logits.min():.1f}, {prev_logits.max():.1f}] "
            f"cond_frames={cond_frames} "
            f"non_cond_frames({len(non_cond_frames)})={non_cond_frames[:10]}{'...' if len(non_cond_frames) > 10 else ''}"
        )

        # 12. Call track_step with point_inputs and prev_sam_mask_logits
        with torch.inference_mode(), torch.autocast("cuda", torch.bfloat16):
            current_out = self.predictor.track_step(
                frame_idx=frame_idx,
                is_init_cond_frame=is_init_cond_frame,
                current_vision_feats=current_vision_feats,
                current_vision_pos_embeds=current_vision_pos_embeds,
                feat_sizes=feat_sizes,
                point_inputs=point_inputs,
                mask_inputs=None,
                output_dict=filtered_output_dict,
                num_frames=session["num_frames"],
                prev_sam_mask_logits=prev_logits_tensor,
            )

        # 13. Extract mask, threshold, resize to original dims
        pred_mask_high_res = current_out["pred_masks_high_res"][0, 0]  # (H, W)
        mask_binary = (pred_mask_high_res > 0).to(torch.uint8).mul(255).cpu().numpy()
        mask_resized = cv2.resize(
            mask_binary,
            (orig_w, orig_h),  # (width, height) for cv2.resize
            interpolation=cv2.INTER_NEAREST,
        )

        # 13. Get updated logits
        pred_masks_low_res = current_out["pred_masks"][0, 0].cpu().numpy()

        # 14. Store new result in cond_frame_outputs
        output_dict["cond_frame_outputs"][frame_idx] = self._make_compact_output(current_out)

        # 15. Return mask, logits, and score (caller saves to storage)
        score = self._extract_score(current_out)
        n_cond_used = len(filtered_output_dict["cond_frame_outputs"])
        n_non_cond_used = len(filtered_output_dict["non_cond_frame_outputs"])
        return mask_resized, pred_masks_low_res, score, n_cond_used, n_non_cond_used

    def propagate(
        self,
        video_id: str,
        frame_idx: int,
        frames,  # Indexable frame source
        masks,  # Indexable mask source
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Propagate tracking to a single frame.

        This is a convenience wrapper that propagates to exactly one frame.
        For batch propagation, use propagate_sequential() instead.

        Args:
            video_id: The video identifier.
            frame_idx: Frame index to propagate to.
            frames: Indexable frame source returning BGR uint8 (H, W, 3).
            masks: Indexable mask source returning uint8 (H, W).

        Returns:
            Tuple of (mask, logits, score) where:
            - mask: Binary mask array (height, width) with dtype uint8, values 0 or 255.
            - logits: Low-res logits array (256, 256).
            - score: Predicted IoU confidence in [0, 1].

        Raises:
            KeyError: If video_id is not open.
            RuntimeError: If no memory exists (need to add a prompt first).
        """
        # Prepare memory for arbitrary frame access
        self._set_memory_frame(video_id, frame_idx, frames, masks)

        frame = frames[frame_idx]
        return self._propagate_single_frame(video_id, frame_idx, frame)

    def propagate_with_box(
        self,
        video_id: str,
        frame_idx: int,
        frame: np.ndarray,
        box_prompt: tuple,
        add_as_conditioning: bool = True,
        use_memory_with_prompt: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Propagate with a box prompt, controlling conditioning storage.

        Args:
            video_id: The video identifier.
            frame_idx: Frame index to propagate to.
            frame: BGR uint8 frame (H, W, 3).
            box_prompt: Bounding box as (x1, y1, x2, y2).
            add_as_conditioning: If True, store in cond_frame_outputs (permanent).
                If False, box guides prediction but result goes to non_cond_frame_outputs.
            use_memory_with_prompt: If True, use memory cross-attention even on
                prompted frames (guidance mode). If False (default), prompted frames
                get fresh segmentation without memory (correction mode).

        Returns:
            Tuple of (mask, logits, score).
        """
        return self._propagate_single_frame(
            video_id, frame_idx, frame,
            box_prompt=box_prompt,
            store_as_cond=add_as_conditioning,
            use_memory_with_prompt=use_memory_with_prompt,
        )

    def propagate_frame(
        self,
        video_id: str,
        frame_idx: int,
        frame: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Propagate to a single frame without prompts, no memory rebuild.

        Unlike propagate(), this does NOT call _set_memory_frame(). Use this
        for sequential forward processing where the sliding window is already
        correct from the previous frame. This matches how propagate_with_detector
        handles coasting frames internally.
        """
        return self._propagate_single_frame(video_id, frame_idx, frame)

    def backtrack_reprop(
        self,
        video_id: str,
        frames,
        final_masks,
        start_idx: int,
        end_idx: int,
        scores=None,
    ) -> None:
        """Re-propagate gap frames after a new anchor.

        Public wrapper around _backtrack_reprop. Only updates final_masks.
        """
        self._backtrack_reprop(
            video_id, frames, final_masks, start_idx, end_idx, scores=scores,
        )

    def set_memory_frame(
        self,
        video_id: str,
        frame_idx: int,
        frames,
        masks,
    ) -> None:
        """Rebuild non-cond memory window before propagating to frame_idx.

        Public wrapper around _set_memory_frame.
        """
        self._set_memory_frame(video_id, frame_idx, frames, masks)

    def evict_conditioning_frame(self, video_id: str, frame_idx: int) -> None:
        """Remove a frame from SAM2's conditioning memory.

        Used by command handler's LRU eviction policy for long videos.
        """
        session = self.sessions[video_id]
        session["cond_frame_indices"].discard(frame_idx)
        session["output_dict"]["cond_frame_outputs"].pop(frame_idx, None)

    def propagate_sequential(
        self,
        video_id: str,
        start_frame: int,
        num_frames: int,
        frames,  # Indexable frame source
        masks,  # Indexable mask source
        on_result: Callable[[int, np.ndarray, np.ndarray, float], None],  # callback(frame_idx, mask, logits, score)
        progress_interval: int = 10,
    ) -> list[int]:
        """Propagate tracking forward from start_frame.

        Uses memory attention to track the object across frames without
        requiring additional point prompts. Frames that are conditioning
        frames (have user prompts) are skipped to preserve user annotations.

        Args:
            video_id: The video identifier.
            start_frame: Frame index to start propagation from.
            num_frames: Maximum number of frames to propagate.
            frames: Indexable frame source returning BGR uint8 (H, W, 3).
            masks: Indexable mask source returning uint8 (H, W).
            on_result: Callback called for each frame with (frame_idx, mask, logits).
                       The caller should save the results in this callback.
            progress_interval: Print progress every N frames (0 to disable).

        Returns:
            List of frame indices that were propagated.

        Raises:
            KeyError: If video_id is not open.
            RuntimeError: If no memory exists (need to add a prompt first).
        """
        # 1. Get session state
        session = self.sessions[video_id]
        cond_frame_indices = session["cond_frame_indices"]
        frame_dims = session["frame_dims"]  # (height, width)
        output_dict = session["output_dict"]

        # 2. Check output_dict has memory
        has_memory = (
            len(output_dict["cond_frame_outputs"]) > 0
            or len(output_dict["non_cond_frame_outputs"]) > 0
        )
        if not has_memory:
            raise RuntimeError(
                "No memory exists for propagation. "
                "Use add_point_prompt() to create an initial mask first."
            )

        # 3. Prepare memory for starting frame
        self._set_memory_frame(video_id, start_frame, frames, masks)

        orig_h, orig_w = frame_dims
        propagated = []

        # 4. Loop for num_frames iterations
        for i in range(num_frames):
            frame_idx = start_frame + i

            # Skip if frame_idx in cond_indices (don't overwrite user prompts)
            if frame_idx in cond_frame_indices:
                continue

            # Try to read frame from frames_source, break on IndexError/KeyError
            try:
                frame_bgr = frames[frame_idx]
            except (IndexError, KeyError):
                break

            # Get image features and prepare backbone features
            _, backbone_out = self._get_image_features(video_id, frame_idx, frame_bgr)
            current_vision_feats, current_vision_pos_embeds, feat_sizes = (
                self._prepare_backbone_features(backbone_out)
            )

            # Call track_step for propagation (no point or mask inputs)
            with torch.inference_mode(), torch.autocast("cuda", torch.bfloat16):
                current_out = self.predictor.track_step(
                    frame_idx=frame_idx,
                    is_init_cond_frame=False,
                    current_vision_feats=current_vision_feats,
                    current_vision_pos_embeds=current_vision_pos_embeds,
                    feat_sizes=feat_sizes,
                    point_inputs=None,
                    mask_inputs=None,
                    output_dict=output_dict,
                    num_frames=session["num_frames"],
                    run_mem_encoder=True,
                )

            # Extract mask from pred_masks_high_res
            pred_mask_high_res = current_out["pred_masks_high_res"][0, 0]  # (H, W)

            # Threshold at 0, convert to uint8 * 255, resize to original dims
            mask_binary = (pred_mask_high_res > 0).to(torch.uint8).mul(255).cpu().numpy()
            mask_resized = cv2.resize(
                mask_binary,
                (orig_w, orig_h),  # (width, height) for cv2.resize
                interpolation=cv2.INTER_NEAREST,
            )

            # Get low-res logits
            pred_masks_low_res = current_out["pred_masks"][0, 0].cpu().numpy()

            # Call callback to save results
            score = self._extract_score(current_out)
            on_result(frame_idx, mask_resized, pred_masks_low_res, score)

            # Store in output_dict["non_cond_frame_outputs"]
            output_dict["non_cond_frame_outputs"][frame_idx] = self._make_compact_output(
                current_out
            )

            # Memory eviction: keep MEM_WINDOW + 1 frames (current + MEM_WINDOW previous)
            eviction_threshold = frame_idx - (self.MEM_WINDOW + 1)
            keys_to_evict = [
                k
                for k in output_dict["non_cond_frame_outputs"]
                if k <= eviction_threshold
            ]
            for k in keys_to_evict:
                del output_dict["non_cond_frame_outputs"][k]

            # Append frame_idx to propagated list
            propagated.append(frame_idx)

            # Progress reporting
            if progress_interval > 0 and len(propagated) % progress_interval == 0:
                print(f"  Propagated {len(propagated)} frames...")

        return propagated

    def propagate_sequential_cond_only(
        self,
        video_id: str,
        start_frame: int,
        num_frames: int,
        frames,
        masks,
        on_result: Callable | None = None,
        progress_interval: int = 50,
    ) -> list[int]:
        """Propagate using only conditioning frame memories (no temporal window).

        Like propagate_sequential(), but clears non_cond_frame_outputs before
        each frame. This prevents temporal drift where propagated frames
        reinforce segmentation errors across the sliding window.

        Each frame is segmented using only the permanent conditioning frame
        memories from user prompts.

        Args:
            video_id: The video identifier.
            start_frame: Frame index to start propagation.
            num_frames: Maximum number of frames to propagate.
            frames: Indexable frame source: frames[idx] -> np.ndarray (H, W, 3).
            masks: Indexable mask source: masks[idx] -> np.ndarray (H, W).
            on_result: Callback called for each frame with (frame_idx, mask, logits, score).
            progress_interval: Print progress every N frames (0 to disable).

        Returns:
            List of frame indices that were propagated.

        Raises:
            KeyError: If video_id is not open.
            RuntimeError: If no conditioning frames exist.
        """
        session = self.sessions[video_id]
        cond_frame_indices = session["cond_frame_indices"]
        frame_dims = session["frame_dims"]
        output_dict = session["output_dict"]

        if len(output_dict["cond_frame_outputs"]) == 0:
            raise RuntimeError(
                "No conditioning frames exist. "
                "Use add_point_prompt() or add_box_prompt() first."
            )

        orig_h, orig_w = frame_dims
        propagated = []

        for i in range(num_frames):
            frame_idx = start_frame + i

            # Skip conditioning frames (don't overwrite user prompts)
            if frame_idx in cond_frame_indices:
                continue

            try:
                frame_bgr = frames[frame_idx]
            except (IndexError, KeyError):
                break

            # Clear non-cond memory so track_step only sees conditioning frames
            output_dict["non_cond_frame_outputs"].clear()
            assert len(output_dict["non_cond_frame_outputs"]) == 0, "non_cond_frame_outputs not empty after clear!"

            # Get image features
            _, backbone_out = self._get_image_features(video_id, frame_idx, frame_bgr)
            current_vision_feats, current_vision_pos_embeds, feat_sizes = (
                self._prepare_backbone_features(backbone_out)
            )

            # track_step with only conditioning memory
            with torch.inference_mode(), torch.autocast("cuda", torch.bfloat16):
                current_out = self.predictor.track_step(
                    frame_idx=frame_idx,
                    is_init_cond_frame=False,
                    current_vision_feats=current_vision_feats,
                    current_vision_pos_embeds=current_vision_pos_embeds,
                    feat_sizes=feat_sizes,
                    point_inputs=None,
                    mask_inputs=None,
                    output_dict=output_dict,
                    num_frames=session["num_frames"],
                    run_mem_encoder=True,
                )

            # Extract mask
            pred_mask_high_res = current_out["pred_masks_high_res"][0, 0]
            mask_binary = (pred_mask_high_res > 0).to(torch.uint8).mul(255).cpu().numpy()
            mask_resized = cv2.resize(
                mask_binary,
                (orig_w, orig_h),
                interpolation=cv2.INTER_NEAREST,
            )

            pred_masks_low_res = current_out["pred_masks"][0, 0].cpu().numpy()
            score = self._extract_score(current_out)

            if on_result is not None:
                on_result(frame_idx, mask_resized, pred_masks_low_res, score)

            # Store in non_cond (will be cleared next iteration)
            output_dict["non_cond_frame_outputs"][frame_idx] = self._make_compact_output(
                current_out
            )

            propagated.append(frame_idx)

            if progress_interval > 0 and len(propagated) % progress_interval == 0:
                print(f"  Propagated {len(propagated)} frames (cond-only)...")

        return propagated

    def _compute_bbox_iou(
        self, mask1: np.ndarray, mask2: np.ndarray
    ) -> float:
        """Compute IoU between bounding boxes of two masks.

        This is faster and more robust than pixel-level mask IoU for
        detecting drift between tracker and detector.

        Args:
            mask1: First binary mask (H, W).
            mask2: Second binary mask (H, W).

        Returns:
            IoU value in [0, 1]. Returns 0 if either mask is empty.
        """
        bbox1 = self._bbox_from_mask(mask1)
        bbox2 = self._bbox_from_mask(mask2)
        if bbox1 is None or bbox2 is None:
            return 0.0

        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        inter_x1 = max(x1_1, x1_2)
        inter_y1 = max(y1_1, y1_2)
        inter_x2 = min(x2_1, x2_2)
        inter_y2 = min(y2_1, y2_2)

        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0

        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    @staticmethod
    def _compute_bbox_iou_from_bboxes(
        bbox_a: tuple[int, int, int, int] | None,
        bbox_b: tuple[float, float, float, float] | None,
    ) -> float:
        """Compute IoU between two bounding boxes directly.

        Unlike _compute_bbox_iou which derives bboxes from masks, this
        takes pre-computed bboxes (e.g. tracker bbox from mask + detector
        bbox from model output).

        Args:
            bbox_a: First bbox (x1, y1, x2, y2) with exclusive x2/y2, or None.
            bbox_b: Second bbox (x1, y1, x2, y2) with exclusive x2/y2, or None.

        Returns:
            IoU value in [0, 1]. Returns 0 if either bbox is None.
        """
        if bbox_a is None or bbox_b is None:
            return 0.0
        ax1, ay1, ax2, ay2 = bbox_a
        bx1, by1, bx2, by2 = bbox_b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union_area = area_a + area_b - inter_area
        return inter_area / union_area if union_area > 0 else 0.0

    def _bbox_from_mask(self, mask: np.ndarray) -> tuple[int, int, int, int] | None:
        """Extract bounding box (x1, y1, x2, y2) from a binary mask.

        Args:
            mask: Binary mask (H, W) uint8.

        Returns:
            (x1, y1, x2, y2) where x2/y2 are exclusive, or None if empty.
        """
        rows = np.any(mask > 127, axis=1)
        cols = np.any(mask > 127, axis=0)
        if not rows.any() or not cols.any():
            return None
        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]
        return (int(x1), int(y1), int(x2 + 1), int(y2 + 1))

    def _encode_crop(self, frame: np.ndarray, bbox: tuple[int, int, int, int]) -> torch.Tensor:
        """Encode a cropped region into a 256-dim L2-normalized feature vector.

        Crops the frame to the bounding box, resizes to 1024x1024, runs through
        SAM2's image encoder, and global-average-pools the stride-64 FPN features.

        Args:
            frame: BGR uint8 frame (H, W, 3).
            bbox: (x1, y1, x2, y2) with exclusive x2/y2.

        Returns:
            L2-normalized feature vector of shape (256,) on self.device.
        """
        x1, y1, x2, y2 = bbox
        crop = frame[y1:y2, x1:x2]

        crop_gpu = torch.from_numpy(crop).to(self.device)
        image_tensor = crop_gpu[..., [2, 1, 0]].permute(2, 0, 1).float().div_(255.0)
        image_tensor = torch.nn.functional.interpolate(
            image_tensor.unsqueeze(0),
            size=(self.INPUT_SIZE, self.INPUT_SIZE),
            mode="bilinear",
            align_corners=False,
        )

        with torch.inference_mode(), torch.autocast("cuda", torch.bfloat16):
            backbone_out = self.predictor.forward_image(image_tensor)

        # Stride-64 FPN level: (1, 256, 16, 16) -> global avg pool -> (256,)
        feat = backbone_out["backbone_fpn"][-1].mean(dim=[2, 3]).squeeze(0)
        return torch.nn.functional.normalize(feat, dim=0)

    def _build_training_embeddings(
        self,
        training_frame_indices: list[int],
        frames,
        masks,
        max_samples: int = 50,
    ) -> torch.Tensor | None:
        """Build reference embeddings from training frame crops.

        Samples up to max_samples frames, crops each to its mask bbox,
        encodes with SAM2's image encoder, returns stacked L2-normalized vectors.

        Args:
            training_frame_indices: Frame indices marked as training data.
            frames: Indexable frame source.
            masks: Indexable mask source (for bbox extraction).
            max_samples: Max training frames to encode.

        Returns:
            (N, 256) tensor of L2-normalized embeddings, or None if no valid crops.
        """
        import random

        if not training_frame_indices:
            return None

        sampled = random.sample(training_frame_indices, min(max_samples, len(training_frame_indices)))

        embeddings = []
        for idx in sampled:
            mask = np.asarray(masks[idx])
            bbox = self._bbox_from_mask(mask)
            if bbox is None:
                continue
            feat = self._encode_crop(frames[idx], bbox)
            embeddings.append(feat)

        if not embeddings:
            return None

        return torch.stack(embeddings, dim=0)

    def _compare_to_training(
        self,
        crop_embedding: torch.Tensor,
        training_embeddings: torch.Tensor,
    ) -> float:
        """Mean cosine similarity between a crop embedding and training embeddings.

        Both inputs must be L2-normalized so dot product = cosine similarity.

        Args:
            crop_embedding: (256,) L2-normalized vector.
            training_embeddings: (N, 256) L2-normalized matrix.

        Returns:
            Mean cosine similarity as float.
        """
        return (training_embeddings @ crop_embedding).mean().item()

    def _propagate_single_frame(
        self,
        video_id: str,
        frame_idx: int,
        frame: np.ndarray,
        mask_prompt: np.ndarray | None = None,
        box_prompt: tuple[float, float, float, float] | None = None,
        store_as_cond: bool | None = None,
        use_memory_with_prompt: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Propagate to a single frame, optionally with a mask or box prompt.

        Args:
            video_id: The video identifier.
            frame_idx: Frame index to propagate to.
            frame: BGR uint8 frame data (H, W, 3).
            mask_prompt: Optional mask to use as prompt (for detector re-prompting).
                Mutually exclusive with box_prompt.
            box_prompt: Optional bounding box (x1, y1, x2, y2) in original image
                pixel coordinates. Encoded as two special points with SAM2 labels
                [2, 3] (top-left, bottom-right). Mutually exclusive with mask_prompt.

        Returns:
            (mask, logits, score) where mask is (H, W) uint8, logits is (256, 256) float32,
            and score is the predicted IoU confidence in [0, 1].
        """
        if mask_prompt is not None and box_prompt is not None:
            raise ValueError("mask_prompt and box_prompt are mutually exclusive")

        session = self.sessions[video_id]
        frame_dims = session["frame_dims"]
        output_dict = session["output_dict"]
        orig_h, orig_w = frame_dims

        is_prompted = mask_prompt is not None or box_prompt is not None

        import time as _time
        _pt = {}  # propagate sub-timings

        # Get image features and prepare backbone features
        try:
            _t0 = _time.perf_counter()
            _, backbone_out = self._get_image_features(video_id, frame_idx, frame)
            _pt["image_features_ms"] = round((_time.perf_counter() - _t0) * 1000, 2)
        except Exception as e:
            raise RuntimeError(f"Failed to get image features for frame {frame_idx}: {e}") from e

        try:
            _t0 = _time.perf_counter()
            current_vision_feats, current_vision_pos_embeds, feat_sizes = (
                self._prepare_backbone_features(backbone_out)
            )
            _pt["prepare_backbone_ms"] = round((_time.perf_counter() - _t0) * 1000, 2)
        except Exception as e:
            raise RuntimeError(f"Failed to prepare backbone features for frame {frame_idx}: {e}") from e

        # Prepare prompt inputs (mask_prompt and box_prompt are mutually exclusive)
        _t0 = _time.perf_counter()
        mask_inputs = None
        point_inputs = None

        if mask_prompt is not None:
            # Resize mask to model input size and convert to tensor
            mask_resized = cv2.resize(
                mask_prompt.astype(np.float32),
                (self.INPUT_SIZE, self.INPUT_SIZE),
                interpolation=cv2.INTER_NEAREST,
            )
            # Convert to binary float (0.0 or 1.0) and add batch/channel dims
            mask_tensor = torch.from_numpy((mask_resized > 127).astype(np.float32))
            mask_inputs = mask_tensor.unsqueeze(0).unsqueeze(0).to(self.device)

        elif box_prompt is not None:
            # Scale box coordinates from original image space to model input space (1024x1024)
            x1, y1, x2, y2 = box_prompt
            scaled_x1 = x1 * self.INPUT_SIZE / orig_w
            scaled_y1 = y1 * self.INPUT_SIZE / orig_h
            scaled_x2 = x2 * self.INPUT_SIZE / orig_w
            scaled_y2 = y2 * self.INPUT_SIZE / orig_h

            # SAM2 box convention: two points with labels [2, 3] (top-left, bottom-right)
            point_coords = torch.tensor(
                [[[scaled_x1, scaled_y1], [scaled_x2, scaled_y2]]],
                dtype=torch.float32,
                device=self.device,
            )  # (1, 2, 2)
            point_labels = torch.tensor(
                [[2, 3]], dtype=torch.int32, device=self.device
            )  # (1, 2)

            point_inputs = {
                "point_coords": point_coords,
                "point_labels": point_labels,
            }
        _pt["prompt_prep_ms"] = round((_time.perf_counter() - _t0) * 1000, 2)

        # Call track_step
        try:
            _t0 = _time.perf_counter()
            with torch.inference_mode(), torch.autocast("cuda", torch.bfloat16):
                current_out = self.predictor.track_step(
                    frame_idx=frame_idx,
                    is_init_cond_frame=is_prompted and not use_memory_with_prompt,
                    current_vision_feats=current_vision_feats,
                    current_vision_pos_embeds=current_vision_pos_embeds,
                    feat_sizes=feat_sizes,
                    point_inputs=point_inputs,
                    mask_inputs=mask_inputs,
                    output_dict=output_dict,
                    num_frames=session["num_frames"],
                    run_mem_encoder=True,
                )
            _pt["track_step_ms"] = round((_time.perf_counter() - _t0) * 1000, 2)
        except Exception as e:
            raise RuntimeError(f"track_step failed for frame {frame_idx}: {e}") from e

        # Extract mask from pred_masks_high_res
        _t0 = _time.perf_counter()
        pred_mask_high_res = current_out["pred_masks_high_res"][0, 0]

        # Threshold at 0, convert to uint8 * 255, resize to original dims
        mask_binary = (pred_mask_high_res > 0).to(torch.uint8).mul(255).cpu().numpy()
        mask_resized = cv2.resize(
            mask_binary,
            (orig_w, orig_h),
            interpolation=cv2.INTER_NEAREST,
        )

        # Get logits
        pred_masks_low_res = current_out["pred_masks"][0, 0].cpu().numpy()
        _pt["postprocess_ms"] = round((_time.perf_counter() - _t0) * 1000, 2)

        # Write sub-timings to file (append mode, same file as handler)
        try:
            with open("/tmp/coseg_propagate_timing.jsonl", "a") as _pf:
                import json as _json
                _pf.write(_json.dumps({"f": frame_idx, **_pt}) + "\n")
        except Exception:
            pass

        # Determine storage destination
        # store_as_cond overrides default is_prompted logic when explicitly set
        should_store_as_cond = is_prompted if store_as_cond is None else store_as_cond

        if should_store_as_cond:
            output_dict["cond_frame_outputs"][frame_idx] = self._make_compact_output(current_out)
            # Only update cond_frame_indices when store_as_cond was explicitly requested.
            # When store_as_cond is None (default), preserve existing behavior where
            # cond_frame_indices is managed by add_point_prompt, not here.
            if store_as_cond is not None:
                session["cond_frame_indices"].add(frame_idx)
        else:
            output_dict["non_cond_frame_outputs"][frame_idx] = self._make_compact_output(current_out)

            # CRITICAL: Preserve the sliding window eviction from the original code.
            # Without this, non_cond_frame_outputs grows unboundedly for long videos.
            eviction_threshold = frame_idx - (self.MEM_WINDOW + 1)
            keys_to_evict = [
                k for k in output_dict["non_cond_frame_outputs"]
                if k <= eviction_threshold
            ]
            for k in keys_to_evict:
                del output_dict["non_cond_frame_outputs"][k]

        score = self._extract_score(current_out)
        return mask_resized, pred_masks_low_res, score

    def _propagate_single_frame_no_store(
        self,
        video_id: str,
        frame_idx: int,
        frame: np.ndarray,
        mask_prompt: np.ndarray,
    ) -> np.ndarray:
        """Propagate with mask prompt but don't write to session masks.

        This is used for final mask generation where we want the corrected
        result but don't want to overwrite the original tracker mask.

        The result IS added to cond_frame_outputs so it influences future
        frame predictions via memory attention.

        Args:
            video_id: The video identifier.
            frame_idx: Frame index to propagate to.
            frame: BGR uint8 frame data (H, W, 3).
            mask_prompt: Mask to use as prompt (required).

        Returns:
            Corrected mask (H, W) uint8.
        """
        session = self.sessions[video_id]
        frame_dims = session["frame_dims"]
        output_dict = session["output_dict"]
        orig_h, orig_w = frame_dims

        # Get image features and prepare backbone features
        _, backbone_out = self._get_image_features(video_id, frame_idx, frame)
        current_vision_feats, current_vision_pos_embeds, feat_sizes = (
            self._prepare_backbone_features(backbone_out)
        )

        # Prepare mask_inputs
        mask_resized = cv2.resize(
            mask_prompt.astype(np.float32),
            (self.INPUT_SIZE, self.INPUT_SIZE),
            interpolation=cv2.INTER_NEAREST,
        )
        mask_tensor = torch.from_numpy((mask_resized > 127).astype(np.float32))
        mask_inputs = mask_tensor.unsqueeze(0).unsqueeze(0).to(self.device)

        # Call track_step
        with torch.inference_mode(), torch.autocast("cuda", torch.bfloat16):
            current_out = self.predictor.track_step(
                frame_idx=frame_idx,
                is_init_cond_frame=True,  # Treat as conditioning frame
                current_vision_feats=current_vision_feats,
                current_vision_pos_embeds=current_vision_pos_embeds,
                feat_sizes=feat_sizes,
                point_inputs=None,
                mask_inputs=mask_inputs,
                output_dict=output_dict,
                num_frames=session["num_frames"],
                run_mem_encoder=True,
            )

        # Extract mask
        pred_mask_high_res = current_out["pred_masks_high_res"][0, 0]
        mask_binary = (pred_mask_high_res > 0).to(torch.uint8).mul(255).cpu().numpy()
        mask_result = cv2.resize(mask_binary, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        # Store in cond_frame_outputs for memory (so future frames benefit)
        output_dict["cond_frame_outputs"][frame_idx] = self._make_compact_output(current_out)

        return mask_result

    def _compute_iou(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Compute IoU between two binary masks.

        Handles empty masks specially:
        - empty-vs-empty = 1.0 (both agree there's nothing)
        - non-empty-vs-empty = 0.0 (no overlap)

        Args:
            mask1: First mask (uint8, values 0 or 255).
            mask2: Second mask (uint8, values 0 or 255).

        Returns:
            IoU score between 0.0 and 1.0.
        """
        bin1 = mask1 > 127
        bin2 = mask2 > 127

        # Handle empty masks
        if not bin1.any() and not bin2.any():
            return 1.0  # Both empty = agreement
        if not bin1.any() or not bin2.any():
            return 0.0  # One empty = no overlap

        intersection = np.logical_and(bin1, bin2).sum()
        union = np.logical_or(bin1, bin2).sum()

        return float(intersection / union) if union > 0 else 0.0

    def _backtrack_reprop(
        self,
        video_id: str,
        frames,
        final_masks,
        start_idx: int,
        end_idx: int,
        scores: dict | None = None,
    ) -> None:
        """Re-propagate frames after a correction, writing only to final_masks.

        Called after adding a new conditioning frame. Uses _set_memory_frame
        to prepare memory context, then propagates forward through the
        specified range.

        Args:
            video_id: The video session ID.
            frames: Indexable frame source.
            final_masks: MutableSequence to write corrected masks to.
            start_idx: First frame to re-propagate (inclusive).
            end_idx: Last frame to re-propagate (inclusive).
        """
        if start_idx > end_idx:
            return

        for idx in range(start_idx, end_idx + 1):
            # Prepare memory for this frame (uses new conditioning frame)
            self._set_memory_frame(video_id, idx, frames, final_masks)

            frame = frames[idx]
            mask, _, score = self._propagate_single_frame(video_id, idx, frame)
            final_masks[idx] = mask
            if scores is not None:
                scores[idx] = score

    def propagate_with_detector(
        self,
        video_id: str,
        num_frames: int,
        frames,  # Sequence - read frames[idx]
        get_detector_bbox,  # Callable[[int, np.ndarray, tuple|None], tuple[tuple|None, float]]
        tracker_masks,  # MutableSequence - write tracker output
        final_masks,  # MutableSequence - write final output
        on_progress: Callable[[int], None] | None = None,
        check_interval: int = 10,
        iou_threshold: float = 0.7,
        scores: dict | None = None,
        detector_bboxes: dict | None = None,
        detector_scores: dict | None = None,
        training_frame_indices: list[int] | None = None,
    ) -> None:
        """Propagate tracking with on-the-fly detector-guided correction.

        Runs the tracker on every frame, invoking the detector only at check_interval
        frames (or when drift is detected). Uses bbox IoU comparison between tracker
        mask bbox and detector bbox to decide when to apply corrections.

        Args:
            video_id: The video identifier.
            num_frames: Number of frames to propagate.
            frames: Sequence of frames - read frames[idx] to get BGR numpy array.
            get_detector_bbox: Callable(frame_idx, frame, tracker_bbox_hint) ->
                (bbox, confidence) or (None, 0.0). The tracker_bbox_hint is the
                current tracker mask's bbox for multi-detection selection.
            tracker_masks: MutableSequence to write tracker output masks.
            final_masks: MutableSequence to write final (corrected) output masks.
            on_progress: Optional callback(frame_idx) for progress reporting.
            check_interval: Run detector every N frames (default 10).
            iou_threshold: Re-prompt when bbox IoU drops below this (default 0.7).
            scores: Optional dict to collect {frame_idx: score}.
            detector_bboxes: Optional dict to collect {frame_idx: (x1,y1,x2,y2)}.
            detector_scores: Optional dict to collect {frame_idx: confidence}.
            training_frame_indices: Frame indices with training data for feature comparison.

        Returns:
            None. Results are written to the provided MutableSequences.

        Raises:
            RuntimeError: If no memory exists and detector finds no object.
        """
        # Get session state
        session = self.sessions[video_id]
        output_dict = session["output_dict"]

        # Find first frame with a detection
        start_frame = 0
        for idx in range(num_frames):
            frame = frames[idx]
            det_bbox, det_conf = get_detector_bbox(idx, frame, None)  # no tracker bbox yet

            if det_bbox is not None:
                if detector_bboxes is not None:
                    detector_bboxes[idx] = det_bbox
                if detector_scores is not None:
                    detector_scores[idx] = det_conf
                start_frame = idx
                break
            else:
                # Write empty to tracker and final for skipped frames
                empty_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                tracker_masks[idx] = empty_mask
                final_masks[idx] = empty_mask
                if on_progress:
                    on_progress(idx)
        else:
            # No object found in entire video - complete successfully
            print(f"  No object detected in any frame")
            return

        # Initialize tracker from first detected frame using box prompt
        frame = frames[start_frame]
        mask, _, score = self._propagate_single_frame(
            video_id, start_frame, frame, box_prompt=det_bbox
        )
        tracker_masks[start_frame] = mask
        final_masks[start_frame] = mask
        if scores is not None:
            scores[start_frame] = score

        last_successful_check = start_frame
        searching = False

        # Build training crop embeddings for feature-space comparison
        training_embeddings = None
        if training_frame_indices:
            print(f"  Building training embeddings from {len(training_frame_indices)} frames...")
            training_embeddings = self._build_training_embeddings(
                training_frame_indices, frames, tracker_masks
            )
            if training_embeddings is not None:
                print(f"  Cached {training_embeddings.shape[0]} training embeddings")

        if on_progress:
            on_progress(start_frame)

        # Main loop
        for frame_idx in range(start_frame + 1, num_frames):
            frame = frames[frame_idx]

            if searching:
                # Searching mode: detect every frame, no tracking
                # Use last successful mask bbox as hint for multi-detection selection
                tracker_bbox_hint = (
                    self._bbox_from_mask(final_masks[last_successful_check])
                    if last_successful_check >= 0
                    else None
                )
                det_bbox, det_conf = get_detector_bbox(frame_idx, frame, tracker_bbox_hint)

                # Store detector results
                if detector_bboxes is not None and det_bbox is not None:
                    detector_bboxes[frame_idx] = det_bbox
                if detector_scores is not None and det_conf > 0:
                    detector_scores[frame_idx] = det_conf

                # Write empty while searching
                h, w = frame.shape[:2]
                empty_mask = np.zeros((h, w), dtype=np.uint8)
                tracker_masks[frame_idx] = empty_mask
                final_masks[frame_idx] = empty_mask

                if det_bbox is not None:
                    # Object reappeared - add as conditioning frame via box prompt
                    mask, _, score = self._propagate_single_frame(
                        video_id, frame_idx, frame, box_prompt=det_bbox
                    )
                    final_masks[frame_idx] = mask
                    if scores is not None:
                        scores[frame_idx] = score

                    # Backtrack and re-propagate
                    self._backtrack_reprop(
                        video_id, frames, final_masks,
                        last_successful_check + 1, frame_idx - 1,
                        scores=scores,
                    )

                    last_successful_check = frame_idx
                    searching = False
            else:
                # Normal mode: propagate tracker
                tracker_mask, _, score = self._propagate_single_frame(video_id, frame_idx, frame)
                tracker_masks[frame_idx] = tracker_mask
                final_masks[frame_idx] = tracker_mask
                if scores is not None:
                    scores[frame_idx] = score

                # Check frame?
                if frame_idx % check_interval == 0:
                    tracker_bbox = self._bbox_from_mask(tracker_mask)
                    det_bbox, det_conf = get_detector_bbox(frame_idx, frame, tracker_bbox)

                    # Store detector results
                    if detector_bboxes is not None and det_bbox is not None:
                        detector_bboxes[frame_idx] = det_bbox
                    if detector_scores is not None and det_conf > 0:
                        detector_scores[frame_idx] = det_conf

                    if det_bbox is not None:
                        iou = self._compute_bbox_iou_from_bboxes(tracker_bbox, det_bbox) if tracker_bbox else 0.0

                        if iou >= iou_threshold:
                            # Tracker is good
                            last_successful_check = frame_idx
                        else:
                            # Drift detected — use feature comparison to decide
                            use_detector = True  # default: trust detector

                            if training_embeddings is not None:
                                if tracker_bbox is not None:
                                    tracker_feat = self._encode_crop(frame, tracker_bbox)
                                    det_bbox_int = (
                                        int(det_bbox[0]),
                                        int(det_bbox[1]),
                                        int(det_bbox[2]),
                                        int(det_bbox[3]),
                                    )
                                    detector_feat = self._encode_crop(frame, det_bbox_int)

                                    tracker_sim = self._compare_to_training(
                                        tracker_feat, training_embeddings
                                    )
                                    detector_sim = self._compare_to_training(
                                        detector_feat, training_embeddings
                                    )

                                    use_detector = detector_sim > tracker_sim

                            if use_detector:
                                # Detector wins — correct with box prompt
                                mask, _, score = self._propagate_single_frame(
                                    video_id, frame_idx, frame, box_prompt=det_bbox
                                )
                                final_masks[frame_idx] = mask
                                if scores is not None:
                                    scores[frame_idx] = score

                                # Backtrack and re-propagate
                                self._backtrack_reprop(
                                    video_id, frames, final_masks,
                                    last_successful_check + 1, frame_idx - 1,
                                    scores=scores,
                                )

                                last_successful_check = frame_idx
                            else:
                                # Tracker wins — no correction needed
                                last_successful_check = frame_idx
                    else:
                        # Detector empty - object disappeared
                        h, w = frame.shape[:2]
                        empty_mask = np.zeros((h, w), dtype=np.uint8)
                        self._propagate_single_frame(
                            video_id, frame_idx, frame, mask_prompt=empty_mask
                        )
                        final_masks[frame_idx] = empty_mask
                        if scores is not None:
                            scores[frame_idx] = 0.0

                        # Backtrack and re-propagate
                        self._backtrack_reprop(
                            video_id, frames, final_masks,
                            last_successful_check + 1, frame_idx - 1,
                            scores=scores,
                        )

                        last_successful_check = frame_idx
                        searching = True

            if on_progress:
                on_progress(frame_idx)

    def simple_propagate_with_detector(
        self,
        video_id: str,
        num_frames: int,
        frames,
        get_detector_bbox,
        tracker_masks,
        final_masks,
        on_progress=None,
        reprompt_interval: int = 30,
        scores: dict | None = None,
        detector_bboxes: dict | None = None,
        detector_scores: dict | None = None,
    ) -> None:
        """Simple batch segmentation: propagate forward, re-prompt with detector every N frames.

        No drift detection, no search mode, no backtracking. Just forward propagation
        with periodic detector re-prompting to keep the tracker on target.

        Args:
            video_id: Video session ID (string).
            num_frames: Total frames in video.
            frames: Indexable frame source, frames[idx] -> np.ndarray BGR.
            get_detector_bbox: Callable(frame_idx, frame) -> (bbox|None, confidence).
            tracker_masks: MutableSequence to write tracker masks.
            final_masks: MutableSequence to write final masks.
            on_progress: Optional callback(frame_idx) for progress reporting.
            reprompt_interval: Re-prompt with detector every N frames.
            scores: Optional dict to collect {frame_idx: sam2_score}.
            detector_bboxes: Optional dict to collect {frame_idx: bbox}.
            detector_scores: Optional dict to collect {frame_idx: confidence}.
        """
        session = self.sessions.get(video_id)
        if session is None:
            raise RuntimeError(f"No session for video {video_id}")

        orig_h, orig_w = session["orig_h"], session["orig_w"]
        cond_frame_indices = set(session.get("cond_frame_indices", []))
        empty_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)

        if scores is None:
            scores = {}
        if detector_bboxes is None:
            detector_bboxes = {}
        if detector_scores is None:
            detector_scores = {}

        # Bootstrap: scan for first detector bbox
        start_frame = None
        for frame_idx in range(num_frames):
            if frame_idx in cond_frame_indices:
                continue
            frame = frames[frame_idx]
            det_bbox, det_conf = get_detector_bbox(frame_idx, frame)
            if det_bbox is not None:
                # Initialize with box prompt
                # _propagate_single_frame already returns mask in original (orig_h, orig_w) resolution
                mask, logits, score = self._propagate_single_frame(
                    video_id, frame_idx, frame, box_prompt=det_bbox
                )
                tracker_masks[frame_idx] = mask
                final_masks[frame_idx] = mask
                scores[frame_idx] = score
                detector_bboxes[frame_idx] = det_bbox
                detector_scores[frame_idx] = det_conf
                start_frame = frame_idx
                # Write empty masks for skipped frames
                for i in range(frame_idx):
                    if i not in cond_frame_indices:
                        tracker_masks[i] = empty_mask
                        final_masks[i] = empty_mask
                break
            else:
                tracker_masks[frame_idx] = empty_mask
                final_masks[frame_idx] = empty_mask

        if start_frame is None:
            # No detection found in any frame
            print(f"[Simple Propagate] No object detected in video {video_id}")
            return

        if on_progress:
            on_progress(start_frame)

        # Track conditioning frames added by reprompting for eviction
        reprompt_cond_frames: list[int] = [start_frame]

        # Forward propagation
        for frame_idx in range(start_frame + 1, num_frames):
            if frame_idx in cond_frame_indices:
                # User conditioning frame, already in memory
                continue

            frame = frames[frame_idx]
            is_reprompt = (frame_idx - start_frame) % reprompt_interval == 0

            if is_reprompt:
                det_bbox, det_conf = get_detector_bbox(frame_idx, frame)
                if det_bbox is not None:
                    # Re-prompt with detector bbox
                    # _propagate_single_frame already returns mask in original (orig_h, orig_w) resolution
                    mask, logits, score = self._propagate_single_frame(
                        video_id, frame_idx, frame, box_prompt=det_bbox
                    )
                    detector_bboxes[frame_idx] = det_bbox
                    detector_scores[frame_idx] = det_conf

                    # Evict oldest reprompt conditioning frame if over limit
                    reprompt_cond_frames.append(frame_idx)
                    while len(reprompt_cond_frames) > self.MEM_WINDOW:
                        old_idx = reprompt_cond_frames.pop(0)
                        self.evict_conditioning_frame(video_id, old_idx)
                else:
                    # No detection, normal tracking
                    mask, logits, score = self._propagate_single_frame(
                        video_id, frame_idx, frame
                    )
            else:
                # Normal tracking
                mask, logits, score = self._propagate_single_frame(
                    video_id, frame_idx, frame
                )

            tracker_masks[frame_idx] = mask
            final_masks[frame_idx] = mask
            scores[frame_idx] = score

            if on_progress and (frame_idx % 50 == 0 or frame_idx == num_frames - 1):
                on_progress(frame_idx)
