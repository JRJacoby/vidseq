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

import sys
from typing import Callable

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

    # Low-res logits size (SAM2 outputs 256x256)
    LOGITS_SIZE = 256

    def __init__(self, device: str | None = None, compile_model: bool = True):
        """Initialize the segmentor and load the SAM2 model.

        Args:
            device: Torch device string ("cuda", "cpu", etc.).
                    Defaults to "cuda" if available, else "cpu".
            compile_model: Whether to use torch.compile for faster inference.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading SAM2 model on {self.device}...")
        self.predictor = build_sam2_video_predictor(
            config_file="configs/sam2.1/sam2.1_hiera_l.yaml",
            ckpt_path="/n/groups/datta/john/repos/sam2/checkpoints/sam2.1_hiera_large.pt",
            device=self.device,
        )
        print("SAM2 model loaded.")

        # Compile model components for faster inference
        if compile_model and self.device == "cuda":
            print("Compiling SAM2 image encoder with torch.compile...")
            # Compile image encoder (the main bottleneck - 55% of inference time)
            self.predictor.image_encoder = torch.compile(
                self.predictor.image_encoder,
                mode="max-autotune",
                fullgraph=True,
            )
            print("SAM2 image encoder compiled.")

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
        """Get image features for a frame, with caching.

        Preprocesses the frame and runs through the SAM2 image encoder.
        Caches only the most recent frame's features to avoid memory bloat.

        Args:
            video_id: The video identifier.
            frame_idx: Frame index (used for caching).
            frame: BGR uint8 frame data (H, W, 3).

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
        loading masks and encoding them into memory. Stops after 6 frames
        or when hitting a gap (empty mask).

        Args:
            video_id: The video session ID.
            frame_idx: Target frame we're about to track/prompt.
            frames: Indexable frame source returning BGR uint8 (H, W, 3).
            masks: Indexable mask source returning uint8 (H, W).
        """
        MEM_WINDOW = 6

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
            if frames_added >= MEM_WINDOW:
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
            "cached_features": {},  # Cache for most recent frame's features
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

        # Clear cached features to free GPU memory
        session["cached_features"].clear()

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

        # Clear from cached_features if it's cached for this frame
        cached = session.get("cached_features", {})
        if cached.get("frame_idx") == frame_idx:
            session["cached_features"] = {}

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

        # Clear cached_features
        session["cached_features"].clear()

    def add_point_prompt(
        self,
        video_id: str,
        frame_idx: int,
        location: tuple[float, float] | list[tuple[float, float]],
        label: int | list[int],
        frames,  # Indexable frame source
        masks,  # Indexable mask source
    ) -> tuple[np.ndarray, np.ndarray]:
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
            Tuple of (mask, logits) where:
            - mask: Binary mask array (height, width) with dtype uint8, values 0 or 255.
            - logits: Low-res logits array (256, 256) for potential refinement.

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

        # 8. Determine is_init_cond_frame (True if no existing memory)
        has_existing_memory = (
            len(output_dict["cond_frame_outputs"]) > 0
            or len(output_dict["non_cond_frame_outputs"]) > 0
        )
        is_init_cond_frame = not has_existing_memory

        # 9. Call track_step with point_inputs
        with torch.inference_mode(), torch.autocast("cuda", torch.bfloat16):
            current_out = self.predictor.track_step(
                frame_idx=frame_idx,
                is_init_cond_frame=is_init_cond_frame,
                current_vision_feats=current_vision_feats,
                current_vision_pos_embeds=current_vision_pos_embeds,
                feat_sizes=feat_sizes,
                point_inputs=point_inputs,
                mask_inputs=None,
                output_dict=output_dict,
                num_frames=session["num_frames"],
            )

        # 10. Extract mask from pred_masks_high_res
        # pred_masks_high_res has shape (1, num_objects, H, W), we want first object
        pred_mask_high_res = current_out["pred_masks_high_res"][0, 0]  # (H, W)

        # 11. Threshold at 0, convert to uint8 * 255, resize to original dims
        mask_binary = (pred_mask_high_res > 0).cpu().numpy().astype(np.uint8) * 255
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

        # 15. Return mask and logits (caller saves to storage)
        return mask_resized, pred_masks_low_res

    def refine_mask(
        self,
        video_id: str,
        frame_idx: int,
        location: tuple[float, float] | list[tuple[float, float]],
        label: int | list[int],
        frames,  # Indexable frame source
        masks,  # Indexable mask source
        prev_logits: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
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
            Tuple of (mask, logits) where:
            - mask: Refined binary mask array (height, width) with dtype uint8.
            - logits: Updated low-res logits array (256, 256).

        Raises:
            KeyError: If video_id is not open.
        """
        # 1. Get session state
        session = self.sessions[video_id]
        frame_dims = session["frame_dims"]  # (height, width)
        output_dict = session["output_dict"]

        # 2. Prepare memory for arbitrary frame access
        self._set_memory_frame(video_id, frame_idx, frames, masks)

        # 3. Get frame from source
        frame = frames[frame_idx]

        # 4. Convert prev_logits to tensor: shape (1, 1, 256, 256), float32, on device
        # SAM2 low-res logits are 256x256
        prev_logits_tensor = torch.from_numpy(prev_logits.astype(np.float32))
        prev_logits_tensor = prev_logits_tensor.unsqueeze(0).unsqueeze(0).to(self.device)

        # 5. Normalize location/label to lists
        if isinstance(location, tuple) and len(location) == 2 and not isinstance(location[0], tuple):
            # Single point: (x, y)
            locations = [location]
        else:
            locations = list(location)

        if isinstance(label, int):
            labels = [label]
        else:
            labels = list(label)

        # 6. Scale points to INPUT_SIZE (1024) space
        orig_h, orig_w = frame_dims
        scaled_points = []
        for x, y in locations:
            scaled_x = x * self.INPUT_SIZE / orig_w
            scaled_y = y * self.INPUT_SIZE / orig_h
            scaled_points.append([scaled_x, scaled_y])

        # 7. Create point_inputs dict with tensors on device
        point_coords = torch.tensor(scaled_points, dtype=torch.float32, device=self.device)
        point_coords = point_coords.unsqueeze(0)  # (1, N, 2) - batch dim
        point_labels = torch.tensor(labels, dtype=torch.int32, device=self.device)
        point_labels = point_labels.unsqueeze(0)  # (1, N)

        point_inputs = {
            "point_coords": point_coords,
            "point_labels": point_labels,
        }

        # 8. Get image features and prepare backbone features
        _, backbone_out = self._get_image_features(video_id, frame_idx, frame)
        current_vision_feats, current_vision_pos_embeds, feat_sizes = (
            self._prepare_backbone_features(backbone_out)
        )

        # 9. Call track_step with point_inputs and prev_sam_mask_logits
        # is_init_cond_frame=False because we have existing context (the previous mask)
        with torch.inference_mode(), torch.autocast("cuda", torch.bfloat16):
            current_out = self.predictor.track_step(
                frame_idx=frame_idx,
                is_init_cond_frame=False,
                current_vision_feats=current_vision_feats,
                current_vision_pos_embeds=current_vision_pos_embeds,
                feat_sizes=feat_sizes,
                point_inputs=point_inputs,
                mask_inputs=None,
                output_dict=output_dict,
                num_frames=session["num_frames"],
                prev_sam_mask_logits=prev_logits_tensor,
            )

        # 10. Extract mask, threshold, resize to original dims
        pred_mask_high_res = current_out["pred_masks_high_res"][0, 0]  # (H, W)
        mask_binary = (pred_mask_high_res > 0).cpu().numpy().astype(np.uint8) * 255
        mask_resized = cv2.resize(
            mask_binary,
            (orig_w, orig_h),  # (width, height) for cv2.resize
            interpolation=cv2.INTER_NEAREST,
        )

        # 11. Get updated logits
        pred_masks_low_res = current_out["pred_masks"][0, 0].cpu().numpy()

        # 12. Update output_dict["cond_frame_outputs"][frame_idx]
        output_dict["cond_frame_outputs"][frame_idx] = self._make_compact_output(current_out)

        # 13. Return mask and logits (caller saves to storage)
        return mask_resized, pred_masks_low_res

    def propagate(
        self,
        video_id: str,
        frame_idx: int,
        frames,  # Indexable frame source
        masks,  # Indexable mask source
    ) -> tuple[np.ndarray, np.ndarray]:
        """Propagate tracking to a single frame.

        This is a convenience wrapper that propagates to exactly one frame.
        For batch propagation, use propagate_sequential() instead.

        Args:
            video_id: The video identifier.
            frame_idx: Frame index to propagate to.
            frames: Indexable frame source returning BGR uint8 (H, W, 3).
            masks: Indexable mask source returning uint8 (H, W).

        Returns:
            Tuple of (mask, logits) where:
            - mask: Binary mask array (height, width) with dtype uint8, values 0 or 255.
            - logits: Low-res logits array (256, 256).

        Raises:
            KeyError: If video_id is not open.
            RuntimeError: If no memory exists (need to add a prompt first).
        """
        # Prepare memory for arbitrary frame access
        self._set_memory_frame(video_id, frame_idx, frames, masks)

        frame = frames[frame_idx]
        return self._propagate_single_frame(video_id, frame_idx, frame)

    def propagate_sequential(
        self,
        video_id: str,
        start_frame: int,
        num_frames: int,
        frames,  # Indexable frame source
        masks,  # Indexable mask source
        on_result: Callable[[int, np.ndarray, np.ndarray], None],  # callback(frame_idx, mask, logits)
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
        # Memory window size for eviction
        MEM_WINDOW = 7

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
            mask_binary = (pred_mask_high_res > 0).cpu().numpy().astype(np.uint8) * 255
            mask_resized = cv2.resize(
                mask_binary,
                (orig_w, orig_h),  # (width, height) for cv2.resize
                interpolation=cv2.INTER_NEAREST,
            )

            # Get low-res logits
            pred_masks_low_res = current_out["pred_masks"][0, 0].cpu().numpy()

            # Call callback to save results
            on_result(frame_idx, mask_resized, pred_masks_low_res)

            # Store in output_dict["non_cond_frame_outputs"]
            output_dict["non_cond_frame_outputs"][frame_idx] = self._make_compact_output(
                current_out
            )

            # Memory eviction: remove frames from non_cond_frame_outputs where key <= frame_idx - MEM_WINDOW
            eviction_threshold = frame_idx - MEM_WINDOW
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
        # Get bounding box from mask1
        rows1 = np.any(mask1 > 127, axis=1)
        cols1 = np.any(mask1 > 127, axis=0)
        if not rows1.any() or not cols1.any():
            return 0.0

        y1_1, y2_1 = np.where(rows1)[0][[0, -1]]
        x1_1, x2_1 = np.where(cols1)[0][[0, -1]]
        # Add 1 to max indices to get proper bbox dimensions
        y2_1 += 1
        x2_1 += 1

        # Get bounding box from mask2
        rows2 = np.any(mask2 > 127, axis=1)
        cols2 = np.any(mask2 > 127, axis=0)
        if not rows2.any() or not cols2.any():
            return 0.0

        y1_2, y2_2 = np.where(rows2)[0][[0, -1]]
        x1_2, x2_2 = np.where(cols2)[0][[0, -1]]
        y2_2 += 1
        x2_2 += 1

        # Compute intersection box
        inter_x1 = max(x1_1, x1_2)
        inter_y1 = max(y1_1, y1_2)
        inter_x2 = min(x2_1, x2_2)
        inter_y2 = min(y2_1, y2_2)

        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0

        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)

        # Compute union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    def _propagate_single_frame(
        self,
        video_id: str,
        frame_idx: int,
        frame: np.ndarray,
        mask_prompt: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Propagate to a single frame, optionally with a mask prompt.

        Args:
            video_id: The video identifier.
            frame_idx: Frame index to propagate to.
            frame: BGR uint8 frame data (H, W, 3).
            mask_prompt: Optional mask to use as prompt (for detector re-prompting).

        Returns:
            (mask, logits) where mask is (H, W) uint8 and logits is (256, 256) float32.
        """
        session = self.sessions[video_id]
        frame_dims = session["frame_dims"]
        output_dict = session["output_dict"]
        orig_h, orig_w = frame_dims

        # Get image features and prepare backbone features
        try:
            _, backbone_out = self._get_image_features(video_id, frame_idx, frame)
        except Exception as e:
            raise RuntimeError(f"Failed to get image features for frame {frame_idx}: {e}") from e

        try:
            current_vision_feats, current_vision_pos_embeds, feat_sizes = (
                self._prepare_backbone_features(backbone_out)
            )
        except Exception as e:
            raise RuntimeError(f"Failed to prepare backbone features for frame {frame_idx}: {e}") from e

        # Prepare mask_inputs if mask_prompt provided
        mask_inputs = None
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

        # Call track_step
        try:
            with torch.inference_mode(), torch.autocast("cuda", torch.bfloat16):
                current_out = self.predictor.track_step(
                    frame_idx=frame_idx,
                    is_init_cond_frame=(mask_prompt is not None),  # Treat mask prompts as conditioning
                    current_vision_feats=current_vision_feats,
                    current_vision_pos_embeds=current_vision_pos_embeds,
                    feat_sizes=feat_sizes,
                    point_inputs=None,
                    mask_inputs=mask_inputs,
                    output_dict=output_dict,
                    num_frames=session["num_frames"],
                    run_mem_encoder=True,
                )
        except Exception as e:
            raise RuntimeError(f"track_step failed for frame {frame_idx}: {e}") from e

        # Extract mask from pred_masks_high_res
        pred_mask_high_res = current_out["pred_masks_high_res"][0, 0]

        # Threshold at 0, convert to uint8 * 255, resize to original dims
        mask_binary = (pred_mask_high_res > 0).cpu().numpy().astype(np.uint8) * 255
        mask_resized = cv2.resize(
            mask_binary,
            (orig_w, orig_h),
            interpolation=cv2.INTER_NEAREST,
        )

        # Get logits
        pred_masks_low_res = current_out["pred_masks"][0, 0].cpu().numpy()

        # Store in output_dict (memory for future frames)
        if mask_prompt is not None:
            output_dict["cond_frame_outputs"][frame_idx] = self._make_compact_output(current_out)
        else:
            output_dict["non_cond_frame_outputs"][frame_idx] = self._make_compact_output(current_out)

        return mask_resized, pred_masks_low_res

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
        mask_binary = (pred_mask_high_res > 0).cpu().numpy().astype(np.uint8) * 255
        mask_result = cv2.resize(mask_binary, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        # Store in cond_frame_outputs for memory (so future frames benefit)
        output_dict["cond_frame_outputs"][frame_idx] = self._make_compact_output(current_out)

        return mask_result

    def propagate_with_detector(
        self,
        video_id: str,
        num_frames: int,
        frames,  # Indexable frame source (frames[idx] -> np.ndarray)
        detector_masks,  # Indexable detector mask source (detector_masks[idx] -> np.ndarray)
        on_result: Callable[[int, np.ndarray, np.ndarray], None],
        on_final_result: Callable[[int, np.ndarray], None] | None = None,
        iou_threshold: float = 0.5,
        progress_callback=None,
    ) -> tuple[list[int], list[int]]:
        """Propagate tracking with detector-guided correction.

        Uses the detector mask to correct SAM2 when tracker drift is detected.
        Drift is detected by comparing bounding box IoU between tracker output
        and detector mask.

        The tracker output is ALWAYS passed to on_result callback.
        When on_final_result is provided:
        - If IoU >= threshold: calls on_final_result with tracker mask
        - If IoU < threshold: re-prompt with detector, calls on_final_result
          with corrected mask (tracker mask via on_result remains original)

        Args:
            video_id: The video identifier.
            num_frames: Number of frames to propagate.
            frames: Indexable frame source (frames[idx] returns BGR numpy array).
            detector_masks: Indexable detector mask source.
            on_result: Callback(frame_idx, mask, logits) called for each tracker result.
            on_final_result: Optional callback(frame_idx, mask) for final (corrected) masks.
            iou_threshold: Re-prompt when bbox IoU drops below this (default 0.5).
            progress_callback: Optional callback(frame_idx, num_frames) for progress.

        Returns:
            Tuple of (propagated_frames, corrected_frames) where:
            - propagated_frames: List of all frame indices that were propagated
            - corrected_frames: List of frame indices where detector correction was applied

        Raises:
            RuntimeError: If no memory exists and detector mask at frame 0 is empty.
        """
        MEM_WINDOW = 7

        session = self.sessions[video_id]
        output_dict = session["output_dict"]

        # Check output_dict has memory (need at least frame 0 initialized)
        has_memory = (
            len(output_dict["cond_frame_outputs"]) > 0
            or len(output_dict["non_cond_frame_outputs"]) > 0
        )
        if not has_memory:
            # Initialize from detector mask on frame 0
            detector_0 = np.array(detector_masks[0])
            if not (detector_0 > 127).any():
                raise RuntimeError(
                    "No memory and detector mask at frame 0 is empty. "
                    "Need either existing memory or detector mask to start."
                )
            print("  Initializing from detector mask at frame 0...")
            try:
                frame_0 = frames[0]
                mask_0, logits_0 = self._propagate_single_frame(
                    video_id, 0, frame_0, mask_prompt=detector_0
                )
                on_result(0, mask_0, logits_0)
                if on_final_result is not None:
                    on_final_result(0, mask_0)
                print("  Frame 0 initialized successfully")
            except Exception as e:
                import traceback
                traceback.print_exc()
                raise RuntimeError(f"Failed to initialize frame 0: {e}") from e

        propagated = []
        corrected_frames = []

        for i in range(num_frames):
            frame_idx = i

            # Skip frame 0 if already initialized
            if frame_idx == 0 and 0 in output_dict["cond_frame_outputs"]:
                propagated.append(frame_idx)
                continue

            # Load detector mask for this frame
            try:
                detector_mask = np.array(detector_masks[frame_idx])
            except (IndexError, KeyError):
                break

            detector_has_mask = (detector_mask > 127).any()

            # Get frame data
            frame = frames[frame_idx]

            # Propagate without prompt to get tracker prediction
            tracker_mask, tracker_logits = self._propagate_single_frame(
                video_id, frame_idx, frame
            )
            tracker_has_mask = (tracker_mask > 127).any()

            # Always call on_result with tracker output
            on_result(frame_idx, tracker_mask, tracker_logits)

            # Decide if we need to correct with detector
            need_correction = False
            iou = 1.0  # Default if not computed

            if detector_has_mask:
                if not tracker_has_mask:
                    # Tracker lost object but detector sees it
                    need_correction = True
                    iou = 0.0
                    print(f"  Frame {frame_idx}: tracker empty, correcting with detector")
                else:
                    # Both have masks - check IoU
                    iou = self._compute_bbox_iou(tracker_mask, detector_mask)
                    if iou < iou_threshold:
                        need_correction = True
                        print(f"  Frame {frame_idx}: IoU={iou:.3f} < {iou_threshold}, correcting")

            # Call on_final_result if provided
            if on_final_result is not None:
                if need_correction:
                    # Get corrected mask without overwriting tracker memory
                    corrected_mask = self._propagate_single_frame_no_store(
                        video_id, frame_idx, frame, mask_prompt=detector_mask
                    )
                    on_final_result(frame_idx, corrected_mask)
                    corrected_frames.append(frame_idx)
                else:
                    # Trust tracker - pass tracker mask to final
                    on_final_result(frame_idx, tracker_mask)

            # Memory eviction
            eviction_threshold = frame_idx - MEM_WINDOW
            keys_to_evict = [
                k
                for k in output_dict["non_cond_frame_outputs"]
                if k <= eviction_threshold
            ]
            for k in keys_to_evict:
                del output_dict["non_cond_frame_outputs"][k]

            propagated.append(frame_idx)

            # Progress callback
            if progress_callback:
                progress_callback(frame_idx, num_frames)

        print(f"  Propagated {len(propagated)} frames, corrected {len(corrected_frames)} frames")
        return propagated, corrected_frames
