"""SAM2++ keypoint tracker for multi-object point tracking across video frames.

================================================================================
PURPOSE
================================================================================

This module provides a SAM2PlusKeypointTracker class for tracking keypoints
(individual points of interest) across video frames. Unlike the segmentation
tracker which produces binary masks, this tracker outputs per-keypoint
coordinates by finding the argmax of predicted heatmap logits.

SAM2++ (SAM2-Plus) extends SAM2 with a point-tracking task that uses Gaussian
heatmaps as mask prompts alongside point prompts for conditioning frames,
then propagates through memory attention to predict keypoint locations on
subsequent frames.

================================================================================
ARCHITECTURE
================================================================================

SAM2++'s keypoint tracking:
- Uses build_sam2_video_predictor_plus with task='point' to build a model
  that predicts heatmaps centered on keypoint locations.
- Each object (keypoint) gets its own independent output_dict for memory,
  allowing multiple keypoints to be tracked simultaneously.
- Conditioning frames provide BOTH point_inputs (point coordinates) AND
  mask_inputs (Gaussian heatmaps) to anchor keypoint identity.
- Propagation frames use memory attention only (no explicit inputs).
- Output coordinates are extracted via argmax on pred_masks_high_res.

================================================================================
USAGE
================================================================================

    # Initialize once (loads SAM2++ model)
    tracker = SAM2PlusKeypointTracker()

    # Open a video session
    tracker.open_video(
        video_id="video_001",
        num_frames=500,
        frame_dims=(1080, 1920),
    )

    # Add a keypoint prompt
    x, y, logits = tracker.add_keypoint_prompt(
        video_id="video_001",
        frame_idx=0,
        obj_id=1,
        x_norm=0.5,
        y_norm=0.3,
        frame=frame_bgr,
        coords_source=my_coords_source,
    )

    # Propagate tracking
    tracker.propagate_sequential(
        video_id="video_001",
        start_frame=1,
        num_frames=499,
        frames=frame_source,
        coords_source=coords_source,
        on_result=my_callback,
    )

    # Close when done
    tracker.close_video("video_001")

"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import torch


class SAM2PlusKeypointTracker:
    """Stateful manager for SAM2++ multi-keypoint tracking sessions.

    Each keypoint is treated as an independent object with its own memory
    bank (output_dict). The tracker manages per-object state and coordinates
    extraction from predicted heatmaps.

    Attributes:
        device: Torch device for model inference.
        predictor: SAM2++ video predictor instance with task='point'.
        sessions: Dict mapping video_id to session state.
    """

    # Model input size (SAM2++ uses 1024x1024)
    INPUT_SIZE = 1024

    # Low-res logits size (SAM2++ outputs 256x256)
    LOGITS_SIZE = 256

    # Memory window size for non-conditioning frames.
    # SAM2's num_maskmem defaults to 7 (1 current + 6 previous frames).
    # We use 6 for loading previous frames.
    MEM_WINDOW = 6

    # Radius for Gaussian heatmap generation on conditioning frames.
    GAUSSIAN_RADIUS = 50

    def __init__(self, device: str | None = None, compile_model: bool = True):
        """Initialize the keypoint tracker and load the SAM2++ model.

        Args:
            device: Torch device string ("cuda", "cpu", etc.).
                    Defaults to "cuda" if available, else "cpu".
            compile_model: Whether to use torch.compile for faster inference.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # SAM2++ requires Hydra configs relative to the sam2_plus_repo directory
        sam2_plus_repo = str(Path(__file__).resolve().parents[3] / "sam2_plus_repo")
        sys.path.insert(0, sam2_plus_repo)

        saved_cwd = os.getcwd()
        os.chdir(sam2_plus_repo)
        try:
            from sam2_plus.build_sam import build_sam2_video_predictor_plus

            print(f"Loading SAM2++ model on {self.device}...")
            self.predictor = build_sam2_video_predictor_plus(
                config_file="configs/sam2.1/sam2.1_hiera_b+_predmasks_decoupled_MAME.yaml",
                ckpt_path="./checkpoints/SAM2-Plus/checkpoint_phase123.pt",
                apply_postprocessing=False,
                hydra_overrides_extra=["++model.non_overlap_masks=false"],
                vos_optimized=False,
                task="point",
            )
            print("SAM2++ model loaded.")
        finally:
            os.chdir(saved_cwd)

        # Compile model components for faster inference
        if compile_model and self.device == "cuda":
            print("Compiling SAM2++ image encoder with torch.compile...")
            self.predictor.image_encoder = torch.compile(
                self.predictor.image_encoder,
            )
            # Run warmup inference to trigger actual compilation (torch.compile is lazy)
            print("Running warmup inference (this may take a minute)...")
            dummy_input = torch.zeros(
                1, 3, self.INPUT_SIZE, self.INPUT_SIZE, device=self.device
            )
            with torch.inference_mode(), torch.autocast("cuda", torch.bfloat16):
                _ = self.predictor.forward_image(dummy_input)
            print("SAM2++ image encoder ready.")

        # Sessions dict: video_id -> session state
        self.sessions: dict[str, dict] = {}

    def open_video(
        self,
        video_id: str,
        num_frames: int,
        frame_dims: tuple[int, int],
        cond_frame_data: list[tuple[int, int, float, float]] | None = None,
        frames=None,
        coords_source=None,
    ) -> None:
        """Create a session for keypoint tracking.

        Initializes a session with per-object output_dicts. If cond_frame_data
        is provided along with frames and coords_source, conditioning frame
        memories will be reconstructed.

        Args:
            video_id: Unique identifier for this video session.
            num_frames: Actual number of frames in the video.
            frame_dims: Tuple of (height, width) for the video frames.
            cond_frame_data: List of (frame_idx, obj_id, x_norm, y_norm) tuples
                for reconstructing conditioning frame memories. Pass None or
                empty for a fresh session.
            frames: Optional indexable frame source (only needed if reconstructing).
            coords_source: Optional indexable coords source (only needed for
                _set_memory_frame during reconstruction).

        Raises:
            ValueError: If video_id already exists.
        """
        if video_id in self.sessions:
            raise ValueError(f"Video '{video_id}' is already open. Close it first.")

        session = {
            "output_dicts": {},        # obj_id -> {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
            "cond_frame_indices": {},   # obj_id -> set of frame_idx
            "frame_dims": frame_dims,
            "num_frames": num_frames,
        }

        self.sessions[video_id] = session

        # Reconstruct conditioning frame memories from stored data
        if cond_frame_data and frames is not None:
            cond_sorted = sorted(cond_frame_data, key=lambda t: (t[0], t[1]))
            print(f"  Reconstructing {len(cond_sorted)} conditioning frame memories...")
            sys.stdout.flush()
            for i, (frame_idx, obj_id, x_norm, y_norm) in enumerate(cond_sorted):
                print(
                    f"    [{i + 1}/{len(cond_sorted)}] Encoding frame {frame_idx}, obj {obj_id}...",
                    end="",
                )
                sys.stdout.flush()
                frame = frames[frame_idx]
                self._encode_stored_keypoint(
                    video_id, frame_idx, obj_id, x_norm, y_norm, frame,
                    store_as_cond=True,
                )
                print(" done")
                sys.stdout.flush()

        total_cond = sum(
            len(s) for s in session["cond_frame_indices"].values()
        )
        total_objs = len(session["output_dicts"])
        print(
            f"Opened video '{video_id}' with "
            f"{total_objs} objects, {total_cond} total cond frames"
        )

    def close_video(self, video_id: str) -> bool:
        """Close a video session and free resources.

        Clears all per-object output_dicts and cond_frame_indices, then
        removes the session from tracking.

        Args:
            video_id: The video identifier to close.

        Returns:
            True if the session was closed, False if video_id not found.
        """
        if video_id not in self.sessions:
            return False

        session = self.sessions[video_id]

        # Clear all per-object memory
        for obj_id in list(session["output_dicts"].keys()):
            od = session["output_dicts"][obj_id]
            od["cond_frame_outputs"].clear()
            od["non_cond_frame_outputs"].clear()
        session["output_dicts"].clear()
        session["cond_frame_indices"].clear()

        del self.sessions[video_id]
        return True

    def add_keypoint_prompt(
        self,
        video_id: str,
        frame_idx: int,
        obj_id: int,
        x_norm: float,
        y_norm: float,
        frame: np.ndarray,
        coords_source,
    ) -> tuple[float, float, np.ndarray]:
        """Add a keypoint prompt to a frame for a specific object.

        Generates a Gaussian heatmap at the given coordinates and runs
        track_step with both point_inputs and mask_inputs.

        Args:
            video_id: The video identifier.
            frame_idx: Index of the frame to annotate.
            obj_id: Object (keypoint) identifier.
            x_norm: Normalized x coordinate in [0, 1].
            y_norm: Normalized y coordinate in [0, 1].
            frame: BGR uint8 frame data (H, W, 3).
            coords_source: Indexable coords source for memory preparation.
                coords_source[idx] -> np.ndarray (K, 2) float32, NaN for empty.

        Returns:
            Tuple of (x_norm, y_norm, logits) where:
            - x_norm: Predicted x coordinate normalized to [0, 1].
            - y_norm: Predicted y coordinate normalized to [0, 1].
            - logits: Low-res logits array (256, 256) for potential refinement.

        Raises:
            KeyError: If video_id is not open.
        """
        session = self.sessions[video_id]

        # Ensure output_dict exists for this obj_id
        self._ensure_obj(session, obj_id)

        # Prepare memory for arbitrary frame access
        self._set_memory_frame(video_id, frame_idx, obj_id, coords_source=coords_source, frames=None)

        # Get image features
        _, backbone_out = self._get_image_features(video_id, frame_idx, frame)
        current_vision_feats, current_vision_pos_embeds, feat_sizes = (
            self._prepare_backbone_features(backbone_out)
        )

        # Generate Gaussian heatmap
        x_model = x_norm * self.INPUT_SIZE
        y_model = y_norm * self.INPUT_SIZE
        gaussian = self._generate_gaussian(x_model, y_model)

        # Create point inputs
        point_inputs = self._point_inputs_from_coords(x_norm, y_norm)

        # Determine is_init_cond_frame (check ALL objects)
        has_any_memory = any(
            od["cond_frame_outputs"] or od["non_cond_frame_outputs"]
            for od in session["output_dicts"].values()
        )
        is_init_cond_frame = not has_any_memory

        # Run track_step
        with torch.inference_mode(), torch.autocast("cuda", torch.bfloat16):
            current_out = self.predictor.track_step(
                frame_idx=frame_idx,
                is_init_cond_frame=is_init_cond_frame,
                current_vision_feats=current_vision_feats,
                current_vision_pos_embeds=current_vision_pos_embeds,
                feat_sizes=feat_sizes,
                point_inputs=point_inputs,
                mask_inputs=gaussian,
                output_dict=session["output_dicts"][obj_id],
                num_frames=session["num_frames"],
            )

        # Extract keypoint coordinates
        pred_x, pred_y, logits = self._extract_keypoint(current_out, session["frame_dims"])

        # Add frame to cond_frame_indices for this obj_id
        session["cond_frame_indices"][obj_id].add(frame_idx)

        # Store compact output
        session["output_dicts"][obj_id]["cond_frame_outputs"][frame_idx] = (
            self._make_compact_output(current_out)
        )

        return pred_x, pred_y, logits

    def refine_keypoint(
        self,
        video_id: str,
        frame_idx: int,
        obj_id: int,
        x_norm: float,
        y_norm: float,
        frame: np.ndarray,
        prev_logits: np.ndarray,
        coords_source,
    ) -> tuple[float, float, np.ndarray]:
        """Refine an existing keypoint with updated coordinates.

        Removes the old conditioning output for this frame/obj, then runs
        track_step with the new point_inputs and previous logits as mask_inputs.

        Args:
            video_id: The video identifier.
            frame_idx: Index of the frame to refine.
            obj_id: Object (keypoint) identifier.
            x_norm: Updated normalized x coordinate in [0, 1].
            y_norm: Updated normalized y coordinate in [0, 1].
            frame: BGR uint8 frame data (H, W, 3).
            prev_logits: Previous low-res logits (256, 256) from prior call.
            coords_source: Indexable coords source for memory preparation.

        Returns:
            Tuple of (x_norm, y_norm, logits) where:
            - x_norm: Predicted x coordinate normalized to [0, 1].
            - y_norm: Predicted y coordinate normalized to [0, 1].
            - logits: Updated low-res logits array (256, 256).

        Raises:
            KeyError: If video_id is not open.
        """
        session = self.sessions[video_id]
        self._ensure_obj(session, obj_id)
        output_dict = session["output_dicts"][obj_id]

        # Pop existing conditioning output for this frame to avoid self-bias
        output_dict["cond_frame_outputs"].pop(frame_idx, None)

        # Prepare memory
        self._set_memory_frame(video_id, frame_idx, obj_id, coords_source=coords_source, frames=None)

        # Determine is_init_cond_frame based on ALL objects
        has_any_memory = any(
            od["cond_frame_outputs"] or od["non_cond_frame_outputs"]
            for od in session["output_dicts"].values()
        )
        is_init_cond_frame = not has_any_memory

        # Get image features
        _, backbone_out = self._get_image_features(video_id, frame_idx, frame)
        current_vision_feats, current_vision_pos_embeds, feat_sizes = (
            self._prepare_backbone_features(backbone_out)
        )

        # Create point inputs
        point_inputs = self._point_inputs_from_coords(x_norm, y_norm)

        # Convert prev_logits to mask_inputs tensor
        prev_logits_tensor = torch.from_numpy(prev_logits.astype(np.float32))
        prev_logits_tensor = prev_logits_tensor.unsqueeze(0).unsqueeze(0).to(self.device)
        prev_logits_tensor = torch.clamp(prev_logits_tensor, -32.0, 32.0)

        # Run track_step with point_inputs and prev_logits as mask_inputs
        with torch.inference_mode(), torch.autocast("cuda", torch.bfloat16):
            current_out = self.predictor.track_step(
                frame_idx=frame_idx,
                is_init_cond_frame=is_init_cond_frame,
                current_vision_feats=current_vision_feats,
                current_vision_pos_embeds=current_vision_pos_embeds,
                feat_sizes=feat_sizes,
                point_inputs=point_inputs,
                mask_inputs=prev_logits_tensor,
                output_dict=output_dict,
                num_frames=session["num_frames"],
            )

        # Extract keypoint coordinates
        pred_x, pred_y, logits = self._extract_keypoint(current_out, session["frame_dims"])

        # Store new result in cond_frame_outputs
        output_dict["cond_frame_outputs"][frame_idx] = self._make_compact_output(current_out)

        return pred_x, pred_y, logits

    def propagate_sequential(
        self,
        video_id: str,
        start_frame: int,
        num_frames: int,
        frames,
        coords_source,
        on_result: Callable[[int, dict[int, tuple[float, float, np.ndarray]]], None] | None = None,
        progress_interval: int = 50,
    ) -> list[int]:
        """Propagate keypoint tracking forward from start_frame.

        For each frame, iterates over all objects that have conditioning frames
        and runs track_step with memory only (no explicit inputs). Extracts
        coordinates via argmax on the predicted heatmap.

        Args:
            video_id: The video identifier.
            start_frame: Frame index to start propagation from.
            num_frames: Maximum number of frames to propagate.
            frames: Indexable frame source returning BGR uint8 (H, W, 3).
            coords_source: Indexable coords source for gap detection.
                coords_source[idx] -> np.ndarray (K, 2) float32, NaN for empty.
            on_result: Optional callback called for each frame with
                (frame_idx, {obj_id: (x_norm, y_norm, logits), ...}).
                The caller should save the results in this callback.
            progress_interval: Print progress every N frames (0 to disable).

        Returns:
            List of frame indices that were propagated.

        Raises:
            KeyError: If video_id is not open.
            RuntimeError: If no memory exists for any object.
        """
        session = self.sessions[video_id]
        frame_dims = session["frame_dims"]

        # Identify objects that have conditioning frames
        active_obj_ids = [
            obj_id for obj_id, cond_set in session["cond_frame_indices"].items()
            if cond_set
        ]
        if not active_obj_ids:
            raise RuntimeError(
                "No conditioning frames exist for any object. "
                "Use add_keypoint_prompt() to create an initial keypoint first."
            )

        propagated = []

        for i in range(num_frames):
            frame_idx = start_frame + i

            # Try to read frame, break on out-of-bounds
            try:
                frame_bgr = frames[frame_idx]
            except (IndexError, KeyError):
                break

            # Get image features once per frame (shared across objects)
            _, backbone_out = self._get_image_features(video_id, frame_idx, frame_bgr)
            current_vision_feats, current_vision_pos_embeds, feat_sizes = (
                self._prepare_backbone_features(backbone_out)
            )

            frame_results: dict[int, tuple[float, float, np.ndarray]] = {}

            for obj_id in active_obj_ids:
                cond_indices = session["cond_frame_indices"].get(obj_id, set())
                output_dict = session["output_dicts"][obj_id]

                # Skip if this frame is a conditioning frame for this object
                if frame_idx in cond_indices:
                    continue

                # Run track_step with no inputs (memory-based propagation)
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
                    )

                # Extract coordinates
                pred_x, pred_y, logits = self._extract_keypoint(current_out, frame_dims)
                frame_results[obj_id] = (pred_x, pred_y, logits)

                # Store compact output in non_cond_frame_outputs
                output_dict["non_cond_frame_outputs"][frame_idx] = (
                    self._make_compact_output(current_out)
                )

                # Memory eviction: keep MEM_WINDOW + 1 frames
                eviction_threshold = frame_idx - (self.MEM_WINDOW + 1)
                keys_to_evict = [
                    k for k in output_dict["non_cond_frame_outputs"]
                    if k <= eviction_threshold
                ]
                for k in keys_to_evict:
                    del output_dict["non_cond_frame_outputs"][k]

            # Call callback with results for this frame
            if on_result is not None and frame_results:
                on_result(frame_idx, frame_results)

            propagated.append(frame_idx)

            # Progress reporting
            if progress_interval > 0 and len(propagated) % progress_interval == 0:
                print(f"  Propagated {len(propagated)} frames...")

        return propagated

    def reset_frame(self, video_id: str, frame_idx: int) -> None:
        """Reset a frame: clear its memory state across all objects.

        Removes the frame from all per-object output_dicts and
        cond_frame_indices.

        Args:
            video_id: The video identifier.
            frame_idx: Frame index to reset.
        """
        session = self.sessions[video_id]

        for obj_id in list(session["output_dicts"].keys()):
            output_dict = session["output_dicts"][obj_id]
            output_dict["cond_frame_outputs"].pop(frame_idx, None)
            output_dict["non_cond_frame_outputs"].pop(frame_idx, None)

            if obj_id in session["cond_frame_indices"]:
                session["cond_frame_indices"][obj_id].discard(frame_idx)

    def reset_video(self, video_id: str) -> None:
        """Reset entire video: clear all per-object output_dicts and indices.

        Args:
            video_id: The video identifier.
        """
        session = self.sessions[video_id]

        for obj_id in list(session["output_dicts"].keys()):
            output_dict = session["output_dicts"][obj_id]
            output_dict["cond_frame_outputs"].clear()
            output_dict["non_cond_frame_outputs"].clear()

        for obj_id in list(session["cond_frame_indices"].keys()):
            session["cond_frame_indices"][obj_id].clear()

    # =========================================================================
    # Internal methods
    # =========================================================================

    def _ensure_obj(self, session: dict, obj_id: int) -> None:
        """Ensure output_dict and cond_frame_indices exist for an object.

        Args:
            session: The session dict for the video.
            obj_id: Object (keypoint) identifier.
        """
        if obj_id not in session["output_dicts"]:
            session["output_dicts"][obj_id] = {
                "cond_frame_outputs": {},
                "non_cond_frame_outputs": {},
            }
        if obj_id not in session["cond_frame_indices"]:
            session["cond_frame_indices"][obj_id] = set()

    def _get_image_features(
        self, video_id: str, frame_idx: int, frame_bgr: np.ndarray
    ) -> tuple[torch.Tensor, dict]:
        """Get image features for a frame.

        Preprocesses the frame and runs through the SAM2++ image encoder.

        Args:
            video_id: The video identifier.
            frame_idx: Frame index (unused, kept for API compatibility).
            frame_bgr: BGR uint8 frame data (H, W, 3).

        Returns:
            (image_tensor, backbone_out) where:
            - image_tensor is the preprocessed image tensor (1, 3, 1024, 1024)
            - backbone_out is the dict from forward_image
        """
        # GPU-accelerated preprocessing
        frame_gpu = torch.from_numpy(frame_bgr).to(self.device)

        # BGR->RGB, HWC->CHW, normalize to [0,1]
        image_tensor = frame_gpu[..., [2, 1, 0]].permute(2, 0, 1).float().div_(255.0)

        # Resize to model input size
        image_tensor = torch.nn.functional.interpolate(
            image_tensor.unsqueeze(0),
            size=(self.INPUT_SIZE, self.INPUT_SIZE),
            mode="bilinear",
            align_corners=False,
        )

        # Run through image encoder
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
        num_feature_levels = self.predictor.num_feature_levels

        feature_maps = backbone_out["backbone_fpn"][-num_feature_levels:]
        vision_pos_enc = backbone_out["vision_pos_enc"][-num_feature_levels:]

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

    def _generate_gaussian(
        self, x_model: float, y_model: float
    ) -> torch.Tensor:
        """Generate a Gaussian heatmap at the given model-space coordinates.

        Uses the generate_gaussian utility from SAM2++ training code to create
        a heatmap in 1024x1024 model space.

        Args:
            x_model: X coordinate in model space (0 to 1024).
            y_model: Y coordinate in model space (0 to 1024).

        Returns:
            Gaussian heatmap tensor of shape (1, 1, 1024, 1024) on device.
        """
        from training.dataset_plus.point.utils import generate_gaussian

        heatmap = generate_gaussian(
            center=(int(x_model), int(y_model)),
            img_size=(self.INPUT_SIZE, self.INPUT_SIZE),
            radius=self.GAUSSIAN_RADIUS,
            sigma=None,
        )

        # Convert to tensor: (1, 1, 1024, 1024) float on device
        heatmap_tensor = torch.from_numpy(heatmap).float()
        heatmap_tensor = heatmap_tensor.unsqueeze(0).unsqueeze(0).to(self.device)

        return heatmap_tensor

    def _extract_keypoint(
        self, current_out: dict, frame_dims: tuple[int, int]
    ) -> tuple[float, float, np.ndarray]:
        """Extract keypoint coordinates from track_step output.

        Finds the argmax of the high-resolution predicted heatmap to get
        the predicted keypoint location, and extracts low-res logits.

        Args:
            current_out: Output dict from track_step containing pred_masks_high_res.
            frame_dims: (height, width) of the original video frame (unused,
                kept for API compatibility since coords are normalized).

        Returns:
            Tuple of (x_norm, y_norm, logits) where:
            - x_norm: Predicted x coordinate normalized to [0, 1].
            - y_norm: Predicted y coordinate normalized to [0, 1].
            - logits: Low-res logits array (256, 256) float32.
        """
        pred_masks_high_res = current_out["pred_masks_high_res"]
        logit_map = pred_masks_high_res[0, 0]  # (H_highres, W_highres)

        max_idx = torch.argmax(logit_map)
        y, x = torch.unravel_index(max_idx, logit_map.shape)

        x_norm = x.item() / logit_map.shape[1]
        y_norm = y.item() / logit_map.shape[0]

        # Get low-res logits (256, 256) for potential refinement
        logits = current_out["pred_masks"][0, 0].float().cpu().numpy()

        return x_norm, y_norm, logits

    def _encode_stored_keypoint(
        self,
        video_id: str,
        frame_idx: int,
        obj_id: int,
        x_norm: float,
        y_norm: float,
        frame: np.ndarray,
        store_as_cond: bool = True,
    ) -> dict | None:
        """Encode a stored keypoint into the memory bank.

        Generates a Gaussian heatmap from the coordinates and runs through
        track_step with both point_inputs and mask_inputs.

        Used to reconstruct conditioning frame memories when opening a video
        that has existing keypoint data, or to encode non-conditioning frame
        memories for arbitrary frame access.

        Args:
            video_id: The video identifier.
            frame_idx: Frame index to encode.
            obj_id: Object (keypoint) identifier.
            x_norm: Normalized x coordinate in [0, 1].
            y_norm: Normalized y coordinate in [0, 1].
            frame: BGR uint8 frame data (H, W, 3).
            store_as_cond: If True, store to cond_frame_outputs and return None.
                If False, return the compact output dict without storing.

        Returns:
            None if store_as_cond=True, otherwise the compact output dict.
        """
        session = self.sessions[video_id]
        self._ensure_obj(session, obj_id)
        output_dict = session["output_dicts"][obj_id]

        # Generate Gaussian heatmap
        x_model = x_norm * self.INPUT_SIZE
        y_model = y_norm * self.INPUT_SIZE
        gaussian = self._generate_gaussian(x_model, y_model)

        # Create point inputs
        point_inputs = self._point_inputs_from_coords(x_norm, y_norm)

        # Get image features
        _, backbone_out = self._get_image_features(video_id, frame_idx, frame)
        current_vision_feats, current_vision_pos_embeds, feat_sizes = (
            self._prepare_backbone_features(backbone_out)
        )

        # Run track_step
        with torch.inference_mode(), torch.autocast("cuda", torch.bfloat16):
            current_out = self.predictor.track_step(
                frame_idx=frame_idx,
                is_init_cond_frame=True,
                current_vision_feats=current_vision_feats,
                current_vision_pos_embeds=current_vision_pos_embeds,
                feat_sizes=feat_sizes,
                point_inputs=point_inputs,
                mask_inputs=gaussian,
                output_dict=output_dict,
                num_frames=session["num_frames"],
            )

        compact = self._make_compact_output(current_out)
        if store_as_cond:
            output_dict["cond_frame_outputs"][frame_idx] = compact
            session["cond_frame_indices"][obj_id].add(frame_idx)
            return None
        else:
            return compact

    def _set_memory_frame(
        self,
        video_id: str,
        frame_idx: int,
        obj_id: int,
        coords_source,
        frames=None,
    ) -> None:
        """Prepare non_cond_frame_outputs for tracking at frame_idx.

        Clears all existing non-cond memory for this object, then walks
        backward from frame_idx-1, checking coords_source for valid data.
        Stops after MEM_WINDOW frames or when hitting a gap (NaN coords).

        Note: This method currently only clears non-cond memory. If frames
        are provided along with coords_source, it would encode them (but
        for keypoint tracking the propagation loop handles this naturally).

        Args:
            video_id: The video session ID.
            frame_idx: Target frame we're about to track/prompt.
            obj_id: Object (keypoint) identifier.
            coords_source: Indexable coords source.
                coords_source[idx] -> np.ndarray (K, 2) float32, NaN for empty.
            frames: Optional indexable frame source. If None, only clears memory
                without encoding previous frames.
        """
        session = self.sessions[video_id]
        self._ensure_obj(session, obj_id)
        output_dict = session["output_dicts"][obj_id]
        cond_indices = session["cond_frame_indices"].get(obj_id, set())

        # Clear all existing non-cond memory
        output_dict["non_cond_frame_outputs"].clear()

        # Early return if no previous frames or no frame source for encoding
        if frame_idx <= 0 or frames is None or coords_source is None:
            return

        # Walk backward, encode into memory
        frames_added = 0
        for prev_idx in range(frame_idx - 1, -1, -1):
            if frames_added >= self.MEM_WINDOW:
                break

            # Skip conditioning frames for this object (handled by SAM2)
            if prev_idx in cond_indices:
                continue

            # Check coords for gap detection
            try:
                coords = np.asarray(coords_source[prev_idx])
            except (IndexError, KeyError):
                break

            # Stop at NaN coords (gap in tracking)
            if np.isnan(coords).any():
                break

            # For keypoint tracking, we need valid x, y to encode
            # coords shape may be (K, 2) - find the column for this obj_id
            # In practice, this is handled by the command layer providing
            # per-object coords. For now, we skip encoding if no frame source.
            # The propagation loop builds memory naturally through sequential access.
            frames_added += 1

    def _point_inputs_from_coords(
        self, x_norm: float, y_norm: float
    ) -> dict[str, torch.Tensor]:
        """Create point_inputs dict matching SAM2++ format.

        Converts normalized coordinates to 1024-space and creates the
        point_coords and point_labels tensors.

        Args:
            x_norm: Normalized x coordinate in [0, 1].
            y_norm: Normalized y coordinate in [0, 1].

        Returns:
            Dict with:
            - point_coords: (1, 1, 2) tensor in 1024 space.
            - point_labels: (1, 1) tensor, value 1 (positive).
        """
        x_model = x_norm * self.INPUT_SIZE
        y_model = y_norm * self.INPUT_SIZE

        point_coords = torch.tensor(
            [[[x_model, y_model]]],
            dtype=torch.float32,
            device=self.device,
        )  # (1, 1, 2)

        point_labels = torch.tensor(
            [[1]],
            dtype=torch.int32,
            device=self.device,
        )  # (1, 1)

        return {
            "point_coords": point_coords,
            "point_labels": point_labels,
        }
