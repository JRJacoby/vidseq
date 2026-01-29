# SAM2 Streaming Segmentor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create a SAM2-based streaming segmentor that mirrors the existing SAM3 StreamingSegmentor API, enabling direct comparison of tracking quality.

**Architecture:** Create `sam2_streaming_segmentor.py` alongside the existing SAM3 version. It will wrap SAM2VideoPredictor's core methods (`track_step`, `forward_image`) while maintaining the same external API (`open_video`, `add_point_prompt`, `refine_mask`, `propagate_sequential`). The TCP worker will be updated to support switching between backends.

**Tech Stack:** SAM2 (from /n/groups/datta/john/repos/sam2), PyTorch, NumPy, OpenCV

---

## Task 1: Create SAM2StreamingSegmentor Base Class

**Files:**
- Create: `vidseq/services/sam2/streaming_segmentor.py`

**Step 1: Create the file with imports and class skeleton**

```python
"""Stateful streaming segmentation manager using SAM2 tracker.

This mirrors the SAM3 StreamingSegmentor API but uses SAM2VideoPredictor
internally for comparison testing.
"""

from __future__ import annotations

import sys
sys.path.insert(0, "/n/groups/datta/john/repos/sam2")

import cv2
import numpy as np
import torch

from sam2.build_sam import build_sam2_video_predictor


class SAM2StreamingSegmentor:
    """Stateful manager for SAM2 video segmentation sessions.

    Mirrors the SAM3 StreamingSegmentor API for comparison testing.
    """

    # SAM2 expects 1024x1024 input (configurable, but this is the default)
    INPUT_SIZE = 1024

    def __init__(self, device: str | None = None):
        """Initialize the segmentor and load the SAM2 model.

        Args:
            device: Torch device string. Defaults to "cuda" if available.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading SAM2 model on {self.device}...")

        # Use SAM2.1 large model (best quality)
        self.predictor = build_sam2_video_predictor(
            config_file="configs/sam2.1/sam2.1_hiera_l.yaml",
            ckpt_path="/n/groups/datta/john/repos/sam2/checkpoints/sam2.1_hiera_large.pt",
            device=self.device,
        )

        # Sessions dict: video_id -> session state
        self.sessions: dict[str, dict] = {}

        print("SAM2 model loaded.")

    def _scale_point(self, x: float, y: float, orig_w: int, orig_h: int) -> tuple[float, float]:
        """Scale point from original frame coords to normalized [0,1] coords.

        SAM2 expects points normalized to [0,1] relative to the image.
        """
        return (x / orig_w, y / orig_h)
```

**Step 2: Verify the file is syntactically correct**

Run: `python -c "import sys; sys.path.insert(0, '/n/groups/datta/john/projects/vidseq'); from vidseq.services.sam2.streaming_segmentor import SAM2StreamingSegmentor"`

Expected: No import errors (model loading will happen but that's fine)

**Step 3: Commit**

```bash
git add vidseq/services/sam2/streaming_segmentor.py
git commit -m "feat: add SAM2StreamingSegmentor skeleton"
```

---

## Task 2: Implement open_video and close_video

**Files:**
- Modify: `vidseq/services/sam2/streaming_segmentor.py`

**Step 1: Add open_video method**

Add after `_scale_point`:

```python
    def open_video(
        self,
        video_id: str,
        frames,  # Indexable returning BGR uint8 (H, W, 3)
        masks,   # Indexable/assignable for binary mask storage
        logits,  # Indexable/assignable for logits storage (256x256 float32)
        frame_dims: tuple[int, int],  # (height, width)
        cond_frame_indices: set[int] | list[int] | None = None,
    ) -> None:
        """Create a session with external frame, mask, and logits sources.

        This initializes the session and reconstructs conditioning frame memories
        from stored masks. Call this before using add_point_prompt or refine_mask.

        Args:
            video_id: Unique identifier for this video session.
            frames: Indexable frame source returning BGR uint8 arrays (H, W, 3).
            masks: Indexable storage for binary masks.
            logits: Indexable storage for low-res logits (256x256 float32).
            frame_dims: Tuple of (height, width) for the video frames.
            cond_frame_indices: Frame indices that have existing user prompts.

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

        height, width = frame_dims

        # Create a wrapper that provides frames to SAM2
        # SAM2 expects frames as torch tensors in RGB, normalized
        class FrameWrapper:
            def __init__(self, frame_source, h, w):
                self.frame_source = frame_source
                self.h = h
                self.w = w

            def __getitem__(self, idx):
                frame_bgr = self.frame_source[idx]
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                # Convert to tensor and normalize to [0, 1]
                tensor = torch.from_numpy(frame_rgb).float() / 255.0
                # HWC -> CHW
                tensor = tensor.permute(2, 0, 1)
                return tensor

        # Create session state that mimics SAM2's inference_state
        session = {
            "frames": frames,
            "frame_wrapper": FrameWrapper(frames, height, width),
            "masks": masks,
            "logits": logits,
            "cond_frame_indices": cond_frame_indices,
            "frame_dims": frame_dims,
            # SAM2's output_dict structure
            "output_dict": {
                "cond_frame_outputs": {},
                "non_cond_frame_outputs": {},
            },
            # Track number of frames (estimate from first access or set externally)
            "num_frames": 10000,  # Large default, will be bounded by actual frames
            # Feature cache for interactive frames
            "cached_features": {},
        }

        self.sessions[video_id] = session

        # Reconstruct conditioning frame memories from stored masks
        cond_to_reconstruct = [idx for idx in sorted(cond_frame_indices)
                               if self._mask_exists(masks, idx)]
        if cond_to_reconstruct:
            print(f"  Reconstructing {len(cond_to_reconstruct)} cond frame memories...")
            for idx in cond_to_reconstruct:
                self._encode_stored_mask(video_id, idx)

    def _mask_exists(self, masks, idx: int) -> bool:
        """Check if a non-empty mask exists at the given index."""
        try:
            mask = masks[idx]
            return mask is not None and np.any(mask > 0)
        except (IndexError, KeyError):
            return False

    def close_video(self, video_id: str) -> bool:
        """Close a video session and release resources.

        Args:
            video_id: The video identifier.

        Returns:
            True if the session was closed, False if it didn't exist.
        """
        if video_id not in self.sessions:
            return False

        # Clear cached features to free GPU memory
        session = self.sessions[video_id]
        session["cached_features"].clear()
        session["output_dict"]["cond_frame_outputs"].clear()
        session["output_dict"]["non_cond_frame_outputs"].clear()

        del self.sessions[video_id]
        return True
```

**Step 2: Add _encode_stored_mask placeholder**

Add after `close_video`:

```python
    def _encode_stored_mask(self, video_id: str, frame_idx: int) -> None:
        """Encode a stored mask into memory features.

        Used to reconstruct conditioning frame memories from stored masks.
        """
        session = self.sessions[video_id]
        masks = session["masks"]
        frame_dims = session["frame_dims"]
        height, width = frame_dims

        # Load the stored mask
        mask = masks[frame_idx]
        if mask is None or not np.any(mask > 0):
            return

        # Convert mask to SAM2's expected format
        # SAM2 expects masks as torch tensors with shape (1, 1, H, W)
        mask_tensor = torch.from_numpy(mask.astype(np.float32) / 255.0)
        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0).to(self.device)

        # Get image features
        frame_bgr = session["frames"][frame_idx]
        image, backbone_out = self._get_image_features(video_id, frame_idx)

        # Run track_step with mask input to encode into memory
        current_vision_feats, current_vision_pos_embeds, feat_sizes = self._prepare_backbone_features(backbone_out)

        output_dict = session["output_dict"]
        current_out = self.predictor.track_step(
            frame_idx=frame_idx,
            is_init_cond_frame=True,
            current_vision_feats=current_vision_feats,
            current_vision_pos_embeds=current_vision_pos_embeds,
            feat_sizes=feat_sizes,
            point_inputs=None,
            mask_inputs=mask_tensor,
            output_dict=output_dict,
            num_frames=session["num_frames"],
            track_in_reverse=False,
            run_mem_encoder=True,
        )

        # Store as conditioning frame output
        output_dict["cond_frame_outputs"][frame_idx] = self._make_compact_output(current_out)
```

**Step 3: Add helper methods for image features**

Add after `_encode_stored_mask`:

```python
    def _get_image_features(self, video_id: str, frame_idx: int):
        """Get image and backbone features for a frame, with caching."""
        session = self.sessions[video_id]

        # Check cache
        cached = session["cached_features"].get(frame_idx)
        if cached is not None:
            return cached

        # Load and preprocess frame
        frame_bgr = session["frames"][frame_idx]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Resize to model input size
        frame_resized = cv2.resize(frame_rgb, (self.INPUT_SIZE, self.INPUT_SIZE))

        # Convert to tensor: HWC -> CHW, normalize to [0, 1]
        image = torch.from_numpy(frame_resized).float() / 255.0
        image = image.permute(2, 0, 1).unsqueeze(0).to(self.device)

        # Run backbone
        with torch.inference_mode():
            backbone_out = self.predictor.forward_image(image)

        # Cache (only keep most recent frame)
        session["cached_features"] = {frame_idx: (image, backbone_out)}

        return image, backbone_out

    def _prepare_backbone_features(self, backbone_out):
        """Prepare backbone features for track_step."""
        # Extract FPN features and position encodings
        backbone_fpn = backbone_out["backbone_fpn"]
        vision_pos_enc = backbone_out["vision_pos_enc"]

        # Get feature sizes
        feat_sizes = [(x.shape[-2], x.shape[-1]) for x in vision_pos_enc]

        # Reshape to (HW, B, C) format expected by track_step
        current_vision_feats = [x.flatten(2).permute(2, 0, 1) for x in backbone_fpn]
        current_vision_pos_embeds = [x.flatten(2).permute(2, 0, 1) for x in vision_pos_enc]

        return current_vision_feats, current_vision_pos_embeds, feat_sizes

    def _make_compact_output(self, current_out):
        """Create a compact version of frame output for storage."""
        return {
            "maskmem_features": current_out.get("maskmem_features"),
            "maskmem_pos_enc": current_out.get("maskmem_pos_enc"),
            "pred_masks": current_out["pred_masks"],
            "obj_ptr": current_out["obj_ptr"],
            "object_score_logits": current_out.get("object_score_logits"),
        }
```

**Step 4: Verify syntax**

Run: `python -c "from vidseq.services.sam2.streaming_segmentor import SAM2StreamingSegmentor; print('OK')"`

**Step 5: Commit**

```bash
git add vidseq/services/sam2/streaming_segmentor.py
git commit -m "feat: add open_video/close_video to SAM2StreamingSegmentor"
```

---

## Task 3: Implement add_point_prompt

**Files:**
- Modify: `vidseq/services/sam2/streaming_segmentor.py`

**Step 1: Add add_point_prompt method**

Add after `_make_compact_output`:

```python
    def add_point_prompt(
        self,
        video_id: str,
        frame_idx: int,
        location: tuple[float, float] | list[tuple[float, float]],
        label: int | list[int],
    ) -> np.ndarray:
        """Add point prompt(s) to a BLANK frame and generate initial mask.

        Args:
            video_id: The video identifier.
            frame_idx: Index of the frame to annotate.
            location: (x, y) point or list of points in original frame coords.
            label: Label(s) for each point. 1=positive, 0=negative.

        Returns:
            Binary mask array (height, width) with dtype uint8, values 0 or 255.
        """
        session = self.sessions[video_id]
        masks_storage = session["masks"]
        logits_storage = session["logits"]
        frame_dims = session["frame_dims"]
        height, width = frame_dims
        output_dict = session["output_dict"]

        # Normalize to lists
        if isinstance(location, tuple) and len(location) == 2 and not isinstance(location[0], tuple):
            locations = [location]
            labels = [label]
        else:
            locations = list(location)
            labels = list(label) if isinstance(label, (list, tuple)) else [label]

        # Scale points to normalized [0, 1] coordinates
        scaled_points = []
        for loc in locations:
            nx, ny = self._scale_point(loc[0], loc[1], width, height)
            scaled_points.append([nx, ny])

        # Convert to tensors
        point_coords = torch.tensor([scaled_points], dtype=torch.float32, device=self.device)
        point_labels = torch.tensor([labels], dtype=torch.int32, device=self.device)

        # Scale to model input size (SAM2 expects points in image_size space)
        point_coords = point_coords * self.INPUT_SIZE

        point_inputs = {
            "point_coords": point_coords,
            "point_labels": point_labels,
        }

        # Get image features
        image, backbone_out = self._get_image_features(video_id, frame_idx)
        current_vision_feats, current_vision_pos_embeds, feat_sizes = self._prepare_backbone_features(backbone_out)

        # Check if this is the first conditioning frame
        has_memory = bool(output_dict["cond_frame_outputs"]) or bool(output_dict["non_cond_frame_outputs"])
        is_init_cond_frame = not has_memory

        # Run tracking with point prompt
        with torch.inference_mode():
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
                track_in_reverse=False,
                run_mem_encoder=True,
            )

        # Extract and save mask
        mask_logits = current_out["pred_masks_high_res"][0, 0].float().cpu().numpy()
        mask_binary = (mask_logits > 0).astype(np.uint8) * 255
        mask_resized = cv2.resize(mask_binary, (width, height), interpolation=cv2.INTER_NEAREST)
        masks_storage[frame_idx] = mask_resized

        # Save low-res logits for refinement
        logits_lowres = current_out["pred_masks"][0, 0].float().cpu().numpy()
        logits_storage[frame_idx] = logits_lowres

        # Store as conditioning frame
        session["cond_frame_indices"].add(frame_idx)
        output_dict["cond_frame_outputs"][frame_idx] = self._make_compact_output(current_out)

        return mask_resized
```

**Step 2: Verify syntax**

Run: `python -c "from vidseq.services.sam2.streaming_segmentor import SAM2StreamingSegmentor; print('OK')"`

**Step 3: Commit**

```bash
git add vidseq/services/sam2/streaming_segmentor.py
git commit -m "feat: add add_point_prompt to SAM2StreamingSegmentor"
```

---

## Task 4: Implement refine_mask

**Files:**
- Modify: `vidseq/services/sam2/streaming_segmentor.py`

**Step 1: Add refine_mask method**

Add after `add_point_prompt`:

```python
    def refine_mask(
        self,
        video_id: str,
        frame_idx: int,
        location: tuple[float, float] | list[tuple[float, float]],
        label: int | list[int],
    ) -> np.ndarray:
        """Refine an existing mask with point prompt(s).

        Uses the previous mask logits as context for refinement.

        Args:
            video_id: The video identifier.
            frame_idx: Index of the frame to refine.
            location: (x, y) point or list of points in original frame coords.
            label: Label(s) for each point. 1=positive, 0=negative.

        Returns:
            Refined binary mask array (height, width) with dtype uint8.
        """
        session = self.sessions[video_id]
        masks_storage = session["masks"]
        logits_storage = session["logits"]
        frame_dims = session["frame_dims"]
        height, width = frame_dims
        output_dict = session["output_dict"]

        # Load previous logits
        prev_logits = logits_storage[frame_idx]
        if prev_logits is None:
            raise RuntimeError(f"No existing mask/logits for frame {frame_idx}. Use add_point_prompt first.")

        prev_logits_tensor = torch.from_numpy(prev_logits).float()
        prev_logits_tensor = prev_logits_tensor.unsqueeze(0).unsqueeze(0).to(self.device)

        # Normalize to lists
        if isinstance(location, tuple) and len(location) == 2 and not isinstance(location[0], tuple):
            locations = [location]
            labels = [label]
        else:
            locations = list(location)
            labels = list(label) if isinstance(label, (list, tuple)) else [label]

        # Scale points
        scaled_points = []
        for loc in locations:
            nx, ny = self._scale_point(loc[0], loc[1], width, height)
            scaled_points.append([nx, ny])

        point_coords = torch.tensor([scaled_points], dtype=torch.float32, device=self.device)
        point_labels = torch.tensor([labels], dtype=torch.int32, device=self.device)
        point_coords = point_coords * self.INPUT_SIZE

        point_inputs = {
            "point_coords": point_coords,
            "point_labels": point_labels,
        }

        # Get image features
        image, backbone_out = self._get_image_features(video_id, frame_idx)
        current_vision_feats, current_vision_pos_embeds, feat_sizes = self._prepare_backbone_features(backbone_out)

        # Run tracking with point + previous mask logits
        with torch.inference_mode():
            current_out = self.predictor.track_step(
                frame_idx=frame_idx,
                is_init_cond_frame=False,  # Not init since we have context
                current_vision_feats=current_vision_feats,
                current_vision_pos_embeds=current_vision_pos_embeds,
                feat_sizes=feat_sizes,
                point_inputs=point_inputs,
                mask_inputs=None,
                output_dict=output_dict,
                num_frames=session["num_frames"],
                track_in_reverse=False,
                run_mem_encoder=True,
                prev_sam_mask_logits=prev_logits_tensor,
            )

        # Extract and save refined mask
        mask_logits = current_out["pred_masks_high_res"][0, 0].float().cpu().numpy()
        mask_binary = (mask_logits > 0).astype(np.uint8) * 255
        mask_resized = cv2.resize(mask_binary, (width, height), interpolation=cv2.INTER_NEAREST)
        masks_storage[frame_idx] = mask_resized

        # Update logits
        logits_lowres = current_out["pred_masks"][0, 0].float().cpu().numpy()
        logits_storage[frame_idx] = logits_lowres

        # Update conditioning frame output
        output_dict["cond_frame_outputs"][frame_idx] = self._make_compact_output(current_out)

        return mask_resized
```

**Step 2: Verify syntax**

Run: `python -c "from vidseq.services.sam2.streaming_segmentor import SAM2StreamingSegmentor; print('OK')"`

**Step 3: Commit**

```bash
git add vidseq/services/sam2/streaming_segmentor.py
git commit -m "feat: add refine_mask to SAM2StreamingSegmentor"
```

---

## Task 5: Implement propagate_sequential

**Files:**
- Modify: `vidseq/services/sam2/streaming_segmentor.py`

**Step 1: Add propagate_sequential method**

Add after `refine_mask`:

```python
    def propagate_sequential(
        self,
        video_id: str,
        start_frame: int,
        num_frames: int,
        progress_interval: int = 10,
    ) -> list[int]:
        """Propagate tracking forward from start_frame.

        Args:
            video_id: The video identifier.
            start_frame: Frame index to start propagation from.
            num_frames: Maximum number of frames to propagate.
            progress_interval: Print progress every N frames (0 to disable).

        Returns:
            List of frame indices that were propagated.
        """
        import sys

        session = self.sessions[video_id]
        frames_source = session["frames"]
        masks_storage = session["masks"]
        cond_indices = session["cond_frame_indices"]
        frame_dims = session["frame_dims"]
        height, width = frame_dims
        output_dict = session["output_dict"]

        # Check that we have memory
        if not output_dict["cond_frame_outputs"] and not output_dict["non_cond_frame_outputs"]:
            raise RuntimeError("Cannot propagate without memory. Add at least one point prompt first.")

        propagated = []

        for i in range(num_frames):
            frame_idx = start_frame + i

            # Skip conditioning frames
            if frame_idx in cond_indices:
                continue

            # Try to read frame
            try:
                frame_bgr = frames_source[frame_idx]
            except (IndexError, KeyError):
                print(f"    End of video at frame {frame_idx}")
                sys.stdout.flush()
                break

            # Get image features
            image, backbone_out = self._get_image_features(video_id, frame_idx)
            current_vision_feats, current_vision_pos_embeds, feat_sizes = self._prepare_backbone_features(backbone_out)

            # Run tracking (no prompt, just memory)
            with torch.inference_mode():
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
                    track_in_reverse=False,
                    run_mem_encoder=True,
                )

            # Save mask
            mask_logits = current_out["pred_masks_high_res"][0, 0].float().cpu().numpy()
            mask_binary = (mask_logits > 0).astype(np.uint8) * 255
            mask_resized = cv2.resize(mask_binary, (width, height), interpolation=cv2.INTER_NEAREST)
            masks_storage[frame_idx] = mask_resized

            # Debug: log object score and mask stats
            obj_score = current_out["object_score_logits"][0, 0].item()
            mask_pixels = int(mask_resized.sum() // 255)
            print(f"  Frame {frame_idx}: obj_score={obj_score:.3f}, mask_pixels={mask_pixels}")

            # Store as non-conditioning frame
            output_dict["non_cond_frame_outputs"][frame_idx] = self._make_compact_output(current_out)

            # Evict old non-cond frames to prevent unbounded memory growth
            # Keep frames within a window of the current frame
            MEM_WINDOW = 7
            cutoff = frame_idx - MEM_WINDOW
            to_remove = [k for k in output_dict["non_cond_frame_outputs"] if k <= cutoff]
            for k in to_remove:
                del output_dict["non_cond_frame_outputs"][k]

            propagated.append(frame_idx)

            # Progress reporting
            if progress_interval > 0 and (i + 1) % progress_interval == 0:
                print(f"    Propagated {i + 1}/{num_frames} frames...")
                sys.stdout.flush()

        if progress_interval > 0 and len(propagated) > 0:
            print(f"    Done. Propagated {len(propagated)} frames.")
            sys.stdout.flush()

        return propagated
```

**Step 2: Verify syntax**

Run: `python -c "from vidseq.services.sam2.streaming_segmentor import SAM2StreamingSegmentor; print('OK')"`

**Step 3: Commit**

```bash
git add vidseq/services/sam2/streaming_segmentor.py
git commit -m "feat: add propagate_sequential to SAM2StreamingSegmentor"
```

---

## Task 6: Add reset_frame Method

**Files:**
- Modify: `vidseq/services/sam2/streaming_segmentor.py`

**Step 1: Add reset_frame method**

Add after `propagate_sequential`:

```python
    def reset_frame(
        self,
        video_id: str,
        frame_idx: int,
    ) -> None:
        """Reset a frame: clear its mask and remove from memory.

        Args:
            video_id: The video identifier.
            frame_idx: Frame index to reset.
        """
        session = self.sessions[video_id]
        masks_storage = session["masks"]
        logits_storage = session["logits"]
        output_dict = session["output_dict"]
        frame_dims = session["frame_dims"]
        height, width = frame_dims

        # Clear mask
        masks_storage[frame_idx] = np.zeros((height, width), dtype=np.uint8)

        # Clear logits
        try:
            logits_storage[frame_idx] = np.zeros((256, 256), dtype=np.float32)
        except:
            pass  # Logits storage might not support assignment

        # Remove from conditioning frames
        session["cond_frame_indices"].discard(frame_idx)

        # Remove from output dicts
        output_dict["cond_frame_outputs"].pop(frame_idx, None)
        output_dict["non_cond_frame_outputs"].pop(frame_idx, None)

        # Clear feature cache for this frame
        session["cached_features"].pop(frame_idx, None)

    def reset_video(self, video_id: str) -> None:
        """Reset entire video: clear all masks and memory.

        Args:
            video_id: The video identifier.
        """
        session = self.sessions[video_id]

        # Clear all memory
        session["output_dict"]["cond_frame_outputs"].clear()
        session["output_dict"]["non_cond_frame_outputs"].clear()
        session["cond_frame_indices"].clear()
        session["cached_features"].clear()
```

**Step 2: Verify syntax**

Run: `python -c "from vidseq.services.sam2.streaming_segmentor import SAM2StreamingSegmentor; print('OK')"`

**Step 3: Commit**

```bash
git add vidseq/services/sam2/streaming_segmentor.py
git commit -m "feat: add reset_frame/reset_video to SAM2StreamingSegmentor"
```

---

## Task 7: Update TCP Worker to Support SAM2 Backend

**Files:**
- Modify: `vidseq/services/sam3/server/commands.py`

**Step 1: Add backend selection logic**

Near the top of the file, after imports, add a backend selection mechanism:

```python
# Backend selection - can be switched via environment variable
import os
SAM_BACKEND = os.environ.get("SAM_BACKEND", "sam3").lower()

if SAM_BACKEND == "sam2":
    from vidseq.services.sam2.streaming_segmentor import SAM2StreamingSegmentor as StreamingSegmentor
else:
    from vidseq.services.sam3.streaming_segmentor import StreamingSegmentor

print(f"[SAM Worker] Using backend: {SAM_BACKEND}")
```

**Step 2: Verify the worker can import both backends**

Run: `SAM_BACKEND=sam2 python -c "from vidseq.services.sam3.server.commands import SAM_BACKEND; print(f'Backend: {SAM_BACKEND}')"`

**Step 3: Commit**

```bash
git add vidseq/services/sam3/server/commands.py
git commit -m "feat: add SAM2 backend selection to TCP worker"
```

---

## Task 8: Integration Test

**Files:**
- None (manual testing)

**Step 1: Start the SAM2 worker**

```bash
cd /n/groups/datta/john/projects/vidseq
SAM_BACKEND=sam2 python -m vidseq.services.sam3.server.worker
```

**Step 2: Test via the frontend**

1. Open the vidseq frontend
2. Create/open a project with a video
3. Add a positive point prompt on frame 0
4. Propagate forward
5. Compare the results to SAM3 (flickering, mask stability, etc.)

**Step 3: Document findings**

Note any differences in:
- Tracking stability (flickering)
- Mask quality
- Speed
- Memory usage

---

## Verification Checklist

After all tasks are complete, verify:

- [ ] SAM2StreamingSegmentor loads without errors
- [ ] `open_video` creates a valid session
- [ ] `add_point_prompt` generates a mask
- [ ] `refine_mask` refines the mask with additional points
- [ ] `propagate_sequential` tracks through the video
- [ ] `reset_frame` and `reset_video` clear state properly
- [ ] TCP worker can switch between SAM2 and SAM3 backends
- [ ] Frontend works with SAM2 backend
