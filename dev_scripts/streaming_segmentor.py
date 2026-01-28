"""Stateful streaming segmentation manager using SAM3 tracker.

================================================================================
PURPOSE
================================================================================

This module provides a stateful class for managing video segmentation sessions.
It wraps the composable functions in custom_sam3.py into a convenient interface
for interactive annotation workflows where:

1. Users click on frames to provide point prompts
2. The tracker maintains memory across frames
3. Masks are written to a provided storage structure (array or h5 dataset)

================================================================================
DESIGN DECISIONS
================================================================================

1. Multi-video support:
   Each video is identified by a video_id and maintains its own memory bank.
   This allows one StreamingSegmentor instance to handle multiple videos
   without interference.

2. Session structure:
   sessions[video_id] = {
       "frames": <indexable>,             # frame source (passed by caller)
       "masks": <array-or-h5>,            # mask storage (passed by caller)
       "cond_frame_indices": set[int],    # frames with user prompts
       "cond_frame_memories": dict,       # {frame_idx: frame_output} - kept in memory
       "frame_dims": (height, width),     # original video dimensions
   }

3. I/O managed by caller:
   The caller provides frame and mask sources via open_video(). This allows
   the caller (e.g., TCP worker) to manage file handles and caching externally.
   StreamingSegmentor is purely a stateful inference wrapper.

4. Memory reconstruction:
   - Conditioning frame memories are stored in-memory (reconstructed on open_video)
   - Non-conditioning memories are reconstructed on-demand from stored masks
   - Memory encoding is path-independent: uses raw backbone features + mask

5. Conditioning frame storage:
   User prompts are always stored as conditioning frames (never evicted).
   This ensures that explicit user annotations remain authoritative anchors
   for the tracker's memory attention.

6. No duplication in memory:
   Each frame is in exactly ONE dict (cond_frame_outputs OR non_cond_frame_outputs).
   When building the rolling window, conditioning frames are skipped since they
   already provide context from cond_frame_outputs. This matches SAM3 behavior.

================================================================================
USAGE
================================================================================

    # Initialize once (loads SAM3 model)
    segmentor = StreamingSegmentor()

    # Open a video session (caller provides frame/mask sources)
    segmentor.open_video(
        video_id="video_001",
        frames=my_frame_source,  # indexable returning BGR uint8 (H, W, 3)
        masks=my_mask_storage,   # indexable/assignable for mask storage
        frame_dims=(1080, 1920),  # (height, width)
        cond_frame_indices={0, 50},  # frames that already have prompts
    )

    # Add a point prompt (frame is read from frames source)
    segmentor.add_point_prompt(
        video_id="video_001",
        frame_idx=150,
        location=(320, 240),  # original video coordinates
        label=1,              # 1=positive, 0=negative
    )

    # Propagate tracking without a prompt
    segmentor.propagate(video_id="video_001", frame_idx=151)

    # Close when done (caller is responsible for closing frame/mask sources)
    segmentor.close_video("video_001")

================================================================================
FUTURE EXTENSIONS
================================================================================

- add_box_prompt(): Box prompt support
- add_mask_prompt(): Mask refinement support
- propagate_backward(): Bidirectional correction (re-run with future memory)
- Custom detector integration

"""

from __future__ import annotations

import sys
sys.path.insert(0, "/n/groups/datta/john/repos/sam3")

import cv2
import numpy as np
import torch

from custom_sam3 import preprocess_image, encode, track


class StreamingSegmentor:
    """Stateful manager for SAM3 video segmentation sessions.

    Handles multiple videos, each with its own memory bank. Provides methods
    for adding prompts, propagating tracking, and managing video sessions.

    Attributes:
        device: Torch device for model inference.
        backbone: SAM3 VL backbone (shared across all sessions).
        tracker: SAM3 tracker (shared across all sessions).
        sessions: Dict mapping video_id to session state.
    """

    # Model input size (SAM3 expects 1008x1008)
    INPUT_SIZE = 1008

    # Number of non-conditioning frames to include in rolling memory window
    MEM_WINDOW = 7

    def __init__(self, device: str | None = None):
        """Initialize the segmentor and load the SAM3 model.

        Args:
            device: Torch device string ("cuda", "cpu", etc.).
                    Defaults to "cuda" if available, else "cpu".
        """
        from sam3.model_builder import build_sam3_video_model

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading SAM3 model on {self.device}...")
        model = build_sam3_video_model(
            apply_temporal_disambiguation=False,
            device=self.device,
        )
        model.eval()

        # Extract the components we need
        # backbone: SAM3VLBackbone - handles image encoding + text encoding
        # tracker: Sam3TrackerPredictor - handles memory attention + mask prediction
        self.backbone = model.detector.backbone
        self.tracker = model.tracker

        # Sessions dict: video_id -> session state
        self.sessions: dict[str, dict] = {}

        print("SAM3 model loaded.")

    def _scale_point(self, x: float, y: float, orig_w: int, orig_h: int) -> tuple[float, float]:
        """Scale point from original frame coords to model input coords.

        SAM3 expects point coordinates in the 1008x1008 input space.

        Args:
            x, y: Point coordinates in original frame space.
            orig_w, orig_h: Original frame dimensions.

        Returns:
            (scaled_x, scaled_y) in 1008x1008 space.
        """
        scaled_x = x * self.INPUT_SIZE / orig_w
        scaled_y = y * self.INPUT_SIZE / orig_h
        return scaled_x, scaled_y

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

    def _read_frame(self, frames, frame_idx: int) -> np.ndarray:
        """Read a specific frame from the frame source.

        Args:
            frames: Indexable frame source returning BGR uint8 arrays.
            frame_idx: Frame index to read.

        Returns:
            BGR frame as numpy array (HWC uint8).
        """
        return frames[frame_idx]

    def _encode_stored_mask(
        self,
        frames,
        masks,
        frame_idx: int,
        frame_dims: tuple[int, int],
    ) -> dict:
        """Encode a stored mask into memory format.

        Reads the frame and mask, then uses track() with mask_input to encode
        the mask into memory features (including obj_ptr from SAM decoder).

        Args:
            frames: Indexable frame source returning BGR uint8 arrays.
            masks: Indexable mask storage.
            frame_idx: Frame index to encode.
            frame_dims: (height, width) of the original video.

        Returns:
            frame_output dict suitable for storing in memory.
        """
        orig_h, orig_w = frame_dims

        # Read frame and encode
        frame = self._read_frame(frames, frame_idx)
        img_tensor = preprocess_image(frame, bgr=True).to(self.device)

        with torch.no_grad():
            backbone_out = encode(self.backbone, img_tensor, captions=["object"])

        # Load and prepare mask
        stored_mask = masks[frame_idx]
        # Resize mask to model input size and convert to tensor
        mask_resized = cv2.resize(
            stored_mask.astype(np.float32),
            (self.INPUT_SIZE, self.INPUT_SIZE),
            interpolation=cv2.INTER_NEAREST,
        )
        # Convert to binary (0/1) tensor: shape (1, 1, H, W)
        mask_tensor = torch.from_numpy((mask_resized > 127).astype(np.float32))
        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0).to(self.device)

        # Use track() with mask_input to encode the mask
        # This uses SAM3's _use_mask_as_output path which:
        # - Uses the mask directly as output (no prediction)
        # - Gets obj_ptr from SAM decoder with mask context
        # - Encodes into memory features
        empty_memory = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
        result = track(
            tracker=self.tracker,
            backbone_out=backbone_out,
            image=img_tensor,
            frame_idx=frame_idx,
            memory=empty_memory,
            is_first_frame=True,  # Don't attend to memory during encoding
            mask_input=mask_tensor,
        )

        return result.frame_output

    def open_video(
        self,
        video_id: str,
        frames,  # Indexable returning BGR uint8 (H, W, 3)
        masks,   # Indexable/assignable for mask storage
        frame_dims: tuple[int, int],  # (height, width)
        cond_frame_indices: set[int] | list[int] | None = None,
    ) -> None:
        """Create a session with external frame and mask sources.

        This initializes the session and reconstructs conditioning frame memories
        from stored masks. Call this before using add_point_prompt or propagate.

        The caller is responsible for managing the lifecycle of frames and masks
        (e.g., opening/closing file handles).

        Args:
            video_id: Unique identifier for this video session.
            frames: Indexable frame source returning BGR uint8 arrays (H, W, 3).
                    Must support frames[frame_idx] access.
            masks: Indexable storage for masks (array or h5 dataset).
                   Will be used for both reading existing masks and writing new ones.
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

        # Create session
        session = {
            "frames": frames,
            "masks": masks,
            "cond_frame_indices": cond_frame_indices,
            "cond_frame_memories": {},
            "frame_dims": frame_dims,
        }

        # Reconstruct conditioning frame memories from stored masks
        cond_to_reconstruct = [idx for idx in sorted(cond_frame_indices)
                               if self._mask_exists(masks, idx)]
        if cond_to_reconstruct:
            import sys
            print(f"  Reconstructing {len(cond_to_reconstruct)} cond frame memories...")
            sys.stdout.flush()
            for i, idx in enumerate(cond_to_reconstruct):
                print(f"    [{i+1}/{len(cond_to_reconstruct)}] Encoding frame {idx}...", end="")
                sys.stdout.flush()
                frame_output = self._encode_stored_mask(frames, masks, idx, frame_dims)
                session["cond_frame_memories"][idx] = frame_output
                print(" done")
                sys.stdout.flush()

        self.sessions[video_id] = session
        print(f"Opened video '{video_id}' with {len(session['cond_frame_memories'])} cond frames")

    def close_video(self, video_id: str) -> bool:
        """Close a video session.

        Note: The caller is responsible for closing the frames and masks sources
        that were passed to open_video(). This method only removes the session
        from StreamingSegmentor's internal tracking.

        Args:
            video_id: The video identifier to close.

        Returns:
            True if the session was closed, False if video_id not found.
        """
        if video_id not in self.sessions:
            return False

        del self.sessions[video_id]
        return True

    def _prepare_memory(
        self,
        video_id: str,
        frame_idx: int,
    ) -> tuple[np.ndarray, dict]:
        """Prepare memory context for tracking at a target frame.

        This method:
        1. Copies conditioning frame memories from the session (already in memory)
        2. Reconstructs up to MEM_WINDOW contiguous non-cond frames before target
        3. Reads and returns the target frame

        Args:
            video_id: The video identifier.
            frame_idx: Target frame index to prepare for.

        Returns:
            (frame_bgr, memory_dict) where:
            - frame_bgr is the target frame as BGR numpy array
            - memory_dict has cond_frame_outputs and non_cond_frame_outputs populated

        Raises:
            KeyError: If video_id is not found.
        """
        session = self.sessions[video_id]
        frames = session["frames"]
        masks = session["masks"]
        cond_indices = session["cond_frame_indices"]
        cond_memories = session["cond_frame_memories"]
        frame_dims = session["frame_dims"]

        # Initialize memory dict with copies of conditioning frame memories
        memory = {
            "cond_frame_outputs": dict(cond_memories),
            "non_cond_frame_outputs": {},
        }

        # Reconstruct up to MEM_WINDOW contiguous non-cond frames before target
        # Walk backward from frame_idx - 1
        # Stop at first gap (no mask) or after collecting MEM_WINDOW frames
        # Skip conditioning frames (they're already in cond_frame_outputs)

        # First, collect which frames we need to encode
        frames_to_encode = []
        for i in range(frame_idx - 1, -1, -1):
            if len(frames_to_encode) >= self.MEM_WINDOW:
                break
            if not self._mask_exists(masks, i):
                break  # Gap found - stop
            if i in cond_indices:
                continue  # Skip cond frames
            frames_to_encode.append(i)

        # Now encode them (silently - progress is shown at higher level)
        for i in frames_to_encode:
            frame_output = self._encode_stored_mask(frames, masks, i, frame_dims)
            memory["non_cond_frame_outputs"][i] = frame_output

        # Read the target frame
        target_frame = self._read_frame(frames, frame_idx)

        return target_frame, memory

    def add_point_prompt(
        self,
        video_id: str,
        frame_idx: int,
        location: tuple[float, float] | list[tuple[float, float]],
        label: int | list[int],
    ) -> None:
        """Add point prompt(s) and update the mask for a frame.

        This is the primary method for interactive annotation. It:
        1. Prepares memory context (cond frames + rolling window of non-cond)
        2. Preprocesses and encodes the target frame
        3. Runs the tracker with the point prompt(s) (attending to memory)
        4. Stores the result as a conditioning frame (in memory and in storage)
        5. Writes the binary mask to the masks storage

        Args:
            video_id: The video identifier (must be opened with open_video first).
            frame_idx: Index of the frame to annotate. The frame will be read
                       from the video internally.
            location: Either a single (x, y) point or a list of (x, y) points
                      in original frame space. Will be scaled to 1008x1008 internally.
            label: Either a single label or a list of labels (one per point).
                   1 = positive (foreground), 0 = negative (background).

        Side Effects:
            - Updates masks[frame_idx] with the binary mask (uint8, 0/255)
            - Adds frame_idx to cond_frame_indices
            - Stores frame_output in cond_frame_memories

        Raises:
            KeyError: If video_id is not found (call open_video first).
            ValueError: If location and label list lengths don't match.
        """
        session = self.sessions[video_id]
        masks_storage = session["masks"]
        frame_dims = session["frame_dims"]
        orig_h, orig_w = frame_dims

        # Normalize to lists for uniform handling
        if isinstance(location, tuple) and len(location) == 2 and not isinstance(location[0], tuple):
            # Single point: (x, y)
            locations = [location]
            labels = [label]
        else:
            # Multiple points: [(x1, y1), (x2, y2), ...]
            locations = list(location)
            labels = list(label) if isinstance(label, (list, tuple)) else [label]

        if len(locations) != len(labels):
            raise ValueError(f"Number of locations ({len(locations)}) must match number of labels ({len(labels)})")

        # Prepare memory context and get target frame
        frame, memory = self._prepare_memory(video_id, frame_idx)

        # Preprocess: BGR -> RGB, resize to 1008x1008, normalize
        img_tensor = preprocess_image(frame, bgr=True).to(self.device)

        # Encode through backbone (image features + text features)
        with torch.no_grad():
            backbone_out = encode(self.backbone, img_tensor, captions=["object"])

        # Scale all points from original coords to model input coords
        scaled_coords = []
        for loc in locations:
            scaled_x, scaled_y = self._scale_point(loc[0], loc[1], orig_w, orig_h)
            scaled_coords.append([scaled_x, scaled_y])

        # Prepare point prompt tensors
        point_coords = torch.tensor(
            [scaled_coords],  # shape: (batch=1, num_points, xy=2)
            dtype=torch.float32,
            device=self.device,
        )
        point_labels = torch.tensor(
            [labels],  # shape: (batch=1, num_points)
            dtype=torch.int32,
            device=self.device,
        )

        # Determine if this is a bootstrap frame (no existing memory)
        has_memory = bool(memory["cond_frame_outputs"]) or bool(memory["non_cond_frame_outputs"])
        is_first_frame = not has_memory

        # Run tracking with point prompt
        result = track(
            tracker=self.tracker,
            backbone_out=backbone_out,
            image=img_tensor,
            frame_idx=frame_idx,
            memory=memory,
            is_first_frame=is_first_frame,
            point_coords=point_coords,
            point_labels=point_labels,
        )

        # Extract mask: high-res logits -> binary -> resize to original resolution
        mask_logits = result.masks_high_res[0, 0].float().cpu().numpy()
        mask_binary = (mask_logits > 0).astype(np.uint8) * 255
        mask_resized = cv2.resize(mask_binary, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        # Write mask to storage
        masks_storage[frame_idx] = mask_resized

        # Update session: add to cond_frame_indices and store memory
        session["cond_frame_indices"].add(frame_idx)
        session["cond_frame_memories"][frame_idx] = result.frame_output

    def propagate(
        self,
        video_id: str,
        frame_idx: int,
    ) -> None:
        """Track object at a frame without a prompt (uses memory only).

        This propagates the tracking from existing memories to a new frame.
        The frame is NOT added as a conditioning frame.

        Args:
            video_id: The video identifier (must be opened with open_video first).
            frame_idx: Index of the frame to track.

        Side Effects:
            - Updates masks[frame_idx] with the binary mask (uint8, 0/255)

        Raises:
            KeyError: If video_id is not found.
            RuntimeError: If no memory exists (need at least one cond frame).
        """
        session = self.sessions[video_id]
        masks_storage = session["masks"]
        frame_dims = session["frame_dims"]
        orig_h, orig_w = frame_dims

        # Prepare memory context and get target frame
        frame, memory = self._prepare_memory(video_id, frame_idx)

        # Check that we have memory to attend to
        if not memory["cond_frame_outputs"] and not memory["non_cond_frame_outputs"]:
            raise RuntimeError(
                f"Cannot propagate without memory. Add at least one point prompt first."
            )

        # Preprocess and encode
        img_tensor = preprocess_image(frame, bgr=True).to(self.device)

        with torch.no_grad():
            backbone_out = encode(self.backbone, img_tensor, captions=["object"])

        # Run tracking WITHOUT prompt (memory attention only)
        result = track(
            tracker=self.tracker,
            backbone_out=backbone_out,
            image=img_tensor,
            frame_idx=frame_idx,
            memory=memory,
            is_first_frame=False,  # Must attend to memory
        )

        # Extract and save mask
        mask_logits = result.masks_high_res[0, 0].float().cpu().numpy()
        mask_binary = (mask_logits > 0).astype(np.uint8) * 255
        mask_resized = cv2.resize(mask_binary, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        masks_storage[frame_idx] = mask_resized

    def propagate_sequential(
        self,
        video_id: str,
        start_frame: int,
        num_frames: int,
        progress_interval: int = 50,
    ) -> list[int]:
        """Efficiently propagate tracking across sequential frames.

        Unlike calling propagate() repeatedly, this method:
        1. Calls _prepare_memory() only once for the start frame
        2. Maintains rolling memory window as it propagates (no re-encoding)
        3. Reads frames sequentially from the frame source
        4. SAM3's track_step automatically updates memory dict

        Args:
            video_id: The video identifier (must be opened with open_video first).
            start_frame: First frame index to propagate.
            num_frames: Number of frames to propagate.
            progress_interval: Print progress every N frames (0 to disable).

        Returns:
            List of frame indices that were actually propagated.
            (Conditioning frames in the range are skipped.)

        Side Effects:
            - Updates masks for propagated frames
            - Does NOT add frames to cond_frame_indices (these are propagated, not prompted)

        Raises:
            KeyError: If video_id is not found.
            RuntimeError: If no memory exists for start_frame.
        """
        import sys

        session = self.sessions[video_id]
        frames_source = session["frames"]
        masks_storage = session["masks"]
        cond_indices = session["cond_frame_indices"]
        frame_dims = session["frame_dims"]
        orig_h, orig_w = frame_dims

        # Prepare initial memory for start_frame
        frame, memory = self._prepare_memory(video_id, start_frame)

        # Check that we have memory to attend to
        if not memory["cond_frame_outputs"] and not memory["non_cond_frame_outputs"]:
            raise RuntimeError(
                "Cannot propagate without memory. Add at least one point prompt first."
            )

        propagated = []

        for i in range(num_frames):
            frame_idx = start_frame + i

            # Read frame from source
            if i > 0:
                try:
                    frame = frames_source[frame_idx]
                except (IndexError, KeyError):
                    print(f"    End of video at frame {frame_idx}")
                    sys.stdout.flush()
                    break

            # Skip conditioning frames (they have user prompts, don't overwrite)
            if frame_idx in cond_indices:
                # The cond frame's memory is already in cond_frame_outputs
                # We just don't overwrite its mask
                continue

            # Encode through backbone
            img_tensor = preprocess_image(frame, bgr=True).to(self.device)

            with torch.no_grad():
                backbone_out = encode(self.backbone, img_tensor, captions=["object"])

            # Run tracking (memory attention only, no prompt)
            # Note: track_step automatically adds result.frame_output to
            # memory["non_cond_frame_outputs"][frame_idx]
            result = track(
                tracker=self.tracker,
                backbone_out=backbone_out,
                image=img_tensor,
                frame_idx=frame_idx,
                memory=memory,
                is_first_frame=False,
            )

            # Save mask
            mask_logits = result.masks_high_res[0, 0].float().cpu().numpy()
            mask_binary = (mask_logits > 0).astype(np.uint8) * 255
            mask_resized = cv2.resize(mask_binary, (orig_w, orig_h),
                                      interpolation=cv2.INTER_NEAREST)
            masks_storage[frame_idx] = mask_resized

            # Evict old frames from non_cond (keep only last MEM_WINDOW)
            # This prevents memory from growing unbounded
            cutoff = frame_idx - self.MEM_WINDOW
            to_remove = [k for k in memory["non_cond_frame_outputs"] if k <= cutoff]
            for k in to_remove:
                del memory["non_cond_frame_outputs"][k]

            propagated.append(frame_idx)

            # Progress reporting
            if progress_interval > 0 and (i + 1) % progress_interval == 0:
                print(f"    Propagated {i + 1}/{num_frames} frames...")
                sys.stdout.flush()

        if progress_interval > 0 and len(propagated) > 0:
            print(f"    Done. Propagated {len(propagated)} frames "
                  f"(skipped {num_frames - len(propagated)} cond frames)")
            sys.stdout.flush()

        return propagated

    def get_session_info(self, video_id: str) -> dict | None:
        """Get information about a video session.

        Args:
            video_id: The video identifier.

        Returns:
            Dict with session info, or None if video_id not found.
            Keys: "num_cond_frames", "cond_frame_indices", "frame_dims"
        """
        if video_id not in self.sessions:
            return None

        session = self.sessions[video_id]
        return {
            "num_cond_frames": len(session["cond_frame_indices"]),
            "cond_frame_indices": sorted(session["cond_frame_indices"]),
            "frame_dims": session["frame_dims"],
        }

    def reset_frame(self, video_id: str, frame_idx: int) -> bool:
        """Remove a frame from memory banks.

        Removes the frame from both conditioning and non-conditioning memory
        dictionaries. Call this when a frame's mask is deleted.

        Args:
            video_id: The video identifier.
            frame_idx: Frame index to remove.

        Returns:
            True if the frame was found and removed, False otherwise.
        """
        if video_id not in self.sessions:
            return False

        session = self.sessions[video_id]
        removed = False

        # Remove from conditioning frames
        if frame_idx in session["cond_frame_indices"]:
            session["cond_frame_indices"].discard(frame_idx)
            removed = True
        if frame_idx in session["cond_frame_memories"]:
            del session["cond_frame_memories"][frame_idx]
            removed = True

        # Remove from non-conditioning rolling buffer (if present)
        # Note: non_cond_frame_memories is rebuilt on-demand in _prepare_memory,
        # but we should clear it if the frame happens to be cached
        if "non_cond_frame_memories" in session:
            if frame_idx in session["non_cond_frame_memories"]:
                del session["non_cond_frame_memories"][frame_idx]
                removed = True

        return removed
