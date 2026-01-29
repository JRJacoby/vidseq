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

import dataclasses
import sys

sys.path.insert(0, "/n/groups/datta/john/repos/sam3")

import cv2
import numpy as np
import torch


# =============================================================================
# SAM3 inference primitives (from custom_sam3.py)
# =============================================================================


def preprocess_image(
    image: np.ndarray,
    *,
    bgr: bool = False,
) -> torch.Tensor:
    """Prepare a raw image for the SAM3 image encoder.

    Steps:
        1. Convert BGR → RGB if needed.
        2. Resize to 1008×1008 (the model's expected input size).
        3. Normalize pixel values from [0, 255] → [0.0, 1.0].
        4. Standardize with mean=0.5, std=0.5 per channel.

    Args:
        image: HWC uint8 numpy array (RGB or BGR).
        bgr: If True, the input is in BGR channel order and will be
             flipped to RGB.

    Returns:
        Float32 tensor of shape (3, 1008, 1008), standardized.
    """
    if bgr:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    resized = cv2.resize(image, (1008, 1008), interpolation=cv2.INTER_LINEAR)

    tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0

    mean = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32).view(3, 1, 1)
    tensor = (tensor - mean) / std

    return tensor


def encode(
    backbone: torch.nn.Module,
    image: torch.Tensor,
    captions: list[str],
) -> dict:
    """Run the SAM3 VL backbone: image encoder + FPN neck + text encoder.

    Args:
        backbone: SAM3VLBackbone instance.
        image: Preprocessed float32 tensor, either (3, H, W) for a single
               image or (B, 3, H, W) for a batch.
        captions: List of text prompts (e.g. ["a dog"]).

    Returns:
        Dict with keys: backbone_fpn, vision_pos_enc, vision_features,
        language_features, language_mask, sam2_backbone_out.
    """
    if image.ndim == 3:
        image = image.unsqueeze(0)

    with torch.inference_mode():
        output = backbone.forward_image(image)
        device = output["vision_features"].device
        output.update(backbone.forward_text(captions, device=device))

    return output


@dataclasses.dataclass
class TrackResult:
    """Output of one frame through the SAM3 tracker."""

    masks: torch.Tensor          # (B, 1, H, W) low-res mask logits
    masks_high_res: torch.Tensor # (B, 1, H_hi, W_hi) high-res mask logits
    obj_ptr: torch.Tensor        # (B, C) object pointer for identity tracking
    score: torch.Tensor          # (B, 1) object-presence score logits
    frame_output: dict           # full current_out — store in memory dict


def track(
    tracker: torch.nn.Module,
    backbone_out: dict,
    image: torch.Tensor,
    frame_idx: int,
    memory: dict,
    *,
    is_first_frame: bool = False,
    point_coords: torch.Tensor | None = None,
    point_labels: torch.Tensor | None = None,
    mask_input: torch.Tensor | None = None,
    prev_mask_logits: torch.Tensor | None = None,
    run_mem_encoder: bool = True,
    use_prev_mem_frame: bool = True,
    num_frames: int = 1000,
) -> TrackResult:
    """Run one video frame through the SAM3 tracker with memory attention.

    Args:
        tracker: Sam3TrackerBase (or subclass) instance.
        backbone_out: Dict returned by encode(). The sam2_backbone_out key
                      is extracted automatically.
        image: Preprocessed image tensor — (3, H, W) or (B, 3, H, W).
        frame_idx: Sequential frame number (0, 1, 2, …).
        memory: Mutable dict with keys "cond_frame_outputs" and
                "non_cond_frame_outputs".
        is_first_frame: If True, memory attention is skipped and the frame
                        is stored as a conditioning frame.
        point_coords: Optional point prompts, shape (B, P, 2).
        point_labels: Optional point labels, shape (B, P). 1=fg, 0=bg.
        mask_input: Optional mask prompt, shape (B, 1, H, W). Used when you
                    want to use the mask directly as output.
        prev_mask_logits: Optional previous mask logits for iterative refinement,
                          shape (B, 1, H, W). Used WITH points to refine a mask.
        run_mem_encoder: Whether to encode this frame into memory. Set to False
                         during interactive refinement (SAM3's approach).
        use_prev_mem_frame: Whether to attend to previous memory frames. Set to
                            False during refinement (SAM3's approach) to rely only
                            on point prompts + prev_mask_logits.
        num_frames: Total number of frames in the video.

    Returns:
        TrackResult with masks, object pointer, score, and frame_output.
    """
    # Use sam2_backbone_out (separate FPN weights for tracking)
    sam2_out = backbone_out["sam2_backbone_out"]

    # Clone FPN features to avoid mutating the original
    fpn = [x.clone() for x in sam2_out["backbone_fpn"]]
    pos = [x.clone() for x in sam2_out["vision_pos_enc"]]

    # Apply conv_s0/conv_s1 projections for mask decoder
    fpn[0] = tracker.sam_mask_decoder.conv_s0(fpn[0])
    fpn[1] = tracker.sam_mask_decoder.conv_s1(fpn[1])

    # Reshape from (B, C, H, W) to (HW, B, C) for track_step
    feat_sizes = [(x.shape[-2], x.shape[-1]) for x in pos]
    vision_feats = [x.flatten(2).permute(2, 0, 1) for x in fpn]
    vision_pos_embeds = [x.flatten(2).permute(2, 0, 1) for x in pos]

    if image.ndim == 3:
        image = image.unsqueeze(0)

    point_inputs = None
    if point_coords is not None:
        point_inputs = {
            "point_coords": point_coords,
            "point_labels": point_labels,
        }

    with torch.inference_mode():
        current_out = tracker.track_step(
            frame_idx=frame_idx,
            is_init_cond_frame=is_first_frame,
            current_vision_feats=vision_feats,
            current_vision_pos_embeds=vision_pos_embeds,
            feat_sizes=feat_sizes,
            image=image,
            point_inputs=point_inputs,
            mask_inputs=mask_input,
            prev_sam_mask_logits=prev_mask_logits,
            output_dict=memory,
            num_frames=num_frames,
            run_mem_encoder=run_mem_encoder,
            use_prev_mem_frame=use_prev_mem_frame,
        )

    return TrackResult(
        masks=current_out["pred_masks"],
        masks_high_res=current_out["pred_masks_high_res"],
        obj_ptr=current_out["obj_ptr"],
        score=current_out["object_score_logits"],
        frame_output=current_out,
    )


# =============================================================================
# StreamingSegmentor class
# =============================================================================


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

    # Low-res logits size (SAM3 outputs 288x288, i.e. 1008/3.5)
    LOGITS_SIZE = 288

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

    def _compute_bbox_iou(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Compute IoU between bounding boxes of two binary masks."""
        def get_bbox(mask):
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            if not rows.any() or not cols.any():
                return None
            y_indices = np.where(rows)[0]
            x_indices = np.where(cols)[0]
            y1, y2 = y_indices[0], y_indices[-1] + 1
            x1, x2 = x_indices[0], x_indices[-1] + 1
            return (x1, y1, x2, y2)

        box1 = get_bbox(mask1)
        box2 = get_bbox(mask2)

        if box1 is None or box2 is None:
            return 0.0

        # Intersection
        xi1 = max(box1[0], box2[0])
        yi1 = max(box1[1], box2[1])
        xi2 = min(box1[2], box2[2])
        yi2 = min(box1[3], box2[3])
        inter_width = max(0, xi2 - xi1)
        inter_height = max(0, yi2 - yi1)
        inter_area = inter_width * inter_height

        # Union
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = area1 + area2 - inter_area

        if union_area <= 0:
            return 0.0

        return inter_area / union_area

    def _add_mask_prompt(
        self,
        video_id: str,
        frame_idx: int,
        mask: np.ndarray,
        memory: dict,
    ) -> tuple[np.ndarray, dict]:
        """
        Use a mask as dense prompt for SAM2, letting it refine based on visual features.

        Args:
            video_id: Video session ID
            frame_idx: Frame index
            mask: Binary mask (H, W) at original resolution
            memory: Current memory dict

        Returns:
            Tuple of (refined_mask at original resolution, frame_output for memory)
        """
        session = self.sessions[video_id]
        frames = session["frames"]
        frame_dims = session["frame_dims"]
        orig_h, orig_w = frame_dims

        # Read and preprocess frame
        frame = self._read_frame(frames, frame_idx)
        img_tensor = preprocess_image(frame, bgr=True).to(self.device)

        # Encode frame through backbone
        with torch.no_grad():
            backbone_out = encode(self.backbone, img_tensor, captions=["object"])

        # Resize mask to model input size
        mask_resized = cv2.resize(
            mask.astype(np.float32),
            (self.INPUT_SIZE, self.INPUT_SIZE),
            interpolation=cv2.INTER_NEAREST
        )
        mask_tensor = torch.from_numpy(mask_resized > 0).float()
        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0).to(self.device)  # (1, 1, H, W)

        # Run tracking with mask as input prompt
        result = track(
            tracker=self.tracker,
            backbone_out=backbone_out,
            image=img_tensor,
            frame_idx=frame_idx,
            memory=memory,
            mask_input=mask_tensor,
            is_first_frame=(frame_idx == 0),
            run_mem_encoder=True,
        )

        # Extract and resize mask back to original resolution
        pred_mask = result.masks  # Use .masks attribute
        pred_mask_np = pred_mask.squeeze().cpu().numpy()

        refined_mask = cv2.resize(
            (pred_mask_np > 0).astype(np.uint8) * 255,
            (orig_w, orig_h),
            interpolation=cv2.INTER_NEAREST
        )

        return refined_mask, result.frame_output

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
        masks,   # Indexable/assignable for binary mask storage
        logits,  # Indexable/assignable for logits storage (256x256 float32)
        frame_dims: tuple[int, int],  # (height, width)
        cond_frame_indices: set[int] | list[int] | None = None,
        detector_masks=None,  # h5py dataset or None
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
            logits: Indexable storage for low-res logits (256x256 float32).
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

        # Create session
        session = {
            "frames": frames,
            "masks": masks,
            "logits": logits,
            "cond_frame_indices": cond_frame_indices,
            "cond_frame_memories": {},
            "frame_dims": frame_dims,
            "detector_masks": detector_masks,
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

    def _clear_non_cond_mem_around_frame(
        self,
        memory: dict,
        frame_idx: int,
        cond_indices: set[int],
    ) -> None:
        """Clear non-conditioning memories around a frame being refined.

        When users provide correction clicks, surrounding frames' non-conditioning
        memories can contain outdated object appearance information that could
        confuse the model. This matches SAM3's _clear_non_cond_mem_around_input.

        Args:
            memory: Memory dict with cond_frame_outputs and non_cond_frame_outputs.
            frame_idx: The frame being refined.
            cond_indices: Set of conditioning frame indices (these are NOT cleared).
        """
        # Clear non-cond frames within MEM_WINDOW of the target frame
        # This prevents stale appearance info from confusing refinement
        window = self.MEM_WINDOW
        frame_idx_begin = frame_idx - window
        frame_idx_end = frame_idx + window

        non_cond = memory.get("non_cond_frame_outputs", {})
        to_remove = [
            t for t in non_cond
            if frame_idx_begin <= t <= frame_idx_end and t not in cond_indices
        ]
        for t in to_remove:
            del non_cond[t]

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
        """Add point prompt(s) to a BLANK frame and generate initial mask.

        Use this for frames that don't have an existing mask. For refining
        an existing mask, use refine_mask() instead.

        This method:
        1. Prepares memory context (cond frames + rolling window of non-cond)
        2. Preprocesses and encodes the target frame
        3. Runs the tracker with the point prompt(s) (attending to memory)
        4. Stores the result as a conditioning frame (in memory and in storage)
        5. Writes both the binary mask and low-res logits to storage

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
            - Updates logits[frame_idx] with low-res logits (for future refinement)
            - Adds frame_idx to cond_frame_indices
            - Stores frame_output in cond_frame_memories

        Raises:
            KeyError: If video_id is not found (call open_video first).
            ValueError: If location and label list lengths don't match.
        """
        session = self.sessions[video_id]
        masks_storage = session["masks"]
        logits_storage = session["logits"]
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

        # Run tracking with point prompt (no mask_input for blank frames)
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

        # Extract and save binary mask (high-res logits -> binary -> resize)
        mask_logits_highres = result.masks_high_res[0, 0].float().cpu().numpy()
        mask_binary = (mask_logits_highres > 0).astype(np.uint8) * 255
        mask_resized = cv2.resize(mask_binary, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        masks_storage[frame_idx] = mask_resized

        # Save low-res logits for future refinement (256x256)
        logits_lowres = result.masks[0, 0].float().cpu().numpy()
        logits_storage[frame_idx] = logits_lowres

        # Update session: add to cond_frame_indices and store memory
        session["cond_frame_indices"].add(frame_idx)
        session["cond_frame_memories"][frame_idx] = result.frame_output

    def refine_mask(
        self,
        video_id: str,
        frame_idx: int,
        location: tuple[float, float] | list[tuple[float, float]],
        label: int | list[int],
    ) -> None:
        """Refine an existing mask with point prompt(s).

        Use this for frames that already have a mask. The previous mask's logits
        are passed as a dense prompt along with the point(s), allowing SAM3
        to refine the segmentation.

        This follows SAM3's iterative refinement pattern where points
        build upon the previous mask state.

        Args:
            video_id: The video identifier (must be opened with open_video first).
            frame_idx: Index of the frame to refine. Must have an existing mask.
            location: Either a single (x, y) point or a list of (x, y) points
                      in original frame space. Will be scaled to 1008x1008 internally.
            label: Either a single label or a list of labels (one per point).
                   1 = positive (foreground), 0 = negative (background).

        Side Effects:
            - Updates masks[frame_idx] with the refined binary mask
            - Updates logits[frame_idx] with new low-res logits
            - Updates cond_frame_memories[frame_idx] with new frame_output

        Raises:
            KeyError: If video_id is not found.
            RuntimeError: If frame_idx doesn't have an existing mask/logits.
            ValueError: If location and label list lengths don't match.
        """
        session = self.sessions[video_id]
        masks_storage = session["masks"]
        logits_storage = session["logits"]
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

        # Verify this frame has existing logits to refine
        prev_logits = logits_storage[frame_idx]
        if not np.any(prev_logits != 0):
            raise RuntimeError(
                f"Frame {frame_idx} has no existing logits. Use add_point_prompt for blank frames."
            )

        # Prepare memory context and get target frame
        frame, memory = self._prepare_memory(video_id, frame_idx)

        # Clear non-conditioning memories around the refined frame
        # This prevents stale appearance info from confusing the refinement
        # (matches SAM3's _clear_non_cond_mem_around_input behavior)
        cond_indices = session["cond_frame_indices"]
        self._clear_non_cond_mem_around_frame(memory, frame_idx, cond_indices)

        # Preprocess: BGR -> RGB, resize to 1008x1008, normalize
        img_tensor = preprocess_image(frame, bgr=True).to(self.device)

        # Encode through backbone
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

        # Prepare previous logits for refinement
        # Logits are 288x288, convert to tensor (1, 1, 288, 288)
        prev_mask_logits = torch.from_numpy(prev_logits).float()
        prev_mask_logits = prev_mask_logits.unsqueeze(0).unsqueeze(0).to(self.device)
        # Clamp to avoid numerical issues (following SAM3's approach)
        prev_mask_logits = torch.clamp(prev_mask_logits, -32.0, 32.0)

        # Run tracking with point prompt AND previous mask logits for refinement
        # Note: prev_mask_logits is different from mask_input - it's used for
        # iterative refinement WITH points, not for using a mask directly as output
        # Key SAM3 settings for refinement:
        # - run_mem_encoder=False: defer memory encoding
        # - use_prev_mem_frame=False: don't attend to previous memory, rely only on
        #   point prompts + prev_mask_logits for refinement
        result = track(
            tracker=self.tracker,
            backbone_out=backbone_out,
            image=img_tensor,
            frame_idx=frame_idx,
            memory=memory,
            is_first_frame=False,
            point_coords=point_coords,
            point_labels=point_labels,
            prev_mask_logits=prev_mask_logits,
            run_mem_encoder=False,
            use_prev_mem_frame=False,  # Don't attend to memory during refinement
        )

        # Extract and save refined binary mask
        mask_logits_highres = result.masks_high_res[0, 0].float().cpu().numpy()
        mask_binary = (mask_logits_highres > 0).astype(np.uint8) * 255
        mask_resized = cv2.resize(mask_binary, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        masks_storage[frame_idx] = mask_resized

        # Save updated low-res logits for future refinement
        logits_lowres = result.masks[0, 0].float().cpu().numpy()
        logits_storage[frame_idx] = logits_lowres

        # Re-encode the stored mask to get proper memory features
        # (Since we used run_mem_encoder=False, result.frame_output is incomplete)
        frames_source = session["frames"]
        frame_output = self._encode_stored_mask(frames_source, masks_storage, frame_idx, frame_dims)
        session["cond_frame_memories"][frame_idx] = frame_output

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

            # Debug: log object score and mask stats
            obj_score = result.score[0, 0].item()
            mask_sum = int(mask_resized.sum() // 255)  # count of non-zero pixels
            print(f"  Frame {frame_idx}: obj_score={obj_score:.3f}, mask_pixels={mask_sum}")

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
