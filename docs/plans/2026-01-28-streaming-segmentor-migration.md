# StreamingSegmentor Migration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the current SAM3 integration with StreamingSegmentor, removing bounding box support and making prompts ephemeral.

**Architecture:** Keep TCP worker pattern for CUDA isolation. Worker manages file handles (video, HDF5) and delegates segmentation to StreamingSegmentor. Frontend simplifies to points-only with ephemeral prompt state. Conditioning frames tracked in SQLite for memory reconstruction.

**Tech Stack:** Python (FastAPI, SQLAlchemy, h5py, OpenCV, PyTorch), TypeScript (Vue 3), SAM3

---

## Task 1: Add reset_frame() and Decouple I/O from StreamingSegmentor

**Files:**
- Modify: `dev_scripts/streaming_segmentor.py`

**Step 1: Add reset_frame method**

Add after `get_session_info()`:

```python
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
```

**Step 2: Update open_video signature to accept frame source**

Change the method signature and implementation:

```python
def open_video(
    self,
    video_id: str,
    frames,  # Indexable returning BGR uint8 (H, W, 3)
    masks,   # Indexable/assignable for mask storage
    frame_dims: tuple[int, int],  # (height, width)
    cond_frame_indices: set[int] | list[int] | None = None,
) -> None:
    """Open a video session with provided frame and mask sources.

    This initializes the session and reconstructs conditioning frame memories
    from stored masks. Call this before using add_point_prompt or propagate.

    Args:
        video_id: Unique identifier for this video session.
        frames: Indexable frame source. Must support __getitem__(idx) returning
                BGR uint8 numpy array of shape (H, W, 3).
        masks: Indexable storage for masks (array or h5 dataset).
               Will be used for both reading existing masks and writing new ones.
        frame_dims: (height, width) of the original video frames.
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
```

**Step 3: Update _read_frame to use frames source**

Replace the method:

```python
def _read_frame(self, frames, frame_idx: int) -> np.ndarray:
    """Read a specific frame from the frame source.

    Args:
        frames: Indexable frame source.
        frame_idx: Frame index to read.

    Returns:
        BGR frame as numpy array (HWC uint8).

    Raises:
        RuntimeError: If frame cannot be read.
    """
    try:
        frame = frames[frame_idx]
        return frame
    except Exception as e:
        raise RuntimeError(f"Failed to read frame {frame_idx}: {e}")
```

**Step 4: Update _encode_stored_mask signature**

```python
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
        frames: Indexable frame source.
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
    empty_memory = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
    result = track(
        tracker=self.tracker,
        backbone_out=backbone_out,
        image=img_tensor,
        frame_idx=frame_idx,
        memory=empty_memory,
        is_first_frame=True,
        mask_input=mask_tensor,
    )

    return result.frame_output
```

**Step 5: Update _prepare_memory to use session frames**

```python
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
    frames_to_encode = []
    for i in range(frame_idx - 1, -1, -1):
        if len(frames_to_encode) >= self.MEM_WINDOW:
            break
        if not self._mask_exists(masks, i):
            break
        if i in cond_indices:
            continue
        frames_to_encode.append(i)

    for i in frames_to_encode:
        frame_output = self._encode_stored_mask(frames, masks, i, frame_dims)
        memory["non_cond_frame_outputs"][i] = frame_output

    # Read the target frame
    target_frame = self._read_frame(frames, frame_idx)

    return target_frame, memory
```

**Step 6: Update add_point_prompt to use session frames**

Change references from `video_cap` to `frames` throughout:

```python
def add_point_prompt(
    self,
    video_id: str,
    frame_idx: int,
    location: tuple[float, float] | list[tuple[float, float]],
    label: int | list[int],
) -> None:
    # ... (docstring unchanged)
    session = self.sessions[video_id]
    masks_storage = session["masks"]
    frame_dims = session["frame_dims"]
    orig_h, orig_w = frame_dims

    # ... (normalize to lists - unchanged)

    # Prepare memory context and get target frame
    frame, memory = self._prepare_memory(video_id, frame_idx)

    # ... (rest unchanged - uses frame from _prepare_memory)
```

**Step 7: Update propagate to use session frames**

Similar changes - the `_prepare_memory` call already handles reading from the correct source.

**Step 8: Update propagate_sequential to use session frames**

Change:
```python
session = self.sessions[video_id]
frames = session["frames"]  # Changed from video_cap
masks_storage = session["masks"]
# ...

for i in range(num_frames):
    frame_idx = start_frame + i

    if i > 0:
        # Read next frame from source
        try:
            frame = frames[frame_idx]
        except (IndexError, Exception):
            print(f"    End of video at frame {frame_idx}")
            sys.stdout.flush()
            break
    # ... rest unchanged
```

**Step 9: Update close_video (remove video_cap.release())**

```python
def close_video(self, video_id: str) -> bool:
    """Close a video session and release resources.

    Args:
        video_id: The video identifier to close.

    Returns:
        True if the session was closed, False if video_id not found.
    """
    if video_id not in self.sessions:
        return False

    # No need to release anything - caller owns the frame/mask sources
    del self.sessions[video_id]
    return True
```

**Step 10: Remove cv2.VideoCapture import usage in __init__ area**

Remove the video file opening code and cv2 VideoCapture usage from `open_video`. Keep cv2 import for `cv2.resize` in mask processing.

**Step 11: Run basic test**

```bash
cd /n/groups/datta/john/projects/vidseq
python -c "from dev_scripts.streaming_segmentor import StreamingSegmentor; print('Import OK')"
```

Expected: `Import OK`

**Step 12: Commit**

```bash
git add dev_scripts/streaming_segmentor.py
git commit -m "feat: add reset_frame() and decouple I/O from StreamingSegmentor

- Add reset_frame(video_id, frame_idx) to clear frame from memory banks
- Change open_video() to accept external frame/mask sources
- Remove internal cv2.VideoCapture management
- Caller now responsible for file handles

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 2: Create VideoFrameSource Wrapper

**Files:**
- Create: `vidseq/services/sam3/frame_source.py`

**Step 1: Create the file**

```python
"""Video frame source wrapper for StreamingSegmentor.

Provides an indexable interface to video frames that avoids unnecessary
seeks when reading sequentially.
"""

from pathlib import Path

import cv2
import numpy as np


class VideoFrameSource:
    """Wrapper around cv2.VideoCapture with position tracking.

    Implements __getitem__ for random access to video frames, but tracks
    current position to avoid unnecessary seeks during sequential reads.

    Attributes:
        frame_count: Total number of frames in the video.
        width: Frame width in pixels.
        height: Frame height in pixels.
    """

    def __init__(self, path: str | Path):
        """Open a video file.

        Args:
            path: Path to the video file.

        Raises:
            ValueError: If video cannot be opened.
        """
        self._path = str(path)
        self._cap = cv2.VideoCapture(self._path)
        if not self._cap.isOpened():
            raise ValueError(f"Failed to open video: {path}")

        self._pos = 0
        self.frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def __getitem__(self, idx: int) -> np.ndarray:
        """Read a frame by index.

        Args:
            idx: Frame index (0-based).

        Returns:
            BGR frame as numpy array of shape (H, W, 3), dtype uint8.

        Raises:
            IndexError: If frame cannot be read.
        """
        if idx < 0 or idx >= self.frame_count:
            raise IndexError(f"Frame index {idx} out of range [0, {self.frame_count})")

        # Only seek if not at the expected position
        if idx != self._pos:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            self._pos = idx

        success, frame = self._cap.read()
        if not success:
            raise IndexError(f"Failed to read frame {idx}")

        self._pos += 1
        return frame

    def __len__(self) -> int:
        """Return total frame count."""
        return self.frame_count

    def close(self) -> None:
        """Release video capture resources."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __del__(self):
        """Ensure resources are released."""
        self.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False
```

**Step 2: Verify import**

```bash
cd /n/groups/datta/john/projects/vidseq
python -c "from vidseq.services.sam3.frame_source import VideoFrameSource; print('Import OK')"
```

Expected: `Import OK`

**Step 3: Commit**

```bash
git add vidseq/services/sam3/frame_source.py
git commit -m "feat: add VideoFrameSource wrapper for StreamingSegmentor

Position-tracking wrapper around cv2.VideoCapture that avoids
unnecessary seeks during sequential frame reads.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 3: Move StreamingSegmentor to vidseq Package

**Files:**
- Move: `dev_scripts/streaming_segmentor.py` → `vidseq/services/sam3/streaming_segmentor.py`
- Modify: `vidseq/services/sam3/__init__.py`

**Step 1: Copy and update imports**

Copy `dev_scripts/streaming_segmentor.py` to `vidseq/services/sam3/streaming_segmentor.py`.

Update the SAM3 import path at the top:

```python
"""Stateful streaming segmentation manager using SAM3 tracker.
...
"""

from __future__ import annotations

import sys
# Update path for SAM3 - this should match your environment
sys.path.insert(0, "/n/groups/datta/john/repos/sam3")

import cv2
import numpy as np
import torch

from custom_sam3 import preprocess_image, encode, track


class StreamingSegmentor:
    # ... rest unchanged
```

**Step 2: Update __init__.py to export**

Add to `vidseq/services/sam3/__init__.py`:

```python
from vidseq.services.sam3.streaming_segmentor import StreamingSegmentor
from vidseq.services.sam3.frame_source import VideoFrameSource

__all__ = ["StreamingSegmentor", "VideoFrameSource"]
```

**Step 3: Verify import from package**

```bash
cd /n/groups/datta/john/projects/vidseq
python -c "from vidseq.services.sam3 import StreamingSegmentor, VideoFrameSource; print('Import OK')"
```

Expected: `Import OK`

**Step 4: Commit**

```bash
git add vidseq/services/sam3/streaming_segmentor.py vidseq/services/sam3/__init__.py
git commit -m "feat: move StreamingSegmentor to vidseq package

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 4: Rewrite TCP Worker Commands

**Files:**
- Modify: `vidseq/services/sam3/server/commands.py`

**Step 1: Replace file contents**

Replace the entire file with new implementation using StreamingSegmentor:

```python
"""SAM3 TCP command handlers using StreamingSegmentor.

Each function handles one command type. The worker maintains a single
StreamingSegmentor instance and manages file handles externally.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import h5py
import numpy as np
from sqlalchemy import update
from sqlalchemy.orm import Session

from vidseq.models.registry import Job
from vidseq.models.video import Video
from vidseq.models.utils import utc_now
from vidseq.services.database_manager import DatabaseManager
from vidseq.services.sam3.frame_source import VideoFrameSource
from vidseq.services.sam3.streaming_segmentor import StreamingSegmentor
from vidseq.services.sam3.utils import encode_mask_rle


@dataclass
class VideoResources:
    """File handles for an open video session."""
    frame_source: VideoFrameSource
    mask_file: h5py.File
    mask_dataset: Any  # h5py.Dataset


# Global state managed by the worker
_segmentor: StreamingSegmentor | None = None
_video_resources: dict[int, VideoResources] = {}


def handle_load_model(checkpoint_path: Path) -> tuple[dict, StreamingSegmentor]:
    """Load SAM3 model via StreamingSegmentor.

    Returns:
        Tuple of (response_dict, segmentor)
    """
    global _segmentor

    print("[SAM3 Worker] Loading SAM3 model via StreamingSegmentor...")

    _segmentor = StreamingSegmentor(device="cuda")

    print("[SAM3 Worker] SAM3 model loaded!")
    return {"type": "status", "status": "ready"}, _segmentor


def _get_mask_path(project_path: Path, video_id: int) -> Path:
    """Get HDF5 mask file path for a video."""
    return project_path / "masks" / f"{video_id}.h5"


def _ensure_mask_dataset(
    mask_path: Path,
    num_frames: int,
    height: int,
    width: int,
) -> tuple[h5py.File, Any]:
    """Open or create HDF5 mask file and dataset.

    Returns:
        (h5py.File, dataset)
    """
    mask_path.parent.mkdir(parents=True, exist_ok=True)

    mask_file = h5py.File(mask_path, "a")

    if "masks" not in mask_file:
        mask_file.create_dataset(
            "masks",
            shape=(num_frames, height, width),
            dtype=np.uint8,
            chunks=(1, height, width),
            fillvalue=0,
        )

    return mask_file, mask_file["masks"]


def handle_init_session(
    params: dict,
    segmentor: StreamingSegmentor,
) -> dict:
    """Initialize video inference session.

    Args:
        params: Command params with video_id, video_path, project_path,
                num_frames, height, width, cond_frame_indices
        segmentor: StreamingSegmentor instance

    Returns:
        Response dict
    """
    global _video_resources

    video_id = params["video_id"]
    video_path = Path(params["video_path"])
    project_path = Path(params["project_path"])
    num_frames = params["num_frames"]
    height = params["height"]
    width = params["width"]
    cond_frame_indices = set(params.get("cond_frame_indices", []))

    print(f"[SAM3 Worker] Initializing session for video {video_id}...")

    if segmentor is None:
        raise RuntimeError("Model not loaded")

    # Close existing session if any
    if video_id in _video_resources:
        _close_video_resources(video_id, segmentor)

    # Open frame source
    frame_source = VideoFrameSource(video_path)

    # Open/create mask file
    mask_path = _get_mask_path(project_path, video_id)
    mask_file, mask_dataset = _ensure_mask_dataset(
        mask_path, num_frames, height, width
    )

    # Store resources
    _video_resources[video_id] = VideoResources(
        frame_source=frame_source,
        mask_file=mask_file,
        mask_dataset=mask_dataset,
    )

    # Initialize StreamingSegmentor session
    segmentor.open_video(
        video_id=str(video_id),
        frames=frame_source,
        masks=mask_dataset,
        frame_dims=(height, width),
        cond_frame_indices=cond_frame_indices,
    )

    return {
        "type": "init_session_result",
        "status": "ok",
        "video_id": video_id,
        "num_frames": num_frames,
        "height": height,
        "width": width,
    }


def handle_add_prompt(
    params: dict,
    segmentor: StreamingSegmentor,
) -> dict:
    """Add point prompt to a frame.

    Args:
        params: Command params with video_id, frame_idx, x, y, label
        segmentor: StreamingSegmentor instance

    Returns:
        Response dict with mask_rle
    """
    video_id = params["video_id"]
    frame_idx = params["frame_idx"]
    x = params["x"]  # normalized [0, 1]
    y = params["y"]  # normalized [0, 1]
    label = params["label"]  # 1=positive, 0=negative

    if segmentor is None:
        raise RuntimeError("Model not loaded")

    if video_id not in _video_resources:
        raise RuntimeError(f"No session for video {video_id}")

    resources = _video_resources[video_id]
    height, width = resources.mask_dataset.shape[1:]

    # Convert normalized coords to pixel coords
    px = x * width
    py = y * height

    # Run segmentation
    segmentor.add_point_prompt(
        video_id=str(video_id),
        frame_idx=frame_idx,
        location=(px, py),
        label=label,
    )

    # Read back the mask for response
    mask = resources.mask_dataset[frame_idx]

    return {
        "type": "add_prompt_result",
        "status": "ok",
        "mask_rle": encode_mask_rle(mask),
        "mask_shape": mask.shape,
        "mask_dtype": str(mask.dtype),
    }


def handle_propagate(
    params: dict,
    segmentor: StreamingSegmentor,
) -> dict:
    """Propagate tracking to a single frame.

    Args:
        params: Command params with video_id, frame_idx
        segmentor: StreamingSegmentor instance

    Returns:
        Response dict with mask_rle
    """
    video_id = params["video_id"]
    frame_idx = params["frame_idx"]

    if segmentor is None:
        raise RuntimeError("Model not loaded")

    if video_id not in _video_resources:
        raise RuntimeError(f"No session for video {video_id}")

    resources = _video_resources[video_id]

    segmentor.propagate(
        video_id=str(video_id),
        frame_idx=frame_idx,
    )

    mask = resources.mask_dataset[frame_idx]

    return {
        "type": "propagate_result",
        "status": "ok",
        "mask_rle": encode_mask_rle(mask),
        "mask_shape": mask.shape,
        "mask_dtype": str(mask.dtype),
    }


def handle_generate_training_masks(
    params: dict,
    segmentor: StreamingSegmentor,
) -> dict:
    """Generate training masks by propagating through video.

    Args:
        params: Command params with video_id, start_frame_idx, max_frames
        segmentor: StreamingSegmentor instance

    Returns:
        Response dict with frames_processed, frame_indices
    """
    video_id = params["video_id"]
    start_frame_idx = params["start_frame_idx"]
    max_frames = params["max_frames"]

    if segmentor is None:
        raise RuntimeError("Model not loaded")

    if video_id not in _video_resources:
        raise RuntimeError(f"No session for video {video_id}")

    frame_indices = segmentor.propagate_sequential(
        video_id=str(video_id),
        start_frame=start_frame_idx,
        num_frames=max_frames,
        progress_interval=50,
    )

    return {
        "type": "generate_training_masks_result",
        "status": "ok",
        "frames_processed": len(frame_indices),
        "frame_indices": frame_indices,
    }


def handle_reset_frame(
    params: dict,
    segmentor: StreamingSegmentor,
) -> dict:
    """Reset a single frame (clear mask and memory).

    Args:
        params: Command params with video_id, frame_idx
        segmentor: StreamingSegmentor instance

    Returns:
        Response dict
    """
    video_id = params["video_id"]
    frame_idx = params["frame_idx"]

    if segmentor is None:
        raise RuntimeError("Model not loaded")

    # Clear from StreamingSegmentor memory
    if video_id in _video_resources:
        segmentor.reset_frame(str(video_id), frame_idx)

        # Clear mask in HDF5
        resources = _video_resources[video_id]
        resources.mask_dataset[frame_idx] = 0

    return {
        "type": "reset_frame_result",
        "status": "ok",
        "frame_idx": frame_idx,
    }


def handle_reset_video(
    params: dict,
    segmentor: StreamingSegmentor,
) -> dict:
    """Reset entire video (clear all masks and memory).

    Args:
        params: Command params with video_id
        segmentor: StreamingSegmentor instance

    Returns:
        Response dict
    """
    video_id = params["video_id"]

    if segmentor is None:
        raise RuntimeError("Model not loaded")

    if video_id in _video_resources:
        # Close StreamingSegmentor session
        segmentor.close_video(str(video_id))

        # Clear all masks in HDF5
        resources = _video_resources[video_id]
        resources.mask_dataset[...] = 0

        # Don't close file handles - session may be reopened

    return {
        "type": "reset_video_result",
        "status": "ok",
    }


def _close_video_resources(video_id: int, segmentor: StreamingSegmentor) -> None:
    """Close and clean up resources for a video."""
    if video_id in _video_resources:
        segmentor.close_video(str(video_id))

        resources = _video_resources.pop(video_id)
        resources.frame_source.close()
        resources.mask_file.close()


def handle_close_session(
    params: dict,
    segmentor: StreamingSegmentor,
) -> dict:
    """Close video session and free resources.

    Args:
        params: Command params with video_id
        segmentor: StreamingSegmentor instance

    Returns:
        Response dict
    """
    video_id = params["video_id"]

    print(f"[SAM3 Worker] Closing session for video {video_id}...")

    if segmentor is not None:
        _close_video_resources(video_id, segmentor)

    return {"type": "close_session_result", "status": "ok"}


def handle_shutdown(segmentor: StreamingSegmentor) -> dict:
    """Shutdown server and clean up all sessions.

    Args:
        segmentor: StreamingSegmentor instance

    Returns:
        Response dict
    """
    global _video_resources

    print("[SAM3 Worker] Shutting down...")

    for video_id in list(_video_resources.keys()):
        try:
            _close_video_resources(video_id, segmentor)
        except Exception:
            pass

    _video_resources.clear()

    return {"type": "shutdown_result", "status": "ok"}


def handle_segment_videos_batch(
    params: dict,
    segmentor: StreamingSegmentor,
    response_callback: Callable[[dict], None],
) -> dict:
    """Segment multiple videos in batch.

    Note: This is a simplified version. The full batch implementation
    with job tracking can be added later if needed.
    """
    # For now, return an error indicating batch is not yet implemented
    # with the new StreamingSegmentor architecture
    return {
        "type": "segment_videos_batch_result",
        "status": "error",
        "error": "Batch segmentation not yet implemented with StreamingSegmentor",
    }
```

**Step 2: Commit**

```bash
git add vidseq/services/sam3/server/commands.py
git commit -m "refactor: rewrite TCP worker commands to use StreamingSegmentor

- Replace SAM3 model direct calls with StreamingSegmentor
- Worker manages VideoFrameSource and HDF5 file handles
- Simplified command interface (points only, no bbox)
- Add handle_reset_frame and handle_reset_video commands
- Remove batch segmentation for now (can add later)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 5: Update TCP Server to Use New Commands

**Files:**
- Modify: `vidseq/services/sam3/server/tcp_server.py`

**Step 1: Update command dispatch**

Update the command handling to pass segmentor instead of model, and update command routing. The key changes are in the command processing loop.

Find the section that dispatches commands and update it to use the new signatures:

```python
# In the command processing section, update dispatch:

if cmd_type == "load_model":
    response, self._segmentor = handle_load_model(checkpoint_path)

elif cmd_type == "init_session":
    response = handle_init_session(params, self._segmentor)

elif cmd_type == "add_prompt":
    response = handle_add_prompt(params, self._segmentor)

elif cmd_type == "propagate":
    response = handle_propagate(params, self._segmentor)

elif cmd_type == "generate_training_masks":
    response = handle_generate_training_masks(params, self._segmentor)

elif cmd_type == "reset_frame":
    response = handle_reset_frame(params, self._segmentor)

elif cmd_type == "reset_video":
    response = handle_reset_video(params, self._segmentor)

elif cmd_type == "close_session":
    response = handle_close_session(params, self._segmentor)

elif cmd_type == "shutdown":
    response = handle_shutdown(self._segmentor)

# Remove: reset_state, clear_frame_prompts (replaced by reset_frame/reset_video)
```

Also update the instance variable from `self._model` to `self._segmentor`.

**Step 2: Update imports**

```python
from vidseq.services.sam3.server.commands import (
    handle_load_model,
    handle_init_session,
    handle_add_prompt,
    handle_propagate,
    handle_generate_training_masks,
    handle_reset_frame,
    handle_reset_video,
    handle_close_session,
    handle_shutdown,
)
```

**Step 3: Commit**

```bash
git add vidseq/services/sam3/server/tcp_server.py
git commit -m "refactor: update TCP server to use new StreamingSegmentor commands

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 6: Update SAM3Service Client Interface

**Files:**
- Modify: `vidseq/services/sam3/service.py`

**Step 1: Simplify the service interface**

Update methods to match the new command interface:
- Remove `_prompts` dict and prompt accumulation
- Remove `add_box_prompt()`
- Simplify `add_point_prompt()` to send single point
- Add `reset_frame()` and `reset_video()` methods
- Update `init_session()` to pass `cond_frame_indices`

Key method updates:

```python
def add_point_prompt(
    self,
    project_id: int,
    video_id: int,
    video_path: Path,
    frame_idx: int,
    x: float,  # normalized [0, 1]
    y: float,
    label: int,  # 1=positive, 0=negative
) -> np.ndarray:
    """Add a point prompt and return the mask."""
    self._ensure_session(project_id, video_id, video_path)

    response = self._send_command({
        "type": "add_prompt",
        "video_id": video_id,
        "frame_idx": frame_idx,
        "x": x,
        "y": y,
        "label": label,
    })

    return self._decode_mask_response(response)


def reset_frame(self, video_id: int, frame_idx: int) -> None:
    """Reset a single frame."""
    self._send_command({
        "type": "reset_frame",
        "video_id": video_id,
        "frame_idx": frame_idx,
    })


def reset_video(self, video_id: int) -> None:
    """Reset entire video."""
    self._send_command({
        "type": "reset_video",
        "video_id": video_id,
    })
```

Remove:
- `_prompts` dict
- `add_box_prompt()`
- `get_prompts_for_frame()`
- `get_all_prompts()`
- Prompt accumulation logic in `add_point_prompt()`

**Step 2: Update init_session to pass conditioning frames**

```python
def init_session(
    self,
    project_id: int,
    video_id: int,
    video_path: Path,
    project_path: Path,
    num_frames: int,
    height: int,
    width: int,
    cond_frame_indices: list[int] | None = None,
) -> dict:
    """Initialize a video session."""
    response = self._send_command({
        "type": "init_session",
        "video_id": video_id,
        "video_path": str(video_path),
        "project_path": str(project_path),
        "num_frames": num_frames,
        "height": height,
        "width": width,
        "cond_frame_indices": cond_frame_indices or [],
    })
    return response
```

**Step 3: Commit**

```bash
git add vidseq/services/sam3/service.py
git commit -m "refactor: simplify SAM3Service for StreamingSegmentor

- Remove prompt accumulation and storage
- Remove add_box_prompt()
- Add reset_frame() and reset_video() methods
- Update init_session() to pass conditioning frames
- Simplify add_point_prompt() to single point

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 7: Update API Routes

**Files:**
- Modify: `vidseq/api/routes/segmentation/inference.py`
- Modify: `vidseq/api/routes/segmentation/state.py`
- Modify: `vidseq/api/routes/segmentation/sessions.py`

**Step 1: Simplify inference.py**

Remove bbox handling and prompt endpoints:

```python
"""Segmentation inference endpoints - point prompts and propagation."""

import logging
from pathlib import Path

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from vidseq.api.dependencies import get_project_folder, get_project_session, get_video
from vidseq.models.video import Video
from vidseq.services import (
    conditioning_service,
    frame_data_service,
    sam3_service,
    segmentation_service,
)

logger = logging.getLogger(__name__)

router = APIRouter()


class PointPromptRequest(BaseModel):
    frame_idx: int
    x: float  # normalized [0, 1]
    y: float  # normalized [0, 1]
    label: int  # 1=positive, 0=negative


class PropagateRequest(BaseModel):
    start_frame_idx: int
    max_frames: int


class PropagateResponse(BaseModel):
    frames_processed: int


@router.post("/projects/{project_id}/videos/{video_id}/segment")
async def run_segmentation(
    project_id: int,
    request: PointPromptRequest,
    video: Video = Depends(get_video),
    session: AsyncSession = Depends(get_project_session),
    project_path: Path = Depends(get_project_folder),
):
    """Run segmentation with a point prompt.

    Point coords should be normalized [0,1].
    """
    video_path = Path(video.path)

    try:
        mask = sam3_service.add_point_prompt(
            project_id=project_id,
            video_id=video.id,
            video_path=video_path,
            frame_idx=request.frame_idx,
            x=request.x,
            y=request.y,
            label=request.label,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Update mask presence index
    has_content = bool(np.any(mask > 0))
    await frame_data_service.set_has_mask(
        session, video.id, request.frame_idx, has_content
    )

    await conditioning_service.add_conditioning_frame(
        session=session,
        video_id=video.id,
        frame_idx=request.frame_idx,
    )

    mask_png = segmentation_service.mask_to_png(mask)
    return Response(content=mask_png, media_type="image/png")


@router.post(
    "/projects/{project_id}/videos/{video_id}/propagate-mask",
    response_model=PropagateResponse,
)
async def propagate_mask(
    project_id: int,
    request: PropagateRequest,
    video: Video = Depends(get_video),
    project_path: Path = Depends(get_project_folder),
):
    """Propagate segmentation mask forward from the given frame."""
    try:
        frames_processed = sam3_service.generate_training_masks(
            project_id=project_id,
            video_id=video.id,
            start_frame_idx=request.start_frame_idx,
            max_frames=request.max_frames,
            project_path=project_path,
            num_frames=video.num_frames,
            height=video.height,
            width=video.width,
        )
    except RuntimeError as e:
        logger.error(f"Propagate mask failed: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))

    return PropagateResponse(frames_processed=frames_processed)


# REMOVED: get_prompts_for_frame and get_all_prompts endpoints
```

**Step 2: Update state.py for new reset commands**

```python
@router.delete("/projects/{project_id}/videos/{video_id}/frame/{frame_idx}")
async def reset_frame(
    project_id: int,
    frame_idx: int,
    video: Video = Depends(get_video),
    session: AsyncSession = Depends(get_project_session),
    project_path: Path = Depends(get_project_folder),
):
    """Reset a single frame - clears mask and conditioning frame status."""
    # Reset in worker (clears StreamingSegmentor memory + HDF5)
    sam3_service.reset_frame(video.id, frame_idx)

    # Clear conditioning frame in DB
    await conditioning_service.remove_conditioning_frame(
        session=session,
        video_id=video.id,
        frame_idx=frame_idx,
    )

    # Clear frame data
    await frame_data_service.clear_frame(session, video.id, frame_idx)

    return {"status": "ok"}


@router.delete("/projects/{project_id}/videos/{video_id}/all-frames")
async def reset_all_frames(
    project_id: int,
    video: Video = Depends(get_video),
    session: AsyncSession = Depends(get_project_session),
    project_path: Path = Depends(get_project_folder),
):
    """Reset all frames - clears all masks and conditioning frames."""
    # Reset in worker
    sam3_service.reset_video(video.id)

    # Clear all conditioning frames in DB
    await conditioning_service.clear_all_conditioning_frames(
        session=session,
        video_id=video.id,
    )

    # Clear all frame data
    await frame_data_service.clear_all_frames(session, video.id)

    return {"status": "ok"}
```

**Step 3: Update sessions.py to pass conditioning frames on init**

Update the session init to fetch and pass conditioning frames:

```python
@router.post("/projects/{project_id}/videos/{video_id}/session")
async def init_session(
    project_id: int,
    video: Video = Depends(get_video),
    session: AsyncSession = Depends(get_project_session),
    project_path: Path = Depends(get_project_folder),
):
    """Initialize a SAM3 session for a video."""
    # Get existing conditioning frames from DB
    cond_frames = await conditioning_service.get_conditioning_frames(
        session=session,
        video_id=video.id,
    )

    result = sam3_service.init_session(
        project_id=project_id,
        video_id=video.id,
        video_path=Path(video.path),
        project_path=project_path,
        num_frames=video.num_frames,
        height=video.height,
        width=video.width,
        cond_frame_indices=[cf.frame_idx for cf in cond_frames],
    )

    return result
```

**Step 4: Commit**

```bash
git add vidseq/api/routes/segmentation/inference.py \
        vidseq/api/routes/segmentation/state.py \
        vidseq/api/routes/segmentation/sessions.py
git commit -m "refactor: update API routes for StreamingSegmentor

- Simplify inference.py to points only
- Remove prompt endpoints
- Update reset endpoints to use new commands
- Pass conditioning frames on session init

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 8: Update Frontend - Remove Bbox from VideoOverlay

**Files:**
- Modify: `frontend/src/components/VideoOverlay.vue`

**Step 1: Remove bbox state and methods**

Remove:
- `drawingBox` ref
- `placedBox` ref
- `editMode` ref
- `editStart` ref
- `HANDLE_SIZE` constant
- `getHitZone()` function
- All bbox logic in `onMouseDown()`, `onMouseMove()`, `onMouseUp()`
- `submitBbox()` function
- Bbox drawing in `render()` (keep stored prompts rendering for backwards compat, will be removed later)
- `bbox-complete` emit

**Step 2: Simplify ToolType**

```typescript
export type ToolType = 'none' | 'positive_point' | 'negative_point'
```

**Step 3: Simplify mouse handlers**

```typescript
function onMouseDown(event: MouseEvent) {
  const coords = getNormalizedCoords(event)
  if (!coords) return

  if (props.activeTool === 'positive_point' || props.activeTool === 'negative_point') {
    pendingPoint.value = { x: coords.x, y: coords.y, type: props.activeTool }
    render()
    emit('point-complete', { x: coords.x, y: coords.y, type: props.activeTool })
  }
}

function onMouseMove(_event: MouseEvent) {
  // No bbox handling needed
}

function onMouseUp(_event: MouseEvent) {
  // No bbox handling needed
}
```

**Step 4: Simplify render()**

Remove bbox drawing sections, keep:
- Mask rendering
- Point prompts rendering
- Pending point animation

**Step 5: Remove bbox-related props**

Update props to remove `showBbox`:

```typescript
const props = defineProps<{
  videoWidth: number
  videoHeight: number
  activeTool: ToolType
  mask: ImageBitmap | null
  prompts: StoredPrompt[]  // Remove bbox prop
  showMask?: boolean
  showPrompts?: boolean
}>()
```

**Step 6: Commit**

```bash
cd frontend && git add src/components/VideoOverlay.vue
git commit -m "refactor: remove bounding box from VideoOverlay

- Remove bbox drawing, editing, and state
- Simplify to points-only tool types
- Remove bbox-complete emit

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 9: Update Frontend - Simplify useSegmentation

**Files:**
- Modify: `frontend/src/composables/useSegmentation.ts`

**Step 1: Remove bbox and prompts state**

Remove:
- `prompts` ref
- `currentPrompts` computed
- `fetchPromptsForFrame()`
- `bboxCache`
- `toggleBboxTool()`
- `handleBboxComplete()`
- Bbox-related imports and types

**Step 2: Simplify ToolType**

```typescript
export type ToolType = 'none' | 'positive_point' | 'negative_point'
```

**Step 3: Update return type**

```typescript
export interface UseSegmentationReturn {
  activeTool: Ref<ToolType>
  currentMask: Ref<ImageBitmap | null>
  isSegmenting: Ref<boolean>
  loadFrameData: (frameIdx: number) => Promise<void>
  seekToFrame: (frameIdx: number) => void
  togglePositivePointTool: () => void
  toggleNegativePointTool: () => void
  handlePointComplete: (point: { x: number; y: number; type: 'positive_point' | 'negative_point' }) => Promise<void>
  handleResetFrame: () => Promise<void>
  handleResetVideo: () => Promise<void>
  clearMaskCache: (startFrame?: number, endFrame?: number) => void
}
```

**Step 4: Simplify loadFrameData**

Remove bbox fetching:

```typescript
const loadFrameData = async (frameIdx: number) => {
  if (!projectId.value || !videoId.value) return

  const cachedMask = maskCache.get(frameIdx)

  if (cachedMask !== undefined) {
    if (frameIdx === intendedFrameIdx.value) {
      currentMask.value = cachedMask
    }
    return
  }

  try {
    const maskBlob = await getMask(projectId.value, videoId.value, frameIdx)

    if (frameIdx !== intendedFrameIdx.value) {
      return
    }

    const bitmap = await createImageBitmap(maskBlob)
    maskCache.set(frameIdx, bitmap)
    currentMask.value = bitmap
  } catch (e) {
    console.error('Failed to load frame data:', e)
  }
}
```

**Step 5: Simplify handlePointComplete**

Remove bbox and prompt fetching:

```typescript
const handlePointComplete = async (point: { x: number; y: number; type: 'positive_point' | 'negative_point' }) => {
  if (!projectId.value || !videoId.value) return

  isSegmenting.value = true

  try {
    const label = point.type === 'positive_point' ? 1 : 0
    const maskBlob = await runSegmentation(
      projectId.value,
      videoId.value,
      currentFrameIdx.value,
      point.x,
      point.y,
      label
    )

    const bitmap = await createImageBitmap(maskBlob)
    currentMask.value = bitmap
    maskCache.set(currentFrameIdx.value, bitmap)
  } catch (e) {
    console.error('Failed to add point:', e)
  } finally {
    isSegmenting.value = false
  }
}
```

**Step 6: Update prefetchMasks to not fetch bboxes**

Remove bbox batch fetching from `prefetchMasks()`.

**Step 7: Commit**

```bash
git add frontend/src/composables/useSegmentation.ts
git commit -m "refactor: simplify useSegmentation for points-only

- Remove bbox state and methods
- Remove prompt storage (now ephemeral)
- Simplify to mask caching only

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 10: Update Frontend - Simplify api.ts

**Files:**
- Modify: `frontend/src/services/api.ts`

**Step 1: Remove bbox and prompt types/functions**

Remove:
- `StoredPrompt` type (or simplify to points only)
- `Bbox` type
- `getBbox()`
- `getBboxesBatch()`
- `getPromptsForFrame()`
- `getAllPrompts()` (if exists)

**Step 2: Simplify runSegmentation**

```typescript
export async function runSegmentation(
  projectId: number,
  videoId: number,
  frameIdx: number,
  x: number,
  y: number,
  label: number,
): Promise<Blob> {
  const response = await fetch(
    `${API_BASE}/projects/${projectId}/videos/${videoId}/segment`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        frame_idx: frameIdx,
        x,
        y,
        label,
      }),
    }
  )

  if (!response.ok) {
    throw new Error(`Segmentation failed: ${response.statusText}`)
  }

  return response.blob()
}
```

**Step 3: Commit**

```bash
git add frontend/src/services/api.ts
git commit -m "refactor: simplify api.ts for points-only segmentation

- Remove bbox endpoints
- Remove prompt endpoints
- Simplify runSegmentation signature

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 11: Update VideoDetail.vue

**Files:**
- Modify: `frontend/src/components/VideoDetail.vue`

**Step 1: Remove bbox tool button from toolbar**

Find the toolbar section and remove the bbox tool button.

**Step 2: Update VideoOverlay usage**

Remove bbox-related props:
- Remove `:bbox` prop
- Remove `:showBbox` prop
- Remove `@bbox-complete` handler

**Step 3: Remove bbox-related state and handlers**

Remove:
- `handleBboxComplete` handler
- Any bbox toggle calls

**Step 4: Commit**

```bash
git add frontend/src/components/VideoDetail.vue
git commit -m "refactor: remove bbox tool from VideoDetail

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 12: Delete Dead Code

**Files:**
- Delete: `vidseq/services/sam3/inference/streaming.py` (LazyVideoFrameLoader)
- Modify: `vidseq/services/sam3/inference/__init__.py` (remove export)
- Clean up any remaining dead imports

**Step 1: Delete LazyVideoFrameLoader**

```bash
rm vidseq/services/sam3/inference/streaming.py
```

**Step 2: Update inference __init__.py**

Remove LazyVideoFrameLoader from exports if present.

**Step 3: Search for dead imports**

```bash
grep -r "LazyVideoFrameLoader" vidseq/
grep -r "add_box_prompt" vidseq/
grep -r "get_prompts" vidseq/
```

Fix any remaining references.

**Step 4: Commit**

```bash
git add -A
git commit -m "chore: delete dead code after StreamingSegmentor migration

- Remove LazyVideoFrameLoader (replaced by VideoFrameSource)
- Clean up dead imports

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 13: Integration Testing

**Step 1: Start backend**

```bash
cd /n/groups/datta/john/projects/vidseq
vidseq
```

Verify server starts without errors.

**Step 2: Start frontend**

```bash
cd frontend
npm run dev
```

**Step 3: Manual testing checklist**

- [ ] Open a project
- [ ] Open a video in VideoDetail
- [ ] SAM3 status shows "Ready"
- [ ] Click positive point tool, click on video
- [ ] Mask appears
- [ ] Click negative point, mask updates
- [ ] Click propagate, masks generate
- [ ] Reset frame clears mask
- [ ] Reset video clears all masks
- [ ] Navigate to different frame, back - mask loads from cache
- [ ] Close and reopen video - conditioning frames restored

**Step 4: Fix any issues found**

Debug and fix issues as they arise.

**Step 5: Final commit**

```bash
git add -A
git commit -m "test: verify StreamingSegmentor integration

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Summary

This plan migrates vidseq from direct SAM3 API calls to using StreamingSegmentor:

1. **Tasks 1-3**: Prepare StreamingSegmentor (add reset_frame, decouple I/O, create VideoFrameSource)
2. **Tasks 4-6**: Update backend (TCP worker, server, service)
3. **Tasks 7**: Update API routes
4. **Tasks 8-11**: Update frontend (remove bbox, simplify state)
5. **Tasks 12-13**: Clean up and test

Key architectural changes:
- StreamingSegmentor handles all SAM3 logic internally
- Worker manages file handles (video, HDF5), passes to StreamingSegmentor
- Prompts are ephemeral (not stored in backend)
- Conditioning frames tracked in SQLite for memory reconstruction
- Points-only interface (bbox removed)
