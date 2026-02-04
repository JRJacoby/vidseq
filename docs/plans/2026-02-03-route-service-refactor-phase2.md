# Route/Service Layer Refactoring - Phase 2 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extract H5 file access, complex workflow logic, and frame extraction from routes into service layer.

**Architecture:** Routes become thin wrappers that parse requests and call services. All H5 file operations, TCP client orchestration, and business logic moves to `frame_data_service.py` and `segmentation_service.py`. Frame extraction is consolidated into `video_service.py`.

**Tech Stack:** FastAPI, SQLAlchemy async, H5PY via array_storage, TCP client for SAM2 worker

---

## Context

**Pattern established in Phase 1:**
- Routes only: parse request → call service → format response
- Services own: database operations, file I/O, business logic
- Exceptions raised by services, translated by global handlers in `server.py`

**Violations to fix in Phase 2:**

| Route | Violation |
|-------|-----------|
| `segmentation/training.py:create_training_range` | H5 file access + bbox computation loop |
| `segmentation/inference.py:submit_prompt` | Conditional workflow logic (refine vs add_point) |
| `segmentation/inference.py:propagate` | Post-processing loop (updating frame data) |
| `videos.py:get_frame` | cv2/PIL frame extraction |
| `cropped_videos.py:get_cropped_video_frame` | Duplicated cv2/PIL frame extraction |

---

## Task 1: Add New Exceptions

**Files:**
- Modify: `vidseq/services/exceptions.py`
- Modify: `vidseq/server.py`

**Step 1: Add exception classes to exceptions.py**

Add these three new exceptions after the existing `VideoFileInvalidError`:

```python
class FrameIndexOutOfRangeError(Exception):
    """Raised when a frame index is outside valid bounds."""

    def __init__(self, frame_idx: int, num_frames: int):
        self.frame_idx = frame_idx
        self.num_frames = num_frames
        super().__init__(f"Frame index {frame_idx} out of range [0, {num_frames})")


class MultiPointWithoutMaskError(Exception):
    """Raised when multi-point prompt submitted without existing mask."""

    def __init__(self):
        super().__init__(
            "Cannot submit multiple points without an existing mask. "
            "Submit a single point first to create a mask."
        )


class MissingMasksError(Exception):
    """Raised when required masks are missing for an operation."""

    def __init__(self, missing_frames: list[int]):
        self.missing_frames = missing_frames
        super().__init__(f"Frames missing masks: {missing_frames}")
```

**Step 2: Add exception handlers to server.py**

Add these imports to the existing imports from `vidseq.services.exceptions`:
- `FrameIndexOutOfRangeError`
- `MultiPointWithoutMaskError`
- `MissingMasksError`

Add these handlers after the existing handlers:

```python
@app.exception_handler(FrameIndexOutOfRangeError)
async def frame_index_out_of_range_handler(request, exc: FrameIndexOutOfRangeError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(MultiPointWithoutMaskError)
async def multi_point_without_mask_handler(request, exc: MultiPointWithoutMaskError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(MissingMasksError)
async def missing_masks_handler(request, exc: MissingMasksError):
    return JSONResponse(
        status_code=400,
        content={
            "detail": {
                "message": "Some frames are missing masks",
                "missing_frames": exc.missing_frames,
            }
        },
    )
```

**Step 3: Verify syntax**

Run: `uv run python -c "from vidseq.services.exceptions import FrameIndexOutOfRangeError, MultiPointWithoutMaskError, MissingMasksError; print('OK')"`

Expected: `OK`

**Step 4: Commit**

```bash
git add vidseq/services/exceptions.py vidseq/server.py
git commit -m "$(cat <<'EOF'
feat(exceptions): add segmentation-related exceptions

- FrameIndexOutOfRangeError for invalid frame indices
- MultiPointWithoutMaskError for multi-point prompts without mask
- MissingMasksError for training range validation

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Add frame_data_service.create_training_range()

**Files:**
- Modify: `vidseq/services/frame_data_service.py`
- Modify: `vidseq/api/routes/segmentation/training.py`

**Background:**
- `frame_data_service.py` already has `mark_training_range()` and `get_missing_tracker_masks_in_range()`
- Route currently opens H5 file and loops through frames computing bboxes
- Service should do all of this; route just validates params and calls service

**Step 1: Add alias for get_missing_masks_in_range**

The route calls `get_missing_masks_in_range` but the function is named `get_missing_tracker_masks_in_range`. Add this alias at the end of the TRACKER MASK PRESENCE section (after `clear_all_has_tracker_mask`):

```python
# Alias for backwards compatibility
get_missing_masks_in_range = get_missing_tracker_masks_in_range
```

**Step 2: Add create_training_range to frame_data_service.py**

Add import at top of file:

```python
from pathlib import Path
```

Add this function after `unmark_training_range`:

```python
async def create_training_range(
    session: AsyncSession,
    project_path: Path,
    video_id: int,
    start_frame: int,
    end_frame: int,
) -> None:
    """Create a training range by computing bboxes from masks and marking frames.

    Validates that all frames in range have tracker masks, then:
    1. Loads masks from H5 file
    2. Computes bbox for each frame
    3. Saves bboxes to database
    4. Marks frames as training

    Args:
        session: Async database session
        project_path: Path to the project folder
        video_id: ID of the video
        start_frame: First frame of range (inclusive)
        end_frame: Last frame of range (inclusive)

    Raises:
        MissingMasksError: If any frames in range are missing masks
    """
    from vidseq.services.array_storage import tracker_masks, compute_bbox_from_mask
    from vidseq.services.exceptions import MissingMasksError

    # Validate all frames have masks
    missing_frames = await get_missing_tracker_masks_in_range(
        session, video_id, start_frame, end_frame
    )
    if missing_frames:
        raise MissingMasksError(missing_frames)

    # Load masks and compute bboxes
    with tracker_masks(project_path, video_id, "r") as masks:
        for frame_idx in range(start_frame, end_frame + 1):
            mask = masks[frame_idx]
            bbox = compute_bbox_from_mask(mask)
            if bbox is not None:
                await save_bbox(session, video_id, frame_idx, bbox)

    # Mark frames as training
    await mark_training_range(session, video_id, start_frame, end_frame)
```

**Step 3: Refactor training.py route**

Replace the `create_training_range` route function with:

```python
@router.post(
    "/projects/{project_id}/videos/{video_id}/training-range",
    status_code=204,
)
async def create_training_range(
    start_frame: int,
    end_frame: int,
    video: Video = Depends(get_video),
    session: AsyncSession = Depends(get_project_session),
    project_path: Path = Depends(get_project_folder),
):
    """
    Mark frame range as training (computes bboxes from masks).

    All frames in range must have masks. Use GET /training-range/validation first.
    """
    await frame_data_service.create_training_range(
        session=session,
        project_path=project_path,
        video_id=video.id,
        start_frame=start_frame,
        end_frame=end_frame,
    )
    return None
```

**Step 4: Remove unused imports from training.py**

Remove these imports (no longer needed):
- `from vidseq.services.array_storage import tracker_masks, compute_bbox_from_mask`

The imports section should become:

```python
"""Training range endpoints - validation, marking, and unmarking."""

from pathlib import Path

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from vidseq.api.dependencies import get_project_folder, get_project_session, get_video
from vidseq.models.video import Video
from vidseq.services import frame_data_service

router = APIRouter()
```

**Step 5: Verify the refactored route works**

Run: `uv run python -c "from vidseq.api.routes.segmentation.training import router; print('Route imports OK')"`

Expected: `Route imports OK`

**Step 6: Commit**

```bash
git add vidseq/services/frame_data_service.py vidseq/api/routes/segmentation/training.py
git commit -m "$(cat <<'EOF'
refactor(training): move bbox computation to service layer

- Add frame_data_service.create_training_range() with H5 access
- Route becomes thin wrapper calling service
- Add get_missing_masks_in_range alias

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Add segmentation_service.submit_prompt()

**Files:**
- Modify: `vidseq/services/segmentation_service.py`
- Modify: `vidseq/api/routes/segmentation/inference.py`

**Background:**
- Route has conditional logic: if existing mask → refine, else → add_point
- Route also handles post-processing: checking for empty mask, resetting state
- All of this should be in service

**Step 1: Add submit_prompt to segmentation_service.py**

Add these imports at top (merge with existing):

```python
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from vidseq.services import segmentation_tcp_client, frame_data_service
from vidseq.services.exceptions import MultiPointWithoutMaskError

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession
```

Add this function after `reset_frame_memory`:

```python
async def submit_prompt(
    session: "AsyncSession",
    project_id: int,
    video_id: int,
    video_path: Path,
    project_path: Path,
    frame_idx: int,
    points: list[dict],
    labels: list[int],
) -> np.ndarray:
    """Submit point prompt(s) for segmentation.

    Workflow determined by existing state:
    - 1 point, no existing mask: creates new mask (add_point_prompt)
    - 1 point, existing mask: refines mask with single point
    - 2+ points, existing mask: refines mask with all points
    - 2+ points, no existing mask: ERROR (can't refine without mask)

    Args:
        session: Async database session
        project_id: ID of the project
        video_id: ID of the video
        video_path: Path to the video file
        project_path: Path to the project folder
        frame_idx: Frame index (0-based)
        points: List of {"x": float, "y": float} normalized coords
        labels: List of labels (1=positive, 0=negative)

    Returns:
        Resulting mask as numpy array

    Raises:
        MultiPointWithoutMaskError: If multi-point submitted without existing mask
        RuntimeError: If TCP client fails
    """
    # Check if this frame already has a mask
    has_existing_mask = await frame_data_service.get_has_tracker_mask(
        session, video_id, frame_idx
    )

    # Validate: multi-point requires existing mask
    if len(points) > 1 and not has_existing_mask:
        raise MultiPointWithoutMaskError()

    if has_existing_mask:
        # Refine existing mask using previous logits as dense prompt
        mask = segmentation_tcp_client.refine_mask(
            project_id=project_id,
            video_id=video_id,
            frame_idx=frame_idx,
            points=points,
            labels=labels,
        )
    else:
        # Create new mask on blank frame (single point only, validated above)
        p = points[0]
        label = labels[0]
        mask = segmentation_tcp_client.add_point_prompt(
            project_id=project_id,
            video_id=video_id,
            video_path=video_path,
            project_path=project_path,
            frame_idx=frame_idx,
            x=p["x"],
            y=p["y"],
            label=label,
        )

    # Update mask presence index
    has_content = bool(np.any(mask > 0))
    await frame_data_service.set_has_tracker_mask(
        session, video_id, frame_idx, has_content
    )

    # If refinement resulted in an empty mask, reset the frame's SAM state
    if has_existing_mask and not has_content:
        segmentation_tcp_client.reset_frame(
            project_id=project_id,
            video_id=video_id,
            project_path=project_path,
            frame_idx=frame_idx,
        )

    return mask
```

**Step 2: Refactor inference.py submit_prompt route**

Replace the entire `submit_prompt` route function:

```python
@router.post("/projects/{project_id}/videos/{video_id}/prompt/{frame_idx}")
async def submit_prompt(
    project_id: int,
    frame_idx: int,
    request: PromptRequest,
    video: Video = Depends(get_video),
    session: AsyncSession = Depends(get_project_session),
    project_path: Path = Depends(get_project_folder),
):
    """
    Submit point prompt(s) for segmentation.

    Point coords should be normalized [0,1].

    Workflow is determined by existing state:
    - 1 point, no existing mask: creates new mask (add_point_prompt)
    - 1 point, existing mask: refines mask with single point
    - 2+ points, existing mask: refines mask with all points
    - 2+ points, no existing mask: ERROR (can't refine without mask)
    """
    video_path = Path(video.path)

    # Convert points to backend format
    points = [{"x": p.x, "y": p.y} for p in request.points]
    labels = [1 if p.type == "positive_point" else 0 for p in request.points]

    mask = await segmentation_service.submit_prompt(
        session=session,
        project_id=project_id,
        video_id=video.id,
        video_path=video_path,
        project_path=project_path,
        frame_idx=frame_idx,
        points=points,
        labels=labels,
    )

    mask_png = segmentation_service.mask_to_png(mask)
    return Response(content=mask_png, media_type="image/png")
```

**Step 3: Clean up inference.py imports**

Remove unused imports. The imports section should become:

```python
"""Segmentation inference endpoints - point prompts and propagation."""

import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from sqlalchemy.ext.asyncio import AsyncSession

from vidseq.api.dependencies import get_project_folder, get_project_session, get_video
from vidseq.models.video import Video
from vidseq.schemas.segmentation import (
    PromptRequest,
    PropagateRequest,
    PropagateResponse,
)
from vidseq.services import (
    frame_data_service,
    segmentation_service,
    segmentation_tcp_client,
)

logger = logging.getLogger(__name__)

router = APIRouter()
```

Note: Keep `numpy` import removed, `HTTPException` is still needed for propagate, and `segmentation_tcp_client` is still needed for propagate (for now).

**Step 4: Verify imports work**

Run: `uv run python -c "from vidseq.services.segmentation_service import submit_prompt; print('Service OK')"`

Expected: `Service OK`

**Step 5: Commit**

```bash
git add vidseq/services/segmentation_service.py vidseq/api/routes/segmentation/inference.py
git commit -m "$(cat <<'EOF'
refactor(inference): move submit_prompt logic to service layer

- Add segmentation_service.submit_prompt() with workflow logic
- Route becomes thin wrapper
- Service handles mask state checks and TCP client calls

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Add segmentation_service.propagate()

**Files:**
- Modify: `vidseq/services/segmentation_service.py`
- Modify: `vidseq/api/routes/segmentation/inference.py`

**Background:**
- Route calls TCP client then loops updating frame_data
- All orchestration should be in service

**Step 1: Add propagate to segmentation_service.py**

Add this function after `submit_prompt`:

```python
async def propagate(
    session: "AsyncSession",
    project_id: int,
    video_id: int,
    project_path: Path,
    start_frame_idx: int,
    max_frames: int,
    num_frames: int,
    height: int,
    width: int,
) -> int:
    """Propagate segmentation masks forward from a frame.

    Requires an active SAM session with a tracked object.

    Args:
        session: Async database session
        project_id: ID of the project
        video_id: ID of the video
        project_path: Path to the project folder
        start_frame_idx: Frame to start propagation from
        max_frames: Maximum number of frames to propagate
        num_frames: Total frames in video
        height: Video height
        width: Video width

    Returns:
        Number of frames processed

    Raises:
        RuntimeError: If propagation fails (no active session, etc.)
    """
    frame_indices = segmentation_tcp_client.generate_training_masks(
        project_id=project_id,
        video_id=video_id,
        start_frame_idx=start_frame_idx,
        max_frames=max_frames,
        project_path=project_path,
        num_frames=num_frames,
        height=height,
        width=width,
    )

    # Update has_tracker_mask for all propagated frames
    for frame_idx in frame_indices:
        await frame_data_service.set_has_tracker_mask(session, video_id, frame_idx, True)

    return len(frame_indices)
```

**Step 2: Refactor inference.py propagate route**

Replace the entire `propagate` route function:

```python
@router.post(
    "/projects/{project_id}/videos/{video_id}/propagation",
    response_model=PropagateResponse,
)
async def propagate(
    project_id: int,
    request: PropagateRequest,
    video: Video = Depends(get_video),
    session: AsyncSession = Depends(get_project_session),
    project_path: Path = Depends(get_project_folder),
):
    """
    Propagate segmentation mask forward from the given frame.

    Requires an active SAM session with a tracked object (submit a point prompt first).
    Saves only masks to HDF5. Does NOT mark frames as training.
    Use POST /training-range to mark frames for training.
    """
    frames_processed = await segmentation_service.propagate(
        session=session,
        project_id=project_id,
        video_id=video.id,
        project_path=project_path,
        start_frame_idx=request.start_frame_idx,
        max_frames=request.max_frames,
        num_frames=video.num_frames,
        height=video.height,
        width=video.width,
    )

    return PropagateResponse(frames_processed=frames_processed)
```

**Step 3: Clean up inference.py imports**

Now we can remove more imports. Final imports section:

```python
"""Segmentation inference endpoints - point prompts and propagation."""

from pathlib import Path

from fastapi import APIRouter, Depends
from fastapi.responses import Response
from sqlalchemy.ext.asyncio import AsyncSession

from vidseq.api.dependencies import get_project_folder, get_project_session, get_video
from vidseq.models.video import Video
from vidseq.schemas.segmentation import (
    PromptRequest,
    PropagateRequest,
    PropagateResponse,
)
from vidseq.services import segmentation_service

router = APIRouter()
```

Note: Removed `logging`, `HTTPException`, `frame_data_service`, and `segmentation_tcp_client` (all now handled in service).

**Step 4: Verify imports work**

Run: `uv run python -c "from vidseq.api.routes.segmentation.inference import router; print('Route OK')"`

Expected: `Route OK`

**Step 5: Commit**

```bash
git add vidseq/services/segmentation_service.py vidseq/api/routes/segmentation/inference.py
git commit -m "$(cat <<'EOF'
refactor(inference): move propagate logic to service layer

- Add segmentation_service.propagate() with TCP orchestration
- Route becomes thin wrapper
- Remove unused imports from route

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Add video_service.extract_frame_as_jpeg()

**Files:**
- Modify: `vidseq/services/video_service.py`
- Modify: `vidseq/api/routes/videos.py`
- Modify: `vidseq/api/routes/cropped_videos.py`

**Background:**
- Both `videos.py:get_frame` and `cropped_videos.py:get_cropped_video_frame` have nearly identical code
- Both use cv2.VideoCapture, read frame, convert BGR→RGB, save as JPEG
- Consolidate into service function

**Step 1: Add extract_frame_as_jpeg to video_service.py**

Add these imports at top (merge with existing):

```python
from io import BytesIO

from PIL import Image
```

Add this function after `get_video_metadata`:

```python
def extract_frame_as_jpeg(
    video_path: Path,
    frame_idx: int,
    num_frames: int | None = None,
    quality: int = 95,
) -> bytes:
    """Extract a specific frame from a video and return as JPEG bytes.

    Args:
        video_path: Path to the video file
        frame_idx: Frame index (0-based)
        num_frames: Total frames in video (for validation). If None, reads from video.
        quality: JPEG quality (1-100)

    Returns:
        JPEG image bytes

    Raises:
        VideoFileNotFoundError: If video file doesn't exist
        FrameIndexOutOfRangeError: If frame_idx is out of bounds
        VideoFileInvalidError: If video cannot be opened or frame cannot be read
    """
    from vidseq.services.exceptions import FrameIndexOutOfRangeError

    if not video_path.exists():
        raise VideoFileNotFoundError(str(video_path))

    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise VideoFileInvalidError(str(video_path), "could not open video")

        # Get frame count if not provided
        if num_frames is None:
            num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Validate frame index
        if frame_idx < 0 or frame_idx >= num_frames:
            raise FrameIndexOutOfRangeError(frame_idx, num_frames)

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            raise VideoFileInvalidError(
                str(video_path), f"could not read frame {frame_idx}"
            )
    finally:
        cap.release()

    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert to PIL Image and then to JPEG bytes
    pil_image = Image.fromarray(frame_rgb)
    img_bytes = BytesIO()
    pil_image.save(img_bytes, format="JPEG", quality=quality)
    img_bytes.seek(0)

    return img_bytes.read()
```

**Step 2: Refactor videos.py get_frame route**

Replace the entire `get_frame` route function:

```python
@router.get("/projects/{project_id}/videos/{video_id}/frame/{frame_idx}")
async def get_frame(
    frame_idx: int,
    video: Video = Depends(get_video),
):
    """
    Extract a specific frame from a video and return it as a JPEG image.

    Args:
        video_id: ID of the video
        frame_idx: Frame index to extract (0-based)

    Returns:
        JPEG image bytes
    """
    video_path = Path(video.path)
    jpeg_bytes = video_service.extract_frame_as_jpeg(
        video_path=video_path,
        frame_idx=frame_idx,
        num_frames=video.num_frames,
    )
    return Response(content=jpeg_bytes, media_type="image/jpeg")
```

**Step 3: Clean up videos.py imports**

Remove unused imports. The imports section should become:

```python
import mimetypes
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse, Response
from sqlalchemy.ext.asyncio import AsyncSession

from vidseq.api.dependencies import get_project_folder, get_project_session, get_video
from vidseq.models.video import Video
from vidseq.schemas.video import VideoCreate, VideoResponse
from vidseq.services import video_service

router = APIRouter()
```

Note: Removed `cv2`, `BytesIO`, and `PIL.Image` - now handled in service.

**Step 4: Refactor cropped_videos.py get_cropped_video_frame route**

Replace the entire `get_cropped_video_frame` route function:

```python
@router.get("/projects/{project_id}/videos/{video_id}/cropped-video/frame/{frame_idx}")
async def get_cropped_video_frame(
    frame_idx: int,
    video: Video = Depends(get_video),
    project_path: Path = Depends(get_project_folder),
):
    """Extract a specific frame from a cropped video and return it as a JPEG image."""
    cropped_path = cropped_video_service.get_cropped_video_path(project_path, video.name)
    jpeg_bytes = video_service.extract_frame_as_jpeg(
        video_path=cropped_path,
        frame_idx=frame_idx,
    )
    return Response(content=jpeg_bytes, media_type="image/jpeg")
```

**Step 5: Clean up cropped_videos.py imports**

Remove unused imports. The imports section should become:

```python
"""API routes for cropped video extraction and streaming."""

import asyncio
import mimetypes
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from vidseq.api.dependencies import get_project_folder, get_project_session, get_video
from vidseq.models.video import Video
from vidseq.services import cropped_video_service, video_service

router = APIRouter()
```

Note: Removed `BytesIO`, `cv2`, and `PIL.Image` - now handled in service.

**Step 6: Verify all imports work**

Run: `uv run python -c "from vidseq.services.video_service import extract_frame_as_jpeg; print('Service OK')"`

Expected: `Service OK`

Run: `uv run python -c "from vidseq.api.routes.videos import router; print('Videos OK')"`

Expected: `Videos OK`

Run: `uv run python -c "from vidseq.api.routes.cropped_videos import router; print('Cropped OK')"`

Expected: `Cropped OK`

**Step 7: Commit**

```bash
git add vidseq/services/video_service.py vidseq/api/routes/videos.py vidseq/api/routes/cropped_videos.py
git commit -m "$(cat <<'EOF'
refactor(video): consolidate frame extraction in service layer

- Add video_service.extract_frame_as_jpeg()
- videos.py and cropped_videos.py now use shared function
- Remove duplicated cv2/PIL code from routes

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Verification Checklist

After all tasks, verify the success criteria from the main refactoring plan:

**Routes should NOT import:**
- [ ] `tracker_masks`, `detector_masks` from array_storage (check training.py)
- [ ] `cv2` (check videos.py, cropped_videos.py)
- [ ] `PIL.Image` (check videos.py, cropped_videos.py)
- [ ] `numpy` (check inference.py)

**Routes should follow pattern:**
- [ ] `training.py:create_training_range` - parse request → call service → return
- [ ] `inference.py:submit_prompt` - parse request → call service → format response
- [ ] `inference.py:propagate` - parse request → call service → return response
- [ ] `videos.py:get_frame` - get video → call service → return response
- [ ] `cropped_videos.py:get_cropped_video_frame` - get path → call service → return response

Run final verification:

```bash
# Check no H5 imports in routes
grep -r "from vidseq.services.array_storage" vidseq/api/routes/ && echo "FAIL: H5 imports in routes" || echo "PASS: No H5 in routes"

# Check no cv2 in routes (except stream helpers which are acceptable)
grep -r "^import cv2\|^from cv2" vidseq/api/routes/ && echo "FAIL: cv2 in routes" || echo "PASS: No cv2 in routes"

# Check no PIL in routes
grep -r "from PIL" vidseq/api/routes/ && echo "FAIL: PIL in routes" || echo "PASS: No PIL in routes"
```

---

## Summary

| Task | Route File | Service Function | Lines Moved |
|------|-----------|------------------|-------------|
| 1 | server.py | N/A (exceptions) | +30 |
| 2 | training.py | frame_data_service.create_training_range | ~25 |
| 3 | inference.py | segmentation_service.submit_prompt | ~45 |
| 4 | inference.py | segmentation_service.propagate | ~20 |
| 5 | videos.py, cropped_videos.py | video_service.extract_frame_as_jpeg | ~60 (deduped) |

**Total:** ~5 route files simplified, ~3 service files expanded, ~180 lines reorganized.
