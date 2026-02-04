# Route/Service Layer Refactoring - Phase 3 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Move remaining business logic from alignment.py routes into AlignmentService methods.

**Architecture:** Three routes have inline DB queries or DatabaseManager access that should be in the service layer. Move this logic to new service methods with matching names. Routes become thin wrappers.

**Tech Stack:** FastAPI, SQLAlchemy async, AlignmentService singleton

---

## Context

**Pattern established in Phase 1-2:**
- Routes only: parse request → call service → format response
- Services own: database operations, file I/O, business logic
- Service method names should match route function names

**Violations to fix:**

| Route | Line | Violation |
|-------|------|-----------|
| `get_alignment_status` | 110-121 | `select(Video)` + filesystem iteration |
| `create_alignment_training` | 322-325 | `select(Video).where(...)` for video_name_map |
| `create_videos_alignment` | 448-449 | `DatabaseManager.get_instance()` access |

---

## Task 1: Move get_alignment_status logic to service

**Files:**
- Modify: `vidseq/services/alignment_service.py`
- Modify: `vidseq/api/routes/alignment.py`

**Step 1: Add get_alignment_status method to AlignmentService**

In `alignment_service.py`, add this method to the `AlignmentService` class (after `get_random_unlabeled_frame`, around line 1516):

```python
async def get_alignment_status(
    self, session: AsyncSession, project_path: Path
) -> dict:
    """Get current alignment status including label count and cropped video status.

    Args:
        session: Async database session
        project_path: Path to the project folder

    Returns:
        Dict with keys: label_count, model_trained, is_training, is_applying, all_videos_cropped
    """
    label_count = await self.get_label_count(session)
    model_trained = self.is_model_trained(project_path)
    is_training = self.is_training()
    is_applying = self.is_applying()

    # Check if all videos have cropping completed
    result = await session.execute(select(Video))
    videos = list(result.scalars().all())
    video_count = len(videos)

    def _count_cropped() -> int:
        return sum(
            1 for v in videos
            if cropped_video_exists(project_path, v.name)
        )

    cropped_count = await asyncio.to_thread(_count_cropped)
    all_videos_cropped = video_count > 0 and cropped_count == video_count

    return {
        "label_count": label_count,
        "model_trained": model_trained,
        "is_training": is_training,
        "is_applying": is_applying,
        "all_videos_cropped": all_videos_cropped,
    }
```

**Step 2: Refactor the route in alignment.py**

Replace the `get_alignment_status` route function (lines 93-137) with:

```python
@router.get("/projects/{project_id}/alignment/status")
async def get_alignment_status(
    project_id: int,
    session: AsyncSession = Depends(get_project_session),
    project_path: Path = Depends(get_project_folder),
) -> AlignmentStatusResponse:
    """Get current alignment training status."""
    service = AlignmentService.get_instance()
    status = await service.get_alignment_status(session, project_path)
    return AlignmentStatusResponse(**status)
```

**Step 3: Remove unused imports from alignment.py**

Remove these imports that are no longer needed in the route file:
- `from sqlalchemy import delete, select` - remove `select` (keep `delete` if used elsewhere, but check - it's not used)

Actually, check if `select` or `delete` are used anywhere else in the file. Looking at the routes, they're not - all DB operations go through the service. Remove the entire import line:

```python
# Remove this line:
from sqlalchemy import delete, select
```

**Step 4: Verify imports work**

Run: `uv run python -c "from vidseq.api.routes.alignment import router; print('Route OK')"`

Expected: `Route OK`

**Step 5: Commit**

```bash
git add vidseq/services/alignment_service.py vidseq/api/routes/alignment.py
git commit -m "$(cat <<'EOF'
refactor(alignment): move get_alignment_status logic to service

- Add AlignmentService.get_alignment_status() method
- Route becomes thin wrapper
- Remove unused sqlalchemy imports from route

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Move create_alignment_training logic to service

**Files:**
- Modify: `vidseq/services/alignment_service.py`
- Modify: `vidseq/api/routes/alignment.py`

**Step 1: Add create_alignment_training method to AlignmentService**

In `alignment_service.py`, add this method to the `AlignmentService` class (after `get_alignment_status`):

```python
async def create_alignment_training(
    self,
    session: AsyncSession,
    project_path: Path,
    epochs: int = 100,
    augment: bool = True,
    early_stop_patience: int = 5,
    lr_patience: int = 3,
) -> dict:
    """Start alignment model training.

    Fetches labels, builds video name map, validates, and starts training
    in a background thread.

    Args:
        session: Async database session
        project_path: Path to the project folder
        epochs: Maximum training epochs
        augment: Whether to use data augmentation
        early_stop_patience: Epochs without improvement before stopping
        lr_patience: Epochs without improvement before reducing LR

    Returns:
        Dict with status and training parameters

    Raises:
        RuntimeError: If training already in progress or no labels available
    """
    if self.is_training():
        raise RuntimeError("Training already in progress")

    # Fetch all labels
    labels = await self.get_all_labels(session)

    if len(labels) == 0:
        raise RuntimeError("No labels available for training")

    # Build video_name_map: video_id -> video.name
    video_ids = list({label.video_id for label in labels})
    result = await session.execute(select(Video).where(Video.id.in_(video_ids)))
    videos = list(result.scalars().all())
    video_name_map = {v.id: v.name for v in videos}

    # Start training in background thread
    asyncio.create_task(
        asyncio.to_thread(
            self.train_model_sync,
            project_path,
            labels,
            video_name_map,
            epochs,
            augment,
            early_stop_patience,
            lr_patience,
        )
    )

    return {"status": "started", "max_epochs": epochs, "augment": augment}
```

**Step 2: Add exception imports and handlers**

Add a new exception to `vidseq/services/exceptions.py`:

```python
class AlignmentTrainingError(Exception):
    """Raised when alignment training cannot be started."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)
```

Add the handler to `vidseq/server.py`. First add the import:

```python
from vidseq.services.exceptions import (
    ...
    AlignmentTrainingError,
)
```

Then add the handler:

```python
@app.exception_handler(AlignmentTrainingError)
async def alignment_training_error_handler(request, exc: AlignmentTrainingError):
    return JSONResponse(status_code=400, content={"detail": exc.message})
```

**Step 3: Update the service method to use the exception**

In the `create_alignment_training` method, change the raises to use the new exception:

```python
from vidseq.services.exceptions import AlignmentTrainingError

# In the method:
if self.is_training():
    raise AlignmentTrainingError("Training already in progress")

if len(labels) == 0:
    raise AlignmentTrainingError("No labels available for training")
```

**Step 4: Refactor the route in alignment.py**

Replace the `create_alignment_training` route function (lines 284-346) with:

```python
@router.post("/projects/{project_id}/alignment/training")
async def create_alignment_training(
    project_id: int,
    epochs: int = 100,
    augment: bool = True,
    early_stop_patience: int = 5,
    lr_patience: int = 3,
    project_path: Path = Depends(get_project_folder),
    session: AsyncSession = Depends(get_project_session),
):
    """Start alignment model training (fire-and-forget).

    Training will run for up to `epochs` (max), but may stop early if loss
    plateaus. Learning rate is automatically reduced on plateau.

    Returns immediately after starting training. Use the SSE stream endpoint
    (/alignment/training/stream) to monitor progress, or the status endpoint
    (/alignment/training/status) to check current state.
    """
    service = AlignmentService.get_instance()
    return await service.create_alignment_training(
        session=session,
        project_path=project_path,
        epochs=epochs,
        augment=augment,
        early_stop_patience=early_stop_patience,
        lr_patience=lr_patience,
    )
```

**Step 5: Remove the helper function from alignment.py**

Delete the `_run_training_in_background` function (lines 265-281) - it's no longer needed since the service handles everything.

**Step 6: Verify imports work**

Run: `uv run python -c "from vidseq.api.routes.alignment import router; print('Route OK')"`

Expected: `Route OK`

**Step 7: Commit**

```bash
git add vidseq/services/alignment_service.py vidseq/services/exceptions.py vidseq/server.py vidseq/api/routes/alignment.py
git commit -m "$(cat <<'EOF'
refactor(alignment): move create_alignment_training logic to service

- Add AlignmentService.create_alignment_training() method
- Add AlignmentTrainingError exception
- Route becomes thin wrapper
- Remove _run_training_in_background helper from route

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Move create_videos_alignment logic to service

**Files:**
- Modify: `vidseq/services/alignment_service.py`
- Modify: `vidseq/api/routes/alignment.py`

**Step 1: Add create_videos_alignment method to AlignmentService**

In `alignment_service.py`, add this import at the top with the other imports:

```python
from vidseq.services.database_manager import DatabaseManager
```

Add this method to the `AlignmentService` class (after `create_alignment_training`):

```python
async def create_videos_alignment(self, project_path: Path) -> dict:
    """Apply alignment to all cropped videos.

    Creates aligned videos in <project>/aligned_videos/ folder.
    Starts processing in a background thread.

    Args:
        project_path: Path to the project folder

    Returns:
        Dict with status

    Raises:
        AlignmentTrainingError: If alignment already in progress or model not trained
    """
    from vidseq.services.exceptions import AlignmentTrainingError

    if self.is_applying():
        raise AlignmentTrainingError("Alignment already in progress")

    if not self.is_model_trained(project_path):
        raise AlignmentTrainingError("Model not trained yet")

    # Get project engine for sync operations
    db_manager = DatabaseManager.get_instance()
    project_engine = db_manager.get_project_engine(project_path)

    # Start alignment in background thread
    asyncio.create_task(
        asyncio.to_thread(
            self.apply_alignment_sync,
            project_path,
            project_engine,
        )
    )

    return {"status": "started"}
```

**Step 2: Refactor the route in alignment.py**

Replace the `create_videos_alignment` route function (lines 424-463) with:

```python
@router.post("/projects/{project_id}/videos/alignment")
async def create_videos_alignment(
    project_id: int,
    project_path: Path = Depends(get_project_folder),
):
    """Apply alignment to all cropped videos.

    Creates aligned videos in <project>/aligned_videos/ folder.
    Returns immediately after starting. Use the SSE stream endpoint
    (/alignment/apply/stream) to monitor progress.
    """
    service = AlignmentService.get_instance()
    return await service.create_videos_alignment(project_path)
```

**Step 3: Remove the helper function from alignment.py**

Delete the `_run_alignment_in_background` function (lines 415-421) - no longer needed.

**Step 4: Remove unused import from alignment.py**

Remove this import line:

```python
from vidseq.services.database_manager import DatabaseManager
```

**Step 5: Verify imports work**

Run: `uv run python -c "from vidseq.api.routes.alignment import router; print('Route OK')"`

Expected: `Route OK`

**Step 6: Commit**

```bash
git add vidseq/services/alignment_service.py vidseq/api/routes/alignment.py
git commit -m "$(cat <<'EOF'
refactor(alignment): move create_videos_alignment logic to service

- Add AlignmentService.create_videos_alignment() method
- Route becomes thin wrapper
- Remove _run_alignment_in_background helper from route
- Remove DatabaseManager import from route

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Clean up alignment.py imports and verify

**Files:**
- Modify: `vidseq/api/routes/alignment.py`

**Step 1: Review and clean imports**

After all refactoring, the imports section of alignment.py should be:

```python
"""API routes for egocentric alignment."""

import asyncio
import json
import logging
import mimetypes
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from vidseq.api.dependencies import get_project_folder, get_project_session, get_video
from vidseq.models.video import Video
from vidseq.services import alignment_service
from vidseq.services.alignment_service import (
    AlignmentService,
    heatmap_to_png,
    load_prediction,
    predictions_exist,
)
from vidseq.services.cropped_video_service import cropped_video_exists
```

Note what's removed:
- `from sqlalchemy import delete, select` - no longer needed
- `from vidseq.services.database_manager import DatabaseManager` - no longer needed
- `from vidseq.models.alignment_label import AlignmentLabel` - check if still needed (used in type hints? No - remove it)

**Step 2: Verify final state**

Run verification checks:

```bash
# Check no select/delete in routes
grep -n "select\|delete" vidseq/api/routes/alignment.py | grep -v "# " | grep -v '"""' || echo "PASS: No raw SQL in routes"

# Check no DatabaseManager in routes
grep -n "DatabaseManager" vidseq/api/routes/alignment.py && echo "FAIL" || echo "PASS: No DatabaseManager in routes"

# Verify imports
uv run python -c "from vidseq.api.routes.alignment import router; print('Route OK')"
```

**Step 3: Commit cleanup**

```bash
git add vidseq/api/routes/alignment.py
git commit -m "$(cat <<'EOF'
chore(alignment): clean up unused imports

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Verification Checklist

After all tasks, verify:

- [ ] `get_alignment_status` route has no `select(Video)` call
- [ ] `create_alignment_training` route has no `select(Video)` call
- [ ] `create_videos_alignment` route has no `DatabaseManager` access
- [ ] No `_run_*_in_background` helper functions in alignment.py
- [ ] All three routes are thin wrappers calling service methods
- [ ] `from sqlalchemy import delete, select` removed from alignment.py
- [ ] `from vidseq.services.database_manager import DatabaseManager` removed from alignment.py

---

## Summary

| Task | Route | Service Method | Lines Moved |
|------|-------|----------------|-------------|
| 1 | get_alignment_status | AlignmentService.get_alignment_status() | ~30 |
| 2 | create_alignment_training | AlignmentService.create_alignment_training() | ~45 |
| 3 | create_videos_alignment | AlignmentService.create_videos_alignment() | ~20 |
| 4 | (cleanup) | N/A | imports |

**Total:** ~95 lines of logic moved from routes to service.
