# Associated Videos Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add associated video support — allowing IR video segmentation to guide depth video segmentation via confidence-gated box prompts, with inline expand UI and a read-only review screen.

**Architecture:** Main videos (IR) map 1:1 to associated videos (depth) via `associated_with_id` FK on the Video model. A new `handle_propagate_with_associated` TCP command handler orchestrates the inference loop, calling thin public wrappers on StreamingSegmentor. The frontend adds inline expand on video cards and a new AssociatedVideoDetail screen.

**Tech Stack:** Python/FastAPI, SQLAlchemy, SQLite, h5py, PyTorch/SAM2, Vue 3/TypeScript, Pinia

**Spec:** `docs/superpowers/specs/2026-03-13-associated-videos-design.md`

---

## Chunk 1: Foundation — Data Model, Storage, Services

### Task 1: Video Model & Schema Changes

**Files:**
- Modify: `vidseq/models/video.py`
- Modify: `vidseq/schemas/video.py`

- [ ] **Step 1: Add columns to Video model**

In `vidseq/models/video.py`, add two new columns after the existing `p95_confidence` field:

```python
is_associated: Mapped[bool] = mapped_column(default=False)
associated_with_id: Mapped[int | None] = mapped_column(
    ForeignKey("videos.id"), default=None
)
```

Add `ForeignKey` to the existing sqlalchemy import line (e.g., `from sqlalchemy import String, Float, Integer, ForeignKey`).

- [ ] **Step 2: Update VideoResponse schema**

In `vidseq/schemas/video.py`, add three new fields to `VideoResponse` (the existing class uses `class Config: from_attributes = True` — keep that convention):

```python
class VideoResponse(BaseModel):
    id: int
    name: str
    path: str
    fps: float
    segmentation_status: str | None = None
    # New fields for associated video support
    is_associated: bool = False
    associated_with_id: int | None = None
    associated_video_id: int | None = None  # Populated by service layer, not from_attributes

    class Config:
        from_attributes = True
```

Note: `associated_video_id` is a reverse-lookup field populated by the service layer (see Task 5). Since `from_attributes=True` tries to read it from the model, we need a default of `None` so it doesn't fail when constructed from a plain Video ORM object.

- [ ] **Step 3: Commit**

```bash
git add vidseq/models/video.py vidseq/schemas/video.py
git commit -m "feat: add is_associated and associated_with_id columns to Video model"
```

---

### Task 2: Migration Script

**Files:**
- Create: `scripts/migrate_associated_videos.py`

- [ ] **Step 1: Write migration script**

```python
#!/usr/bin/env python3
"""Add associated video columns to existing project databases.

Usage: uv run python scripts/migrate_associated_videos.py /path/to/project
"""
import sqlite3
import sys
from pathlib import Path


def migrate(project_dir: Path) -> None:
    db_path = project_dir / "vidseq.db"
    if not db_path.exists():
        print(f"Error: {db_path} not found")
        sys.exit(1)

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Check existing columns
    cursor.execute("PRAGMA table_info(videos)")
    existing_cols = {row[1] for row in cursor.fetchall()}

    if "is_associated" not in existing_cols:
        cursor.execute(
            "ALTER TABLE videos ADD COLUMN is_associated INTEGER NOT NULL DEFAULT 0"
        )
        print("Added column: is_associated")
    else:
        print("Column is_associated already exists, skipping")

    if "associated_with_id" not in existing_cols:
        cursor.execute(
            "ALTER TABLE videos ADD COLUMN associated_with_id INTEGER DEFAULT NULL"
        )
        print("Added column: associated_with_id")
    else:
        print("Column associated_with_id already exists, skipping")

    conn.commit()
    conn.close()
    print("Migration complete.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: uv run python scripts/migrate_associated_videos.py /path/to/project")
        sys.exit(1)
    migrate(Path(sys.argv[1]))
```

- [ ] **Step 2: Commit**

```bash
git add scripts/migrate_associated_videos.py
git commit -m "feat: add migration script for associated video columns"
```

---

### Task 3: Array Storage — `is_associated` param

**Files:**
- Modify: `vidseq/services/array_storage.py`

- [ ] **Step 1: Add `is_associated` param to `create_video_segmentation_arrays()`**

Find the function signature (around line 252) and add the param:

```python
def create_video_segmentation_arrays(
    project_path: Path,
    video_id: int,
    num_frames: int,
    height: int,
    width: int,
    logits_size: int = 256,
    is_associated: bool = False,
) -> None:
```

Inside the function, wrap the tracker_logits and detector_masks creation blocks with `if not is_associated:`:

```python
    # 1. Tracker masks: tracker_masks.h5  (always created)
    tracker_masks_path = _construct_h5_path(project_path, video_id, "tracker_masks.h5")
    with open_h5_with_lock(tracker_masks_path, mode="w") as f:
        f.create_dataset(
            "data",
            shape=(num_frames, height, width),
            dtype=np.uint8,
            chunks=(1, height, width),
            fillvalue=0,
        )

    if not is_associated:
        # 2. Tracker logits: tracker_logits.h5  (not needed for associated videos)
        tracker_logits_path = _construct_h5_path(project_path, video_id, "tracker_logits.h5")
        with open_h5_with_lock(tracker_logits_path, mode="w") as f:
            f.create_dataset(
                "data",
                shape=(num_frames, logits_size, logits_size),
                dtype=np.float32,
                chunks=(1, logits_size, logits_size),
                fillvalue=0.0,
            )

        # 3. Detector: detector_masks.h5  (not needed for associated videos)
        detector_masks_path = _construct_h5_path(project_path, video_id, "detector_masks.h5")
        with open_h5_with_lock(detector_masks_path, mode="w") as f:
            f.create_dataset(
                "data",
                shape=(num_frames, height, width),
                dtype=np.uint8,
                chunks=(1, height, width),
                compression="gzip",
                fillvalue=0,
            )

    # 4. Final: final_masks.h5  (always created)
    final_masks_path = _construct_h5_path(project_path, video_id, "final_masks.h5")
    with open_h5_with_lock(final_masks_path, mode="w") as f:
        f.create_dataset(
            "data",
            shape=(num_frames, height, width),
            dtype=np.uint8,
            chunks=(1, height, width),
            fillvalue=0,
        )
```

- [ ] **Step 2: Add `is_associated` param to `reset_video_segmentation_arrays()`**

Find the function (around line 365) and add the param:

```python
def reset_video_segmentation_arrays(
    project_path: Path,
    video_id: int,
    num_frames: int,
    height: int,
    width: int,
    logits_size: int = 256,
    is_associated: bool = False,
) -> None:
```

Pass `is_associated` through to the create call:

```python
    delete_video_segmentation_arrays(project_path, video_id)
    create_video_segmentation_arrays(
        project_path, video_id, num_frames, height, width, logits_size,
        is_associated=is_associated,
    )
```

Note: `delete_video_segmentation_arrays` does NOT need the param — it checks file existence before unlinking and handles missing files gracefully.

- [ ] **Step 3: Verify no callers break**

Search for all callers of `create_video_segmentation_arrays` and `reset_video_segmentation_arrays`. They all use positional args or don't pass `is_associated`, so they default to `False` (existing behavior unchanged).

```bash
cd /n/groups/datta/john/projects/vidseq && grep -rn "create_video_segmentation_arrays\|reset_video_segmentation_arrays" vidseq/
```

- [ ] **Step 4: Commit**

```bash
git add vidseq/services/array_storage.py
git commit -m "feat: add is_associated param to array storage creation/reset"
```

---

### Task 4: Frame Data Service — `load_all_scores()`

**Files:**
- Modify: `vidseq/services/frame_data_service.py`

- [ ] **Step 1: Add `load_all_scores()` function**

Add this function near the other score-related functions (after `load_scores_in_range`):

```python
async def load_all_scores(
    session: AsyncSession,
    video_id: int,
) -> dict[int, float]:
    """Load all scores for a video as {frame_idx: score}.

    Returns all FrameData rows for the video as a dict. Frames without
    FrameData rows are absent from the dict. The caller should treat
    missing keys the same as scores <= 0 (below threshold).
    """
    result = await session.execute(
        select(FrameData.frame_idx, FrameData.score)
        .where(FrameData.video_id == video_id)
    )
    return {row.frame_idx: row.score for row in result.all()}
```

Make sure `select` and `FrameData` are already imported (they should be — check the existing imports at the top of the file).

- [ ] **Step 2: Commit**

```bash
git add vidseq/services/frame_data_service.py
git commit -m "feat: add load_all_scores function for associated video workflow"
```

---

### Task 5: Filesystem Route — `accept` filter

**Files:**
- Modify: `vidseq/api/routes/filesystem.py`

- [ ] **Step 1: Add `accept` query param**

The current `list_directory` endpoint takes only `path: str = Query(...)`. Add an optional `accept` param (preserve existing `Query()` usage):

```python
@router.get("/filesystem/list", response_model=list[DirectoryEntry])
async def list_directory(
    path: str = Query(..., description="Directory path to list"),
    accept: str | None = Query(None, description="Comma-separated file extensions to filter (e.g., '.json')"),
):
```

After the existing directory listing logic (where entries are built and sorted), add filtering before returning:

```python
    if accept:
        # Parse comma-separated extensions, normalize to lowercase with dot prefix
        accepted_exts = set()
        for ext in accept.split(","):
            ext = ext.strip().lower()
            if not ext.startswith("."):
                ext = "." + ext
            accepted_exts.add(ext)

        # Filter: keep directories (for navigation) + files matching extensions
        entries = [
            e for e in entries
            if e.is_directory or Path(e.name).suffix.lower() in accepted_exts
        ]

    return entries
```

- [ ] **Step 2: Commit**

```bash
git add vidseq/api/routes/filesystem.py
git commit -m "feat: add accept filter to filesystem list endpoint"
```

---

## Chunk 2: Video Service & Routes

### Task 6: Video Service — `get_all_videos()` filter + route enrichment

**Files:**
- Modify: `vidseq/services/video_service.py`
- Modify: `vidseq/api/routes/videos.py`

- [ ] **Step 1: Add `is_associated` filter to `get_all_videos()`**

Find `get_all_videos()` (around line 181). Add a `WHERE is_associated = False` filter. Keep returning `list[Video]` — do NOT change the return type (arhmm.py and other callers depend on ORM attribute access):

```python
async def get_all_videos(
    session: AsyncSession,
) -> list[Video]:
    """Get all main videos (excludes associated videos)."""
    result = await session.execute(
        select(Video)
        .where(Video.is_associated == False)
        .order_by(Video.id)
    )
    return list(result.scalars().all())
```

- [ ] **Step 2: Update the route handler to enrich with `associated_video_id`**

The `associated_video_id` reverse-lookup is populated in the route handler (not the service) since it is a presentation concern. In `vidseq/api/routes/videos.py`, find the handler that calls `get_all_videos()` and update it:

```python
@router.get("/projects/{project_id}/videos", response_model=list[VideoResponse])
async def list_videos(
    project_id: int,
    session: AsyncSession = Depends(get_project_session),
):
    videos = await video_service.get_all_videos(session)

    # Build associated_video_id lookup in a single query (avoids N+1)
    assoc_result = await session.execute(
        select(Video.associated_with_id, Video.id)
        .where(Video.is_associated == True)
    )
    assoc_lookup = {row[0]: row[1] for row in assoc_result.all()}

    # Enrich VideoResponse with the reverse-lookup field
    responses = []
    for video in videos:
        resp = VideoResponse.model_validate(video)
        resp.associated_video_id = assoc_lookup.get(video.id)
        responses.append(resp)

    return responses
```

This keeps `get_all_videos()` returning `list[Video]`, preserving all existing callers (arhmm.py, etc.) unchanged.

- [ ] **Step 3: Commit**

```bash
git add vidseq/services/video_service.py vidseq/api/routes/videos.py
git commit -m "feat: filter associated videos from get_all_videos, populate associated_video_id in route"
```

---

### Task 7: Video Service — Associated Video Ingestion

**Files:**
- Modify: `vidseq/services/video_service.py`

- [ ] **Step 1: Add `add_associated_videos()` function**

Add this function to `video_service.py`. Follow the pattern of the existing `add_videos()` function but with JSON validation:

```python
async def add_associated_videos(
    session: AsyncSession,
    project_path: Path,
    mapping: dict[str, str],
) -> list[Video]:
    """Add associated videos from a JSON mapping of main_path -> associated_path.

    Validates all pairs, creates Video rows with is_associated=True, and creates
    H5 files (tracker_masks + final_masks only, no detector_masks or logits).

    Raises ValueError on any validation failure (fail-fast).
    """
    # 1. Load all existing videos by path for lookup
    result = await session.execute(
        select(Video).where(Video.is_associated == False)
    )
    main_videos = {v.path: v for v in result.scalars().all()}

    # 2. Check for existing associated videos
    result = await session.execute(
        select(Video).where(Video.is_associated == True)
    )
    existing_assoc = {v.associated_with_id for v in result.scalars().all()}

    # 3. Validate all pairs — extract metadata once per video via get_video_metadata()
    seen_assoc_paths: set[str] = set()
    pairs: list[tuple[Video, VideoMetadata, str]] = []

    for main_path, assoc_path in mapping.items():
        # Main video must exist
        main_video = main_videos.get(main_path)
        if main_video is None:
            raise ValueError(f"Main video not found: {main_path}")

        # No duplicate associations
        if main_video.id in existing_assoc:
            raise ValueError(f"Main video already has an associated video: {main_path}")

        # No duplicate associated paths
        if assoc_path in seen_assoc_paths:
            raise ValueError(f"Duplicate associated video path: {assoc_path}")
        seen_assoc_paths.add(assoc_path)

        # Extract metadata (validates existence, readability, fps/frames/dims > 0)
        try:
            meta = get_video_metadata(assoc_path)
        except VideoFileNotFoundError:
            raise ValueError(f"Associated video file not found: {assoc_path}")
        except VideoFileInvalidError as e:
            raise ValueError(f"Cannot open associated video {assoc_path}: {e}")

        # Frame count must match
        if meta.num_frames != main_video.num_frames:
            raise ValueError(
                f"Frame count mismatch for {assoc_path}: "
                f"associated has {meta.num_frames}, main has {main_video.num_frames}"
            )

        pairs.append((main_video, meta, assoc_path))

    # 4. All validation passed — create Video rows (reuse metadata from validation)
    created: list[Video] = []
    for main_video, meta, assoc_path in pairs:
        video = Video(
            name=Path(assoc_path).name,
            path=assoc_path,
            fps=meta.fps,
            height=meta.height,
            width=meta.width,
            num_frames=meta.num_frames,
            is_associated=True,
            associated_with_id=main_video.id,
        )
        session.add(video)
        created.append(video)

    await session.commit()

    # 5. Create H5 files (need video.id from commit)
    for video in created:
        await session.refresh(video)
        create_video_segmentation_arrays(
            project_path=project_path,
            video_id=video.id,
            num_frames=video.num_frames,
            height=video.height,
            width=video.width,
            is_associated=True,
        )

    return created
```

Add `from vidseq.services.array_storage import create_video_segmentation_arrays` to the imports if not already present. `get_video_metadata`, `VideoMetadata`, `VideoFileNotFoundError`, and `VideoFileInvalidError` are already defined in this file.

- [ ] **Step 2: Commit**

```bash
git add vidseq/services/video_service.py
git commit -m "feat: add associated video ingestion with fail-fast validation"
```

---

### Task 8: Video Service — Cascade Delete & Reset

**Files:**
- Modify: `vidseq/services/video_service.py`

- [ ] **Step 1: Update `delete_videos()` to cascade to associated videos**

Find `delete_videos()`. At the start of the function (after fetching the requested videos), add:

```python
    # Find associated videos for cascade deletion
    assoc_result = await session.execute(
        select(Video).where(Video.associated_with_id.in_(video_ids))
    )
    assoc_videos = list(assoc_result.scalars().all())
    assoc_ids = [v.id for v in assoc_videos]

    # Merge associated video IDs into the deletion batch
    all_video_ids = list(video_ids) + assoc_ids
    all_videos = videos + assoc_videos
```

Then replace **every** subsequent reference to `video_ids` with `all_video_ids` and `videos` with `all_videos` for the rest of the function — this covers closing sessions, deleting ConditioningFrame/FrameData/AlignmentLabel DB records, deleting files via `shutil.rmtree`, and deleting Video DB rows. Associated videos will be closed/cleaned up alongside the main videos they belong to.

- [ ] **Step 2: Update `reset_video()` to cascade to associated video**

In `reset_video()`, after the main video's segmentation has been reset (after `reset_video_segmentation_arrays` call and FrameData deletion), add:

```python
    # Cascade: reset associated video's co-segmentation if it exists
    assoc_result = await session.execute(
        select(Video).where(
            Video.associated_with_id == video.id,
            Video.segmentation_status == "segmented",
        )
    )
    assoc_video = assoc_result.scalar_one_or_none()
    if assoc_video is not None:
        reset_video_segmentation_arrays(
            project_path=project_path,
            video_id=assoc_video.id,
            num_frames=assoc_video.num_frames,
            height=assoc_video.height,
            width=assoc_video.width,
            is_associated=True,
        )
        await session.execute(
            delete(FrameData).where(FrameData.video_id == assoc_video.id)
        )
        assoc_video.segmentation_status = None
```

- [ ] **Step 3: Update `delete_videos_segmentation()` to cascade**

In `delete_videos_segmentation()`, inside the `for video in videos:` loop, after the main video's reset logic and before moving to the next video, add the same cascade:

```python
        # Cascade: reset associated video's co-segmentation
        assoc_result = await session.execute(
            select(Video).where(
                Video.associated_with_id == video.id,
                Video.segmentation_status == "segmented",
            )
        )
        assoc_video = assoc_result.scalar_one_or_none()
        if assoc_video is not None:
            reset_video_segmentation_arrays(
                project_path=project_path,
                video_id=assoc_video.id,
                num_frames=assoc_video.num_frames,
                height=assoc_video.height,
                width=assoc_video.width,
                is_associated=True,
            )
            await session.execute(
                delete(FrameData).where(FrameData.video_id == assoc_video.id)
            )
            assoc_video.segmentation_status = None
```

Note: `project_path` is available in the function scope — check the existing function signature to confirm the variable name matches.

- [ ] **Step 3: Commit**

```bash
git add vidseq/services/video_service.py
git commit -m "feat: cascade delete and segmentation reset to associated videos"
```

---

### Task 9: Video Routes — Associated Video Endpoints

**Files:**
- Modify: `vidseq/api/routes/videos.py`

- [ ] **Step 1: Add GET endpoint for associated video**

```python
@router.get(
    "/projects/{project_id}/videos/{video_id}/associated",
    response_model=VideoResponse,
)
async def get_associated_video(
    project_id: int,
    video_id: int,
    session: AsyncSession = Depends(get_project_session),
):
    """Get the associated video for a main video, or 404 if none."""
    result = await session.execute(
        select(Video).where(Video.associated_with_id == video_id)
    )
    assoc = result.scalar_one_or_none()
    if assoc is None:
        raise HTTPException(status_code=404, detail="No associated video found")
    return VideoResponse.model_validate(assoc)
```

- [ ] **Step 2: Add POST endpoint for JSON ingestion**

The frontend sends a `json_path` (server-side path selected via FilePickerModal). The backend reads and validates the JSON file:

```python
from pydantic import BaseModel

class AssociatedVideoRequest(BaseModel):
    json_path: str

@router.post(
    "/projects/{project_id}/videos/associated",
    response_model=list[VideoResponse],
    status_code=201,
)
async def add_associated_videos(
    project_id: int,
    body: AssociatedVideoRequest,
    session: AsyncSession = Depends(get_project_session),
    project_path: Path = Depends(get_project_folder),
):
    """Add associated videos from a JSON mapping file on disk."""
    import json

    json_file = Path(body.json_path)
    if not json_file.exists():
        raise HTTPException(status_code=400, detail=f"File not found: {body.json_path}")

    try:
        mapping = json.loads(json_file.read_text())
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

    if not isinstance(mapping, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in mapping.items()
    ):
        raise HTTPException(
            status_code=400,
            detail="Expected flat {string: string} mapping",
        )

    try:
        videos = await video_service.add_associated_videos(
            session=session,
            project_path=project_path,
            mapping=mapping,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return [VideoResponse.model_validate(v) for v in videos]
```

- [ ] **Step 3: Commit**

```bash
git add vidseq/api/routes/videos.py
git commit -m "feat: add associated video GET and POST endpoints"
```

---

## Chunk 3: Segmentor & Inference Pipeline

### Task 10: StreamingSegmentor — `store_as_cond` and Public Wrappers

**Files:**
- Modify: `vidseq/services/segmentation_model/streaming_segmentor.py`

- [ ] **Step 1: Add `store_as_cond` param to `_propagate_single_frame()`**

Find `_propagate_single_frame()` (around line 1145). Add the parameter:

```python
    def _propagate_single_frame(
        self,
        video_id: str,
        frame_idx: int,
        frame: np.ndarray,
        mask_prompt: np.ndarray | None = None,
        box_prompt: tuple | None = None,
        store_as_cond: bool | None = None,
    ) -> tuple[np.ndarray, np.ndarray, float]:
```

Find the storage logic (around line 1263) where it decides cond vs non-cond:

```python
        if is_prompted:
            output_dict["cond_frame_outputs"][frame_idx] = self._make_compact_output(current_out)
        else:
            output_dict["non_cond_frame_outputs"][frame_idx] = self._make_compact_output(current_out)
```

Replace the entire storage block (lines ~1262-1275) with:

```python
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
```

**Important:** `is_init_cond_frame` (passed to SAM2's `track_step`) must remain based on `is_prompted`, NOT `store_as_cond`. This ensures the box prompt always guides inference regardless of where the result is stored. Do NOT change the `is_init_cond_frame` assignment.

**Behavior note:** When `store_as_cond=None` (default for all existing callers), this code produces identical behavior to the original — the `is_prompted` branch stores in cond but does NOT add to `cond_frame_indices` (that's handled by `add_point_prompt` separately). The `cond_frame_indices.add` only fires when `store_as_cond` is explicitly set by the new associated video workflow.

- [ ] **Step 2: Add five public wrapper methods**

Add these methods to the `StreamingSegmentor` class, after the existing `propagate()` method:

```python
    def propagate_with_box(
        self,
        video_id: str,
        frame_idx: int,
        frame: np.ndarray,
        box_prompt: tuple,
        add_as_conditioning: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Propagate with a box prompt, controlling conditioning storage.

        Args:
            video_id: The video identifier.
            frame_idx: Frame index to propagate to.
            frame: BGR uint8 frame (H, W, 3).
            box_prompt: Bounding box as (x1, y1, x2, y2).
            add_as_conditioning: If True, store in cond_frame_outputs (permanent).
                If False, box guides prediction but result goes to non_cond_frame_outputs.

        Returns:
            Tuple of (mask, logits, score).
        """
        return self._propagate_single_frame(
            video_id, frame_idx, frame,
            box_prompt=box_prompt,
            store_as_cond=add_as_conditioning,
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
```

- [ ] **Step 3: Verify existing callers are unaffected**

The `store_as_cond=None` default preserves existing behavior. All existing callers of `_propagate_single_frame` don't pass `store_as_cond`, so they get the default.

```bash
grep -n "_propagate_single_frame" vidseq/services/segmentation_model/streaming_segmentor.py
```

- [ ] **Step 4: Commit**

```bash
git add vidseq/services/segmentation_model/streaming_segmentor.py
git commit -m "feat: add store_as_cond param and five public wrappers on StreamingSegmentor"
```

---

### Task 11: Command Handler — `handle_propagate_with_associated()`

**Files:**
- Modify: `vidseq/services/segmentation_commands.py`

- [ ] **Step 1: Add the handler function**

Add this function after `handle_propagate_with_detector()`:

```python
def handle_propagate_with_associated(
    params: dict,
    segmentor: StreamingSegmentor,
    response_callback,
) -> dict:
    """Propagate associated video segmentation using main video's bbox prompts.

    The main video's scores determine when to provide box prompts from its masks.
    Uses sparse conditioning + LRU eviction for memory management on long videos.
    """
    from collections import deque

    video_id = params["video_id"]
    main_video_id = params["main_video_id"]
    project_path = Path(params["project_path"])
    confidence_threshold = params["confidence_threshold"]
    main_height = params["main_height"]
    main_width = params["main_width"]
    main_video_scores = params["main_video_scores"]  # {str(frame_idx): float}
    # Note: JSON keys are strings, convert to int
    main_scores = {int(k): v for k, v in main_video_scores.items()}
    num_frames = params["num_frames"]
    cond_frame_interval = params.get("cond_frame_interval", 50)
    max_cond_frames = params.get("max_cond_frames", 32)

    if video_id not in _video_resources:
        raise RuntimeError(f"No session for video {video_id}")

    resources = _video_resources[video_id]

    # Check that at least one frame exceeds threshold
    has_valid = any(
        s > confidence_threshold
        for s in main_scores.values()
        if s > 0
    )
    if not has_valid:
        raise RuntimeError(
            "Main video has no segmentation data above confidence threshold"
        )

    # Compute bbox rescaling factors
    assoc_height, assoc_width = resources.height, resources.width
    x_scale = assoc_width / main_width if main_width != assoc_width else 1.0
    y_scale = assoc_height / main_height if main_height != assoc_height else 1.0
    needs_rescale = x_scale != 1.0 or y_scale != 1.0

    # LRU tracking for conditioning frames
    cond_lru: deque[int] = deque()
    prompted_count = 0  # Counts prompted frames for interval logic

    depth_scores: dict[int, float] = {}

    # Open H5 files: main video read-only, depth video read-write
    # This direct H5 access is a deliberate exception — see spec for justification
    with tracker_masks(project_path, main_video_id, "r") as main_masks, \
         tracker_masks(project_path, video_id, "a") as depth_trk, \
         final_masks(project_path, video_id, "a") as depth_fin:

        frame_source = resources.frame_source

        # 1. Find first frame above threshold
        start_frame = None
        for idx in range(num_frames):
            score = main_scores.get(idx, -1.0)
            if score > 0 and score > confidence_threshold:
                start_frame = idx
                break

        if start_frame is None:
            raise RuntimeError(
                "Main video has no segmentation data above confidence threshold"
            )

        # 2. Get bbox from main video's mask for the start frame
        main_mask = np.asarray(main_masks[start_frame])
        bbox = compute_bbox_from_mask(main_mask)
        if bbox is None:
            raise RuntimeError(f"Main video mask at frame {start_frame} is empty")

        if needs_rescale:
            bbox = np.array([
                bbox[0] * x_scale, bbox[1] * y_scale,
                bbox[2] * x_scale, bbox[3] * y_scale,
            ], dtype=np.float32)

        # Initialize with box prompt (always conditioning)
        frame = frame_source[start_frame]
        mask, _, score = segmentor.propagate_with_box(
            str(video_id), start_frame, frame,
            box_prompt=tuple(bbox),
            add_as_conditioning=True,
        )
        depth_trk[start_frame] = mask
        depth_fin[start_frame] = mask
        depth_scores[start_frame] = score
        cond_lru.append(start_frame)
        prompted_count = 1
        last_anchor_frame = start_frame
        was_below_threshold = False

        if response_callback:
            response_callback({
                "type": "progress",
                "frame_idx": start_frame,
                "total": num_frames,
            })

        # 3. Main loop: propagate forward
        for frame_idx in range(start_frame + 1, num_frames):
            frame = frame_source[frame_idx]
            main_score = main_scores.get(frame_idx, -1.0)
            is_above = main_score > 0 and main_score >= confidence_threshold

            if is_above:
                # Get bbox from main video
                main_mask = np.asarray(main_masks[frame_idx])
                bbox = compute_bbox_from_mask(main_mask)

                if bbox is not None:
                    if needs_rescale:
                        bbox = np.array([
                            bbox[0] * x_scale, bbox[1] * y_scale,
                            bbox[2] * x_scale, bbox[3] * y_scale,
                        ], dtype=np.float32)

                    # 4. Check for confidence recovery (backtrack)
                    if was_below_threshold:
                        # Re-condition with new anchor
                        mask, _, score = segmentor.propagate_with_box(
                            str(video_id), frame_idx, frame,
                            box_prompt=tuple(bbox),
                            add_as_conditioning=True,
                        )
                        depth_fin[frame_idx] = mask
                        depth_scores[frame_idx] = score
                        cond_lru.append(frame_idx)

                        # Backtrack gap frames
                        if last_anchor_frame + 1 <= frame_idx - 1:
                            segmentor.backtrack_reprop(
                                str(video_id), frame_source, depth_fin,
                                last_anchor_frame + 1, frame_idx - 1,
                                scores=depth_scores,
                            )

                        was_below_threshold = False
                    else:
                        # Normal prompted frame
                        prompted_count += 1
                        add_cond = (prompted_count % cond_frame_interval) == 0
                        mask, _, score = segmentor.propagate_with_box(
                            str(video_id), frame_idx, frame,
                            box_prompt=tuple(bbox),
                            add_as_conditioning=add_cond,
                        )

                        if add_cond:
                            cond_lru.append(frame_idx)
                            # LRU eviction
                            while len(cond_lru) > max_cond_frames:
                                oldest = cond_lru.popleft()
                                segmentor.evict_conditioning_frame(
                                    str(video_id), oldest
                                )

                    depth_trk[frame_idx] = mask
                    depth_fin[frame_idx] = mask
                    depth_scores[frame_idx] = score
                    last_anchor_frame = frame_idx
                else:
                    # Main mask is empty despite score above threshold
                    mask, _, score = segmentor.propagate_frame(
                        str(video_id), frame_idx, frame
                    )
                    depth_trk[frame_idx] = mask
                    depth_fin[frame_idx] = mask
                    depth_scores[frame_idx] = score
                    was_below_threshold = True
            else:
                # Coast: propagate without prompt
                mask, _, score = segmentor.propagate_frame(
                    str(video_id), frame_idx, frame
                )
                depth_trk[frame_idx] = mask
                depth_fin[frame_idx] = mask
                depth_scores[frame_idx] = score
                was_below_threshold = True

            # Progress callback
            if response_callback and (frame_idx % 50 == 0 or frame_idx == num_frames - 1):
                response_callback({
                    "type": "progress",
                    "frame_idx": frame_idx,
                    "total": num_frames,
                })

    return {
        "type": "propagate_with_associated_result",
        "status": "ok",
        "scores": [[idx, s] for idx, s in depth_scores.items()],
    }
```

Make sure these imports are at the top of `segmentation_commands.py`:
- `from vidseq.services.array_storage import tracker_masks, final_masks, compute_bbox_from_mask`
- `import numpy as np` (should already be there)

- [ ] **Step 2: Commit**

```bash
git add vidseq/services/segmentation_commands.py
git commit -m "feat: add handle_propagate_with_associated command handler"
```

---

### Task 12: TCP Server & Client Integration

**Files:**
- Modify: `vidseq/services/segmentation_tcp_server.py`
- Modify: `vidseq/services/segmentation_tcp_client.py`

- [ ] **Step 1: Register command in TCP server dispatcher**

In `segmentation_tcp_server.py`, find the `_handle_command()` if/elif chain. Add a new branch before the `else` clause:

```python
        elif cmd_type == "propagate_with_associated":
            if self._segmentor is None:
                raise RuntimeError("Model not loaded")
            result = handle_propagate_with_associated(
                cmd,
                self._segmentor,
                response_callback,
            )
```

Add the import at the top:
```python
from vidseq.services.segmentation_commands import handle_propagate_with_associated
```

(Add it alongside the other handler imports.)

- [ ] **Step 2: Add client method for associated propagation**

In `segmentation_tcp_client.py`, add a method to the client class:

```python
    def propagate_with_associated(
        self,
        video_id: int,
        main_video_id: int,
        project_path: Path,
        num_frames: int,
        confidence_threshold: float,
        main_height: int,
        main_width: int,
        main_video_scores: dict[int, float],
        cond_frame_interval: int = 50,
        max_cond_frames: int = 32,
    ) -> dict:
        """Run associated video propagation via TCP."""
        return self._send_streaming({
            "type": "propagate_with_associated",
            "video_id": video_id,
            "main_video_id": main_video_id,
            "project_path": str(project_path),
            "num_frames": num_frames,
            "confidence_threshold": confidence_threshold,
            "main_height": main_height,
            "main_width": main_width,
            "main_video_scores": main_video_scores,
            "cond_frame_interval": cond_frame_interval,
            "max_cond_frames": max_cond_frames,
        }, timeout=3600.0)  # 1 hour for long videos
```

- [ ] **Step 3: Add module-level wrapper function**

Task 13 calls `segmentation_tcp_client.propagate_with_associated(...)` at the module level. The existing pattern (e.g., `init_session`, `close_session`) uses module-level functions that delegate to the singleton. Add at the bottom of `segmentation_tcp_client.py`, alongside the other module-level wrappers:

```python
def propagate_with_associated(
    video_id: int,
    main_video_id: int,
    project_path: Path,
    num_frames: int,
    confidence_threshold: float,
    main_height: int,
    main_width: int,
    main_video_scores: dict[int, float],
    cond_frame_interval: int = 50,
    max_cond_frames: int = 32,
) -> dict:
    """Module-level wrapper for associated video propagation."""
    return SegmentationService.get_instance().propagate_with_associated(
        video_id=video_id,
        main_video_id=main_video_id,
        project_path=project_path,
        num_frames=num_frames,
        confidence_threshold=confidence_threshold,
        main_height=main_height,
        main_width=main_width,
        main_video_scores=main_video_scores,
        cond_frame_interval=cond_frame_interval,
        max_cond_frames=max_cond_frames,
    )
```

- [ ] **Step 4: Commit**

```bash
git add vidseq/services/segmentation_tcp_server.py vidseq/services/segmentation_tcp_client.py
git commit -m "feat: register propagate_with_associated in TCP server and client"
```

---

### Task 13: Segmentation Service — Batch Co-Segmentation Orchestration

**Files:**
- Modify: `vidseq/services/segmentation_service.py`

- [ ] **Step 1: Add `co_segment_videos()` function**

Follow the pattern of `segment_all_videos()`:

```python
async def co_segment_videos(
    session: AsyncSession,
    project_id: int,
    project_path: Path,
    video_ids: list[int],
    confidence_threshold: float = 0.9,
) -> None:
    """Batch co-segmentation of associated videos.

    For each main video:
    1. Load its scores from DB
    2. Find its associated video
    3. Init session for the associated video
    4. Run propagate_with_associated
    5. Save results to DB
    6. Close session

    Continues on per-video errors.
    """
    from vidseq.services import frame_data_service

    # Fetch main videos
    result = await session.execute(
        select(Video).where(Video.id.in_(video_ids)).order_by(Video.id)
    )
    main_videos = list(result.scalars().all())
    if not main_videos:
        raise ValueError("No videos found for the given IDs")

    # Fetch associated videos
    result = await session.execute(
        select(Video).where(
            Video.associated_with_id.in_(video_ids),
            Video.is_associated == True,
        )
    )
    assoc_by_main = {v.associated_with_id: v for v in result.scalars().all()}

    for main_video in main_videos:
        assoc_video = assoc_by_main.get(main_video.id)
        if assoc_video is None:
            print(f"[Co-Segment] Video {main_video.id} has no associated video, skipping")
            continue

        try:
            # Load main video scores
            main_scores = await frame_data_service.load_all_scores(
                session, main_video.id
            )

            # Init session for associated video
            segmentation_tcp_client.init_session(
                project_id=project_id,
                video_id=assoc_video.id,
                video_path=Path(assoc_video.path),
                project_path=project_path,
                num_frames=assoc_video.num_frames,
                height=assoc_video.height,
                width=assoc_video.width,
                cond_frame_indices=[],
            )

            # Run propagation
            result = segmentation_tcp_client.propagate_with_associated(
                video_id=assoc_video.id,
                main_video_id=main_video.id,
                project_path=project_path,
                num_frames=assoc_video.num_frames,
                confidence_threshold=confidence_threshold,
                main_height=main_video.height,
                main_width=main_video.width,
                main_video_scores=main_scores,
            )

            if result.get("status") == "ok":
                # Save scores to DB
                scores = result.get("scores", [])
                if scores:
                    await frame_data_service.save_scores_batch(
                        session, assoc_video.id, scores
                    )

                # Update mask flags
                processed_frames = [s[0] for s in scores]
                await frame_data_service.set_has_tracker_mask(
                    session, assoc_video.id, processed_frames, True
                )
                await frame_data_service.set_has_final_mask(
                    session, assoc_video.id, processed_frames, True
                )

                assoc_video.segmentation_status = "segmented"
                print(f"[Co-Segment] Video {main_video.id} -> {assoc_video.id} complete")
            else:
                print(f"[Co-Segment] Failed for video {main_video.id}: {result.get('error')}")

            # Close session
            segmentation_tcp_client.close_session(project_id, assoc_video.id)

        except Exception as e:
            print(f"[Co-Segment] Error for video {main_video.id}: {e}")
            try:
                segmentation_tcp_client.close_session(project_id, assoc_video.id)
            except Exception:
                pass

    await session.commit()
```

Add the necessary imports at the top of `segmentation_service.py` if not already present:

```python
from vidseq.models.video import Video
from sqlalchemy import select
from vidseq.services import frame_data_service, segmentation_tcp_client
```

Check the existing file — `Video`, `select`, and `segmentation_tcp_client` may already be imported.

- [ ] **Step 2: Commit**

```bash
git add vidseq/services/segmentation_service.py
git commit -m "feat: add co_segment_videos batch orchestration"
```

---

### Task 14: Segmentation Route — Co-Segmentation Endpoint

**Files:**
- Modify or create: `vidseq/api/routes/segmentation/sessions.py` (or appropriate file in segmentation/)

- [ ] **Step 1: Add the endpoint**

Find the file that contains `create_videos_segmentation` (the existing batch segmentation endpoint). Add the new endpoint nearby:

Define a Pydantic request model (add to the same file or to `vidseq/api/schemas.py` following the existing `VideoSelectionRequest` pattern):

```python
from pydantic import BaseModel

class CoSegmentationRequest(BaseModel):
    video_ids: list[int]
    confidence_threshold: float = 0.9

@router.post("/projects/{project_id}/videos/associated/segmentation")
async def co_segment_associated_videos(
    project_id: int,
    request: CoSegmentationRequest,
    session: AsyncSession = Depends(get_project_session),
    project_path: Path = Depends(get_project_folder),
):
    """Start co-segmentation for associated videos of selected main videos."""
    if not request.video_ids:
        raise HTTPException(status_code=400, detail="No video IDs provided")

    try:
        await segmentation_service.co_segment_videos(
            session=session,
            project_id=project_id,
            project_path=project_path,
            video_ids=request.video_ids,
            confidence_threshold=request.confidence_threshold,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"status": "ok"}
```

**Important:** This endpoint must be registered BEFORE the `/projects/{project_id}/videos/associated` POST endpoint in the route ordering, since FastAPI matches routes by specificity. Since this is in a different router (segmentation/ vs videos), this is handled naturally.

- [ ] **Step 2: Commit**

```bash
git add vidseq/api/routes/segmentation/
git commit -m "feat: add co-segmentation endpoint for associated videos"
```

---

## Chunk 4: Frontend

### Task 15: Frontend API Functions

**Files:**
- Modify: `frontend/src/services/api.ts`

- [ ] **Step 1: Update Video interface**

Add new fields to the existing `Video` interface:

```typescript
export interface Video {
    id: number
    name: string
    path: string
    fps: number
    segmentation_status: 'in_progress' | 'segmented' | null
    min_confidence?: number
    p50_confidence?: number
    p95_confidence?: number
    // Associated video fields
    is_associated: boolean
    associated_with_id: number | null
    associated_video_id: number | null
}
```

- [ ] **Step 2: Add new API functions**

```typescript
export async function getAssociatedVideo(
    projectId: number,
    videoId: number,
): Promise<Video> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/${videoId}/associated`,
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to get associated video'))
    }
    return response.json()
}

export async function addAssociatedVideos(
    projectId: number,
    jsonPath: string,
): Promise<Video[]> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/associated`,
        {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ json_path: jsonPath }),
        },
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to add associated videos'))
    }
    return response.json()
}

export async function coSegmentVideos(
    projectId: number,
    videoIds: number[],
    confidenceThreshold: number = 0.9,
): Promise<void> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/associated/segmentation`,
        {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                video_ids: videoIds,
                confidence_threshold: confidenceThreshold,
            }),
        },
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to co-segment videos'))
    }
}
```

- [ ] **Step 3: Commit**

```bash
cd frontend && git add src/services/api.ts && cd ..
git commit -m "feat: add associated video API functions"
```

---

### Task 16: FilePickerModal — `accept` and `singleSelect` Props

**Files:**
- Modify: `frontend/src/components/FilePickerModal.vue`

- [ ] **Step 1: Add props**

Update the props definition:

```typescript
const props = defineProps<{
    initialPath?: string
    accept?: string[]
    singleSelect?: boolean
}>()
```

- [ ] **Step 2: Pass `accept` to backend via `getDirectoryListing`**

The component uses `getDirectoryListing(path)` from `api.ts`. First, update `getDirectoryListing` in `api.ts` to accept an optional `accept` param:

```typescript
export async function getDirectoryListing(
    path: string,
    accept?: string[],
): Promise<DirectoryEntry[]> {
    let url = `${API_BASE}/filesystem/list?path=${encodeURIComponent(path)}`
    if (accept && accept.length > 0) {
        url += `&accept=${encodeURIComponent(accept.join(','))}`
    }
    const response = await fetch(url)
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to list directory'))
    }
    return response.json()
}
```

Then in `FilePickerModal.vue`, pass `props.accept` through in the `loadDirectory` call:

```typescript
entries.value = await getDirectoryListing(path, props.accept)
```

- [ ] **Step 3: Implement `singleSelect`**

Find the `handleEntrySelect` function (line 58). When `singleSelect` is true, replace selection instead of toggling. Note: `selectedPaths` is a `string[]` (not a Set):

```typescript
const handleEntrySelect = (path: string) => {
    if (props.singleSelect) {
        selectedPaths.value = [path]
    } else {
        // existing toggle logic
        const index = selectedPaths.value.indexOf(path)
        if (index === -1) {
            selectedPaths.value.push(path)
        } else {
            selectedPaths.value.splice(index, 1)
        }
    }
}
```

- [ ] **Step 4: Commit**

```bash
cd frontend && git add src/components/FilePickerModal.vue && cd ..
git commit -m "feat: add accept and singleSelect props to FilePickerModal"
```

---

### Task 17: VideoPipeline — Expand/Collapse, Sidebar Buttons

**Files:**
- Modify: `frontend/src/components/VideoPipeline.vue`

- [ ] **Step 1: Add imports and state**

Add to the imports:

```typescript
import { addAssociatedVideos, coSegmentVideos } from '@/services/api'
```

Add state variables:

```typescript
// Associated videos
const expandedVideoIds = ref(new Set<number>())
const isCoSegmenting = ref(false)
const confidenceThreshold = ref(0.9)
const showAssociatedFilePicker = ref(false)
const isAddingAssociated = ref(false)

const toggleExpand = (videoId: number) => {
    const next = new Set(expandedVideoIds.value)
    if (next.has(videoId)) {
        next.delete(videoId)
    } else {
        next.add(videoId)
    }
    expandedVideoIds.value = next
}

// Selected videos that have associated videos
const selectedWithAssociated = computed(() =>
    videos.value.filter(
        v => selectedVideoIds.value.has(v.id) && v.associated_video_id != null
    )
)
```

- [ ] **Step 2: Add "Add Associated Videos" button in sidebar**

In the template, add a new sidebar section after the existing ones:

```html
<h4 class="sidebar-section-title">Associated Videos</h4>
<button
    class="sidebar-button"
    @click="showAssociatedFilePicker = true"
    :disabled="isAddingAssociated"
>
    {{ isAddingAssociated ? 'Adding...' : 'Add Associated Videos' }}
</button>
```

- [ ] **Step 3: Add "Co-Segment" button with confidence threshold input**

```html
<div class="co-segment-controls">
    <button
        class="sidebar-button"
        @click="handleCoSegment"
        :disabled="isCoSegmenting || selectedWithAssociated.length === 0"
    >
        {{ isCoSegmenting ? 'Co-Segmenting...' : `Co-Segment ${selectedWithAssociated.length} Videos` }}
    </button>
    <label class="threshold-label">
        Confidence:
        <input
            type="number"
            v-model.number="confidenceThreshold"
            min="0"
            max="1"
            step="0.05"
            class="threshold-input"
        />
    </label>
</div>
```

- [ ] **Step 4: Add handlers**

Note: `loadVideos()` is the existing function name in VideoPipeline for refreshing the video list — check the actual function name and use it.

```typescript
const handleCoSegment = async () => {
    if (!projectId.value || isCoSegmenting.value) return
    isCoSegmenting.value = true
    try {
        const ids = selectedWithAssociated.value.map(v => v.id)
        await coSegmentVideos(projectId.value, ids, confidenceThreshold.value)
        await loadVideos()
    } catch (e: any) {
        alert(e.message || 'Co-segmentation failed')
    } finally {
        isCoSegmenting.value = false
    }
}

const handleAssociatedFileSelected = async (paths: string[]) => {
    showAssociatedFilePicker.value = false
    if (!projectId.value || paths.length === 0) return

    isAddingAssociated.value = true
    try {
        // Pass the JSON file path to the backend, which reads + validates it
        await addAssociatedVideos(projectId.value, paths[0])
        await loadVideos()
    } catch (e: any) {
        alert(e.message || 'Failed to add associated videos')
    } finally {
        isAddingAssociated.value = false
    }
}
```

- [ ] **Step 5: Add expand/collapse in video cards**

In the video card template, the existing `.video-item` uses `display: flex; align-items: center` for a single row. The expanded section must appear below the row. Restructure the card with a row wrapper:

```html
<div
    v-for="video in sortedVideos"
    :key="video.id"
    class="video-item"
    :class="[getStatusClass(video), { selected: selectedVideoIds.has(video.id) }]"
>
    <!-- Wrap existing row content -->
    <div class="video-item-row">
        <input type="checkbox" ... />
        <div class="video-info">...</div>
        <div class="video-actions">
            <!-- existing buttons -->

            <!-- Associated video expand toggle -->
            <button
                v-if="video.associated_video_id"
                class="expand-toggle"
                @click.stop="toggleExpand(video.id)"
            >
                {{ expandedVideoIds.has(video.id) ? '▾ 1 associated' : '▸ 1 associated' }}
            </button>
        </div>
    </div>

    <!-- Expanded associated video info (below the row) -->
    <div
        v-if="video.associated_video_id && expandedVideoIds.has(video.id)"
        class="associated-video-expanded"
    >
        <router-link
            :to="{ name: 'associatedVideo', params: { id: projectId, videoId: video.id } }"
            class="associated-link"
        >
            View Associated Video →
        </router-link>
    </div>
</div>
```

**CSS change:** Move the existing `.video-item` row flex styles to `.video-item-row`, and make `.video-item` use `flex-direction: column`:

```css
.video-item {
    display: flex;
    flex-direction: column;
}

.video-item-row {
    display: flex;
    align-items: center;
    /* move existing .video-item padding/gap styles here */
}
```

- [ ] **Step 6: Add FilePickerModal for associated videos**

```html
<FilePickerModal
    v-if="showAssociatedFilePicker"
    :initial-path="project?.path"
    :accept="['.json']"
    :single-select="true"
    @files-selected="handleAssociatedFileSelected"
    @cancel="showAssociatedFilePicker = false"
/>
```

- [ ] **Step 7: Add CSS**

```css
.associated-video-expanded {
    padding: 8px 16px 8px 40px;
    background: var(--bg-secondary, #f5f5f5);
    border-top: 1px solid var(--border-color, #e0e0e0);
}

.expand-toggle {
    background: none;
    border: none;
    cursor: pointer;
    font-size: 0.85em;
    color: var(--text-secondary);
    padding: 2px 6px;
}

.co-segment-controls {
    display: flex;
    flex-direction: column;
    gap: 6px;
}

.threshold-label {
    font-size: 0.85em;
    display: flex;
    align-items: center;
    gap: 6px;
}

.threshold-input {
    width: 60px;
    padding: 2px 4px;
}
```

- [ ] **Step 8: Commit**

```bash
cd frontend && git add src/components/VideoPipeline.vue && cd ..
git commit -m "feat: add associated video expand/collapse and sidebar controls"
```

---

### Task 18: AssociatedVideoDetail Component

**Files:**
- Create: `frontend/src/components/AssociatedVideoDetail.vue`
- Modify: `frontend/src/router/index.ts`

- [ ] **Step 1: Create the component**

Model after `CroppedVideoDetail.vue` / `AlignedVideoDetail.vue`. This is a read-only review screen:

```vue
<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import {
    getAssociatedVideo,
    getVideoStreamUrl,
    type Video,
} from '@/services/api'
import { useVideoPlayback } from '@/composables/useVideoPlayback'

const route = useRoute()
const router = useRouter()

const projectId = computed(() => Number(route.params.id))
const mainVideoId = computed(() => Number(route.params.videoId))

const associatedVideo = ref<Video | null>(null)
const isLoading = ref(true)
const error = ref<string | null>(null)

// View modes — spec requires three: main_bbox, tracker_mask, final_mask
type ViewMode = 'main_bbox' | 'tracker_mask' | 'final_mask'
const viewMode = ref<ViewMode>('final_mask')

// useVideoPlayback returns: videoRef, currentTime, duration, isPlaying, videoWidth,
// videoHeight, onTimeUpdate, onLoadedMetadata, onPlay, onPause, seek, togglePlay,
// setMetadataCallback. It does NOT return currentFrame/totalFrames — compute locally.
const {
    videoRef,
    currentTime,
    duration,
    isPlaying,
    onTimeUpdate,
    onLoadedMetadata,
    onPlay,
    onPause,
    togglePlay,
} = useVideoPlayback()

// Compute frame index from time + fps (matches CroppedVideoDetail pattern)
const currentFrameIdx = computed(() => {
    if (!associatedVideo.value?.fps) return 0
    return Math.floor(currentTime.value * associatedVideo.value.fps)
})

const totalFrames = computed(() => {
    if (!associatedVideo.value?.fps || !duration.value) return 0
    return Math.floor(duration.value * associatedVideo.value.fps)
})

const videoStreamUrl = computed(() => {
    if (!projectId.value || !associatedVideo.value) return ''
    return getVideoStreamUrl(projectId.value, associatedVideo.value.id)
})

onMounted(async () => {
    try {
        associatedVideo.value = await getAssociatedVideo(
            projectId.value,
            mainVideoId.value,
        )
    } catch (e: any) {
        error.value = e.message || 'Failed to load associated video'
    } finally {
        isLoading.value = false
    }
})

const handleBack = () => {
    router.push({ name: 'project', params: { id: projectId.value } })
}
</script>

<template>
    <div class="detail-container">
        <div class="detail-header">
            <button class="back-button" @click="handleBack">← Back</button>
            <h2 v-if="associatedVideo">{{ associatedVideo.name }} (Associated)</h2>
        </div>

        <div v-if="isLoading" class="loading">Loading...</div>
        <div v-else-if="error" class="error">{{ error }}</div>
        <div v-else-if="associatedVideo" class="detail-content">
            <!-- Action bar -->
            <div class="action-bar">
                <div class="view-mode-selector">
                    <label>
                        <input type="radio" v-model="viewMode" value="main_bbox" />
                        Main BBox
                    </label>
                    <label>
                        <input type="radio" v-model="viewMode" value="tracker_mask" />
                        Tracker Mask
                    </label>
                    <label>
                        <input type="radio" v-model="viewMode" value="final_mask" />
                        Final Mask
                    </label>
                </div>
            </div>

            <!-- Video player -->
            <div class="video-player-area">
                <video
                    ref="videoRef"
                    :src="videoStreamUrl"
                    @timeupdate="onTimeUpdate"
                    @loadedmetadata="onLoadedMetadata"
                    @play="onPlay"
                    @pause="onPause"
                />
                <canvas ref="overlayCanvasRef" class="overlay-canvas" />
            </div>

            <!-- Playback controls -->
            <div class="playback-controls">
                <button @click="togglePlay">{{ isPlaying ? 'Pause' : 'Play' }}</button>
                <span>Frame {{ currentFrameIdx }} / {{ totalFrames }}</span>
            </div>
        </div>
    </div>
</template>

<style scoped>
.detail-container {
    padding: 16px;
    height: 100%;
    display: flex;
    flex-direction: column;
}

.detail-header {
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 16px;
}

.back-button {
    padding: 6px 12px;
    cursor: pointer;
}

.action-bar {
    margin-bottom: 12px;
}

.view-mode-selector {
    display: flex;
    gap: 16px;
}

.video-player-area {
    position: relative;
    flex: 1;
}

.video-player-area video {
    width: 100%;
    max-height: 70vh;
}

.overlay-canvas {
    position: absolute;
    top: 0;
    left: 0;
    pointer-events: none;
}

.playback-controls {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 8px 0;
}
</style>
```

**Note:** This is a minimal scaffold. The full implementation will need to:
1. Load and render mask overlays based on `viewMode` (reuse patterns from `useSegmentation` composable)
2. Add a data track timeline for scores (reuse `DataTrack.vue`)
3. For "Main BBox" view mode: fetch the main video's mask for the current frame, compute the bbox, and draw it on the overlay canvas. This requires loading the main video's data as well — the component should also fetch the main video to get its ID for mask API calls.

These details follow existing patterns in `CroppedVideoDetail.vue` and `VideoDetail.vue`. The implementer should reference those files to wire up the canvas overlay rendering and data track.

- [ ] **Step 2: Register the route**

In `frontend/src/router/index.ts`, add the import and route:

```typescript
import AssociatedVideoDetail from '@/components/AssociatedVideoDetail.vue'
```

Add the route inside the project children array, alongside the existing cropped/aligned routes:

```typescript
{
    path: 'video/:videoId/associated',
    name: 'associatedVideo',
    component: AssociatedVideoDetail,
},
```

- [ ] **Step 3: Commit**

```bash
cd frontend && git add src/components/AssociatedVideoDetail.vue src/router/index.ts && cd ..
git commit -m "feat: add AssociatedVideoDetail component and route"
```

---

## Task Dependencies

```
Independent (can run in parallel):
  Task 1: Video Model & Schema
  Task 2: Migration Script
  Task 3: Array Storage
  Task 4: Frame Data Service
  Task 5: Filesystem Route
  Task 10: StreamingSegmentor Wrappers

Sequential chains:
  Task 1 + 3 → Task 6 → Task 7 → Task 8 → Task 9
  Task 10 → Task 11 → Task 12 → Task 13 → Task 14
  Task 5 + 9 → Task 15 → Task 16 → Task 17 → Task 18
```

Tasks 1-5 and Task 10 can all run in parallel as they modify independent files.
