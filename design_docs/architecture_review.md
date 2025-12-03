# VidSeq Architecture Review

*Review Date: December 2, 2025*

This document identifies architectural issues in the VidSeq codebase, organized by severity.

## Severity Scale

| Level | Description |
|-------|-------------|
| 1 | Nitpicking |
| 2 | Could cause confusion trying to read/understand, or more complex than needed |
| 3 | Could make changes/updates/extensions later difficult |
| 4 | Lots of tangled logic - will DEFINITELY make changes difficult due to implicit coupling |
| 5 | Actively bad/harmful design. Confusing to readers. Couples things that have no business being coupled. Anti-patterns. |

---

## Severity 5 (Actively Bad/Harmful)

### ~~Issue 1: Module-Level Global State in `sam3_service.py`~~ ✅ FIXED

**Status:** Resolved on 2025-12-02

Converted to a thread-safe singleton class `SAM3Service` with:
- Encapsulated instance state
- `reset_instance()` method for testing
- Backwards-compatible module-level convenience functions

---

## Severity 4 (Tangled Logic / Implicit Coupling)

### ~~Issue 2: Video Metadata Extraction Duplicated in 3 Places~~ ✅ FIXED

**Status:** Resolved on 2025-12-02

Created `vidseq/services/video_metadata.py` with:
- `VideoMetadata` dataclass (frozen, immutable)
- `get_video_metadata()` function with proper error handling
- `VideoMetadataError` exception class

Updated consumers:
- `segmentation.py` - uses `video_metadata.get_video_metadata()`
- `videos.py` - uses `get_video_metadata()` with error handling
- `sam3streaming.py` - uses `VideoMetadata` dataclass, exposes via `.metadata` property

---

### ~~Issue 3: `database.py` Mixes Concerns and Uses Global State~~ ✅ FIXED

**Status:** Resolved on 2025-12-02

Split `database.py` into two modules with clear responsibilities:

- `vidseq/services/database_manager.py` — Singleton class that owns database lifecycle:
  - `DatabaseManager.init_registry()` / `init_project()` / `shutdown()`
  - No HTTP/FastAPI knowledge
  - Testable in isolation
  
- `vidseq/api/dependencies.py` — FastAPI dependency injection:
  - `get_registry_session()`, `get_project_session()`, `get_project_folder()`
  - HTTP-aware error handling (`HTTPException`)
  - Uses `DatabaseManager` under the hood

---

### Issue 4: Inconsistent Frame Index Naming Across API

The codebase uses three different names for the same concept:

| Location | Name |
|----------|------|
| URL path params | `frame_number` |
| Query params | `frame_idx` |
| SAM3 internal | `frame_index` |
| Model/Schema | `frame_idx` |

Example from `segmentation.py`:

```python
@router.delete(
    "/projects/{project_id}/videos/{video_id}/frame/{frame_number}",
)
async def reset_frame(
    video_id: int,
    frame_number: int,
    ...
):
    ...
    await prompt_storage.delete_prompts_for_frame(
        session=session,
        video_id=video_id,
        frame_idx=frame_number,  # <- conversion here
    )
```

This inconsistency makes the API confusing and creates translation layers that add bugs.

**Fix:** Pick ONE name (`frame_idx`) and use it consistently everywhere.

---

## Severity 3 (Will Make Changes Difficult)

### Issue 5: `sam3streaming.py` in Wrong Location

`LazyVideoFrameLoader` is at the package root (`vidseq/sam3streaming.py`) instead of in `services/`.

```python
class LazyVideoFrameLoader:
    """
    Lazy frame loader that provides frames on-demand from a video file.
    
    Implements __getitem__ and __len__ to be compatible with SAM3's 
    inference_state["images"] access pattern.
    """
```

This breaks the consistent module organization where all services are in `services/`.

**Fix:** Move to `vidseq/services/video_loader.py`.

---

### Issue 6: Empty `jobs/` Directory

The `jobs/` directory exists but contains only `__pycache__`. This is confusing - it suggests background job processing was planned but not implemented.

**Fix:** Either implement job workers here, or delete the directory.

---

### Issue 7: `Prompt` Interface Duplicated in Frontend

The `Prompt` interface is defined twice:

1. `frontend/src/services/api.ts:130-137`
2. `frontend/src/components/VideoOverlay.vue:4-10`

```typescript
// In VideoOverlay.vue
export interface Prompt {
  id: number
  video_id: number
  frame_idx: number
  type: string
  details: Record<string, number>
}
```

**Fix:** Import from a single location (e.g., `api.ts` or a dedicated `types.ts`).

---

### Issue 8: Segmentation Route Has Utility Functions That Don't Belong

`segmentation.py` contains utility functions that should be in services:

```python
def _get_video_metadata(video_path: Path) -> tuple[int, int, int]:
    """Get video metadata: num_frames, height, width."""
    cap = cv2.VideoCapture(str(video_path))
    ...

def _mask_to_png(mask) -> bytes:
    """Convert a numpy mask to PNG bytes."""
    ...
```

Routes should only handle HTTP concerns, not business logic.

**Fix:** Move to `services/video_utils.py` or similar.

---

### Issue 9: Duplicated Video Lookup Pattern

The pattern of looking up a video by ID and raising 404 is repeated:

- `segmentation.py:35-43` (`_get_video` helper)
- `videos.py:72-78` (inline)
- `videos.py:87-92` (inline)

**Fix:** Create a shared dependency or service method.

---

## Severity 2 (Could Cause Confusion)

### Issue 10: Overloaded "Session" Term

The word "session" means two different things:
1. SQLAlchemy `AsyncSession` for database transactions
2. SAM3 `VideoSessionInfo` for GPU inference state

In `segmentation.py`, this causes confusing parameter names:

```python
async def init_video_session(
    video_id: int,
    session: AsyncSession = Depends(get_project_session),  # database session
):
    # Creates a SAM3 "video session"
    session_info = sam3_service.init_session(video_id, video_path)
```

**Fix:** Rename SAM3 sessions to "inference context" or "prediction session".

---

### Issue 11: Model File Naming Doesn't Match Contents

`models/project.py` contains `Video` model, not just project-related models:

```python
class Video(Base):
    __tablename__ = "videos"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String)
    path: Mapped[str] = mapped_column(String)
    fps: Mapped[float] = mapped_column(Float)
```

The design docs mention a `video.py` model file that doesn't exist.

**Fix:** Either rename to `models/project_db.py` (clarifying it's "project database models") or split into `video.py`.

---

### Issue 12: `Job` Model in `registry.py`

`Job` is in `registry.py` alongside `Project`:

```python
class Job(Base):
    __tablename__ = "jobs"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    type: Mapped[str] = mapped_column(String)
    status: Mapped[str] = mapped_column(String)
    ...
```

While both are in the registry database, mixing them makes it harder to understand the domain. Jobs have very different lifecycles than projects.

**Fix:** Consider `models/registry/project.py` and `models/registry/job.py`.

---

### Issue 13: Empty `__init__.py` Files Don't Export

All `__init__.py` files in `models/`, `schemas/`, `services/`, `api/`, `api/routes/` are empty. This means you can't do:

```python
from vidseq.models import Project, Video
```

**Fix:** Add explicit exports to `__init__.py` files.

---

### Issue 14: Magic Numbers

```python
# In sam3streaming.py
IMAGE_SIZE = 1008
IMG_MEAN = (0.5, 0.5, 0.5)
IMG_STD = (0.5, 0.5, 0.5)

# In cli.py
uvicorn.run("vidseq.server:app", host='0.0.0.0', port=8000, reload=True)
```

These should be in a config file or environment variables.

**Fix:** Create a `config.py` or use environment variables with defaults.

---

## Severity 1 (Nitpicks)

### Issue 15: Unused Import in `database.py`

```python
import asyncio  # Never used
```

---

### Issue 16: Inconsistent Error Handling Style in Frontend

Some API calls use `.catch(() => ({}))` pattern, others don't:

```typescript
// Pattern 1: Catches JSON parse errors
const error = await response.json().catch(() => ({}))
throw new Error(error.detail || 'Failed to create project for unknown reason.')

// Pattern 2: Uses statusText directly
throw new Error(`Failed to fetch projects: ${response.statusText}`)
```

**Fix:** Standardize on one error handling pattern.

---

### Issue 17: Two `now()` Functions for Timestamps

```python
# In registry.py
def now():
    return datetime.now(timezone.utc)

# In prompt.py
created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
```

One uses `timezone.utc`, the other uses deprecated `datetime.utcnow`.

**Fix:** Use a single shared `utc_now()` function.

---

## Summary

| Severity | Count | Remaining | Key Themes |
|----------|-------|-----------|------------|
| 5 | 1 | 0 ✅ | Global mutable state |
| 4 | 4 | 2 | DRY violations, naming inconsistency, mixed concerns |
| 3 | 5 | 5 | Misplaced files, dead code, duplicated types |
| 2 | 5 | 5 | Confusing terminology, poor organization |
| 1 | 3 | 3 | Minor inconsistencies |

---

## Recommended Priority Order

### High Priority (Do First)
1. ~~**Replace global state in `sam3_service.py` with a proper class**~~ ✅ Done
2. ~~**Consolidate video metadata extraction into one place**~~ ✅ Done
3. **Standardize on `frame_idx` naming everywhere** - API clarity

### Medium Priority
4. Move `sam3streaming.py` to `services/`
5. Create shared video lookup dependency
6. ~~Clean up `database.py` concerns~~ ✅ Done
7. Add exports to `__init__.py` files

### Low Priority
8. Remove empty `jobs/` directory
9. Fix minor inconsistencies (unused imports, error handling patterns)
10. Consolidate frontend `Prompt` interface

