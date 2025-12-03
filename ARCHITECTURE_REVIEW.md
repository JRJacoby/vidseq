# Architecture Review: VidSeq

## Summary

The project has a reasonable overall structure with clear separation between backend (FastAPI/SQLAlchemy) and frontend (Vue 3). However, there are several patterns that could cause maintenance headaches, DRY violations, and coupling issues.

---

## Severity Legend

| Level | Meaning |
|-------|---------|
| **5** | Actively bad/harmful design. Confusing to readers. Couples things that have no business being coupled. Anti-patterns. |
| **4** | Lots of tangled logic - will DEFINITELY make changes to one part of the code difficult later because of implicit coupling. |
| **3** | Could make making changes/updates/extensions later difficult. |
| **2** | Could cause some confusion trying to read/understand, or more complex than it needs to be. |
| **1** | Nitpicking. |

---

## Issues

### Severity 5 - Actively Harmful

#### 1. Service layer coupled to HTTP concerns

**File:** `vidseq/services/video_service.py` (lines 72-92)

```python
async def get_video_by_id(session: AsyncSession, video_id: int) -> Video:
    ...
    from fastapi import HTTPException
    ...
    raise HTTPException(status_code=404, detail=f"Video {video_id} not found")
```

**Problem:** The service layer directly imports and raises `HTTPException`. This violates separation of concerns - services should be transport-agnostic and raise domain exceptions. Routes should catch those and convert to HTTP responses.

**Impact:** Makes it impossible to reuse this service in non-HTTP contexts (CLI, background jobs, tests without mocking HTTP).

**Fix:** Create domain exceptions (e.g., `VideoNotFoundError`) and catch/convert them in route handlers.

---

### Severity 4 - Will Cause Coupling Pain

#### 2. Business logic embedded in route file

**File:** `vidseq/api/routes/segmentation.py` (lines 33-91)

```python
def _mask_to_png(mask) -> bytes:
    """Convert a numpy mask to PNG bytes."""
    img = Image.fromarray(mask)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


async def _run_segmentation_and_save(
    project_path: Path,
    video: Video,
    frame_idx: int,
    prompts: list[Prompt],
) -> bytes:
    """Run SAM3 segmentation with prompts and save the mask."""
    # ... 50 lines of business logic
```

**Problem:** The route file contains significant business logic (`_run_segmentation_and_save`, `_mask_to_png`). This should live in a service (e.g., `segmentation_service.py`).

**Impact:** If you want to run segmentation from a CLI or background job, you'd need to import from a routes module.

**Fix:** Create `vidseq/services/segmentation_service.py` and move this logic there.

---

#### 3. Session state split across process boundary

**Files:** `vidseq/services/sam3_service.py` and `vidseq/services/sam3_worker.py`

```python
# sam3_service.py line 66
self._sessions: dict[int, VideoSessionInfo] = {}

# sam3_worker.py line 108
sessions: Dict[int, Tuple[str, Any]] = {}  # video_id -> (session_id, loader)
```

**Problem:** Video session state is maintained in two places with different representations:
- Main process: `VideoSessionInfo` (lightweight metadata)
- Worker process: `(session_id, loader)` tuple

**Impact:** Implicit coupling where both must agree on what constitutes a "session". If someone modifies one, they might forget the other.

**Fix:** Define a clear contract/protocol for session state. Consider documenting the relationship explicitly.

---

#### 4. Prompt types are stringly-typed

**Files:** `vidseq/models/prompt.py`, `vidseq/schemas/segmentation.py`, `frontend/src/services/api.ts`

```python
# models/prompt.py line 18
type: Mapped[str] = mapped_column(String, nullable=False)
```

```typescript
// api.ts line 132-137
export interface Prompt {
    ...
    type: string
```

**Problem:** Prompt types (`"bbox"`, `"positive_point"`, etc.) are raw strings throughout. A typo like `"bbbox"` would silently create invalid data.

**Impact:** No compile-time or runtime type safety for prompt types.

**Fix:** Create a Python `PromptType` enum and a TypeScript string union type.

---

### Severity 3 - Will Make Changes Difficult

#### 5. Duplicate project lookup pattern

**Files:** `vidseq/api/dependencies.py`, `vidseq/api/routes/projects.py`

The same query pattern appears in at least 4 places:

```python
# dependencies.py lines 26-32
result = await session.execute(
    select(Project).where(Project.id == project_id)
)
project = result.scalar_one_or_none()
if not project:
    raise HTTPException(status_code=404, detail=f"Project {project_id} not found")

# routes/projects.py lines 31-37 (same pattern)
# routes/projects.py delete_project (same pattern)
```

**Problem:** DRY violation. Any change to project lookup logic requires updating multiple locations.

**Fix:** Create a single `get_project_by_id()` function in a service or repository module.

---

#### 6. VideoMetadata constructed in two places

**Files:** `vidseq/services/video_service.py`, `vidseq/services/sam3streaming.py`

`get_video_metadata()` extracts metadata properly with error handling, but `LazyVideoFrameLoader` creates its own `VideoMetadata` directly:

```python
# sam3streaming.py lines 44-49
self._metadata = VideoMetadata(
    num_frames=int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    height=int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    width=int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    fps=self._cap.get(cv2.CAP_PROP_FPS),
)
```

**Problem:** Duplicates the metadata extraction logic without the error handling that `get_video_metadata()` provides.

**Fix:** Have `LazyVideoFrameLoader` use `get_video_metadata()` or factor out shared extraction logic.

---

#### 7. Two DeclarativeBase classes with identical names

**Files:** `vidseq/models/registry.py`, `vidseq/models/project_db.py`

```python
# registry.py lines 9-10
class Base(DeclarativeBase):
    pass

# project_db.py lines 5-6
class Base(DeclarativeBase):
    pass
```

**Problem:** Both are called `Base`. This is confusing when reading code that imports from both modules.

**Fix:** Rename to `RegistryBase` and `ProjectBase` (or similar descriptive names).

---

#### 8. SAM3 worker command types are magic strings

**File:** `vidseq/services/sam3_worker.py` (lines 119-135)

```python
if cmd_type == "load_model":
    ...
elif cmd_type == "init_session":
    ...
elif cmd_type == "add_bbox_prompt":
```

**Problem:** Command types are string literals. A typo in a command string would silently fail or fall through to the unknown command handler.

**Fix:** Create a `CommandType` enum and use it on both sides of the IPC boundary.

---

#### 9. Mask service has unwieldy function signatures

**File:** `vidseq/services/mask_service.py` (lines 70-77)

```python
def load_mask(
    project_path: Path,
    video_id: int,
    frame_idx: int,
    num_frames: int,
    height: int,
    width: int,
) -> np.ndarray:
```

**Problem:** Every call site must pass 6 parameters. The video metadata (`num_frames`, `height`, `width`) always travel together.

**Fix:** Accept a `VideoMetadata` object instead of individual dimensions, or create a `MaskContext` dataclass.

---

#### 10. Title bar duplicated across views

**Files:** `frontend/src/views/HomeView.vue`, `frontend/src/views/ProjectView.vue`

```html
<!-- Both files have identical markup -->
<header class="title-bar">
  <h1 class="title-bar-title">VidSeq - Animal Behavior Modeling from Raw Video</h1>
</header>
```

**Problem:** DRY violation. Changing the title requires editing multiple files.

**Fix:** Create a shared `AppHeader.vue` component.

---

### Severity 2 - Could Cause Confusion

#### 11. Inconsistent service module patterns

**Directory:** `vidseq/services/`

| Module | Pattern |
|--------|---------|
| `database_manager.py` | Singleton class |
| `sam3_service.py` | Singleton class + module-level wrapper functions |
| `video_service.py` | Pure functions |
| `mask_service.py` | Pure functions |
| `prompt_service.py` | Pure functions |

The sam3 module provides both class methods AND wrapper functions:

```python
# sam3_service.py lines 286-288
def get_status() -> dict:
    """Get the current SAM3 loading status."""
    return SAM3Service.get_instance().get_status()
```

**Problem:** Creates two ways to do the same thing. Inconsistent patterns make the codebase harder to navigate.

**Fix:** Pick one pattern per service type and stick with it. Document the reasoning.

---

#### 12. Duplicate singleton implementation

**Files:** `vidseq/services/database_manager.py`, `vidseq/services/sam3_service.py`

Both implement nearly identical singleton patterns with `_instance`, `_lock`, `__new__`, `get_instance()`, `reset_instance()`.

**Problem:** Code duplication. Changes to singleton behavior require updating multiple classes.

**Fix:** Create a shared `Singleton` metaclass or base class.

---

#### 13. Frontend types duplicate backend schemas

**File:** `frontend/src/services/api.ts`

```typescript
export interface Project {
    id: number
    name: string
    path: string
    created_at: string
    updated_at: string
}
```

**Problem:** This mirrors the Python Pydantic schema exactly. There's no mechanism to ensure they stay in sync.

**Fix:** Consider code generation from FastAPI's OpenAPI spec (e.g., using `openapi-typescript`).

---

#### 14. Pydantic Config uses deprecated style

**Files:** All schema files in `vidseq/schemas/`

```python
class Config:
    from_attributes = True
```

**Problem:** This is Pydantic v1 style. Pydantic v2 recommends the new approach.

**Fix:** Update to:
```python
model_config = ConfigDict(from_attributes=True)
```

---

#### 15. DirectoryEntry has inconsistent naming

**File:** `vidseq/schemas/filesystem.py`

```python
is_directory: bool = Field(alias="isDirectory")
```

**Problem:** The Python field uses `is_directory` but exposes `isDirectory` via alias. This dual naming makes code harder to follow.

**Fix:** Either use consistent snake_case with automatic camelCase serialization, or document the convention clearly.

---

#### 16. VideoDetail component is too large

**File:** `frontend/src/components/VideoDetail.vue` (559 lines)

The component handles:
- Video loading/metadata
- Playback controls
- Frame data loading
- SAM3 status polling
- Session management
- Segmentation tool state
- All UI rendering

**Problem:** Too many responsibilities in one component. Hard to test and maintain.

**Fix:** Extract into composables:
- `useVideoPlayback()`
- `useSAM3Session()`
- `useSegmentationTools()`

---

#### 17. CSS patterns repeated across components

**Files:** `VideoPipeline.vue`, `JobsList.vue`, `FilePickerModal.vue`, `RecentProjects.vue`

```css
.loading-state,
.empty-state {
  padding: 2rem;
  text-align: center;
  color: #666;
}
```

**Problem:** Same CSS patterns duplicated in multiple components.

**Fix:** Create global utility classes or a shared styles file.

---

#### 18. Jobs endpoint polls with new sessions

**File:** `vidseq/api/routes/jobs.py` (lines 34-41)

```python
while True:
    factory = db_manager.get_registry_session_factory()
    async with factory() as session:
        result = await session.execute(
            select(Job).order_by(Job.created_at.desc())
        )
```

**Problem:** Creates a new session factory reference and session on every poll iteration. Inefficient and could mask connection issues.

**Fix:** Consider using database change notifications or a pub/sub pattern instead of polling.

---

### Severity 1 - Nitpicking

#### 19. `project_id` URL parameter unused in some routes

**File:** `vidseq/api/routes/segmentation.py` (lines 122-125)

```python
@router.delete("/projects/{project_id}/videos/{video_id}/session")
async def close_video_session(
    video_id: int,
):  # project_id not used
```

**Problem:** The URL includes `{project_id}` but it's not used in the handler.

**Fix:** Either use it for validation (ensure video belongs to project) or remove it from the URL.

---

#### 20. Hardcoded magic numbers

**Files:** Various

| Value | Location | Purpose |
|-------|----------|---------|
| `1008` | `sam3streaming.py` | IMAGE_SIZE |
| `8192` | `videos.py` | Streaming chunk size |
| `120.0` | `sam3_service.py` | Timeout seconds |

**Problem:** Magic numbers without clear context.

**Fix:** Define as named constants with documentation.

---

#### 21. Empty `jobs/` directory

**Directory:** `vidseq/jobs/`

The directory only contains `__pycache__`.

**Problem:** Empty module suggests incomplete implementation or leftover scaffolding.

**Fix:** Either add the intended job logic or remove the directory.

---

#### 22. Empty `__init__.py` files

**Files:** `vidseq/api/__init__.py`, `vidseq/api/routes/__init__.py`

These files exist but are empty.

**Problem:** Minor inconsistency - some `__init__.py` files export items, others don't.

**Fix:** Either add exports for cleaner imports, or remove if using implicit namespace packages.

---

#### 23. `services/__init__.py` has incomplete exports

**File:** `vidseq/services/__init__.py`

Exports some items but not all. The `sam3_service` module is imported but individual items are exported inconsistently.

**Problem:** Inconsistent import experience.

**Fix:** Either export everything publicly needed, or nothing (let consumers import from submodules directly).

---

## Recommendations Summary

| Priority | Action |
|----------|--------|
| **High** | Create a domain exceptions module - Replace HTTP exceptions in services with domain-specific exceptions like `VideoNotFoundError`, `ProjectNotFoundError`. |
| **High** | Extract segmentation service - Move `_run_segmentation_and_save` and related logic from routes to a proper service. |
| **High** | Create enums for typed fields - `PromptType`, `JobStatus`, `SAM3Status`, `CommandType` should be Python enums with TypeScript counterparts. |
| **Medium** | Unify singleton pattern - Create a `Singleton` metaclass or base class. |
| **Medium** | Create shared Vue components - `AppHeader`, `LoadingState`, `EmptyState` components to reduce duplication. |
| **Medium** | Decompose VideoDetail - Extract video playback, segmentation tools, and SAM3 management into composables. |
| **Low** | Consider OpenAPI codegen - Generate TypeScript types from FastAPI's OpenAPI schema to keep frontend/backend in sync. |
| **Low** | Update to Pydantic v2 style - Use `model_config` instead of inner `Config` class. |

