# Export Seg Detector Masks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an "Export Seg Detector Masks" button that exports selected videos' detector masks to a single H5 file keyed by absolute video path.

**Architecture:** Four-layer change mirroring the existing "Export BBoxes" feature: service function reads per-video detector_masks.h5 files and consolidates into one export H5; route exposes it via POST; frontend API client calls it; Vue button triggers it.

**Tech Stack:** Python (h5py, numpy, SQLAlchemy), FastAPI, TypeScript/Vue 3

**Spec:** `docs/superpowers/specs/2026-04-06-export-seg-detector-masks-design.md`

---

### Task 1: Add `export_detector_masks()` service function

**Files:**
- Modify: `vidseq/services/export_service.py`

- [ ] **Step 1: Add the import**

At the top of `export_service.py`, add the `detector_masks` context manager import after the existing imports:

```python
from vidseq.services.array_storage import detector_masks
```

- [ ] **Step 2: Add the `export_detector_masks()` function**

Append this function after the existing `export_detector_bboxes()` function (after line 88):

```python
async def export_detector_masks(
    session: AsyncSession,
    project_path: str | Path,
    video_ids: list[int],
) -> tuple[str, int]:
    """Export detector masks for selected videos to a single H5 file.

    Each video's masks are stored under its absolute path as key.
    Masks are binarized (0/1 uint8).

    Returns (absolute_h5_path, total_frame_count).
    """
    import h5py

    project_path = Path(project_path)

    # 1. Query video metadata
    result = await session.execute(
        select(Video.id, Video.path, Video.num_frames, Video.height, Video.width)
        .where(Video.id.in_(video_ids))
    )
    video_rows = result.all()
    if not video_rows:
        raise ValueError("No videos found for the given IDs")

    video_info = {
        row.id: (row.path, row.num_frames, row.height, row.width)
        for row in video_rows
    }
    missing_ids = set(video_ids) - set(video_info)
    if missing_ids:
        raise ValueError(f"Videos not found: {sorted(missing_ids)}")

    # 2. Create export file
    exports_dir = project_path / "exports"
    exports_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    h5_path = exports_dir / f"seg_detector_masks_{timestamp}.h5"

    total_frames = 0
    chunk_size = 256

    with h5py.File(h5_path, "w") as out_f:
        for vid_id in sorted(video_info):
            path, num_frames, height, width = video_info[vid_id]

            # Create dataset keyed by absolute video path
            ds = out_f.create_dataset(
                path,
                shape=(num_frames, height, width),
                dtype=np.uint8,
                chunks=(1, height, width),
            )

            # Chunked copy with binarization
            with detector_masks(project_path, vid_id) as src:
                for start in range(0, num_frames, chunk_size):
                    end = min(start + chunk_size, num_frames)
                    chunk = src[start:end]
                    ds[start:end] = (chunk > 0).astype(np.uint8)

            total_frames += num_frames

    return str(h5_path), total_frames
```

- [ ] **Step 3: Verify the import works**

Run: `cd /n/groups/datta/john/projects/vidseq && uv run python -c "from vidseq.services.export_service import export_detector_masks; print('OK')"`

Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add vidseq/services/export_service.py
git commit -m "feat: add export_detector_masks service function"
```

---

### Task 2: Add POST route for detector mask export

**Files:**
- Modify: `vidseq/api/routes/exports.py`

- [ ] **Step 1: Add the new route**

Append this route after the existing `create_detector_bboxes_export` endpoint (after line 38):

```python
@router.post(
    "/projects/{project_id}/exports/detector-masks",
    response_model=ExportResponse,
)
async def create_detector_masks_export(
    request: VideoSelectionRequest,
    project_path: Path = Depends(get_project_folder),
    session: AsyncSession = Depends(get_project_session),
):
    """Export seg detector masks to H5 for selected videos."""
    if not request.video_ids:
        raise HTTPException(status_code=400, detail="video_ids must not be empty")

    try:
        h5_path, frame_count = await export_service.export_detector_masks(
            session=session,
            project_path=project_path,
            video_ids=request.video_ids,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return ExportResponse(path=h5_path, row_count=frame_count)
```

- [ ] **Step 2: Commit**

```bash
git add vidseq/api/routes/exports.py
git commit -m "feat: add POST /exports/detector-masks route"
```

---

### Task 3: Add frontend API function

**Files:**
- Modify: `frontend/src/services/api.ts`

- [ ] **Step 1: Add `exportDetectorMasks()` function**

Add this function directly after the existing `exportDetectorBboxes()` function (after line 1491):

```typescript
export async function exportDetectorMasks(
    projectId: number,
    videoIds: number[],
): Promise<{ path: string; row_count: number }> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/exports/detector-masks`,
        {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ video_ids: videoIds }),
        },
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to export seg detector masks'))
    }
    return response.json()
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/services/api.ts
git commit -m "feat: add exportDetectorMasks API client function"
```

---

### Task 4: Add UI button and handler in VideoPipeline

**Files:**
- Modify: `frontend/src/components/VideoPipeline.vue`

- [ ] **Step 1: Add `exportDetectorMasks` to the import block**

In the import from `api.ts` (near line 27), add `exportDetectorMasks` alongside the existing `exportDetectorBboxes`:

```typescript
  exportDetectorBboxes,
  exportDetectorMasks,
```

- [ ] **Step 2: Add state ref**

After the existing `isExportingBboxes` ref (line 112), add:

```typescript
const isExportingSegMasks = ref(false)
```

- [ ] **Step 3: Add handler function**

After the existing `handleExportBboxes` function (after line 522), add:

```typescript
const handleExportSegMasks = async () => {
  if (!projectId.value || isExportingSegMasks.value) return
  isExportingSegMasks.value = true
  try {
    const result = await exportDetectorMasks(projectId.value, selectedVideoIdsList.value)
    alert(`Exported ${result.row_count} frames to:\n${result.path}`)
  } catch (e: any) {
    console.error('Failed to export seg detector masks:', e)
    alert(e.message || 'Failed to export seg detector masks')
  } finally {
    isExportingSegMasks.value = false
  }
}
```

- [ ] **Step 4: Add button to template**

After the "Apply Seg Detector" button closing tag (after line 925), add:

```vue
<button
  class="sidebar-button"
  @click="handleExportSegMasks"
  :disabled="isExportingSegMasks || selectedCount === 0"
>
  <span class="button-label">{{ isExportingSegMasks ? 'Exporting...' : 'Export Seg Masks' }}</span>
</button>
```

- [ ] **Step 5: Verify frontend compiles**

Run: `cd /n/groups/datta/john/projects/vidseq/frontend && npm run type-check`

Expected: No errors

- [ ] **Step 6: Commit**

```bash
git add frontend/src/components/VideoPipeline.vue
git commit -m "feat: add Export Seg Masks button to VideoPipeline"
```
