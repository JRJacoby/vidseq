# Export Detector Bboxes to CSV — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Export standard detector bounding box results to a CSV file in the project's `exports/` directory, triggered from the pipeline sidebar.

**Architecture:** New `export_service.py` handles all logic (DB query + pandas DataFrame + CSV write). A thin route in `exports.py` calls the service. Frontend adds a button to VideoPipeline that calls the new endpoint and alerts the file path.

**Tech Stack:** Python (pandas, SQLAlchemy async), FastAPI, Vue 3 (Composition API), TypeScript

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `vidseq/services/export_service.py` | Create | All export logic: query DB, build DataFrame, write CSV |
| `vidseq/schemas/export.py` | Create | `ExportResponse` Pydantic model |
| `vidseq/api/routes/exports.py` | Create | Thin route passthrough |
| `vidseq/server.py` | Modify (line 22, 57) | Import + register exports router |
| `frontend/src/services/api.ts` | Modify | Add `exportDetectorBboxes()` function |
| `frontend/src/components/VideoPipeline.vue` | Modify | Add button + handler + state |
| `pyproject.toml` | Modify | Add pandas dependency |

---

### Task 1: Add pandas dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Install pandas**

```bash
cd /n/groups/datta/john/projects/vidseq && uv add pandas
```

- [ ] **Step 2: Verify installation**

```bash
uv run python -c "import pandas; print(pandas.__version__)"
```

Expected: prints a version number (e.g., `2.2.x`).

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "deps: add pandas for CSV export"
```

---

### Task 2: Create ExportResponse schema

**Files:**
- Create: `vidseq/schemas/export.py`

- [ ] **Step 1: Create the schema file**

```python
"""Schemas for export endpoints."""

from pydantic import BaseModel


class ExportResponse(BaseModel):
    """Response from an export operation."""

    path: str
    row_count: int
```

- [ ] **Step 2: Commit**

```bash
git add vidseq/schemas/export.py
git commit -m "feat: add ExportResponse schema"
```

---

### Task 3: Create export service

**Files:**
- Create: `vidseq/services/export_service.py`

- [ ] **Step 1: Create the service file**

```python
"""Service for exporting data to files."""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from vidseq.models.frame_data import FrameData
from vidseq.models.video import Video


async def export_detector_bboxes(
    session: AsyncSession,
    project_path: str | Path,
    video_ids: list[int],
) -> tuple[str, int]:
    """Export detector bounding boxes for selected videos to CSV.

    Every frame for each selected video gets a row. Frames without
    detector bboxes have empty coordinate columns.

    Returns (absolute_csv_path, row_count).
    """
    project_path = Path(project_path)

    # 1. Query video metadata
    result = await session.execute(
        select(Video.id, Video.path, Video.num_frames).where(Video.id.in_(video_ids))
    )
    video_rows = result.all()
    if not video_rows:
        raise ValueError("No videos found for the given IDs")

    video_info = {row.id: (row.path, row.num_frames) for row in video_rows}

    # 2. Query detector bboxes
    result = await session.execute(
        select(
            FrameData.video_id,
            FrameData.frame_idx,
            FrameData.detector_bbox_x1,
            FrameData.detector_bbox_y1,
            FrameData.detector_bbox_x2,
            FrameData.detector_bbox_y2,
        ).where(FrameData.video_id.in_(video_ids))
    )
    bbox_rows = result.all()

    # 3. Build complete frame index (per-video ranges, since frame counts differ)
    index_parts = []
    for vid_id in sorted(video_info):
        _, num_frames = video_info[vid_id]
        index_parts.append(
            pd.DataFrame({"video_id": vid_id, "frame_idx": np.arange(num_frames)})
        )
    full_df = pd.concat(index_parts, ignore_index=True)

    # 4. Build bbox DataFrame and left-merge
    if bbox_rows:
        bbox_df = pd.DataFrame(
            bbox_rows, columns=["video_id", "frame_idx", "x1", "y1", "x2", "y2"]
        )
        full_df = full_df.merge(bbox_df, on=["video_id", "frame_idx"], how="left")
    else:
        full_df[["x1", "y1", "x2", "y2"]] = np.nan

    # 5. Map video_id -> absolute path
    full_df["video_full_path"] = full_df["video_id"].map(
        {vid_id: path for vid_id, (path, _) in video_info.items()}
    )

    # 6. Reorder columns to match spec
    full_df = full_df[["video_id", "video_full_path", "frame_idx", "x1", "y1", "x2", "y2"]]

    # 7. Write CSV
    exports_dir = project_path / "exports"
    exports_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = exports_dir / f"detector_bboxes_{timestamp}.csv"
    full_df.to_csv(csv_path, index=False)

    return str(csv_path), len(full_df)
```

- [ ] **Step 2: Verify the module imports cleanly**

```bash
uv run python -c "from vidseq.services import export_service; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add vidseq/services/export_service.py
git commit -m "feat: add export_service with detector bbox CSV export"
```

---

### Task 4: Create exports route

**Files:**
- Create: `vidseq/api/routes/exports.py`
- Modify: `vidseq/server.py`

- [ ] **Step 1: Create the route file**

```python
"""API routes for data export."""

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from vidseq.api.dependencies import get_project_folder, get_project_session
from vidseq.api.schemas import VideoSelectionRequest
from vidseq.schemas.export import ExportResponse
from vidseq.services import export_service

router = APIRouter()


@router.post(
    "/projects/{project_id}/exports/detector-bboxes",
    response_model=ExportResponse,
)
async def create_detector_bboxes_export(
    request: VideoSelectionRequest,
    project_path: Path = Depends(get_project_folder),
    session: AsyncSession = Depends(get_project_session),
):
    """Export detector bounding boxes to CSV for selected videos."""
    if not request.video_ids:
        raise HTTPException(status_code=400, detail="video_ids must not be empty")

    try:
        csv_path, row_count = await export_service.export_detector_bboxes(
            session=session,
            project_path=project_path,
            video_ids=request.video_ids,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return ExportResponse(path=csv_path, row_count=row_count)
```

- [ ] **Step 2: Register the router in server.py**

In `vidseq/server.py`, add the import at line 22 (append `, exports` to the existing import):

Change:
```python
from vidseq.api.routes import alignment, arhmm, cropped_videos, detector, filesystem, pca, projects, segmentation, videos
```
To:
```python
from vidseq.api.routes import alignment, arhmm, cropped_videos, detector, exports, filesystem, pca, projects, segmentation, videos
```

Then add the router registration after line 57 (after the detector router):
```python
app.include_router(exports.router, prefix="/api", tags=["exports"])
```

- [ ] **Step 3: Verify the server starts**

```bash
timeout 5 uv run uvicorn vidseq.server:app --host 0.0.0.0 --port 18765 || true
```

Expected: server starts without import errors (will time out or ctrl-c, that's fine).

- [ ] **Step 4: Commit**

```bash
git add vidseq/api/routes/exports.py vidseq/server.py
git commit -m "feat: add exports route for detector bbox CSV"
```

---

### Task 5: Add frontend API function

**Files:**
- Modify: `frontend/src/services/api.ts`

- [ ] **Step 1: Add the exportDetectorBboxes function**

Add this function near the other detector-related functions (after `applySegDetector` or at the end of the detector section):

```typescript
export async function exportDetectorBboxes(
    projectId: number,
    videoIds: number[],
): Promise<{ path: string; row_count: number }> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/exports/detector-bboxes`,
        {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ video_ids: videoIds }),
        },
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to export detector bboxes'))
    }
    return response.json()
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/services/api.ts
git commit -m "feat: add exportDetectorBboxes API function"
```

---

### Task 6: Add Export Bboxes button to VideoPipeline

**Files:**
- Modify: `frontend/src/components/VideoPipeline.vue`

- [ ] **Step 1: Add the import**

In the imports from `@/services/api`, add `exportDetectorBboxes` to the import list.

- [ ] **Step 2: Add reactive state**

Add alongside the other `isApplying*` refs (near `isApplyingDetector`, `isApplyingObb`, `isApplyingSeg`):

```typescript
const isExportingBboxes = ref(false)
```

- [ ] **Step 3: Add handler function**

Add near the other `handleApply*` functions (near `handleApplyDetector`):

```typescript
const handleExportBboxes = async () => {
  if (!projectId.value || isExportingBboxes.value) return
  isExportingBboxes.value = true
  try {
    const result = await exportDetectorBboxes(projectId.value, selectedVideoIdsList.value)
    alert(`Exported ${result.row_count} rows to:\n${result.path}`)
  } catch (e: any) {
    console.error('Failed to export detector bboxes:', e)
    alert(e.message || 'Failed to export detector bboxes')
  } finally {
    isExportingBboxes.value = false
  }
}
```

- [ ] **Step 4: Add button to template**

Insert after the "Apply Detector" button (after line 868 in `VideoPipeline.vue`, before the `<h4>OBB Detector</h4>` heading):

```vue
          <button
            class="sidebar-button"
            @click="handleExportBboxes"
            :disabled="isExportingBboxes || selectedCount === 0"
          >
            <span class="button-label">{{ isExportingBboxes ? 'Exporting...' : 'Export Bboxes' }}</span>
          </button>
```

- [ ] **Step 5: Verify frontend builds**

```bash
cd /n/groups/datta/john/projects/vidseq/frontend && npm run type-check
```

Expected: no type errors.

- [ ] **Step 6: Commit**

```bash
cd /n/groups/datta/john/projects/vidseq
git add frontend/src/services/api.ts frontend/src/components/VideoPipeline.vue
git commit -m "feat: add Export Bboxes button to pipeline sidebar"
```
