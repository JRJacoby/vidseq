# Export Detector Bboxes to CSV

## Overview

Add the ability to export standard detector bounding box results to a CSV file saved in the project folder. This is the first export feature in VidSeq and establishes the `exports/` directory convention.

## CSV Format

**File location:** `{project_path}/exports/detector_bboxes_{YYYYMMDD_HHMMSS}.csv`

**Columns:**

| Column | Type | Description |
|--------|------|-------------|
| `video_id` | int | Video ID from the database |
| `video_full_path` | string | Absolute path to the original video file on disk (`Video.path`) |
| `frame_idx` | int | 0-based frame index |
| `x1` | float or empty | `detector_bbox_x1` — left edge |
| `y1` | float or empty | `detector_bbox_y1` — top edge |
| `x2` | float or empty | `detector_bbox_x2` — right edge |
| `y2` | float or empty | `detector_bbox_y2` — bottom edge |

**Row coverage:** Every frame for every selected video is represented. If a frame has no detector bbox, the four coordinate columns are empty. Rows are sorted by `video_id` then `frame_idx`.

**Frame range:** Determined by each video's `num_frames` metadata (0 to num_frames - 1).

**Excluded columns:** `detector_score` is intentionally omitted — this export provides geometry only. Score can be added as a separate export if needed.

## API Endpoint

**`POST /projects/{project_id}/exports/detector-bboxes`**

**Request body:**
```json
{ "video_ids": [1, 2, 3] }
```

Uses the existing `VideoSelectionRequest` schema, consistent with all other pipeline actions.

**Response model:** `ExportResponse` (new Pydantic model in `vidseq/schemas/export.py`):
```json
{
  "path": "/absolute/path/to/project/exports/detector_bboxes_20260403_143022.csv",
  "row_count": 15000
}
```

**Errors:**
- 404 if project not found
- 400 if `video_ids` is empty or contains invalid IDs

Validation of `video_ids` happens in the route layer (before calling the service), consistent with other routes that use `VideoSelectionRequest`.

## Service Layer

**New file:** `vidseq/services/export_service.py`

**Function:** `async def export_detector_bboxes(session: AsyncSession, project_path: str, video_ids: list[int]) -> tuple[str, int]`

**Logic:**
1. Query `Video` table for `id`, `path`, `num_frames` for the selected video IDs
2. Query `FrameData` table for `video_id`, `frame_idx`, `detector_bbox_x1`, `detector_bbox_y1`, `detector_bbox_x2`, `detector_bbox_y2` for those video IDs. The `IN` clause contains only video IDs (not per-frame parameters), so no chunking is needed.
3. Build a complete frame index DataFrame. Since videos have different frame counts, construct per-video ranges and concatenate (not `MultiIndex.from_product`):
   ```python
   rows = []
   for vid_id, num_frames in video_info:
       rows.append(pd.DataFrame({
           "video_id": vid_id,
           "frame_idx": np.arange(num_frames),
       }))
   full_index = pd.concat(rows, ignore_index=True)
   ```
4. Left-merge the bbox query results onto the complete frame index on `(video_id, frame_idx)`
5. Map `video_id` → `Video.path` to populate the `video_full_path` column. `Video.path` stores absolute paths.
6. Create `{project_path}/exports/` directory if it doesn't exist
7. Write CSV via `df.to_csv(path, index=False)` — pandas naturally writes empty strings for NaN bbox values
8. Return `(absolute_csv_path, row_count)`

**Design constraint:** The service function is self-contained — it takes a DB session and project path, not request/response objects. Usable from scripts without web infrastructure.

**Expected scale:** Typical usage is 1-20 videos with 1K-50K frames each, producing up to ~1M rows. The pandas vectorized merge and `to_csv()` handle this efficiently. The synchronous CSV write blocks the event loop briefly but is acceptable at this scale (<2 seconds).

## Route Layer

**New file:** `vidseq/api/routes/exports.py`

**Registered under:** `/projects/{project_id}/exports` in `vidseq/server.py`

**Route function:** Thin passthrough — resolves project path from DB, gets a session, calls `export_service.export_detector_bboxes()`, returns the result as `ExportResponse`.

## Frontend

**Location:** "Export Bboxes" button in `VideoPipeline.vue`, in the Detector section alongside existing detector actions.

**Behavior:**
- Disabled when no videos are selected or export is in progress
- Button label: "Export Bboxes" → "Exporting..." while in progress
- On success: `alert()` showing the returned file path
- On error: `alert()` with error message

No guard against exporting videos with no detector results — a CSV with empty bbox columns is valid output.

**API function:** `exportDetectorBboxes(projectId: number, videoIds: number[])` added to `frontend/src/services/api.ts`, following the same POST + video_ids pattern as other pipeline actions.

## New Dependency

**pandas** — added to `pyproject.toml` via `uv add pandas`. Used for vectorized DataFrame construction, merge, and CSV writing. Provides significant speedup over Python-loop `csv.writer` at scale (hundreds of thousands to millions of rows).

## File Changes Summary

| File | Change |
|------|--------|
| `vidseq/services/export_service.py` | New — export logic |
| `vidseq/schemas/export.py` | New — `ExportResponse` Pydantic model |
| `vidseq/api/routes/exports.py` | New — route passthrough |
| `vidseq/server.py` | Register exports router |
| `frontend/src/services/api.ts` | Add `exportDetectorBboxes()` |
| `frontend/src/components/VideoPipeline.vue` | Add "Export Bboxes" button |
| `pyproject.toml` | Add pandas dependency |
