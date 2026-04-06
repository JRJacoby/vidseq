# Export Seg Detector Masks

## Overview

Add an "Export Seg Detector Masks" button under the Seg Detector section in VideoPipeline. Exports selected videos' detector masks to a single H5 file keyed by absolute video path, with binary 0/1 mask datasets.

## Output Format

- **File:** `exports/seg_detector_masks_{timestamp}.h5` inside the project folder
- **Structure:** Top-level keys are absolute video paths (e.g., `/data/videos/mouse01.mp4`). Each key maps to a dataset of shape `(num_frames, height, width)`, dtype `uint8`, values 0 or 1.
- **Chunking:** `(1, H, W)` per frame
- **Compression:** None

## Backend Service

`export_detector_masks()` in `vidseq/services/export_service.py`:

1. Query video metadata (id, path, num_frames) from DB for selected video IDs
2. Create export H5 file
3. For each video:
   - Open source `detector_masks.h5` via `detector_masks()` context manager
   - Read in chunks of 256 frames
   - Binarize: `(chunk > 0).astype(np.uint8)`
   - Write to export file under the video's absolute path key
4. Return `(absolute_h5_path, total_frame_count)`

## Backend Route

`POST /projects/{project_id}/exports/detector-masks` in `vidseq/api/routes/exports.py`:

- Request body: `VideoSelectionRequest` (has `video_ids: list[int]`)
- Response: `ExportResponse` (has `path: str`, `row_count: int` — row_count is total frames)
- Validation: 400 if `video_ids` empty

## Frontend API

`exportDetectorMasks(projectId, videoIds)` in `frontend/src/services/api.ts`:

- POST to `/api/projects/{projectId}/exports/detector-masks`
- Body: `{ video_ids: videoIds }`
- Returns `{ path, row_count }`

## Frontend UI

In `frontend/src/components/VideoPipeline.vue`, under the "Seg Detector" heading after the Apply Seg Detector button:

- State: `isExportingSegMasks` ref
- Handler: `handleExportSegMasks()` — same pattern as `handleExportBboxes()`
- Button: disabled when exporting or no videos selected
- Alert on success: frame count and path
