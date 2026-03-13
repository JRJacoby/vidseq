# Associated Videos

## Problem

Azure depth cameras produce two video streams per session: a high-def IR stream (easy to segment but has a bright center artifact from the IR beam) and a depth estimation stream (contains the actual data needed, but too noisy for SAM2 to track reliably). The two failure modes are complementary — IR segmentation fails under the bright spot, but depth tracking is fine there; depth tracking struggles in general, but IR bounding boxes can rescue it.

Users need a way to leverage the high-quality IR segmentation to guide depth video segmentation, using IR bboxes as SAM2 prompts only when the IR segmentation is trustworthy.

## Solution

Add "associated videos" — a 1:1 mapping between main videos (IR) and associated videos (depth). Associated videos are full Video rows with their own storage, but flagged to stay hidden from the main pipeline list. The main video card shows an inline expander to reveal its associated video.

A new propagation workflow segments the depth video by:
1. Propagating SAM2 forward on the depth video frame by frame
2. Using the main video's bounding boxes as SAM2 box prompts, but only on frames where the main video's predicted IoU exceeds a confidence threshold (default 0.9)
3. When the main video's confidence recovers after a low-confidence gap, rewinding and re-propagating the gap frames with the new anchor

## Design

### Data Model

**Video model additions:**
- `is_associated: bool = False` — hides video from pipeline list
- `associated_with_id: int | None = None` — FK to the main Video's ID

A new `get_main_videos()` function returns only videos where `is_associated = False`. The existing `get_all_videos()` remains unchanged for callers like ARHMM that need all videos. The pipeline list uses `get_main_videos()`.

**VideoResponse schema** (`vidseq/schemas/video.py`) adds `is_associated: bool` and `associated_with_id: int | None` so the frontend can render associations.

**Fetching associated videos:** The existing `GET /projects/{project_id}/videos` endpoint (which uses `get_main_videos` for the pipeline) does not return associated videos. A new `GET /projects/{project_id}/videos/{video_id}/associated` endpoint returns the associated video (or 404 if none). The frontend fetches this for each main video to know whether to show the expand toggle.

Associated videos get their own `array_data/{video_id}/` directory with the standard H5 files (tracker_masks, tracker_logits, detector_masks, final_masks), created at ingestion time just like any other video.

**Cascade on deletion:** When a main video is deleted via `delete_videos()`, its associated video is also deleted (DB row, H5 directory, all data). The delete function queries for associated videos before deleting and includes them in the batch.

### JSON Ingestion

Users click "Add Associated Videos" on the pipeline sidebar. This opens the existing `FilePickerModal` with two new props: `accept: ['.json']` (filter to JSON files only) and `single-select: true`.

The JSON file format is a flat object mapping main video filepaths to associated video filepaths:

```json
{
  "/data/session01/ir_video.avi": "/data/session01/depth_video.avi",
  "/data/session02/ir_video.avi": "/data/session02/depth_video.avi"
}
```

**Validation — fail fast on any error:**
- Valid JSON, flat `{string: string}` structure
- Every key matches an existing video's `path` in the project
- Every value is a valid, readable video file on disk
- No duplicate associations (main video doesn't already have an associated video)
- Associated video has the same `num_frames` as the main video (required for frame-aligned propagation)
- On any error → 400 with specific message (e.g., "Main video not found: /path/to/ir.avi")

On success, for each pair: extract video metadata from the associated file, create a Video row with `is_associated=True` and `associated_with_id` pointing to the main video, and create H5 files.

**API endpoint:** `POST /projects/{project_id}/videos/associated` with multipart or JSON body containing the mapping.

### Frontend — Pipeline List

**Video card expansion (Option C: Inline Expand):**
- Main video cards that have an associated video show a clickable "1 associated" toggle
- Expanding reveals the associated video inline — indented, with its name and a link to open its detail screen
- Collapse/expand state is purely local UI
- Videos without associations are unchanged

**"Co-Segment" batch action:**
- New sidebar button: "Co-Segment N Videos"
- Active only when selected main videos have associated videos
- Confidence threshold input (default 0.9) next to the button, like the alignment epochs input
- Triggers `POST /projects/{project_id}/videos/associated/segmentation` with `{video_ids: [...], confidence_threshold: 0.9}`
- Fire-and-forget with progress polling, similar to "Segment N Videos"

**"Add Associated Videos" button:**
- New sidebar button, opens `FilePickerModal` with `accept: ['.json']` and `single-select: true`
- On success, refreshes the video list

### Frontend — Associated Video Detail Screen

**Route:** `/project/:id/video/:videoId/associated`

**Component:** `AssociatedVideoDetail.vue`, modeled after `CroppedVideoDetail.vue` / `AlignedVideoDetail.vue`

**Layout:**
- Video player + overlay canvas + data track (scores/confidence timeline)
- Simplified action bar with only a view mode switcher:
  - **Main BBox** — bounding box from the main video's segmentation overlaid on the depth video (shows nothing on frames where main video had no mask)
  - **Tracker Mask** — the depth video's SAM2 tracking mask
  - **Final Mask** — corrected final mask

No segmentation tools, no point prompts, no training data marking. This is a read-only review screen.

The associated video's scores (SAM2 predicted IoU from the depth propagation) are displayed in the data track timeline, same as the main video detail screen.

### Inference Workflow — Associated Video Propagation

**New method: `propagate_with_associated()` in StreamingSegmentor**

Follows the same structural pattern as `propagate_with_detector()` but uses pre-computed data from the main video as the prompt source rather than running live detection.

**Inputs:**
- Depth video frames (via `VideoFrameSource`)
- Main video's scores dict (`{frame_idx: float}` — predicted IoU per frame, loaded from DB by FastAPI side and passed via TCP params)
- Main video's tracker masks in H5 (bboxes computed on-the-fly via `_bbox_from_mask()`)
- Confidence threshold (default 0.9)
- Output targets: both `tracker_masks` and `final_masks` (like `propagate_with_detector`)

**Algorithm:**
1. Scan forward to find the first frame where the main video's score > threshold
2. Initialize SAM2 on the depth video at that frame using the main video's bbox as a box prompt
3. Propagate forward frame by frame on the depth video:
   - Look up the main video's score for the current frame
   - If main score >= threshold → provide main video's bbox as a box prompt (re-condition SAM2)
   - If main score < threshold → propagate without a prompt (SAM2 coasts on depth)
   - Track `last_anchor_frame` — the last frame where a bbox prompt was provided
   - Write masks to both `tracker_masks` and `final_masks`
4. When main score transitions from < threshold back to >= threshold → new anchor. Call `_backtrack_reprop()` to re-propagate the gap frames (from `last_anchor_frame + 1` to `current_frame - 1`), then continue forward
5. Store results: depth masks to `tracker_masks.h5` and `final_masks.h5`, depth SAM2 scores to DB via `frame_data_service`
6. Update `has_tracker_mask` and `has_final_mask` flags on the associated video's FrameData rows
7. Set `segmentation_status = "segmented"` on the associated Video row

**Why compute bboxes on-the-fly from masks (not pre-stored):**
The bbox computation from a mask is trivial (`_bbox_from_mask()` — find min/max nonzero pixels). Computing on-the-fly avoids adding storage to the main segmentation workflow. The main video's H5 file is already on disk.

**Service-layer / worker separation:**
The FastAPI service layer loads the main video's scores from DB and passes them as a dict in the TCP command params. The worker never touches the database. The worker reads the main video's H5 masks directly (H5 files are accessible from any process). Only the associated (depth) video needs a SAM2 session — the main video's data is read from storage only.

**TCP command:** `handle_propagate_with_associated` in `segmentation_commands.py`. Receives `video_id` (associated/depth), `main_video_id`, `main_video_scores` (dict), `project_path`, and `confidence_threshold`. Opens depth video resources and main video's H5 masks, then drives the segmentor.

**Progress reporting:** Same pattern as propagate-with-detector — periodic callbacks with frame index, streamed to frontend.

### Migration Script

`scripts/migrate_associated_videos.py` — standalone script for existing projects.

- Takes a project directory path as CLI argument
- Connects directly to `<project_dir>/vidseq.db` with plain `sqlite3` (no SQLAlchemy, no app startup)
- Checks if columns already exist via `PRAGMA table_info(videos)` (idempotent)
- Runs `ALTER TABLE videos ADD COLUMN is_associated BOOLEAN DEFAULT 0`
- Runs `ALTER TABLE videos ADD COLUMN associated_with_id INTEGER`
- Prints what it did, exits

Usage: `uv run python scripts/migrate_associated_videos.py /path/to/project`

## Scope

**Files modified:**

Backend:
- `vidseq/models/video.py` — add `is_associated`, `associated_with_id` columns
- `vidseq/schemas/video.py` — add `is_associated`, `associated_with_id` to VideoResponse
- `vidseq/services/video_service.py` — add `get_main_videos()`, associated video ingestion, cascade delete
- `vidseq/api/routes/videos.py` — new `POST /videos/associated`, `GET /videos/{id}/associated` endpoints; use `get_main_videos()` for pipeline list
- `vidseq/services/segmentation_model/streaming_segmentor.py` — new `propagate_with_associated()` method
- `vidseq/services/segmentation_commands.py` — new `handle_propagate_with_associated` TCP handler
- `vidseq/services/segmentation_service.py` — orchestrate batch co-segmentation
- `vidseq/services/segmentation_tcp_client.py` — new client method for associated propagation
- `vidseq/api/routes/segmentation/` — new `POST /videos/associated/segmentation` endpoint

Frontend:
- `frontend/src/components/VideoPipeline.vue` — expand/collapse on cards, new sidebar buttons
- `frontend/src/components/AssociatedVideoDetail.vue` — new component
- `frontend/src/components/FilePickerModal.vue` — add `accept` and `singleSelect` props
- `frontend/src/services/api.ts` — new API functions
- `frontend/src/router/index.ts` — new route

Scripts:
- `scripts/migrate_associated_videos.py` — DB migration for existing projects

**No changes to:**
- Existing segmentation workflow (propagate, propagate-with-detector)
- H5 file lifecycle (associated videos use standard creation)
- Detector training/inference
- Alignment, PCA, cropping pipelines
