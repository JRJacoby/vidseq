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

The existing `get_all_videos()` adds a filter for `WHERE is_associated = False`. All whole-project operations (pipeline list, ARHMM, batch segmentation, etc.) should only operate on main videos. Associated videos are only accessed individually through their main video's relationship.

**VideoResponse schema** (`vidseq/schemas/video.py`) adds `is_associated: bool` and `associated_with_id: int | None` so the frontend can render associations. Note: `num_frames`, `height`, `width` are NOT needed in VideoResponse — the frontend derives frame information from the `<video>` element's duration and fps, not from the API. This applies to the associated video detail screen as well.

**Fetching associated videos:** The existing `GET /projects/{project_id}/videos` endpoint does not return associated videos (filtered out by `get_all_videos`). A new `GET /projects/{project_id}/videos/{video_id}/associated` endpoint returns the associated video (or 404 if none). The frontend fetches this for each main video to know whether to show the expand toggle.

Associated videos get their own `array_data/{video_id}/` directory with H5 files created at ingestion time. Only tracker_masks, tracker_logits, and final_masks are created — no detector_masks, since there is no detector workflow for associated videos. The `create_video_segmentation_arrays()` function gets a `skip_detector: bool = False` param; the associated video ingestion path passes `skip_detector=True`. The absence of detector_masks.h5 acts as a safety net: any accidental detector operation on an associated video crashes immediately rather than silently succeeding with empty data.

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
- No duplicate associated paths (same depth video path cannot be mapped to multiple main videos)
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
- Confidence threshold input (default 0.9, range 0.0–1.0, step 0.05) next to the button, like the alignment epochs input. 0.0 means always prompt (every frame with a main bbox), 1.0 means never prompt (SAM2 coasts entirely).
- Triggers `POST /projects/{project_id}/videos/associated/segmentation` with `{video_ids: [...], confidence_threshold: 0.9}`
- Synchronous-blocking like existing `segment_all_videos` — the endpoint blocks until all videos are processed. No job queue or polling mechanism.

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

**Data flow:** The route uses the main video's ID (`/project/:id/video/:videoId/associated`). On mount, the component calls `GET /videos/{mainVideoId}/associated` to fetch the associated video's data (including its own `id`). It then uses the associated video's ID for all subsequent API calls — video streaming (`GET /videos/{assocId}/stream`), mask loading, score fetching, etc.

The associated video's scores (SAM2 predicted IoU from the depth propagation) are displayed in the data track timeline, same as the main video detail screen.

### Inference Workflow — Associated Video Propagation

**No new method on StreamingSegmentor.** The orchestration logic (threshold checking, deciding when to prompt) lives entirely in the command handler (`handle_propagate_with_associated` in `segmentation_commands.py`), calling existing segmentor primitives: `add_box_prompt()` for initialization and re-conditioning, `_propagate_single_frame()` for forward propagation, and `_backtrack_reprop()` for gap re-propagation. This keeps the segmentor storage-agnostic — it has no knowledge of "main video scores" or "associated video" concepts.

**Inputs to the command handler:**
- `video_id` (associated/depth video), `main_video_id`, `project_path`, `confidence_threshold`
- `main_video_scores` dict (`{frame_idx: float}` — `FrameData.score` field, which is SAM2 predicted IoU, loaded from DB by FastAPI side and passed via TCP params). For very long videos (~100K frames), this dict is 1-2MB of JSON in the TCP message — acceptable but worth noting.
- The depth video's H5 files are accessed through the existing `_video_resources[video_id]` pattern (session must already be initialized)
- The main video's tracker_masks are opened directly via `tracker_masks(project_path, main_video_id, "r")` as a context manager for the duration of the propagation — this is a new pattern for command handlers, since normally they only access files for the video they have a session for, but it's straightforward since we only need read access to the main video's masks
- Bboxes computed on-the-fly by the command handler via `array_storage.compute_bbox_from_mask()` — trivial computation (find min/max nonzero pixels), avoids adding storage to the main segmentation workflow
- Output targets: `tracker_masks` and `final_masks`. Logits are NOT written — they are only persisted during interactive operations (point prompts, refine_mask), not during batch propagation. This matches the existing `propagate_with_detector` pattern.

**Algorithm (implemented in the command handler):**
1. Scan forward to find the first frame where the main video's score > threshold. Frames before this point are skipped (no masks produced) — reverse-temporal propagation is not implemented. This matches `propagate_with_detector` behavior.
2. Initialize SAM2 on the depth video at that frame using the main video's bbox as a box prompt (via `add_box_prompt()`)
3. Propagate forward frame by frame on the depth video:
   - Look up the main video's score for the current frame
   - If main score >= threshold → provide main video's bbox as a box prompt (re-condition SAM2)
   - If main score < threshold → propagate without a prompt via `_propagate_single_frame()` (SAM2 coasts on depth)
   - Track `last_anchor_frame` — the last frame where a bbox prompt was provided
   - Write masks to both `tracker_masks` (original predictions) and `final_masks` (corrected predictions)
4. When main score transitions from < threshold back to >= threshold → new anchor. Call `_backtrack_reprop()` to re-propagate the gap frames (from `last_anchor_frame + 1` to `current_frame - 1`), then continue forward. `_backtrack_reprop` only updates `final_masks` — `tracker_masks` retains the original forward-pass predictions so the user can compare before/after correction.
5. Store results: depth masks to `tracker_masks.h5` and `final_masks.h5` (as described above), depth SAM2 scores to DB via `frame_data_service`
6. Update `has_tracker_mask` and `has_final_mask` flags on the associated video's FrameData rows
7. Set `segmentation_status = "segmented"` on the associated Video row

**Service-layer / worker separation:**
The FastAPI service layer loads the main video's scores from DB and passes them as a dict in the TCP command params. The worker never touches the database. The worker reads the main video's H5 masks directly (H5 files are accessible from any process). Only the associated (depth) video needs a SAM2 session — the main video's data is read from storage only.

**Edge case — unsegmented main video:** If the main video's tracker_masks are all zeros (video added but not segmented), no frames will pass the confidence threshold. The handler should detect this early (check that at least one frame in `main_video_scores` exceeds the threshold) and raise a clear error: "Main video has no segmentation data above confidence threshold."

**SAM2 session lifecycle:** The service layer (in `segmentation_service.py`) manages init_session → propagate_with_associated → close_session per depth video, same as `segment_all_videos` does for normal propagation. The handler assumes an existing session for the depth video.

**delete_frame_data / reset_video on associated videos:** These functions open detector_masks.h5, which doesn't exist for associated videos. This is not a concern because no code path invokes them on associated videos: the associated video detail screen is read-only (no reset/delete buttons), and main video deletion cascades via `shutil.rmtree` on the entire `array_data/` directory. Auditing all Video queries for associated-safety is out of scope — operations by video ID work correctly on associated videos if ever invoked directly.

**Gap frame behavior during backtrack:** During the forward pass, frames where SAM2 "coasts" (no bbox prompt) get masks written to both tracker_masks and final_masks. When `_backtrack_reprop` runs after a confidence recovery, it re-propagates the gap frames with the new anchor context and overwrites only final_masks. The coasting masks in tracker_masks are preserved as-is. The existing coasting masks in final_masks serve as context for `_set_memory_frame` during re-propagation — they are not zeroed before backtracking. This matches the existing `propagate_with_detector` behavior.

**Progress reporting:** Same pattern as propagate-with-detector — periodic callbacks with frame index, streamed to frontend.

### Migration Script

`scripts/migrate_associated_videos.py` — standalone script for existing projects.

- Takes a project directory path as CLI argument
- Connects directly to `<project_dir>/vidseq.db` with plain `sqlite3` (no SQLAlchemy, no app startup)
- Checks if columns already exist via `PRAGMA table_info(videos)` (idempotent)
- Runs `ALTER TABLE videos ADD COLUMN is_associated INTEGER NOT NULL DEFAULT 0` (SQLite stores BOOLEAN as INTEGER; matches SQLAlchemy `Mapped[bool]` with `default=False`)
- Runs `ALTER TABLE videos ADD COLUMN associated_with_id INTEGER DEFAULT NULL` (matches SQLAlchemy `Mapped[int | None]` with `default=None`)
- Prints what it did, exits

Usage: `uv run python scripts/migrate_associated_videos.py /path/to/project`

## Scope

**Files modified:**

Backend:
- `vidseq/models/video.py` — add `is_associated`, `associated_with_id` columns
- `vidseq/schemas/video.py` — add `is_associated`, `associated_with_id` to VideoResponse
- `vidseq/services/video_service.py` — filter `get_all_videos()`, associated video ingestion, cascade delete
- `vidseq/api/routes/videos.py` — new `POST /videos/associated`, `GET /videos/{id}/associated` endpoints
- `vidseq/services/segmentation_commands.py` — new `handle_propagate_with_associated` TCP handler (orchestrates existing segmentor primitives; no changes to streaming_segmentor.py)
- `vidseq/services/segmentation_service.py` — orchestrate batch co-segmentation (alongside existing `segment_all_videos`)
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
