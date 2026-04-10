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

## Intended User Flow

1. **Segment IR videos.** User adds IR videos to the project and segments them using the normal SAM2 workflow (interactive prompts, propagation, detector-assisted tracking). They review and refine until the IR segmentation is good enough — the predicted IoU scores on each frame reflect segmentation quality.
2. **Create association mapping.** User creates a JSON file mapping each IR video's filepath to its corresponding depth video filepath. One file covers all pairs in the project.
3. **Import associated videos.** User clicks "Add Associated Videos" in the pipeline sidebar, selects the JSON file. The system validates all pairs and creates the associated video records + H5 files.
4. **Co-segment.** User selects the IR videos they want to process, sets the confidence threshold (default 0.9), and clicks "Co-Segment." The system uses the IR segmentation's bounding boxes to guide SAM2 on each depth video.
5. **Review results.** User expands a video card to see its associated video, clicks through to the associated video detail screen. They review the depth segmentation quality by comparing tracker masks, final masks, and the main video's bounding box overlay. The score timeline shows where the depth tracking was confident vs where it coasted.

## Design

### Data Model

**Video model additions:**
- `is_associated: bool = False` — hides video from pipeline list
- `associated_with_id: int | None = None` — FK to the main Video's ID

The existing `get_all_videos()` adds a filter for `WHERE is_associated = False`. All whole-project operations (pipeline list, ARHMM, batch segmentation, etc.) should only operate on main videos. Associated videos are only accessed individually through their main video's relationship.

**VideoResponse schema** (`vidseq/schemas/video.py`) adds `is_associated: bool`, `associated_with_id: int | None`, and `associated_video_id: int | None` so the frontend can render associations. The `associated_video_id` field is a reverse-lookup (finding the associated video whose `associated_with_id` points to this video) — it can't come from `from_attributes` since it's not a column on the Video model. Instead, `get_all_videos()` in `video_service.py` performs a single subquery (`SELECT id FROM videos WHERE associated_with_id = v.id`) and returns the results as a list of dicts (or a lightweight DTO) with `associated_video_id` already populated. The route passes these through to VideoResponse without additional logic. This avoids N+1 API calls: the frontend checks `associated_video_id` on each video card to decide whether to show the expand toggle. Note: `num_frames`, `height`, `width` are NOT needed in VideoResponse — the frontend derives frame information from the `<video>` element's duration and fps, not from the API. This applies to the associated video detail screen as well.

**Fetching associated videos:** The existing `GET /projects/{project_id}/videos` endpoint does not return associated videos (filtered out by `get_all_videos`), but includes `associated_video_id` on main videos for the expand toggle. A new `GET /projects/{project_id}/videos/{video_id}/associated` endpoint returns the full associated video data (needed by the detail screen), or 404 if none.

Associated videos get their own `array_data/{video_id}/` directory with H5 files created at ingestion time. Only tracker_masks and final_masks are created — no detector_masks (no detector workflow) and no tracker_logits (no interactive segmentation, logits are never written during batch propagation). The `array_storage.create_video_segmentation_arrays()` function gets an `is_associated: bool = False` param; the associated video ingestion path passes `is_associated=True`. When `is_associated=True`, the function creates only tracker_masks.h5 and final_masks.h5, skipping detector_masks.h5 and tracker_logits.h5. The `reset_video_segmentation_arrays()` function also takes `is_associated` and propagates it through the delete/recreate cycle, so resetting an associated video does not accidentally create files it doesn't need. The `delete_video_segmentation_arrays()` function does NOT need this param — it already handles missing files gracefully (checks existence before unlinking).

**Cascade on deletion:** When a main video is deleted via `delete_videos()`, its associated video is also deleted (DB row, H5 directory, all data). The `delete_videos()` function in `video_service.py` queries `WHERE associated_with_id IN (requested_video_ids)` to find associated videos, adds their IDs to the deletion batch, then proceeds with the combined list.

### JSON Ingestion

Users click "Add Associated Videos" on the pipeline sidebar. This opens the existing `FilePickerModal` with two new props: `accept: ['.json']` (filter to JSON files only) and `singleSelect: true`.

**FilePickerModal `accept` prop implementation:** The `accept` prop is passed as a query parameter to the filesystem list endpoint (`GET /filesystem/list?path=/some/dir&accept=.json`). The `accept` query param is a comma-separated list of extensions (e.g., `accept=.json` or `accept=.json,.csv`). The backend filters the directory listing server-side, returning only files whose extension matches any of the accepted values (case-insensitive). Directories are always returned (so the user can navigate). Non-matching files are excluded from the response entirely (not shown disabled). When `accept` is not provided (the default for the existing video-adding use case), no filtering is applied — all files are shown regardless of extension.

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
- New sidebar button, opens `FilePickerModal` with `accept: ['.json']` and `singleSelect: true`
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

**Thin public wrappers on StreamingSegmentor.** The orchestration logic (threshold checking, deciding when to prompt) lives entirely in the command handler (`handle_propagate_with_associated` in `segmentation_commands.py`). The handler calls segmentor primitives, but the methods it needs (`_propagate_single_frame`, `_backtrack_reprop`, `_set_memory_frame`) are currently private. Rather than calling private methods directly or duplicating the full `propagate_with_detector` workflow, we add thin public wrappers to `streaming_segmentor.py`:

- `propagate_with_box(video_id, frame_idx, frame, box_prompt, add_as_conditioning=True)` → wraps `_propagate_single_frame(video_id, frame_idx, frame, box_prompt=box_prompt)`, returns `(mask, logits, score)`. The `add_as_conditioning` param controls storage: when `True`, the frame goes to `cond_frame_outputs` AND is added to `cond_frame_indices` (permanent); when `False`, the box prompt still guides the current frame's prediction but the result goes to `non_cond_frame_outputs` (ephemeral, subject to sliding window eviction) and `cond_frame_indices` is NOT updated. Both sides of this must be correct: `cond_frame_indices` tracks which frames are conditioning so that `_set_memory_frame` skips them — if a frame is in `cond_frame_outputs` but not in `cond_frame_indices`, it could get re-encoded as non-cond memory, creating duplicates. Internally, this is implemented via a new `store_as_cond: bool | None = None` param on `_propagate_single_frame` that overrides the default `is_prompted` storage logic. When `None` (default), existing behavior is preserved — all existing callers are unaffected. Note: the `is_init_cond_frame` flag passed to SAM2's `track_step` remains `True` whenever a box_prompt is provided (since `is_prompted=True`), regardless of `store_as_cond`. This is correct — the box prompt should always guide SAM2's prediction during inference; `store_as_cond` only controls where the result is stored afterward.
- `propagate_frame(video_id, frame_idx, frame)` → wraps `_propagate_single_frame(video_id, frame_idx, frame)` with no prompts, returns `(mask, logits, score)`. This is used for coasting frames (no bbox prompt) during sequential forward processing. Unlike `propagate()`, it does NOT call `_set_memory_frame()` — the sliding window is already correct from the previous frame in sequential processing. This matches how `propagate_with_detector` handles its coasting frames internally. Using `propagate()` for coasting would re-encode 6 frames from H5 on every frame, which is an O(6 * inference_time) overhead per frame — catastrophic on 100K-frame videos where most frames are coasting.
- `backtrack_reprop(video_id, frames, final_masks, start_idx, end_idx, scores=None)` → wraps `_backtrack_reprop()`, no return value. `scores` is optional (defaults to None), matching the private method's signature.
- `set_memory_frame(video_id, frame_idx, frames, masks)` → wraps `_set_memory_frame()`, no return value
- `evict_conditioning_frame(video_id, frame_idx)` → removes a frame from SAM2's conditioning memory. Implementation: `session["cond_frame_indices"].discard(frame_idx)` + `session["output_dict"]["cond_frame_outputs"].pop(frame_idx, None)`. Used by the command handler's LRU eviction policy.

The new public methods are generic primitives usable by any future workflow. The segmentor remains storage-agnostic — it has no knowledge of "main video scores" or "associated video" concepts.

**Inputs to the command handler:**
- `video_id` (associated/depth video), `main_video_id`, `project_path`, `confidence_threshold`, `main_height`, `main_width` (for bbox rescaling when dimensions differ), `cond_frame_interval` (default 50), `max_cond_frames` (default 32)
- `main_video_scores` dict (`{frame_idx: float}` — `FrameData.score` field, which is SAM2 predicted IoU, loaded from DB by FastAPI side and passed via TCP params). Note: `FrameData.score` defaults to -1.0 (not None), so frames without segmentation data will have score -1.0. The handler treats any score <= 0 as "no data" (equivalent to below threshold). For very long videos (~100K frames), this dict is 1-2MB of JSON in the TCP message. The alternative — having the worker read scores from the DB — would violate the CLAUDE.md separation rule ("the worker should not touch the database"). The JSON overhead is acceptable for the architectural clarity it provides.
- The depth video's H5 files are accessed through the existing `_video_resources[video_id]` pattern (session must already be initialized)
- The main video's tracker_masks are opened directly via `tracker_masks(project_path, main_video_id, "r")` as a context manager for the duration of the propagation — this is a new pattern for command handlers, since normally they only access files for the video they have a session for. This direct access is justified because the alternative (passing full-resolution masks for ~100K frames over TCP) would be impractical (potentially GBs of data). The worker and FastAPI processes share the filesystem, and we only need read access. Note: the read-mode H5 cache in `array_storage` checks file mtime for staleness, so concurrent modifications to the main video's masks would be detected. In practice, no one edits the main video during batch co-segmentation. This direct H5 access from the command handler is a deliberate exception to the norm (handlers normally only access files for the video they have a session for). Future handlers should NOT copy this pattern without similar justification — i.e., transferring the data over TCP is impractical due to size.
- Bboxes computed on-the-fly by the command handler via `array_storage.compute_bbox_from_mask()` — trivial computation (find min/max nonzero pixels), avoids adding storage to the main segmentation workflow. The bbox is in the main video's pixel coordinates `[x1, y1, x2, y2]`. If the main and associated videos have different dimensions, the bbox must be rescaled to the associated video's resolution: `x_scale = assoc_width / main_width`, `y_scale = assoc_height / main_height`, applied to all four coordinates. The command handler reads both videos' dimensions from `_video_resources` and the main video's metadata (passed via TCP params as `main_height`, `main_width`).
- Output targets: `tracker_masks` and `final_masks`. Logits are NOT written — they are only persisted during interactive operations (point prompts, refine_mask), not during batch propagation. This matches the existing `propagate_with_detector` pattern.
- The handler opens three H5 files simultaneously: main video's tracker_masks (read mode, cached), depth video's tracker_masks (append mode, locked), and depth video's final_masks (append mode, locked). These are different files in different directories (`array_data/{main_id}/` vs `array_data/{assoc_id}/`), so there is no lock contention.

**Algorithm (implemented in the command handler):**
1. Scan forward to find the first frame where the main video's score > threshold (treating scores <= 0 as "no data"). Frames before this point are skipped (no masks produced) — reverse-temporal propagation is not implemented.
2. Initialize SAM2 on the depth video at that frame using the main video's (possibly rescaled) bbox as a box prompt via `segmentor.propagate_with_box(add_as_conditioning=True)` — the first frame is always conditioning.
3. Propagate forward frame by frame on the depth video:
   - Look up the main video's score for the current frame
   - If main score >= threshold → provide main video's bbox as a box prompt. Every `COND_FRAME_INTERVAL`th prompted frame (default 50) uses `add_as_conditioning=True` (persists in SAM2 memory, pushed to LRU); all other prompted frames use `add_as_conditioning=False` (box guides prediction but doesn't persist as conditioning). See "Memory management" below.
   - If main score < threshold → propagate without a prompt via `segmentor.propagate_frame()` (SAM2 coasts on depth using its sliding window — no `_set_memory_frame` overhead)
   - Track `last_anchor_frame` — the last frame where a bbox prompt was provided
   - Write masks to both `tracker_masks` (original predictions) and `final_masks` (corrected predictions)
4. When main score transitions from < threshold back to >= threshold → new anchor. Call `segmentor.backtrack_reprop()` to re-propagate the gap frames (from `last_anchor_frame + 1` to `current_frame - 1`), then continue forward. `backtrack_reprop` only updates `final_masks` — `tracker_masks` retains the original forward-pass predictions so the user can compare before/after correction.
5. Return results: the worker's final TCP response includes `scores: list[dict]` (`[{"frame_idx": int, "score": float}, ...]`) containing the depth video's SAM2 predicted IoU for each processed frame, same shape as `propagate_with_detector_result`. Depth masks are already written to `tracker_masks.h5` and `final_masks.h5` by the handler during propagation.
6. **Service layer (not worker):** On receiving the TCP response, the service layer calls `save_scores_batch()` to persist scores to DB, updates `has_tracker_mask` and `has_final_mask` flags on the associated video's FrameData rows, and sets `segmentation_status = "segmented"` on the associated Video row. This matches the existing `segment_all_videos` pattern where DB operations happen in `segmentation_service.py` after the TCP call returns — the worker never touches the database.

**Service-layer / worker separation:**
The FastAPI service layer loads the main video's scores from DB and passes them as a dict in the TCP command params. A new function `load_all_scores(video_id, session) -> dict[int, float]` is added to `frame_data_service.py` — it queries all FrameData rows for the video and returns `{row.frame_idx: row.score for row in results}`. Frames without FrameData rows are absent from the dict; the handler treats missing keys the same as scores <= 0 (below threshold). The worker never touches the database. The worker reads the main video's H5 masks directly (H5 files are accessible from any process). Only the associated (depth) video needs a SAM2 session — the main video's data is read from storage only.

**Edge case — unsegmented main video:** If the main video's tracker_masks are all zeros (video added but not segmented), no frames will pass the confidence threshold. The handler should detect this early (check that at least one frame in `main_video_scores` exceeds the threshold) and raise a clear error: "Main video has no segmentation data above confidence threshold."

**SAM2 session lifecycle:** The service layer (in `segmentation_service.py`) manages init_session → propagate_with_associated → close_session per depth video, same as `segment_all_videos` does for normal propagation. The handler assumes an existing session for the depth video. The init_session call passes `cond_frame_indices=[]` since the associated video has no prior segmentation data.

**Cascade on main video segmentation reset:** When a main video's segmentation is reset (via `reset_video()` or `delete_videos_segmentation()`), the associated video's co-segmentation data is now stale (it was generated from the now-cleared main segmentation). The reset functions in `video_service.py` check for an associated video and, if one exists with `segmentation_status = "segmented"`, also reset its segmentation arrays and clear its FrameData/scores. This is likely in practice — users iterate on IR segmentation and then re-run co-segmentation.

**Per-video error handling during batch co-segmentation:** Follows the existing `segment_all_videos` pattern: try/catch per video with continue-on-error. If one video fails (e.g., main video has no segmentation above threshold), the error is logged and the batch continues with the remaining videos. The final response includes per-video success/failure status.

**delete_frame_data / reset_video on associated videos:** Associated videos do not have detector_masks.h5. If `delete_frame_data()` were called on an associated video, h5py would open in `"a"` mode (creating a bare H5 file), but the `detector_masks` context manager would immediately crash with `KeyError: 'data'` since the file has no datasets. So accidental invocation crashes rather than silently succeeding — a safe failure mode. However, no code path in the application invokes these on associated videos: the associated video detail screen is read-only (no reset/delete buttons), and main video deletion cascades via `shutil.rmtree` on the entire `array_data/` directory. No route-level guards are added for associated videos on these endpoints; this is an accepted gap. If `reset_video_segmentation_arrays` is ever called on an associated video, the `is_associated` param (see Data Model section) prevents detector_masks.h5 and tracker_logits.h5 from being recreated.

**Which masks source for coasting context:** When `backtrack_reprop` needs to re-propagate gap frames, `set_memory_frame` reads from the depth video's `tracker_masks` (the original forward-pass predictions, not the corrected `final_masks`). This is consistent: tracker_masks always has the unmodified forward-pass output, while final_masks gets overwritten by backtracking.

**Gap frame behavior during backtrack:** During the forward pass, frames where SAM2 "coasts" (no bbox prompt) get masks written to both tracker_masks and final_masks. When `backtrack_reprop` runs after a confidence recovery, it re-propagates the gap frames with the new anchor context and overwrites only final_masks. The coasting masks in tracker_masks are preserved as-is. The existing coasting masks in final_masks serve as context for `set_memory_frame` during re-propagation — they are not zeroed before backtracking. This is inspired by the existing `propagate_with_detector` backtracking pattern, but differs in that our workflow uses pre-computed scores and box prompts from stored masks, rather than real-time detector inference with IoU comparison at check intervals. The implementer should not assume identical logic — refer to the algorithm steps above, not the `propagate_with_detector` source.

**Progress reporting:** The TCP client uses `_send_streaming` (not `_send_and_wait`) to receive periodic progress callbacks from the worker, matching the `propagate_with_detector` pattern. The handler calls `response_callback({"type": "progress", "frame_idx": idx})` at regular intervals. The frontend displays per-video progress as a frame counter (e.g., "Processing video 2/5: frame 1234/50000"), matching the existing `segment_all_videos` UI pattern. No progress bar — just text status in the sidebar.

**Memory management for long videos.** The current interactive workflow assumes a small number of human-set conditioning frames (5-20). SAM2 stores these permanently in `cond_frame_outputs` (a dict of `{frame_idx: compact_memory_state}` containing GPU tensors). In the associated video workflow, thousands of frames receive box prompts — if every prompted frame became conditioning, GPU memory would grow unboundedly and cross-attention over all conditioning frames would slow down.

The solution combines two strategies:

1. **Sparse conditioning (every Nth frame).** Only every `COND_FRAME_INTERVAL`th prompted frame (default 50) is stored as a conditioning frame via `propagate_with_box(add_as_conditioning=True)`. All other prompted frames use `add_as_conditioning=False` — the box prompt still guides SAM2's prediction for that frame, but the result goes to `non_cond_frame_outputs` where it's naturally evicted by the sliding window (`MEM_WINDOW=6`). This avoids over-constraining SAM2 while still providing periodic anchor points.

2. **LRU eviction on conditioning frames.** The command handler maintains a `collections.deque` of conditioning frame indices. When a new conditioning frame is added and the deque exceeds `MAX_COND_FRAMES` (default 32), the handler evicts the oldest via `segmentor.evict_conditioning_frame(video_id, oldest_idx)`. Eviction is trivial: `cond_frame_indices.discard(idx)` + `cond_frame_outputs.pop(idx, None)`.

Both `COND_FRAME_INTERVAL` (50) and `MAX_COND_FRAMES` (32) are passed via TCP params so they can be tuned without code changes. The defaults give ~1 conditioning frame per 50 prompted frames, with at most 32 in memory at any time — covering a window of ~1600 frames of context.

Existing workflows are unaffected: the `store_as_cond` param on `_propagate_single_frame` defaults to `None` (preserving current behavior), and the LRU logic lives entirely in the command handler.

### Migration Script

`scripts/migrate_associated_videos.py` — standalone script for existing projects.

- Takes a project directory path as CLI argument
- Connects directly to `<project_dir>/vidseq.db` with plain `sqlite3` (no SQLAlchemy, no app startup)
- Checks if columns already exist via `PRAGMA table_info(videos)` (idempotent)
- Runs `ALTER TABLE videos ADD COLUMN is_associated INTEGER NOT NULL DEFAULT 0` (SQLite stores BOOLEAN as INTEGER; matches SQLAlchemy `Mapped[bool]` with `default=False`)
- Runs `ALTER TABLE videos ADD COLUMN associated_with_id INTEGER DEFAULT NULL` (matches SQLAlchemy `Mapped[int | None]` with `default=None`). Note: SQLite `ALTER TABLE ADD COLUMN` cannot add foreign key constraints, so no FK is enforced at the DB level for migrated databases. The SQLAlchemy model will have `ForeignKey("videos.id")` which is enforced for newly created databases. This is acceptable for SQLite where FK enforcement is optional.
- Prints what it did, exits

Usage: `uv run python scripts/migrate_associated_videos.py /path/to/project`

## Scope

**Files modified:**

Backend:
- `vidseq/models/video.py` — add `is_associated`, `associated_with_id` columns
- `vidseq/schemas/video.py` — add `is_associated`, `associated_with_id`, `associated_video_id` to VideoResponse
- `vidseq/services/video_service.py` — filter `get_all_videos()`, associated video ingestion, cascade delete
- `vidseq/services/array_storage.py` — add `is_associated` param to `create_video_segmentation_arrays()` and `reset_video_segmentation_arrays()` (skips detector_masks.h5 and tracker_logits.h5 for associated videos)
- `vidseq/services/frame_data_service.py` — add `load_all_scores(video_id, session)` function
- `vidseq/api/routes/videos.py` — new `POST /videos/associated`, `GET /videos/{id}/associated` endpoints
- `vidseq/services/segmentation_model/streaming_segmentor.py` — add five thin public wrappers: `propagate_with_box()`, `propagate_frame()`, `backtrack_reprop()`, `set_memory_frame()`, `evict_conditioning_frame()`; add `store_as_cond` param to `_propagate_single_frame()`
- `vidseq/services/segmentation_commands.py` — new `handle_propagate_with_associated` TCP handler
- `vidseq/services/segmentation_tcp_server.py` — register `"propagate_with_associated"` command type in `_handle_command()` dispatcher
- `vidseq/services/segmentation_service.py` — orchestrate batch co-segmentation (alongside existing `segment_all_videos`)
- `vidseq/services/segmentation_tcp_client.py` — new client method for associated propagation (uses `_send_streaming` for progress)
- `vidseq/api/routes/segmentation/` — new `POST /videos/associated/segmentation` endpoint. The REST path mixes video and segmentation concerns in the URL, but the handler file lives in `api/routes/segmentation/` which is the correct location for segmentation-related logic.

Frontend:
- `frontend/src/components/VideoPipeline.vue` — expand/collapse on cards, new sidebar buttons
- `frontend/src/components/AssociatedVideoDetail.vue` — new component
- `frontend/src/components/FilePickerModal.vue` — add `accept` and `singleSelect` props
- `frontend/src/services/api.ts` — new API functions
- `frontend/src/router/index.ts` — new route

Scripts:
- `scripts/migrate_associated_videos.py` — DB migration for existing projects

**No changes to:**
- Existing segmentation workflow logic (propagate, propagate-with-detector — unchanged behavior)
- Detector training/inference
- Alignment, PCA, cropping pipelines
