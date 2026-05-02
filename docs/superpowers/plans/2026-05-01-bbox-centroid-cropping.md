# Bbox-Centroid Cropping Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a second cropping mode that places the crop window using YOLO bbox detections (non-OBB) instead of mask centroids. Same output paths as the existing mask-based crop (overwrites), but no `cropped_masks.h5` is written.

**Architecture:** Parallel mode in the existing `cropped_video_service.py`. Shares `_reencode_to_h264`, the `CroppedVideoService` singleton (so only one extraction runs at a time across both modes), and `crop_config.json` (with per-mode keys). Reads `frame_data.detector_bbox_*` directly from the DB — last-write-wins semantics, no source-detector filtering. Frontend exposes the two modes as side-by-side buttons.

**Tech Stack:** FastAPI, SQLAlchemy, scipy.ndimage (`median_filter`, `gaussian_filter1d`), numpy.interp, OpenCV, Vue 3.

**Spec:** `docs/superpowers/specs/2026-05-01-bbox-centroid-cropping-design.md`

---

### Task 1: Per-mode crop config persistence

**Files:** Modify `vidseq/services/cropped_video_service.py`

- [ ] **Step 1.1: Add per-mode read/write helpers**

  Add new helpers `get_saved_crop_size_mask`, `save_crop_size_mask`, `get_saved_crop_size_bbox`, `save_crop_size_bbox` that read/write `crop_size_mask` / `crop_size_bbox` keys in `crop_config.json` respectively.

  In `get_saved_crop_size_mask`, fall back to the legacy `crop_size` key if `crop_size_mask` is missing — old projects saved the mask-mode value under the bare `crop_size` name. Don't fall back in the bbox helper; bbox mode is new so an absent key just means "not yet computed."

- [ ] **Step 1.2: Update existing call sites**

  Replace the existing `get_saved_crop_size`/`save_crop_size` calls in `extract_all_cropped_videos` (the mask-mode path) with the new mask-specific helpers. Keep the old function names around as thin aliases that call the mask helpers, in case anything else in the codebase imports them — `rg get_saved_crop_size` first.

- [ ] **Step 1.3: Update `clear_crop_size`**

  `clear_crop_size(project_path)` should delete the whole `crop_config.json` file (clears both keys at once). Document that this is intentional — there's no per-mode clear in the UI today.

---

### Task 2: Centroid signal computation

**Files:** Modify `vidseq/services/cropped_video_service.py` (add new functions; do not touch existing mask-mode functions)

- [ ] **Step 2.1: Bulk-load bboxes for a video**

  New `async def load_bboxes_for_video(session, video_id, num_frames) -> np.ndarray` that runs a single SQL query selecting `(frame_idx, detector_bbox_x1, detector_bbox_y1, detector_bbox_x2, detector_bbox_y2)` from `frame_data` for the given `video_id`, ordered by `frame_idx`. Returns a `(num_frames, 4)` float32 array indexed by frame_idx, with NaN in rows that had NULL bboxes (or that didn't appear in the table at all).

  Allocate the output as `np.full((num_frames, 4), np.nan, dtype=np.float32)` and populate only the rows returned by the query. Frames with no `frame_data` row → stay NaN (treated as no detection).

- [ ] **Step 2.2: Centroids + NULL fill**

  Synchronous helper `compute_smoothed_centroids(bboxes: np.ndarray, fps: float) -> np.ndarray`:

  1. Compute raw centroids: `cx = (x1+x2)/2`, `cy = (y1+y2)/2`. Shape `(N, 2)`, NaN where bbox was NaN.
  2. If every row is NaN, raise `ValueError("video has no valid bboxes")`. The caller turns this into a per-video hard-fail message.
  3. Linear-interpolate NaN rows: for each axis (cx, cy) independently, use `np.interp` over the indices of valid rows.
  4. Hold-nearest at boundaries: leading NaNs fill with the first valid centroid, trailing NaNs with the last. (`np.interp` already does this by default — its `left=`/`right=` default to the endpoint values — but verify.)
  5. 5-frame median: `scipy.ndimage.median_filter(arr, size=(5, 1), mode='nearest')` applied to the `(N, 2)` array. Shape preserved.
  6. Gaussian: `sigma = math.ceil(fps / 10)` frames. Apply `scipy.ndimage.gaussian_filter1d(arr, sigma=sigma, axis=0, mode='nearest')`.
  7. Return the smoothed `(N, 2)` array.

  Add `import math` and `from scipy.ndimage import median_filter, gaussian_filter1d` at the top of the file.

---

### Task 3: Bbox-mode global crop size

**Files:** Modify `vidseq/services/cropped_video_service.py`

- [ ] **Step 3.1: New `compute_global_crop_size_bbox`**

  ```python
  async def compute_global_crop_size_bbox(
      session: AsyncSession,
      videos: list,
  ) -> int:
      """Compute the global square crop size for bbox-centroid mode.

      Pools all non-NULL detector bboxes across the selected videos.
      Returns ceil(max(p99(widths), p99(heights)) * 1.15) as a square crop.
      """
  ```

  Implementation:
  - Single SQL query across all video_ids: `SELECT detector_bbox_x1, detector_bbox_y1, detector_bbox_x2, detector_bbox_y2 FROM frame_data WHERE video_id IN (...) AND detector_bbox_x1 IS NOT NULL`.
  - Compute `widths = x2 - x1` and `heights = y2 - y1` arrays.
  - `p99_w = np.percentile(widths, 99)`, `p99_h = np.percentile(heights, 99)`.
  - `square = max(p99_w, p99_h)`.
  - `crop_size = int(math.ceil(square * 1.15))`.
  - Log the inputs (count of bboxes, p99 values) and the output. Return.
  - Edge case: empty result set (no non-NULL bboxes anywhere across the selection) → raise `ValueError("no detector bboxes across selected videos; run the detector first")`. The gating in Task 5 should make this unreachable, but defend in depth.

---

### Task 4: Per-video bbox-mode extraction

**Files:** Modify `vidseq/services/cropped_video_service.py`

- [ ] **Step 4.1: New `process_single_video_bbox`**

  ```python
  def process_single_video_bbox(
      project_path: Path,
      video,
      crop_size: int,
      smoothed_centroids: np.ndarray,  # (num_frames, 2), already smoothed
  ) -> bool:
  ```

  Mostly a stripped-down copy of `process_single_video` / `_process_frames_with_masks`:

  - Open video with `cv2.VideoCapture`, create `cv2.VideoWriter` to a `.temp.mp4` with `mp4v`, fps from cap.
  - Iterate frames:
    - `cx, cy = smoothed_centroids[frame_idx]` rounded to int.
    - Compute `x1, y1, x2, y2` crop bounds the same way the existing function does.
    - Allocate `cropped_frame = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)` (black).
    - Compute clipped source/destination regions and copy.
    - **Do not** write any mask data. Do not open `cropped_masks.h5`.
    - `writer.write(cropped_frame)`.
  - Re-encode via existing `_reencode_to_h264` to the final path. Atomic-rename via `_tmp.mp4` like the existing function.
  - Return `True`/`False` on success/failure.

  This function does not need to be async. It runs in the same background thread the existing per-video processing uses.

---

### Task 5: Bbox-mode service entrypoint

**Files:** Modify `vidseq/services/cropped_video_service.py`

- [ ] **Step 5.1: Gating helper**

  ```python
  async def videos_with_no_bboxes(session, video_ids: list[int]) -> list[int]:
      """Return the subset of video_ids that have zero non-NULL detector bboxes."""
  ```

  Single SQL query: `SELECT video_id, COUNT(*) FROM frame_data WHERE video_id IN (...) AND detector_bbox_x1 IS NOT NULL GROUP BY video_id`. Compare against the input list to find missing/zero-count video IDs.

- [ ] **Step 5.2: New `create_videos_extraction_bbox`**

  Mirrors the existing `create_videos_extraction` but with bbox-mode gating and dispatch:

  - Fetch videos by ID.
  - Validate gating: call `videos_with_no_bboxes`. If non-empty, raise `ValueError("Detector hasn't been run on: <names>. Run the bbox or seg detector before bbox-centroid cropping.")`. **Note:** unlike the mask-mode entrypoint, do **not** require `segmentation_status == "segmented"` — bbox mode doesn't use masks.
  - Filter out already-cropped (same `cropped_video_exists` check). The cropped-video file path is shared between modes, so this filter behaves the same.
  - Dispatch to `CroppedVideoService.extract_all_cropped_videos_bbox(...)`.

- [ ] **Step 5.3: Singleton extraction method**

  Add `extract_all_cropped_videos_bbox(self, project_path, videos, ...)` to the `CroppedVideoService` class. Reuse the same `_is_extracting` flag — only one extraction (either mode) runs at a time.

  Inside the background `_extract` closure:

  1. Get-or-compute crop size via `get_saved_crop_size_bbox` / `compute_global_crop_size_bbox` / `save_crop_size_bbox`. Note: `compute_global_crop_size_bbox` is async — the worker thread needs to run it via `asyncio.run(...)` since the thread isn't inside an event loop. (Same pattern: open a fresh project DB session inside the thread to call it.)
  2. For each video: load bboxes (`load_bboxes_for_video`, also async), compute smoothed centroids (`compute_smoothed_centroids`), call `process_single_video_bbox`. Catch `ValueError` from the all-NaN case and log `"Skipping video <name>: <error>"`, continue with the rest. Other failures: log and continue (matches existing behavior).
  3. Print a final summary like the mask-mode path does.

  Open question for the implementer: the existing path uses `videos: list` typed as the SQLAlchemy model and pulls `num_frames` / `name` directly from the model. Do the same here — fetch fully-loaded `Video` objects in `create_videos_extraction_bbox` before handing off to the singleton, and let the worker thread pass them through without needing DB access for video metadata. Only the bbox queries need a DB session inside the thread.

---

### Task 6: API route

**Files:** Modify `vidseq/api/routes/cropped_videos.py`

- [ ] **Step 6.1: New `POST /projects/{project_id}/videos/bbox-extraction` route**

  Mirror the existing `create_videos_extraction` route exactly, but call `cropped_video_service.create_videos_extraction_bbox(...)`. Same `VideoSelectionRequest` schema. Same 400-on-`ValueError` translation.

  Path bikeshed: existing is `/projects/{id}/videos/extraction`. New is `/projects/{id}/videos/bbox-extraction`. (We could rename the existing to `/videos/mask-extraction` for symmetry but that's a breaking change for any existing client; leave it.)

---

### Task 7: Frontend API wrapper

**Files:** Modify `frontend/src/services/api.ts`

- [ ] **Step 7.1: Add `createVideosExtractionBbox`**

  Add a new exported async function paralleling `createVideosExtraction`, hitting the new bbox-extraction endpoint with the same request/response shape.

---

### Task 8: Frontend buttons

**Files:** Modify `frontend/src/components/VideoPipeline.vue`

- [ ] **Step 8.1: Add bbox-extraction state**

  Add a new ref `isExtractingBbox = ref(false)` alongside `isExtracting`. Don't reuse a single flag — the user can't run both simultaneously (backend enforces the singleton lock), but separate UI flags make it obvious which button started a run.

  Add a new handler `handleExtractCroppedVideosBbox` paralleling the existing handler, calling `createVideosExtractionBbox`.

- [ ] **Step 8.2: Update sidebar buttons**

  In the `Cropped Videos` section, replace the single button with two buttons side by side. Container is a flex row with gap.

  ```html
  <h4 class="sidebar-section-title">Cropped Videos</h4>
  <div class="crop-mode-buttons">
    <button
      class="sidebar-button extract-button"
      @click="handleExtractCroppedVideos"
      :disabled="isExtracting || isExtractingBbox || selectedCount === 0"
    >
      <span class="button-label">Crop (mask)</span>
    </button>
    <button
      class="sidebar-button extract-button"
      @click="handleExtractCroppedVideosBbox"
      :disabled="isExtracting || isExtractingBbox || selectedCount === 0"
    >
      <span class="button-label">Crop (bbox)</span>
    </button>
  </div>
  ```

  Add minimal CSS:
  ```css
  .crop-mode-buttons {
    display: flex;
    gap: 0.5rem;
  }
  .crop-mode-buttons .sidebar-button {
    flex: 1;
  }
  ```

  The existing button label was dynamic (`Extract ${selectedCount} Cropped Videos`). Two buttons in a 250px sidebar can't fit that long label, so the static `Crop (mask)` / `Crop (bbox)` is the right choice. The selected-count is already visible elsewhere on the page; users won't lose it.

- [ ] **Step 8.3: Mutual disabling**

  Both buttons disable on either flag plus `selectedCount === 0`. While one mode is running, the other is disabled too — the backend rejects concurrent runs anyway, but disabling is clearer UX.

---

### Task 9: Manual testing checklist

- [ ] Apply the bbox detector to a small project (2-3 videos). Click `Crop (bbox)`. Verify cropped videos appear in `cropped_videos/`, are H.264, and the animal is visibly centered with the expected smoothing (no per-frame jitter).
- [ ] Verify `crop_config.json` now contains `crop_size_bbox` (not just `crop_size`).
- [ ] Click `Crop (mask)` on the same project (with masks present). Verify cropped videos overwrite cleanly and the file is the mask-mode output. `crop_config.json` should now contain both `crop_size_mask` and `crop_size_bbox`.
- [ ] On a project with masks but **no detector run**, verify `Crop (bbox)` errors with the "Detector hasn't been run" message before any work happens.
- [ ] On a video where the bbox detector failed on a small contiguous run of frames (e.g. 50–100 frames in the middle), verify cropping completes and the cropped video looks reasonable through the gap (interpolated centroid + smoothing).
- [ ] On a video where the bbox detector found absolutely nothing, verify it is skipped with a clear log message and the rest of the batch continues.
- [ ] Verify no `cropped_masks.h5` is created or modified by the bbox-mode run.
- [ ] Verify the singleton lock: try clicking `Crop (mask)` while `Crop (bbox)` is still running (or vice versa). Both buttons should be disabled until the running extraction finishes.

---

## Notes for the implementer

- **Async/sync split.** Bbox loading, the gating query, and global-crop-size compute all need a DB session, so they're async. The per-video processing loop (cv2 reads, scipy smoothing, ffmpeg re-encode) is sync and runs in the existing background thread. Use `asyncio.run(...)` inside the thread for the async parts, mirroring whatever pattern is used in similar background-thread code elsewhere in the project (`rg asyncio.run vidseq` to find examples).

- **Reuse, don't refactor.** Don't change anything about the existing mask-mode path beyond Step 1.2 (which only touches the persistence helpers). The two modes coexist; risk-of-regression on the working mask path should be near zero.

- **Don't add detector-source filtering.** The spec is explicit: read `detector_bbox_*` as-is, last-write-wins. The shared-column ambiguity is logged on TODO.md as a separate fix.

- **Edge-case math.** `np.percentile` raises if given an empty array; `np.interp` requires at least one valid point in the input. Both are guarded by the gating step (≥1 non-NULL bbox per video) plus the all-NaN raise inside `compute_smoothed_centroids`. Don't add redundant guards.
