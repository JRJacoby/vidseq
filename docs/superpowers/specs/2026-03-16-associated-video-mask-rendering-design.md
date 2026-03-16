# Associated Video Mask Rendering

Wire up mask overlay rendering in `AssociatedVideoDetail.vue`. The component currently has a video player and view mode selector but doesn't display masks.

## View Modes

Three read-only view modes, no interactive prompts (`activeTool` stays `'none'`, no point handlers wired to `VideoOverlay`):

- **final_mask**: Final mask overlay on the associated video. Reuse `useSegmentation` composable pointed at the associated video's ID with `maskViewMode = 'final'`.
- **tracker_mask**: Tracker mask overlay on the associated video. Same as above with `maskViewMode = 'tracker'`.
- **main_bbox**: Bounding box from the **main** video's tracker mask, drawn on the associated video. Uses a new backend endpoint (see below). Coordinates rescaled from main video native dimensions to associated video native dimensions using `mainVideo.width`/`mainVideo.height` and `associatedVideo.width`/`associatedVideo.height` from the DB model. If the main video has no tracker masks (unsegmented), nothing is displayed.

## Backend: Bbox Endpoint

New endpoints on the segmentation mask routes. Route functions named `get_tracker_mask_bbox` and `get_tracker_mask_bboxes` per naming conventions.

**Single frame:**

`GET /projects/{project_id}/videos/{video_id}/segmentation/tracker-mask-bboxes/{frame_idx}`

Returns `{ bbox: { x1, y1, x2, y2 } | null }` in the video's native pixel coordinates. Implementation: read frame from tracker_masks H5, call `compute_bbox_from_mask`, return result. Returns `{ bbox: null }` if the mask is empty.

**Batch (for prefetch):**

`GET /projects/{project_id}/videos/{video_id}/segmentation/tracker-mask-bboxes?start_frame=N&count=100`

Returns `{ bboxes: [{ frame_idx, x1, y1, x2, y2 }, ...] }`. Only includes frames where the mask is non-empty.

Note: these paths use `tracker-mask-bboxes` (not nested under `tracker-masks/{frame_idx}`) to avoid route collision with the existing `tracker-masks/{frame_idx}` PNG endpoint.

## Frontend Changes

### AssociatedVideoDetail.vue

**Template:**
- Add a `.video-wrapper` div (position: relative, display: inline-block) around the `<video>` element, matching `VideoDetail.vue`'s pattern. This allows `VideoOverlay` to position its canvas absolutely over just the video.
- Import and render `VideoOverlay` inside `.video-wrapper`. Pass `videoWidth`/`videoHeight` from `useVideoPlayback` (destructure these from the existing call). Pass `currentMask` from `useSegmentation` when in tracker/final mode, or `null` in main_bbox mode. Pass the rescaled bbox as the `detectorBbox` prop when in main_bbox mode (intentional reuse of this prop — add a code comment noting it's a main video bbox, not a detector bbox).

**Mask modes (tracker_mask, final_mask):**
- Import and use `useSegmentation` composable for these modes. Initialize with the associated video's ID and project ID. Pass a computed `maskViewMode` that maps `'tracker_mask' → 'tracker'` and `'final_mask' → 'final'`.
- Wire `setMetadataCallback` to trigger the first mask load when the video is ready (matching `VideoDetail.vue`'s pattern).
- Call `useSegmentation.seekToFrame()` on seek events. The composable's `syncMaskToVideo` rAF loop handles smooth playback.
- When switching away from these modes, the `maskViewMode` watcher in `useSegmentation` clears the cache and reloads — no additional cleanup needed.

**Main bbox mode:**
- Fetch bboxes from the **main video** (using `mainVideoId`, not `associatedVideo.id`) via the new bbox endpoint.
- Cache in a local `Map<number, {x1,y1,x2,y2} | null>`.
- Implement a lightweight rAF playback sync loop (matching `useSegmentation`'s pattern): on each animation frame during playback, compute the current frame index, look up the cached bbox, rescale coordinates, and set a reactive `currentBbox` ref.
- Prefetch strategy: batch-fetch 100 bboxes ahead, trigger prefetch when fewer than 100 frames remain in cache (same thresholds as `useSegmentation`).
- On seek: fetch single bbox for the target frame if not cached.
- No cache size limit needed — bboxes are 4 floats each, negligible memory even for long videos.
- When switching to this mode, clear `currentMask` (set to null). When switching away, stop the bbox rAF loop.

### api.ts

Add two functions:

```typescript
getTrackerMaskBbox(projectId, videoId, frameIdx) → Promise<{x1,y1,x2,y2} | null>
getTrackerMaskBboxes(projectId, videoId, startFrame, count) → Promise<Array<{frame_idx, x1, y1, x2, y2}>>
```

### VideoOverlay.vue

No changes needed — it already renders masks via `drawImage` and bboxes via `strokeRect`.

## What This Does NOT Include

- No interactive segmentation (prompts, point clicking)
- No detector mask mode (associated videos don't have detector masks)
- No mask editing
- No DataTrack confidence visualization (future work)
