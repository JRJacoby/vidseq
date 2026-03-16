# Associated Video Mask Rendering

Wire up mask overlay rendering in `AssociatedVideoDetail.vue`. The component currently has a video player and view mode selector but doesn't display masks.

## View Modes

Three read-only view modes, no interactive prompts:

- **final_mask**: Final mask overlay on the associated video. Reuse `useSegmentation` composable pointed at the associated video's ID with `maskViewMode = 'final'`.
- **tracker_mask**: Tracker mask overlay on the associated video. Same as above with `maskViewMode = 'tracker'`.
- **main_bbox**: Bounding box from the main video's tracker mask, drawn on the associated video. Uses a new backend endpoint (see below). Coordinates rescaled from main video dimensions to associated video dimensions.

## Backend: Bbox Endpoint

New endpoint on the segmentation mask routes:

`GET /projects/{project_id}/videos/{video_id}/segmentation/tracker-masks/{frame_idx}/bbox`

Returns `{ bbox: { x1, y1, x2, y2 } | null }` in the video's native pixel coordinates. Implementation: read frame from tracker_masks H5, call `compute_bbox_from_mask`, return result. Returns `null` if the mask is empty.

Batch version for prefetch:

`GET /projects/{project_id}/videos/{video_id}/segmentation/tracker-masks/bboxes?start_frame=N&count=100`

Returns `{ bboxes: [{ frame_idx, x1, y1, x2, y2 }, ...] }`. Only includes frames where the mask is non-empty.

## Frontend Changes

### AssociatedVideoDetail.vue

- Import and use `useSegmentation` composable for tracker/final mask modes. Initialize with the associated video's ID and project ID. Pass a computed `maskViewMode` that maps `'tracker_mask' → 'tracker'` and `'final_mask' → 'final'`.
- Import and render `VideoOverlay` component inside `.video-container`, positioned over the video. Pass `currentMask` from `useSegmentation` and `videoWidth`/`videoHeight` from `useVideoPlayback`.
- For `main_bbox` mode: don't use `useSegmentation`. Instead, fetch bboxes from the **main video** using the new bbox endpoint. Cache in a local `Map<number, {x1,y1,x2,y2} | null>`. On frame change, look up the bbox, rescale by `(assocWidth/mainWidth, assocHeight/mainHeight)`, and pass to `VideoOverlay` as the `detectorBbox` prop (which already renders stroked rectangles).
- Wire up playback sync: call `useSegmentation.seekToFrame()` on seek events and use the existing `syncMaskToVideo` rAF loop during playback, but only when in tracker/final mode.

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
