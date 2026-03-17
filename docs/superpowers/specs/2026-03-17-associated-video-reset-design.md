# Associated Video Segmentation Reset

Add a reset button to the AssociatedVideoDetail screen that clears co-segmentation results without touching the main video.

## Backend

`DELETE /projects/{project_id}/videos/{video_id}/associated-segmentation`

1. Verify `video.is_associated == True`, else 400.
2. Call `reset_video_segmentation_arrays(project_path, video_id, num_frames, height, width, is_associated=True)` — recreates tracker_masks.h5 and final_masks.h5, skips detector/logits.
3. Delete all `frame_data` rows for the video.
4. Set `video.segmentation_status = None`.
5. Commit.

Does not touch the main video's data.

## Frontend

### api.ts

```typescript
resetAssociatedSegmentation(projectId, videoId) → Promise<void>
```

DELETE request to the endpoint above.

### AssociatedVideoDetail.vue

Add "Reset Segmentation" button in the action bar sidebar. Disabled when `segmentation_status` is null. On click: call API, reload associated video metadata, clear mask cache.
