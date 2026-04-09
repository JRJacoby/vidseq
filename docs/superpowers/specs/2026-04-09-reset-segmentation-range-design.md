# Reset Segmentation Range Design

**Date:** 2026-04-09
**Branch:** feat/associated-videos

## Overview

Add the ability to click on a contiguous block of tracker-masked frames in the DataTrack and delete all segmentation data (masks, logits, conditioning frames, frame_data, SAM memory) for that range. Uses the same click-to-select + Delete key interaction as training range management.

## Backend

### New endpoint

`DELETE /projects/{project_id}/videos/{video_id}/segmentation/range?start_frame={start}&end_frame={end}`

Calls the existing `video_service.delete_frame_data()` for each frame in `[start_frame, end_frame]` inclusive. This reuses the same per-frame cleanup logic (H5 zeroing, DB deletes, SAM memory reset) that the single-frame reset already uses.

Add a new service function `delete_frame_data_range()` in `video_service.py` that batches the DB deletes (single DELETE WHERE frame_idx BETWEEN) and loops the H5 zeroing + SAM memory reset per frame.

### Route location

Add to `vidseq/api/routes/videos.py` alongside the existing `DELETE .../segmentation/{frame_idx}` endpoint.

## Frontend

### DataTrack changes

The tracker masked ranges (blue blocks) become selectable when NOT in marking mode:

- Click on a blue (tracker masked) range block → selects it (`selectedMaskedRange` state, white dashed border + pulse animation, same style as selected training ranges)
- Click elsewhere or press Escape → deselects
- Press Delete/Backspace while a masked range is selected → emit `'unmark-masked'` event with `(startFrame, endFrame)`

This mirrors exactly how training ranges work: click to select, Delete to remove.

### VideoDetail changes

- Handle the `'unmark-masked'` event from DataTrack
- Call `DELETE /segmentation/range?start_frame=...&end_frame=...`
- Clear mask cache for that range
- Refresh frame ranges

### api.ts

Add `deleteSegmentationRange(projectId, videoId, startFrame, endFrame)` function.

## Scope

- No new mode button needed
- No new composable needed
- Reuses existing visual patterns (selected range styling)
- ~3 files changed: `videos.py` route, `video_service.py` batch function, `DataTrack.vue` selection logic, `VideoDetail.vue` handler, `api.ts` function
