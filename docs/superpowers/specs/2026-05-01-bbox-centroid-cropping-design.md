# Bbox-Centroid Cropping — Design

**Date:** 2026-05-01
**Status:** In progress — being built up piece by piece.

## Overview

A new cropping mode that uses YOLO bbox detections (non-OBB) as the sole signal for placing the crop window. Distinct from any existing mask-based or OBB-based cropping path.

## Gating

Videos are eligible for this operation if they have at least one frame with a non-NULL `detector_bbox_*` — i.e., the bbox or seg detector has been run and produced at least one detection on the video. Selecting a video that doesn't meet this gate raises a clear "detector hasn't been run on video X" error before processing begins.

(No `segmentation_status` requirement — this mode does not use masks.)

## Coexistence with existing cropping

This is a **separate parallel mode** alongside the existing mask-based crop (`cropped_video_service.py`), not a replacement. The user selects between modes from the frontend.

**Cropped video output** is the same path for both modes:
- `cropped_videos/<name>_cropped.mp4`

so the two modes share the output path. Both gate their extraction on `cropped_video_exists(...)`, so **the first mode to write a given video wins**: clicking the other mode afterward returns `status: "skipped"` and does nothing. To switch modes for an already-cropped video, **delete the existing cropped video file first** and re-run. There is no in-app one-click "switch modes" today — that would require either per-mode output paths or a sentinel file recording which mode produced the artifact, both of which were considered out of scope for this iteration.

**This mode produces only the cropped video** — no `cropped_masks.h5`. The existing mask-based mode produces both; this one writes the video only.

**Crop size is persisted per-mode** in `crop_config.json`:
- `crop_size_mask` — used by the existing mask-centroid pipeline.
- `crop_size_bbox` — used by this new bbox-centroid pipeline.

Each mode reuses its own previously-computed value across runs and is unaffected by the other's saved value. Existing single-key `crop_size` from older project folders should be migrated to `crop_size_mask` on first read (or recomputed). `clear_crop_size` should clear both keys, or be split into per-mode clears.

## Algorithm

### 1. Per-frame centroid from bbox

For each frame in each selected video, compute the centroid of the YOLO bbox stored in `frame_data.detector_bbox_{x1,y1,x2,y2}`. The column is whatever was last written by either the bbox detector or the seg detector — no source filtering, last-write-wins. The user is responsible for ensuring the desired detector ran most recently on each selected video.

**NULL bbox handling**: Frames where `detector_bbox_*` is NULL (no detection / sub-threshold confidence) get their centroid filled in by **linear interpolation** between the surrounding valid centroids, before the smoothing pass.

- **Boundary NULLs** (leading or trailing NULL runs with no neighbor on one side): hold the nearest valid centroid (forward-fill from the first valid centroid backward, last valid centroid forward).
- **Entire video has zero valid bboxes**: hard-fail for that video. Surface a clear error naming the video.
- **No max-gap threshold**: linear interpolation is applied across any gap size. The user is responsible for ensuring the detector was trained well enough to avoid pathologically long gaps.

### 2. Smooth the centroid signal

Per video, smooth the centroid time series in two stages:

1. **5-frame median filter** on the raw centroid signal.
2. **Gaussian filter** on the median-filtered signal, with `sigma = ceil(fps / 10)` frames (i.e. 0.1 s in time, rounded up to the nearest whole frame using each video's known fps).

The output of stage 2 is the **true centroid** used for cropping.

Both filters use `mode='nearest'` for boundary handling (replicate the edge value when the kernel extends past the start or end of the signal). Consistent with the boundary-NULL rule (hold-nearest-valid) and avoids pulling smoothed centroids toward (0,0) the way zero-padding would.

### 3. Crop window size (one value per operation, shared across all selected videos)

1. Across every frame of every selected video, gather bbox widths and bbox heights.
2. Take the **99th percentile** of widths and the **99th percentile** of heights (use 99p, not true max, to avoid outliers). Frames with NULL `detector_bbox_*` are excluded from the percentile pool.
3. Take the smaller of the two 99p values and set it equal to the larger → a **square** crop window.
4. Add **15%** padding to that size for safety.

This single square size is the crop window for the operation.

### 4. Crop

For each frame, crop a window of the size computed in step 3, centered on the smoothed centroid from step 2.

When the crop window extends past a frame edge (centroid near a boundary, or crop window larger than the source video), **black-pad**: copy the available source pixels into the crop and leave the rest as zeros. The crop is always perfectly centered on the smoothed centroid; black borders appear when the animal is near a frame edge or the source video is smaller than the crop window. Matches the existing mask-based pipeline.

## Frontend

In the VideoPipeline sidebar's `Cropped Videos` section, two buttons side by side:

- `Crop (mask)` — existing mask-centroid pipeline (rename from `Extract Cropped Videos`).
- `Crop (bbox)` — new bbox-centroid pipeline.

## Known limitations

- **No alignment / PCA support after bbox-mode.** The downstream alignment pipeline (`alignment_service.py`) reads `cropped_masks.h5` for per-frame mask data and dimensions. Bbox-mode does not write that file. Running alignment on a video cropped in bbox-mode will either crash (file missing) or, worse, silently corrupt output if a stale `cropped_masks.h5` from a previous mask-mode run happens to be present. Bbox-mode is therefore a **pipeline leaf** today — it is the first of a planned set of rewrites in this area, and the alignment stage will be replaced with a bbox-compatible version separately.
- **Mode switch requires manual file deletion.** See "Coexistence with existing cropping" above.
