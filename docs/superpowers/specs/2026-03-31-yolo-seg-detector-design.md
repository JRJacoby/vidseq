# YOLO Segmentation Detector

## Problem

The existing detector produces bounding boxes only. For some workflows, per-frame segmentation masks are needed without the cost of running SAM2. YOLO11x-seg can produce instance segmentation masks at real-time speed, providing a fast alternative to SAM2 for mask generation.

## Design

Add a parallel "Seg Detector" system that trains YOLO11x-seg on the existing training masks (converted to polygon labels) and applies it to produce per-frame segmentation masks. Mirrors the existing Train Detector / Apply Detector workflow. Masks are written to the pre-existing `detector_masks.h5` files. A new "Seg" view mode in VideoDetail displays them.

### Model

Pretrained: `yolo11x-seg.pt` (62M params, ~2-3GB VRAM at inference). Added to `PRETRAINED_MODELS` in `detector_model.py`:
```python
PRETRAINED_MODELS = {
    "rtdetr": "rtdetr-x.pt",
    "yolo": "yolo11n.pt",
    "obb": "yolo11n-obb.pt",
    "seg": "yolo11x-seg.pt",  # Extra-large for best quality
}
```

Trained weights saved to `project_path/models/seg_detector.pt`.

### Training Label Format

YOLO segmentation labels use polygon contour points (normalized to [0,1]):
```
0 x1 y1 x2 y2 x3 y3 ...
```

New function `_mask_to_polygon_label()` in `detector_service.py`:
1. `cv2.findContours` on the binary mask
2. Take the largest contour by area (handles disconnected regions)
3. Simplify with `cv2.approxPolyDP` (epsilon = 0.001 * perimeter) to keep polygon manageable
4. If resulting polygon has fewer than 3 points, return `None` (degenerate)
5. Normalize all points to [0,1] by image dimensions
6. Output: `"0 x1 y1 x2 y2 ..."` per the YOLO segmentation format
7. Return `None` if mask is empty (same as `_mask_to_yolo_bbox`)

### Detector Model

Add `detect_seg()` to `detector_model.py`:
```python
def detect_seg(
    model,
    frame: np.ndarray,
    conf: float = DETECTION_CONF_THRESHOLD,
) -> list[dict]:
    """Run segmentation detection on a single BGR frame.

    Returns:
        List of detections sorted by confidence (descending).
        Each detection: {"bbox": (x1,y1,x2,y2), "mask": np.ndarray, "conf": float}
        bbox in pixel coords. mask is at YOLO's internal resolution (~160x160),
        needs cv2.resize to original frame dimensions before storage.
    """
    results = model(frame, conf=conf, verbose=False)
    detections = []
    if len(results) > 0 and results[0].boxes is not None and results[0].masks is not None:
        boxes = results[0].boxes
        masks = results[0].masks
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().tolist()
            mask = masks.data[i].cpu().numpy()
            detections.append({
                "bbox": (x1, y1, x2, y2),
                "mask": mask,
                "conf": boxes.conf[i].item(),
            })
    detections.sort(key=lambda d: d["conf"], reverse=True)
    return detections
```

Used by `_apply_to_training_data` for post-training single-frame inference. The batch inference path in `handle_apply_detector` extracts results inline from the Ultralytics result object (same pattern as bbox/OBB batch paths).

### Training

Extends `DetectorService` with `training_type="seg"`.

**Code sites that need a third branch for `"seg"` (currently binary `"obb"` vs else):**

1. **`_train_sync` — detector type resolution** (line ~246): Add `elif training_type == "seg": detector_type = "seg"` before the else that reads from config.

2. **`_write_yolo_dataset` — label dispatch** (line ~453): Add `elif detector_type == "seg": yolo_line = _mask_to_polygon_label(mask, img_h, img_w)`.

3. **`_train_sync` — model type branching** (line ~304): Add seg case with `train_workers = 4`, `train_batch = max(batch_size, 8)`, `train_name = "seg_train"`.

4. **`_train_sync` — task kwarg** (line ~380): Add `elif detector_type == "seg": train_kwargs["task"] = "segment"`.

5. **`_train_sync` — weights destination** (line ~388): Change to `weights_dest = {"obb": "obb_detector.pt", "seg": "seg_detector.pt"}.get(training_type, "detector.pt")`.

6. **`_apply_to_training_data` — weights path** (line ~519): Same dict lookup for weights name.

7. **`_apply_to_training_data` — inference branch** (line ~555): Add `elif training_type == "seg":` branch that calls `detect_seg()`, extracts mask + bbox + conf, writes mask to `detector_masks.h5`, sets `has_detector_mask` flag.

**`VALID_DETECTOR_TYPES`**: Not updated — seg is not a project-level detector config choice (like rtdetr/yolo). It's a separate parallel system. The seg training path hardcodes `detector_type = "seg"` and bypasses the config read, same as OBB.

**New method on `DetectorService`:**
```python
def seg_model_exists(self, project_path: Path) -> bool:
    return (project_path / "models" / "seg_detector.pt").exists()
```

**Post-training apply** (`_apply_to_training_data` for seg): Runs `detect_seg()` per training frame. For each detection, writes mask to `detector_masks.h5` (opened with `detector_masks(project_path, video_id, "a")` context manager), saves bbox/score to DB, AND calls `set_has_detector_mask_batch_sync` to set the DB flag. This ensures `segDetectorMasksExist()` returns true immediately after training.

### Apply (Inference)

Extends `handle_apply_detector` with `detector_type="seg"`. Key difference from bbox-only: **the handler opens `detector_masks.h5` and writes masks during the batch inference loop** (masks are too large to accumulate and return via TCP). This is an intentional exception to the CLAUDE.md norm that the GPU worker should not touch H5 files — justified because streaming full-resolution masks over TCP would be prohibitively expensive. The existing propagation handlers (`handle_propagate_with_detector`) already write to H5 from the worker, so this is a well-established exception.

**Per-video structure**: One `with detector_masks(project_path, video_id, "a") as mask_data:` context wrapping that video's batch loop. `orig_w` and `orig_h` come from `frame_source.width` and `frame_source.height` (already available on the `VideoFrameSource` object).

Per-frame extraction in batch loop:
```python
if detector_type == "seg":
    if result.masks is not None and len(result.masks) > 0:
        best_i = result.boxes.conf.argmax()
        mask = result.masks.data[best_i].cpu().numpy()
        mask_binary = (mask > 0.5).astype(np.uint8) * 255
        mask_resized = cv2.resize(mask_binary, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        mask_data[frame_idx] = mask_resized
        x1, y1, x2, y2 = result.boxes.xyxy[best_i].cpu().tolist()
        conf = result.boxes.conf[best_i].item()
        scores.append([idx, conf])
        bboxes.append([idx, x1, y1, x2, y2])
    else:
        scores.append([idx, 0.0])
```

Uses `INTER_LINEAR` for smoother mask edges when upscaling from ~160px to original resolution.

Returns via TCP: bboxes + scores in the same `detector_scores` / `detector_bboxes` keys (same format as existing detector). Masks are already written to H5.

Service layer (`apply_seg_detector`): After TCP call, saves bboxes/scores to DB AND sets `has_detector_mask=True` for all processed frames. Add async `set_has_detector_mask_batch()` to `frame_data_service.py` (the existing batch version is sync-only).

### Shared DB Columns

The seg detector writes to the same `detector_bbox_*`, `detector_score`, and `has_detector_mask` columns as the bbox detector. Running "Apply Seg Detector" overwrites any existing bbox detector results, and vice versa. This is an accepted limitation — the two detectors are not expected to be used simultaneously on the same video. The `has_detector_mask` flag is only set by the seg detector (the bbox detector does not set it), so it serves as an implicit indicator of which detector was last applied.

### "Detector" vs "Seg" View Mode Interaction

The existing "Detector" view mode and the new "Seg" view mode both read from `detector_masks.h5`. However, this does not cause user-visible confusion because:

- **"Detector" mode** (`maskViewMode === 'detector'`) fetches a **bbox** via `getDetectorBbox` in `fetchMaskForFrame` and returns `null` for the mask. It renders the bbox as a rectangle overlay. It does NOT render the H5 mask data even though the batch prefetch fetches it.
- **"Seg" mode** (`maskViewMode === 'seg'`) fetches the actual **mask** via `getDetectorMask` in `fetchMaskForFrame` and renders it as a mask overlay. No bbox overlay.

So "Detector" mode always shows bboxes (from DB), and "Seg" mode always shows masks (from H5). They read different things even though the H5 file is shared. No conflict.

### TCP Client

New `apply_seg_detector()` module-level wrapper that calls `apply_detector(project_path, videos, detector_type="seg")`.

### API Endpoints

All relative to `/projects/{project_id}`:

| Method | Path | Description |
|--------|------|-------------|
| GET | `/detection/seg/status` | Seg model exists + training status (checks `_training_type == "seg"`) |
| POST | `/detection/seg/training` | Start seg training |
| DELETE | `/detection/seg/training` | Stop seg training |
| GET | `/detection/seg/training` | Seg training progress |
| GET | `/detection/seg/training/stream` | SSE for seg training |
| POST | `/videos/seg-detection` | Apply seg detector to selected videos |
| GET | `/videos/{id}/seg-detector-masks/exists` | Check `has_detector_mask` flags in DB (not H5 file existence) |

Existing endpoints for detector masks (`/detector-masks/{frame_idx}`, `/detector-masks`) already serve from `detector_masks.h5` — no new mask-serving endpoints needed.

### Frontend: VideoDetail — New "Seg" View Mode

Add `'seg'` to `MaskViewMode` in both `VideoDetail.vue` and `useSegmentation.ts`:
```
'tracker' | 'detector' | 'final' | 'obb' | 'seg'
```

**`useSegmentation.ts` changes** — three code paths must handle `'seg'`:

1. **Batch prefetch (`batchFn` selection)**: When `maskViewMode === 'seg'`, use `getDetectorMasks` (reads from `detector_masks.h5`, returns PNG blobs).

2. **Single-frame load (`fetchMaskForFrame`)**: When `maskViewMode === 'seg'`, call `getDetectorMask(projectId, videoId, frameIdx)` (the single-frame PNG endpoint at `/detector-masks/{frame_idx}`). Return the mask blob for rendering. This is different from `'detector'` mode which calls `getDetectorBbox` and returns null.

3. **Frame data load (`loadFrameData`)**: When `maskViewMode === 'seg'`, skip the bbox fetch, just load the mask from cache or fetch.

**Availability check**: New `segDetectorMasksExist(projectId, videoId)` API call on mount, checking `GET /videos/{id}/seg-detector-masks/exists`. The "Seg" dropdown option disabled with "(not available)" when false.

### Frontend: VideoPipeline Sidebar

New "Seg Detector" section below OBB Detector:
- "Train Seg Detector" button
- "Apply Seg Detector" button (visible when model exists)

New `useSegDetector.ts` composable mirroring `useObbDetector.ts` but pointing at seg endpoints. The composable handles training status only (no apply state). Apply uses a local `isApplyingSeg` ref in VideoPipeline (same pattern as other apply buttons).

Seg detector scores are displayed via the existing detector score track in DataTrack (shared `detector_score` column) — no new score composable or DataTrack props needed.

## Files Changed

| File | Change |
|------|--------|
| `vidseq/services/detector_model.py` | Add `"seg"` to PRETRAINED_MODELS, add `detect_seg()` |
| `vidseq/services/detector_service.py` | Add `_mask_to_polygon_label()`, 7 branch additions for seg in `_train_sync`/`_write_yolo_dataset`/`_apply_to_training_data`, add `seg_model_exists()` |
| `vidseq/services/frame_data_service.py` | Add async `set_has_detector_mask_batch()` |
| `vidseq/services/segmentation_commands.py` | Extend `handle_apply_detector` for seg: open H5 per video, write masks during batch loop |
| `vidseq/services/segmentation_tcp_client.py` | Add `apply_seg_detector()` wrapper |
| `vidseq/services/segmentation_service.py` | Add `apply_seg_detector()` orchestration with `has_detector_mask` batch update |
| `vidseq/api/routes/detector.py` | Seg status, training, apply, mask-exists endpoints |
| `frontend/src/services/api.ts` | Seg API functions + `segDetectorMasksExist()` |
| `frontend/src/composables/useSegDetector.ts` | New composable |
| `frontend/src/components/VideoPipeline.vue` | Seg Detector sidebar section |
| `frontend/src/components/VideoDetail.vue` | Add "Seg" view mode option, availability check |
| `frontend/src/composables/useSegmentation.ts` | Handle `'seg'` in `batchFn`, `fetchMaskForFrame`, `loadFrameData` |

## Out of Scope

- Multiple segmentation instances per frame (always takes top-1 detection)
- Configurable model size (hardcoded to yolo11x-seg)
- Seg detector masks used as SAM2 prompts
- OBB + segmentation combined
- Reset/clear seg detector results endpoint (add to TODO if needed)
