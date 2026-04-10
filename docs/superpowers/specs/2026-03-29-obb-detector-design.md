# OBB (Oriented Bounding Box) Detector

## Problem

Animals oriented diagonally produce axis-aligned bounding boxes that capture excessive background noise. An oriented bounding box tightly fits the animal regardless of pose angle, reducing noise in downstream analysis. This is a separate tool from the existing axis-aligned detector — it is used standalone (via Apply), never as SAM2 box prompts or for drift detection.

## Design

Add a parallel OBB detector system: own DB columns, own model weights, own training label export, own sidebar section, own overlay view mode. Shares the training frame data source, the `DetectorService` singleton (thread/progress infrastructure), and the generalized `apply_detector` TCP command with the existing detector.

### Database: 9 New Columns on FrameData

```
obb_x1, obb_y1  (corner 1, Float, nullable)
obb_x2, obb_y2  (corner 2, Float, nullable)
obb_x3, obb_y3  (corner 3, Float, nullable)
obb_x4, obb_y4  (corner 4, Float, nullable)
obb_score        (Float, default -1.0)
```

Corners are in absolute pixel coordinates, clockwise order as returned by Ultralytics `obb.xyxyxyxy`. No new H5 files — OBB is bbox-only. Separate model weights at `project_path/models/obb_detector.pt`.

No new indexes — score queries use the existing `(video_id, frame_idx)` unique index, matching the pattern of the axis-aligned detector columns which also have no score-specific index.

### Migration Script

Standalone script at `scripts/migrate_add_obb_columns.py`. Takes a project path argument, finds `vidseq.db` under that path, runs `ALTER TABLE frame_data ADD COLUMN` for each of the 9 new columns. Idempotent (checks if columns exist before adding).

### Training

**Label export**: New function `_mask_to_obb_label()` in `detector_service.py`:
1. Find contours of the binary mask with `cv2.findContours`
2. Compute minimum-area rotated rectangle with `cv2.minAreaRect`
3. Get 4 corner points with `cv2.boxPoints`
4. Normalize to [0, 1] by image dimensions
5. Output: `0 x1 y1 x2 y2 x3 y3 x4 y4` (YOLO OBB label format)

**Model variant**: `yolo11n-obb.pt` as pretrained base. Added to `PRETRAINED_MODELS` in `detector_model.py`:
```python
PRETRAINED_MODELS = {
    "rtdetr": "rtdetr-x.pt",
    "yolo": "yolo11n.pt",
    "obb": "yolo11n-obb.pt",
}
```

**Training flow**: The existing `DetectorService` handles OBB training. One training job at a time — blocks if either OBB or axis-aligned training is running. Changes:

- `train()` gains a `training_type: str` parameter (`"detector"` or `"obb"`), passed from the API route. The service stores this in a new `_training_type` field so status endpoints can report which type is training.
- `_train_sync` branches on `training_type`:
  - `"obb"`: uses `_mask_to_obb_label()` for label export, `task="obb"` for Ultralytics, output to `obb_train/`, weights saved to `obb_detector.pt`
  - `"detector"`: existing behavior unchanged (reads `detector_config.json` for rtdetr/yolo choice)
- Post-training apply (`_apply_to_training_data`): runs for OBB training too, using `detect_obb()` for single-frame inference, saving to the new `obb_*` columns via `_chunked_upsert_sync`.

**Config**: No config file needed — there's only one OBB model variant (`yolo11n-obb.pt`), so training and apply hardcode `"obb"` as the type. No radio buttons or variant selection on the frontend.

**Status**: `DetectorService` gains:
- `obb_model_exists(project_path)` — checks `obb_detector.pt`
- `_training_type: str | None` field — set when training starts, cleared when done. The existing `_is_training` flag still gates mutual exclusion. Status endpoints check `_training_type` to report whether it's their type training.

### Apply (Inference)

Generalize the existing `apply_detector` TCP command:
- Add `detector_type` param: `"detector"` (default) or `"obb"`
- Worker loads appropriate weights file based on type (`detector.pt` or `obb_detector.pt`)
- For OBB: extracts from `result.obb.xyxyxyxy` (4 corners, 8 values) instead of `result.boxes.xyxy` (2 corners, 4 values)
- Returns `obb_scores` and `obb_bboxes` keys in result (each bbox is `[frame_idx, x1, y1, x2, y2, x3, y3, x4, y4]` — 9 values)

Service layer (`apply_obb_detector` in `segmentation_service.py`) saves to new columns via `save_obb_bboxes_batch()` and `save_obb_scores_batch()` in `frame_data_service.py`. `save_obb_bboxes_batch` accepts `list[list]` with 9 values per entry (frame_idx + 8 corner coords).

TCP client: `apply_detector` method gains optional `detector_type` param. New module-level wrapper `apply_obb_detector` calls it with `detector_type="obb"`.

API endpoint: `POST /projects/{project_id}/videos/obb-detection` (separate from existing `/videos/detection`).

Frontend: `applyObbDetector(projectId, videoIds)` function in `api.ts`.

Re-applying overwrites existing OBB data via upsert — same as the axis-aligned detector. No separate "clear OBB" operation needed; row deletion on video delete handles cleanup automatically.

### Frontend: VideoPipeline Sidebar

New "OBB Detector" section below existing "Detector" section:
- "Train OBB Detector" button — triggers training via `POST /projects/{project_id}/detection/obb/training` with selected video IDs (same selection mechanism as existing detector). Navigates to detector training page.
- "Apply OBB Detector" button — visible when OBB model exists, applies to selected videos

New composable `useObbDetector.ts` — standalone copy of `useDetector.ts` pointing at OBB endpoints. No shared logic extraction (the composables are small, ~100 lines, duplication is fine).

### Frontend: VideoDetail Overlay

Add `'obb'` to `MaskViewMode` type: `'tracker' | 'detector' | 'final' | 'obb'`

**`useSegmentation` composable handling**: When `maskViewMode === 'obb'`, the composable skips mask fetching entirely (no batch prefetch, no single-frame mask load). Instead, `VideoDetail.vue` fetches OBB bboxes directly via a new `getObbBbox(projectId, videoId, frameIdx)` API call and passes them to `VideoOverlay` as a new `obbBbox` prop.

**`VideoOverlay.vue`**: New optional prop `obbBbox: { corners: [number, number][] } | null`. When present, renders as a polygon using `ctx.beginPath()` + `moveTo` + `lineTo` through the 4 corners + `closePath()` + `stroke()`. Color: cyan `rgba(0, 188, 212, 0.8)`.

**Availability check**: New `obbBboxesExist(projectId, videoId)` API call on mount, similar to `detectorMasksExist`. The "OBB" option in the dropdown is disabled with "(not available)" when no OBB data exists.

### Frontend: Score Display

OBB scores displayed as a third line in DataTrack alongside tracker and detector scores:
- New prop `obbScores` on DataTrack component
- New color (cyan to match overlay)
- Fetched via `getObbScoresDownsampled()` API function
- Hover tooltip shows all three score types when present

### API Endpoints (New)

All paths are relative to `/projects/{project_id}`.

| Method | Path | File | Description |
|--------|------|------|-------------|
| GET | `/detection/obb/status` | `detector.py` | OBB model exists + training status |
| POST | `/detection/obb/training` | `detector.py` | Start OBB training |
| DELETE | `/detection/obb/training` | `detector.py` | Stop OBB training |
| GET | `/detection/obb/training` | `detector.py` | OBB training progress |
| GET | `/detection/obb/training/stream` | `detector.py` | SSE for OBB training |
| POST | `/videos/obb-detection` | `detector.py` | Apply OBB detector to selected videos |
| GET | `/videos/{id}/obb-bboxes/{frame_idx}` | `detector.py` | Single frame OBB bbox |
| GET | `/videos/{id}/obb-bboxes` | `detector.py` | Batch OBB bboxes |
| GET | `/videos/{id}/segmentation/obb-scores-downsampled` | `segmentation/frames.py` | Downsampled OBB scores |

The OBB scores endpoint goes in `segmentation/frames.py` to match the existing `detector-scores-downsampled` endpoint location.

### Detector Model Changes

Add OBB-specific `detect_obb()` function to `detector_model.py`. Used by `_apply_to_training_data` for the post-training apply step (single-frame inference):
```python
def detect_obb(model, frame, conf=DETECTION_CONF_THRESHOLD) -> list[dict]:
    results = model(frame, conf=conf, verbose=False)
    detections = []
    if len(results) > 0 and results[0].obb is not None:
        obb = results[0].obb
        for i in range(len(obb)):
            corners = obb.xyxyxyxy[i].cpu().tolist()  # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
            conf_val = obb.conf[i].item()
            detections.append({"corners": corners, "conf": conf_val})
    detections.sort(key=lambda d: d["conf"], reverse=True)
    return detections
```

Batched inference in `handle_apply_detector` does inline OBB extraction from `result.obb.xyxyxyxy` (same as the existing inline extraction from `result.boxes.xyxy` — the `detect()` / `detect_obb()` functions are not used in the batch path).

## Files Changed

| File | Change |
|------|--------|
| `vidseq/models/frame_data.py` | 9 new columns |
| `scripts/migrate_add_obb_columns.py` | New migration script |
| `vidseq/services/detector_model.py` | Add `"obb"` to PRETRAINED_MODELS, add `detect_obb()` |
| `vidseq/services/detector_service.py` | OBB label export, `training_type` param, OBB post-training apply, `obb_model_exists` |
| `vidseq/services/frame_data_service.py` | `save_obb_bboxes_batch`, `save_obb_scores_batch`, `get_obb_bbox`, `get_obb_bboxes_batch`, `get_obb_scores_downsampled` |
| `vidseq/services/segmentation_commands.py` | Generalize `handle_apply_detector` for OBB |
| `vidseq/services/segmentation_tcp_client.py` | Pass `detector_type` through, new `apply_obb_detector` wrapper |
| `vidseq/services/segmentation_service.py` | New `apply_obb_detector` orchestration |
| `vidseq/api/routes/detector.py` | OBB status, training, apply, bbox endpoints |
| `vidseq/api/routes/segmentation/frames.py` | OBB scores downsampled endpoint |
| `frontend/src/services/api.ts` | OBB API functions, OBB types |
| `frontend/src/composables/useObbDetector.ts` | New composable |
| `frontend/src/components/VideoPipeline.vue` | OBB sidebar section |
| `frontend/src/components/VideoDetail.vue` | OBB view mode, OBB bbox fetching |
| `frontend/src/components/VideoOverlay.vue` | Polygon rendering for OBB |
| `frontend/src/components/DataTrack.vue` | OBB score track |

## Out of Scope

- OBB as SAM2 box prompts (not needed — OBB is standalone only)
- Oriented IoU for drift detection (OBB not used with propagate_with_detector)
- RT-DETR OBB variant (RT-DETR doesn't support OBB)
- Configurable OBB model variant (only yolo11n-obb.pt)
- Auto-migration on DB open (manual script only)
