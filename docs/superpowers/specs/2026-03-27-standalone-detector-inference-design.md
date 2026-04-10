# Standalone Detector Inference

## Problem

The detector can only run inference as part of `segment_all`, which couples it to the full SAM2 segmentation pipeline. Users sometimes need just bounding boxes — for example, to verify detector quality or to get bboxes for downstream use without the cost of SAM2 propagation. YOLO/RT-DETR alone is much faster than the combined pipeline.

## Design

Add a standalone "Apply Detector" operation that runs the trained detector on every frame of selected videos, producing bboxes and confidence scores stored in the existing `FrameData` database columns.

### API

**Endpoint:** `POST /projects/{project_id}/videos/detection`

**Request body:**
```json
{ "video_ids": [1, 2, 3] }
```

**Response:**
```json
{ "videos_processed": 3 }
```

Requires a trained detector model (`project_path/models/detector.pt`). Returns 400 if no model exists.

### Service Layer

New function `apply_detector()` in `segmentation_service.py`:
- Validates detector model exists
- Calls TCP client `apply_detector()` (streaming, with progress)
- Saves returned bboxes and scores to DB via existing `save_detector_bboxes_batch()` and `save_detector_scores_batch()`

### TCP Command

New `apply_detector` command on the GPU worker:

1. Load detector on GPU via `load_finetuned(model_path, device="cuda")`
2. For each video:
   - Open video with `VideoFrameSource`
   - Read frames in batches of 32
   - Run `model(batch)` — Ultralytics natively supports list-of-frames batched inference
   - Extract best detection per frame (highest confidence)
   - Send progress callback every 50 frames
3. Return all bboxes and scores grouped by video
4. Cleanup: `del detector; torch.cuda.empty_cache()`

**Handler signature:**
```python
def handle_apply_detector(
    params: dict,
    segmentor: StreamingSegmentor,
    response_callback: Callable | None = None,
) -> dict
```

The handler does not use `segmentor` for inference but accepts it for the standard handler signature. It loads the detector model independently — no active SAM session required.

**Result format:**
```json
{
  "type": "apply_detector_result",
  "status": "ok",
  "detector_scores": { "1": [[0, 0.95], [1, 0.87], ...], "2": [...] },
  "detector_bboxes": { "1": [[0, 100, 200, 300, 400], ...], "2": [...] }
}
```

Bbox format: `[frame_idx, x1, y1, x2, y2]`. Score format: `[frame_idx, score]`. Keys are video ID strings.

### Detector Model

No changes to `detector_model.py`. The existing `detect()` function works per-frame. For batched inference, the handler calls `model(batch_of_frames)` directly (Ultralytics API) and extracts results per image, bypassing the single-frame `detect()` wrapper.

### Frontend

**VideoPipeline.vue:** New "Apply Detector" button in the detector sidebar section, below "Train Detector". Enabled when a trained detector model exists and videos are selected. Disabled during application with text "Applying...".

**api.ts:** New `applyDetector(projectId, videoIds)` function calling `POST /videos/detection`.

### Storage

Reuses existing `FrameData` columns — no schema changes:
- `detector_bbox_x1`, `detector_bbox_y1`, `detector_bbox_x2`, `detector_bbox_y2`
- `detector_score`

Frames with no detection get `detector_score = 0.0` and null bbox columns.

### Streaming & Timeout

Uses `_send_streaming()` with progress callbacks every 50 frames per video. Timeout: 3600s (1 hour) to handle long videos. Progress message format:
```json
{ "type": "progress", "frame_idx": 150, "total": 5000, "video_id": 1 }
```

## Files Changed

| File | Change |
|------|--------|
| `vidseq/api/routes/detector.py` | New `POST /videos/detection` endpoint |
| `vidseq/services/segmentation_service.py` | New `apply_detector()` function |
| `vidseq/services/segmentation_tcp_client.py` | New `apply_detector()` method using `_send_streaming` |
| `vidseq/services/segmentation_commands.py` | New `handle_apply_detector()` with batched inference |
| `vidseq/services/segmentation_tcp_server.py` | Route `apply_detector` command to handler |
| `frontend/src/services/api.ts` | New `applyDetector()` function |
| `frontend/src/components/VideoPipeline.vue` | "Apply Detector" button |

## Out of Scope

- Detector mask generation (detector is bbox-only)
- Progress bar on frontend (simple "Applying..." indicator, same as existing patterns)
- Configurable batch size (hardcoded at 32)
- Running on CPU (GPU worker only)
