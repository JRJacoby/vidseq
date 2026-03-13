# Multi-Detector Support (RT-DETR + YOLO)

## Problem

The detector is currently hardcoded to RT-DETR-X (~86M params). For some workflows, a lighter model like YOLO11n (~3M params) would be preferable — faster training, faster inference, lower GPU memory. Users need the ability to choose.

## Solution

Add a radio selector in the VideoPipeline sidebar to switch between RT-DETR and YOLO11n. One active detector at a time — switching doesn't delete the old model or clear results, but the next training run uses the selected model type. The choice is stored in a per-project JSON config file.

## Design

### Config Storage

A JSON file at `<project>/models/detector_config.json`:

```json
{"detector_type": "rtdetr"}
```

Two values: `"rtdetr"` or `"yolo"`. If the file doesn't exist, default to `"rtdetr"` (backward compatible with existing projects).

Helper functions in `detector_service.py` (service layer owns filesystem/config concerns):
- `read_detector_config(project_path) -> str`
- `write_detector_config(project_path, detector_type)`

This is a JSON file rather than a DB column — pragmatic choice to avoid a migration for a single setting.

### Model Layer (`detector_model.py`)

**Key simplification:** Ultralytics' `YOLO()` class auto-detects the model architecture from the checkpoint file. `YOLO("rtdetr-x.pt")` returns an `RTDETR` instance; `YOLO("yolo11n.pt")` returns a YOLO instance. This means:

- Replace `load_pretrained()` with `load_pretrained(detector_type, device)` — loads `RTDETR("rtdetr-x.pt")` or `YOLO("yolo11n.pt")` based on `detector_type`.
- Replace `load_finetuned()` with a generic version that uses `YOLO(path)` — auto-detects architecture from the saved weights. No config needed at inference time.
- Generalize type annotation on `detect()` from `model: RTDETR` to accept either type.

The existing `detect()` and `pick_best_detection()` function logic works unchanged — both model types return the same Ultralytics results interface (`boxes.xyxy`, `boxes.conf`, `boxes.cls`).

**Confidence threshold:** The current `DETECTION_CONF_THRESHOLD = 0.25` was set low for RT-DETR. Since both code paths take the top-1 detection, 0.25 works fine for YOLO too (just passes through more candidates). Update the comment to reflect this is a shared threshold.

### Service Layer (`detector_service.py`)

`_train_sync()` reads the config to decide which pretrained model to fine-tune. Both use the same Ultralytics `train()` API and the same YOLO-format dataset. Differences:

| | RT-DETR | YOLO |
|---|---|---|
| Base model | `rtdetr-x.pt` | `yolo11n.pt` |
| workers | 0 (unpicklable lambdas) | 4 |
| batch | 2 | 8 |
| train name | `"rtdetr_train"` | `"yolo_train"` |

The `name` parameter and corresponding `best_pt` path must vary by model type.

The trained model is saved to the same path (`models/detector.pt`) regardless of type — retraining with a different model type overwrites the previous model.

`_apply_to_training_data()` also loads the model via `load_finetuned()`. Since the updated `load_finetuned()` auto-detects architecture, no branching is needed here.

### Inference (`segmentation_commands.py`)

`handle_propagate_with_detector` calls `load_finetuned()` which auto-detects from the checkpoint. No config read needed at inference time. Everything downstream (bbox format, pick_best_detection, SAM2 box prompting) is identical.

### API

**New endpoint:**

`PUT /projects/{project_id}/detection/config` with body `{"detector_type": "rtdetr" | "yolo"}`. Validates the value, writes the config file. Returns 200 with the updated config.

**Extended endpoint:**

`GET /projects/{project_id}/detection/status` — add `detector_type` field to the response (read from config, defaulting to `"rtdetr"`).

### Frontend

**VideoPipeline.vue:**

- Rename section heading from "DINOv2 Detector" to "Detector"
- Add radio group below heading: "RT-DETR" / "YOLO"
- Radio calls `PUT /detection/config` on change (immediate, no save button)
- Current value loaded from `GET /detection/status` response

**api.ts:**

- New `updateDetectionConfig(projectId, detectorType)` function
- Extend `DetectorStatus` interface to include `detector_type: string`

**useDetector composable:**

- Add `detectorType` ref populated from `checkStatus()`
- Add `setDetectorType()` that calls the new API and refreshes status

**DetectorTraining.vue:** No changes — training progress display is model-agnostic.

**Video detail screen, bbox display, score tracks:** No changes — one active detector, same output format.

## Scope

**Files modified:**

- `vidseq/services/detector_model.py` — generalize load functions, update type annotations
- `vidseq/services/detector_service.py` — config read/write helpers, branch on detector type in training, update train name/best_pt path
- `vidseq/services/segmentation_commands.py` — no logic changes needed (auto-detect from checkpoint), but update import if `load_finetuned` signature changes
- `vidseq/api/routes/detector.py` — new PUT config endpoint, extend status response
- `vidseq/schemas/detector.py` — new config request/response schemas
- `frontend/src/services/api.ts` — new config function, extend DetectorStatus
- `frontend/src/composables/useDetector.ts` — detectorType ref, setDetectorType
- `frontend/src/components/VideoPipeline.vue` — radio selector, rename heading

**No changes to:**

- Database models or schema
- `detector_masks.h5` lifecycle
- SAM2 integration (box prompting, propagation logic)
- DetectorTraining.vue
- Video detail screen
