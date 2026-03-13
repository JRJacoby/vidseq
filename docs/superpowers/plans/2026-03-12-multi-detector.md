# Multi-Detector Support Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow users to switch between RT-DETR and YOLO11n detectors via a radio selector in the VideoPipeline sidebar.

**Architecture:** Config file stores detector type choice. Training branches on type to select pretrained model and hyperparameters. Inference auto-detects model architecture from checkpoint via Ultralytics' `YOLO()` loader. Frontend adds radio group to sidebar.

**Tech Stack:** Python/FastAPI, Ultralytics (YOLO, RTDETR), Vue 3 Composition API, TypeScript

---

## Chunk 1: Backend

### Task 1: Config helpers in detector_service.py

**Files:**
- Modify: `vidseq/services/detector_service.py`

- [ ] **Step 1: Add config read/write functions**

Add these two module-level functions after the `_mask_to_yolo_bbox` function (after line 62) in `detector_service.py`:

```python
import json

VALID_DETECTOR_TYPES = ("rtdetr", "yolo")
DEFAULT_DETECTOR_TYPE = "rtdetr"


def read_detector_config(project_path: Path) -> str:
    """Read detector type from project config. Defaults to 'rtdetr'."""
    config_path = project_path / "models" / "detector_config.json"
    if not config_path.exists():
        return DEFAULT_DETECTOR_TYPE
    try:
        data = json.loads(config_path.read_text())
        dtype = data.get("detector_type", DEFAULT_DETECTOR_TYPE)
        return dtype if dtype in VALID_DETECTOR_TYPES else DEFAULT_DETECTOR_TYPE
    except (json.JSONDecodeError, OSError):
        return DEFAULT_DETECTOR_TYPE


def write_detector_config(project_path: Path, detector_type: str) -> None:
    """Write detector type to project config."""
    if detector_type not in VALID_DETECTOR_TYPES:
        raise ValueError(f"Invalid detector type: {detector_type}")
    config_dir = project_path / "models"
    config_dir.mkdir(exist_ok=True)
    config_path = config_dir / "detector_config.json"
    config_path.write_text(json.dumps({"detector_type": detector_type}))
```

Also add `import json` to the top-level imports (after `import logging`).

- [ ] **Step 2: Commit**

```bash
git add vidseq/services/detector_service.py
git commit -m "feat: add detector config read/write helpers"
```

### Task 2: Generalize detector_model.py

**Files:**
- Modify: `vidseq/services/detector_model.py`

- [ ] **Step 1: Update the entire file**

Replace the contents of `detector_model.py`:

```python
"""Detector model for bounding box detection (RT-DETR or YOLO)."""

from pathlib import Path

import numpy as np
from ultralytics import YOLO


# Confidence threshold for detections.
# Set low enough for RT-DETR's transformer queries (which produce lower raw
# scores than YOLO anchors). Both code paths take the top-1 detection, so
# a shared low threshold is functionally fine for YOLO as well.
DETECTION_CONF_THRESHOLD = 0.25

# Map config names to pretrained model identifiers
PRETRAINED_MODELS = {
    "rtdetr": "rtdetr-x.pt",
    "yolo": "yolo11n.pt",
}


def load_pretrained(detector_type: str, device: str = "cuda"):
    """Load a COCO-pretrained detector model.

    Args:
        detector_type: "rtdetr" or "yolo"
        device: Device to load onto
    """
    model_name = PRETRAINED_MODELS[detector_type]
    model = YOLO(model_name)
    model.to(device)
    return model


def load_finetuned(weights_path: str | Path, device: str = "cuda"):
    """Load a fine-tuned detector from checkpoint.

    Uses YOLO() which auto-detects the architecture from the checkpoint.
    Works for both RT-DETR and YOLO weights.
    """
    model = YOLO(str(weights_path))
    model.to(device)
    return model


def detect(
    model,
    frame: np.ndarray,
    conf: float = DETECTION_CONF_THRESHOLD,
) -> list[dict]:
    """Run detection on a single BGR frame.

    Args:
        model: Ultralytics model instance (RTDETR or YOLO).
        frame: BGR uint8 numpy array (H, W, 3).
        conf: Confidence threshold.

    Returns:
        List of detections sorted by confidence (descending).
        Each detection: {"bbox": (x1, y1, x2, y2), "conf": float, "cls": int}
        bbox coordinates are in pixel space of the original frame.
    """
    results = model(frame, conf=conf, verbose=False)
    detections = []
    if len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().tolist()
            detections.append({
                "bbox": (x1, y1, x2, y2),
                "conf": boxes.conf[i].item(),
                "cls": int(boxes.cls[i].item()),
            })
    # Sort by confidence descending
    detections.sort(key=lambda d: d["conf"], reverse=True)
    return detections


def pick_best_detection(
    detections: list[dict],
    tracker_bbox: tuple[int, int, int, int] | None = None,
) -> tuple[tuple[float, float, float, float], float] | tuple[None, float]:
    """Pick the best detection, preferring highest IoU with tracker bbox.

    Args:
        detections: List of detection dicts from detect().
        tracker_bbox: (x1, y1, x2, y2) of tracker's current mask bbox, or None.

    Returns:
        (bbox, confidence) or (None, 0.0) if no detections.
    """
    if not detections:
        return None, 0.0

    if tracker_bbox is None:
        # No tracker reference — take highest confidence
        best = detections[0]
        return best["bbox"], best["conf"]

    # Pick detection with highest IoU to tracker bbox
    best_iou = -1.0
    best_det = detections[0]  # fallback to highest conf
    tx1, ty1, tx2, ty2 = tracker_bbox

    for det in detections:
        dx1, dy1, dx2, dy2 = det["bbox"]
        inter_x1 = max(tx1, dx1)
        inter_y1 = max(ty1, dy1)
        inter_x2 = min(tx2, dx2)
        inter_y2 = min(ty2, dy2)
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        tracker_area = (tx2 - tx1) * (ty2 - ty1)
        det_area = (dx2 - dx1) * (dy2 - dy1)
        union_area = tracker_area + det_area - inter_area
        iou = inter_area / union_area if union_area > 0 else 0.0
        if iou > best_iou:
            best_iou = iou
            best_det = det

    return best_det["bbox"], best_det["conf"]
```

Key changes:
- Import `YOLO` instead of `RTDETR` (YOLO auto-detects both architectures)
- `load_pretrained` takes `detector_type` parameter, looks up model name from `PRETRAINED_MODELS` dict
- `load_finetuned` uses generic `YOLO()` loader — auto-detects architecture from checkpoint
- `detect()` type annotation generalized from `model: RTDETR` to `model` (accepts either)
- Removed `torch` import (not needed in this module)

- [ ] **Step 2: Commit**

```bash
git add vidseq/services/detector_model.py
git commit -m "feat: generalize detector_model for RTDETR and YOLO"
```

### Task 3: Branch training on detector type

**Files:**
- Modify: `vidseq/services/detector_service.py`

- [ ] **Step 1: Update `_train_sync` to read config and branch**

In `_train_sync`, replace the import and model creation section. Change:

```python
    def _train_sync(
        self,
        ...
    ) -> None:
        """Synchronous training implementation using Ultralytics RT-DETR."""
        from ultralytics import RTDETR
        from vidseq.services import segmentation_service
```

to:

```python
    def _train_sync(
        self,
        ...
    ) -> None:
        """Synchronous training implementation using Ultralytics."""
        from vidseq.services.detector_model import load_pretrained
        from vidseq.services import segmentation_service
```

Then replace the model creation and training call. Change:

```python
            logger.info(
                f"Starting RT-DETR training: max_epochs={max_epochs}, "
                f"batch_size={batch_size}, lr={lr}"
            )
```

to:

```python
            # Read detector type from config
            detector_type = read_detector_config(project_path)

            logger.info(
                f"Starting {detector_type.upper()} training: max_epochs={max_epochs}, "
                f"batch_size={batch_size}, lr={lr}"
            )
```

Replace model creation line:

```python
            model = RTDETR("rtdetr-x.pt")
```

with:

```python
            model = load_pretrained(detector_type, device="cpu")
```

(Model loaded on CPU for training — Ultralytics handles device placement during `train()`.)

Replace the training hyperparameters block. Change:

```python
            model.train(
                data=str(yaml_path),
                epochs=max_epochs,
                imgsz=640,
                batch=batch_size,
                lr0=lr,
                optimizer="AdamW",
                patience=early_stop_patience,
                single_cls=True,
                device=0,
                workers=0,  # RT-DETR transforms contain unpicklable lambdas
                plots=False,  # avoid accumulating plot data across epochs
                project=str(model_save_dir),
                name="rtdetr_train",
                exist_ok=True,
                verbose=False,
            )

            # Copy best weights to standard location
            best_pt = model_save_dir / "rtdetr_train" / "weights" / "best.pt"
```

to:

```python
            # Per-model training hyperparameters
            if detector_type == "yolo":
                train_workers = 4
                train_batch = max(batch_size, 8)
                train_name = "yolo_train"
            else:
                train_workers = 0  # RT-DETR transforms contain unpicklable lambdas
                train_batch = batch_size
                train_name = "rtdetr_train"

            model.train(
                data=str(yaml_path),
                epochs=max_epochs,
                imgsz=640,
                batch=train_batch,
                lr0=lr,
                optimizer="AdamW",
                patience=early_stop_patience,
                single_cls=True,
                device=0,
                workers=train_workers,
                plots=False,
                project=str(model_save_dir),
                name=train_name,
                exist_ok=True,
                verbose=False,
            )

            # Copy best weights to standard location
            best_pt = model_save_dir / train_name / "weights" / "best.pt"
```

- [ ] **Step 2: Update docstring and class docstring**

Change the class docstring from:

```python
class DetectorService:
    """Singleton service for RT-DETR detector training and inference."""
```

to:

```python
class DetectorService:
    """Singleton service for detector training and inference (RT-DETR or YOLO)."""
```

Change the module docstring from:

```python
"""Detector Service for RT-DETR-based object detection training and inference."""
```

to:

```python
"""Detector Service for object detection training and inference."""
```

- [ ] **Step 3: Commit**

```bash
git add vidseq/services/detector_service.py
git commit -m "feat: branch detector training on config type"
```

### Task 4: API endpoint for config + extend status

**Files:**
- Modify: `vidseq/api/routes/detector.py`
- Modify: `vidseq/schemas/detector.py` (optional — schemas are inline in detector.py)

- [ ] **Step 1: Add config request/response models and extend status**

In `detector.py` routes, add the import at the top:

```python
from vidseq.services.detector_service import DetectorService, read_detector_config, write_detector_config
```

(Replace the existing `from vidseq.services.detector_service import DetectorService`.)

Add `detector_type` to `DetectorStatusResponse`:

```python
class DetectorStatusResponse(BaseModel):
    """Response for detector status endpoint."""

    model_exists: bool
    is_training: bool
    detector_type: str
```

Add new request model after existing models:

```python
class DetectorConfigRequest(BaseModel):
    """Request body for setting detector type."""

    detector_type: str
```

- [ ] **Step 2: Update get_detection_status to include detector_type**

Change the status endpoint to:

```python
@router.get(
    "/projects/{project_id}/detection/status",
    response_model=DetectorStatusResponse,
)
async def get_detection_status(
    project_path: Path = Depends(get_project_folder),
):
    service = DetectorService.get_instance()
    return DetectorStatusResponse(
        model_exists=service.model_exists(project_path),
        is_training=service.is_training(),
        detector_type=read_detector_config(project_path),
    )
```

- [ ] **Step 3: Add PUT config endpoint**

Add after the status endpoint:

```python
@router.put("/projects/{project_id}/detection/config")
async def update_detection_config(
    body: DetectorConfigRequest,
    project_path: Path = Depends(get_project_folder),
):
    """Set the detector type for this project."""
    if body.detector_type not in ("rtdetr", "yolo"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid detector_type: {body.detector_type}. Must be 'rtdetr' or 'yolo'.",
        )
    write_detector_config(project_path, body.detector_type)
    return {"detector_type": body.detector_type}
```

- [ ] **Step 4: Commit**

```bash
git add vidseq/api/routes/detector.py
git commit -m "feat: add detection config endpoint, extend status with detector_type"
```

## Chunk 2: Frontend

### Task 5: API function and type update

**Files:**
- Modify: `frontend/src/services/api.ts`

- [ ] **Step 1: Extend DetectorStatus interface**

Change (around line 1140):

```typescript
export interface DetectorStatus {
    model_exists: boolean
    is_training: boolean
}
```

to:

```typescript
export interface DetectorStatus {
    model_exists: boolean
    is_training: boolean
    detector_type: string
}
```

- [ ] **Step 2: Add updateDetectionConfig function**

Add after `getDetectionStatus` (around line 1176):

```typescript
export async function updateDetectionConfig(projectId: number, detectorType: string): Promise<void> {
    const response = await fetch(`${API_BASE}/projects/${projectId}/detection/config`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ detector_type: detectorType })
    })
    if (!response.ok) throw new Error(await getErrorMessage(response, 'Failed to update detector config'))
}
```

- [ ] **Step 3: Commit**

```bash
git add frontend/src/services/api.ts
git commit -m "feat: add updateDetectionConfig API function"
```

### Task 6: Update useDetector composable

**Files:**
- Modify: `frontend/src/composables/useDetector.ts`

- [ ] **Step 1: Add detectorType ref and setDetectorType**

Update the imports:

```typescript
import { ref, watch, onUnmounted, type Ref } from 'vue'
import {
    getDetectionStatus,
    createDetectionTraining,
    deleteDetectionTraining,
    updateDetectionConfig,
    type DetectorStatus,
} from '@/services/api'
```

Update the interface:

```typescript
export interface UseDetectorReturn {
    isTraining: Ref<boolean>
    modelExists: Ref<boolean>
    detectorType: Ref<string>
    startTraining: (maxEpochs?: number, videoIds?: number[]) => Promise<void>
    stopTraining: () => Promise<void>
    checkStatus: () => Promise<void>
    setDetectorType: (type: string) => Promise<void>
}
```

Add `detectorType` ref and update `checkStatus` and add `setDetectorType`:

```typescript
export function useDetector(projectId: Ref<number | null>): UseDetectorReturn {
    const isTraining = ref(false)
    const modelExists = ref(false)
    const detectorType = ref('rtdetr')

    const checkStatus = async () => {
        if (!projectId.value) return
        try {
            const status = await getDetectionStatus(projectId.value)
            modelExists.value = status.model_exists
            isTraining.value = status.is_training
            detectorType.value = status.detector_type
        } catch (e) {
            console.error('Failed to check detector status:', e)
        }
    }

    const setDetectorType = async (type: string) => {
        if (!projectId.value) return
        try {
            await updateDetectionConfig(projectId.value, type)
            detectorType.value = type
        } catch (e) {
            console.error('Failed to set detector type:', e)
        }
    }
```

Update the return statement:

```typescript
    return {
        isTraining,
        modelExists,
        detectorType,
        startTraining,
        stopTraining,
        checkStatus,
        setDetectorType,
    }
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/composables/useDetector.ts
git commit -m "feat: add detectorType to useDetector composable"
```

### Task 7: Radio selector in VideoPipeline.vue

**Files:**
- Modify: `frontend/src/components/VideoPipeline.vue`

- [ ] **Step 1: Add detectorType and setDetectorType to destructured composable**

Change the `useDetector` destructure (around line 89):

```typescript
const {
  isTraining: isDetectorTraining,
  modelExists: detectorModelExists,
  startTraining: startDetectorTraining,
  checkStatus: checkDetectorStatus,
} = useDetector(projectId)
```

to:

```typescript
const {
  isTraining: isDetectorTraining,
  modelExists: detectorModelExists,
  detectorType,
  startTraining: startDetectorTraining,
  checkStatus: checkDetectorStatus,
  setDetectorType,
} = useDetector(projectId)
```

- [ ] **Step 2: Rename section heading and add radio group**

In the template, change:

```html
          <h4 class="sidebar-section-title">DINOv2 Detector</h4>
```

to:

```html
          <h4 class="sidebar-section-title">Detector</h4>
          <div class="detector-type-selector">
            <label class="detector-radio">
              <input type="radio" value="rtdetr" :checked="detectorType === 'rtdetr'" @change="setDetectorType('rtdetr')" :disabled="isDetectorTraining" />
              RT-DETR
            </label>
            <label class="detector-radio">
              <input type="radio" value="yolo" :checked="detectorType === 'yolo'" @change="setDetectorType('yolo')" :disabled="isDetectorTraining" />
              YOLO
            </label>
          </div>
```

- [ ] **Step 3: Add styles for radio selector**

Add to the `<style scoped>` section:

```css
.detector-type-selector {
  display: flex;
  gap: 1rem;
  padding: 0.25rem 0;
}

.detector-radio {
  display: flex;
  align-items: center;
  gap: 0.3rem;
  font-size: 0.85rem;
  color: #555;
  cursor: pointer;
}

.detector-radio input[type="radio"] {
  cursor: pointer;
}

.detector-radio input[type="radio"]:disabled {
  cursor: not-allowed;
}
```

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/VideoPipeline.vue
git commit -m "feat: add detector type radio selector to VideoPipeline sidebar"
```
