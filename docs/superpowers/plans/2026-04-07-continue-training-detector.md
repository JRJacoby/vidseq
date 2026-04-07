# Continue Training Detector — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a "Continue Training" button to the detector pipeline that fine-tunes from the previous best checkpoint instead of pretrained weights.

**Architecture:** Single boolean flag (`from_checkpoint`) threaded through the existing training pipeline — TrainRequest → route → service → _train_sync. Model loading branches on this flag: checkpoint path if true, pretrained if false. Frontend adds one button and one handler.

**Tech Stack:** FastAPI/Pydantic (backend), Ultralytics YOLO (model loading), Vue 3 Composition API (frontend)

**Spec:** `docs/superpowers/specs/2026-04-07-continue-training-detector-design.md`

---

### Task 1: Backend — Add `from_checkpoint` to TrainRequest and route

**Files:**
- Modify: `vidseq/api/routes/detector.py:34-44` (TrainRequest)
- Modify: `vidseq/api/routes/detector.py:107-135` (create_detection_training route)

- [ ] **Step 1: Add `from_checkpoint` field to `TrainRequest`**

In `vidseq/api/routes/detector.py`, add the field to the Pydantic model:

```python
class TrainRequest(BaseModel):
    """Request body for training endpoint."""

    model_config = {"extra": "ignore"}

    video_ids: list[int]
    max_epochs: int = 100
    batch_size: int = 4
    lr: float = 1e-4
    early_stop_patience: int = 20
    from_checkpoint: bool = False
```

- [ ] **Step 2: Pass `from_checkpoint` through in the route**

In the `create_detection_training` function (line 107), add `from_checkpoint` to the `service.train()` call:

```python
service.train(
    project_path=project_path,
    video_ids=request.video_ids,
    max_epochs=request.max_epochs,
    batch_size=request.batch_size,
    lr=request.lr,
    early_stop_patience=request.early_stop_patience,
    from_checkpoint=request.from_checkpoint,
)
```

- [ ] **Step 3: Commit**

```bash
git add vidseq/api/routes/detector.py
git commit -m "feat: add from_checkpoint flag to detector training request"
```

---

### Task 2: Backend — Thread `from_checkpoint` through service and load checkpoint

**Files:**
- Modify: `vidseq/services/detector_service.py:228-270` (train method)
- Modify: `vidseq/services/detector_service.py:272-370` (_train_sync method)

- [ ] **Step 1: Add `from_checkpoint` parameter to `train()`**

In `vidseq/services/detector_service.py`, update the `train()` method signature (line 228) to accept and pass through the flag:

```python
def train(
    self,
    project_path: Path,
    video_ids: list[int],
    max_epochs: int = 100,
    batch_size: int = 2,
    lr: float = 1e-4,
    early_stop_patience: int = 20,
    training_type: str = "detector",
    from_checkpoint: bool = False,
) -> bool:
```

And in the `_train_thread` closure inside `train()`, pass it to `_train_sync`:

```python
def _train_thread():
    try:
        self._train_sync(
            project_path,
            video_ids,
            max_epochs,
            batch_size,
            lr,
            early_stop_patience,
            training_type=training_type,
            from_checkpoint=from_checkpoint,
        )
```

- [ ] **Step 2: Add `from_checkpoint` parameter to `_train_sync()`**

Update the `_train_sync()` signature (line 272):

```python
def _train_sync(
    self,
    project_path: Path,
    video_ids: list[int],
    max_epochs: int,
    batch_size: int,
    lr: float,
    early_stop_patience: int,
    training_type: str = "detector",
    from_checkpoint: bool = False,
) -> None:
```

- [ ] **Step 3: Add `load_finetuned` import**

At line 283, the method already imports `load_pretrained`. Add `load_finetuned` to the import:

```python
from vidseq.services.detector_model import load_pretrained, load_finetuned
```

- [ ] **Step 4: Replace model loading with checkpoint-aware logic**

The current code at line 370 is:
```python
model = load_pretrained(detector_type, device="cpu")
```

This line must move AFTER the per-model hyperparameter block (lines 399-415) where `train_name` is determined, because we need `train_name` to build the checkpoint path. Replace line 370 with the model loading block after line 415:

```python
            # Per-model training hyperparameters
            if detector_type == "obb":
                train_workers = 4
                train_batch = max(batch_size, 8)
                train_name = "obb_train"
            elif detector_type == "seg":
                train_workers = 4
                train_batch = max(batch_size, 8)
                train_name = "seg_train"
            elif detector_type == "yolo":
                train_workers = 4
                train_batch = max(batch_size, 8)
                train_name = "yolo_train"
            else:
                train_workers = 0  # RT-DETR transforms contain unpicklable lambdas
                train_batch = batch_size
                train_name = "rtdetr_train"

            # Load model — from checkpoint or pretrained
            if from_checkpoint:
                checkpoint_path = project_path / "models" / train_name / "weights" / "best.pt"
                if checkpoint_path.exists():
                    model = load_finetuned(checkpoint_path, device="cpu")
                    logger.info(f"Fine-tuning from checkpoint: {checkpoint_path}")
                else:
                    model = load_pretrained(detector_type, device="cpu")
                    logger.warning("Checkpoint not found, falling back to pretrained weights")
            else:
                model = load_pretrained(detector_type, device="cpu")
```

Remove the old `model = load_pretrained(detector_type, device="cpu")` that was at line 370. The callbacks (check_stop, on_epoch_end) and model_save_dir setup that were between line 370 and the hyperparameter block remain unchanged — they don't depend on model loading order.

- [ ] **Step 5: Verify backend starts**

```bash
vidseq &
sleep 3
curl -s http://localhost:8000/api/health | head -1
kill %1
```

Expected: server starts without import errors.

- [ ] **Step 6: Commit**

```bash
git add vidseq/services/detector_service.py
git commit -m "feat: support from_checkpoint flag in detector training service"
```

---

### Task 3: Frontend — Add `fromCheckpoint` parameter to API and composable

**Files:**
- Modify: `frontend/src/services/api.ts:1378-1398`
- Modify: `frontend/src/composables/useDetector.ts:10-18,47-57`

- [ ] **Step 1: Add `fromCheckpoint` to `createDetectionTraining()`**

In `frontend/src/services/api.ts` at line 1378, update the function:

```typescript
export async function createDetectionTraining(
    projectId: number,
    maxEpochs: number,
    videoIds: number[],
    lrPatience: number = 10,
    earlyStopPatience: number = 20,
    fromCheckpoint: boolean = false,
): Promise<void> {
    const response = await fetch(`${API_BASE}/projects/${projectId}/detection/training`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            video_ids: videoIds,
            max_epochs: maxEpochs,
            lr_patience: lrPatience,
            early_stop_patience: earlyStopPatience,
            from_checkpoint: fromCheckpoint,
        }),
    })
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to start detection training'))
    }
}
```

- [ ] **Step 2: Update `UseDetectorReturn` interface**

In `frontend/src/composables/useDetector.ts`, update the interface at line 10:

```typescript
export interface UseDetectorReturn {
    isTraining: Ref<boolean>
    modelExists: Ref<boolean>
    detectorType: Ref<string>
    startTraining: (maxEpochs?: number, videoIds?: number[], fromCheckpoint?: boolean) => Promise<void>
    stopTraining: () => Promise<void>
    checkStatus: () => Promise<void>
    setDetectorType: (type: string) => Promise<void>
}
```

- [ ] **Step 3: Update `startTraining()` implementation**

In the same file, update the `startTraining` function at line 47:

```typescript
const startTraining = async (maxEpochs: number = 1000, videoIds: number[] = [], fromCheckpoint: boolean = false) => {
    if (!projectId.value || isTraining.value) return
    isTraining.value = true
    try {
        await createDetectionTraining(projectId.value, maxEpochs, videoIds, 10, 20, fromCheckpoint)
    } catch (e) {
        console.error('Failed to start detector training:', e)
        isTraining.value = false
        throw e
    }
}
```

- [ ] **Step 4: Verify frontend compiles**

```bash
cd frontend && npm run type-check
```

Expected: no type errors.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/services/api.ts frontend/src/composables/useDetector.ts
git commit -m "feat: add fromCheckpoint parameter to detector training API and composable"
```

---

### Task 4: Frontend — Add "Continue Training" button to VideoPipeline

**Files:**
- Modify: `frontend/src/components/VideoPipeline.vue:452-462` (handler)
- Modify: `frontend/src/components/VideoPipeline.vue:879-885` (template)

- [ ] **Step 1: Add `handleContinueTrainDetector` handler**

In `frontend/src/components/VideoPipeline.vue`, add the handler right after `handleTrainDetector` (after line 462):

```typescript
const handleContinueTrainDetector = async () => {
  if (!projectId.value || isDetectorTraining.value) return
  try {
    await startDetectorTraining(1000, selectedVideoIdsList.value, true)
    router.push(`/project/${projectId.value}/detector`)
  } catch (e: any) {
    console.error('Failed to continue detector training:', e)
    alert(e.message || 'Failed to continue detector training')
  }
}
```

- [ ] **Step 2: Add "Continue Training" button to template**

After the "Train Detector" button (line 885), add:

```html
          <button
            class="sidebar-button"
            @click="handleContinueTrainDetector"
            :disabled="!detectorModelExists || isDetectorTraining || selectedCount === 0"
          >
            <span class="button-label">{{ isDetectorTraining ? 'Training...' : 'Continue Training' }}</span>
          </button>
```

- [ ] **Step 3: Verify frontend compiles**

```bash
cd frontend && npm run type-check
```

Expected: no type errors.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/VideoPipeline.vue
git commit -m "feat: add Continue Training button to detector pipeline UI"
```

---

### Task 5: Manual verification

- [ ] **Step 1: Start backend and frontend**

```bash
vidseq &
cd frontend && npm run dev &
```

- [ ] **Step 2: Verify button state**

Open a project with no prior detector training. Confirm the "Continue Training" button is visible but disabled (grayed out). The "Train Detector" button should still work normally.

- [ ] **Step 3: Train a detector from scratch**

Click "Train Detector" with at least one video selected. Wait for training to complete. After completion, confirm:
- `detector.pt` exists in the project's `models/` directory
- "Continue Training" button is now enabled

- [ ] **Step 4: Continue training from checkpoint**

Click "Continue Training". Verify:
- Training starts and navigates to the detector training page
- Backend logs show "Fine-tuning from checkpoint: ..." (not pretrained loading)
- Training runs to completion with the progress chart updating

- [ ] **Step 5: Final commit (if any fixes needed)**

If any adjustments were made during verification, commit them.
