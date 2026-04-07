# Continue Training Detector — Design Spec

**Date:** 2026-04-07
**Scope:** Standard YOLO detector only (not OBB/Seg)
**Branch:** feat/associated-videos

## Summary

Add a "Continue Training" button to the detector section of the video pipeline action bar. When clicked, it starts a fresh training run that initializes from the previous best checkpoint weights instead of the COCO-pretrained backbone. This is fine-tuning from checkpoint — fresh optimizer, new LR schedule, same hyperparameters.

## Current State

- Training always starts from pretrained weights via `load_pretrained()` (`detector_model.py:24`)
- Previous training runs save `best.pt` to `project_path/models/yolo_train/weights/best.pt`
- Final model is copied to `project_path/models/detector.pt`
- `DetectorService.model_exists()` checks for `detector.pt` — this already indicates a prior run completed
- `load_finetuned()` (`detector_model.py:37`) already loads from an arbitrary weights path

## Design

### Backend

#### Schema (`vidseq/api/routes/detector.py`)

Add `from_checkpoint: bool = False` to `TrainRequest`:

```python
class TrainRequest(BaseModel):
    video_ids: list[int]
    max_epochs: int = 100
    batch_size: int = 4
    lr: float = 1e-4
    early_stop_patience: int = 20
    from_checkpoint: bool = False  # NEW
```

#### Route (`vidseq/api/routes/detector.py`)

Pass `from_checkpoint` through to service in `create_detection_training()`:

```python
service.train(
    project_path=project_path,
    video_ids=request.video_ids,
    max_epochs=request.max_epochs,
    batch_size=request.batch_size,
    lr=request.lr,
    early_stop_patience=request.early_stop_patience,
    from_checkpoint=request.from_checkpoint,  # NEW
)
```

#### Service (`vidseq/services/detector_service.py`)

`train()` accepts `from_checkpoint: bool = False` and passes it to `_train_sync()`.

In `_train_sync()`, replace the model loading at line 370:

```python
# Currently:
model = load_pretrained(detector_type, device="cpu")

# Becomes:
if from_checkpoint:
    checkpoint_path = project_path / "models" / f"{train_name}" / "weights" / "best.pt"
    if checkpoint_path.exists():
        model = load_finetuned(checkpoint_path, device="cpu")
        logger.info(f"Resuming from checkpoint: {checkpoint_path}")
    else:
        model = load_pretrained(detector_type, device="cpu")
        logger.warning("Checkpoint not found, falling back to pretrained weights")
else:
    model = load_pretrained(detector_type, device="cpu")
```

`train_name` is already computed just below (`"yolo_train"` for standard YOLO). The checkpoint path resolution needs to happen after `train_name` is determined — move model loading after the per-model hyperparameter block (lines 400-430) where `train_name` is set.

Everything else (dataset prep, callbacks, training loop, saving, post-training apply) remains unchanged.

### Frontend

#### API (`frontend/src/services/api.ts`)

Add `fromCheckpoint` parameter to `createDetectionTraining()`:

```typescript
export async function createDetectionTraining(
    projectId: number,
    maxEpochs: number,
    videoIds: number[],
    lrPatience: number = 10,
    earlyStopPatience: number = 20,
    fromCheckpoint: boolean = false,  // NEW
): Promise<void> {
    // ... existing code, add to body:
    body: JSON.stringify({
        video_ids: videoIds,
        max_epochs: maxEpochs,
        lr_patience: lrPatience,
        early_stop_patience: earlyStopPatience,
        from_checkpoint: fromCheckpoint,  // NEW
    }),
```

#### Composable (`frontend/src/composables/useDetector.ts`)

Add `fromCheckpoint` parameter to `startTraining()`:

```typescript
const startTraining = async (
    maxEpochs: number = 1000,
    videoIds: number[] = [],
    fromCheckpoint: boolean = false,
) => {
    // ...
    await createDetectionTraining(projectId.value, maxEpochs, videoIds, 10, 20, fromCheckpoint)
}
```

Update the `UseDetectorReturn` interface accordingly.

#### UI (`frontend/src/components/VideoPipeline.vue`)

Add handler:

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

Add button after the existing "Train Detector" button (line 885):

```html
<button
    class="sidebar-button"
    @click="handleContinueTrainDetector"
    :disabled="!detectorModelExists || isDetectorTraining || selectedCount === 0"
>
    <span class="button-label">{{ isDetectorTraining ? 'Training...' : 'Continue Training' }}</span>
</button>
```

- Visible always, disabled when no prior model exists (`!detectorModelExists`)
- Also disabled during active training or when no videos are selected
- Uses same training progress view (DetectorTraining.vue) — no changes needed there

## Files Changed

| File | Change |
|------|--------|
| `vidseq/api/routes/detector.py` | Add `from_checkpoint` to `TrainRequest`, pass through in route |
| `vidseq/services/detector_service.py` | Accept `from_checkpoint` in `train()` and `_train_sync()`, conditional model loading |
| `frontend/src/services/api.ts` | Add `fromCheckpoint` param to `createDetectionTraining()` |
| `frontend/src/composables/useDetector.ts` | Add `fromCheckpoint` param to `startTraining()` |
| `frontend/src/components/VideoPipeline.vue` | Add handler and button |

## Out of Scope

- OBB and Seg detector continue-training (same pattern, future work)
- Custom hyperparameters for continue-training runs
- True resume (optimizer state preservation)
- Training history across runs
