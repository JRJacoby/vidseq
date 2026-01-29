# DINOv2 Detector Training & Verification

**Goal:** Add a DINOv2-based detector for segmentation that can be trained on user-annotated masks, verified against training data, and later used in detector-tracker mode.

**Scope:** This design covers training and verification only. Detector-tracker inference mode is deferred.

---

## Architecture

### Model Components

- **Backbone:** DINOv2-giant (1.1B params, frozen during training)
- **Decoder:** Lightweight convolutional decoder (~3M trainable params)

### Decoder Architecture

```
Input: (B, 1536, H/14, W/14)  # DINOv2 patch features
  ↓ Conv 1536→512, 3×3, BatchNorm, ReLU
  ↓ Upsample 2×
  ↓ Conv 512→256, 3×3, BatchNorm, ReLU
  ↓ Upsample 2×
  ↓ Conv 256→128, 3×3, BatchNorm, ReLU
  ↓ Upsample 2×
  ↓ Conv 128→64, 3×3, BatchNorm, ReLU
  ↓ Bilinear upsample to original (trimmed) resolution
  ↓ Conv 64→1, 1×1 (raw logits)
Output: (B, 1, H_trim, W_trim)
```

### Input Preprocessing

- Resize longest edge to ~518, preserve aspect ratio
- Trim dimensions to nearest multiple of 14 (DINOv2 patch size)
- ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

### Training

- **Targets:** Binary masks from tracker (threshold at 127)
- **Loss:** 0.5 × BCE + 0.5 × Dice
- **Optimizer:** AdamW, initial lr=1e-4, weight_decay=1e-4
- **Max epochs:** 1000
- **LR patience:** 10 epochs (reduce by 0.25× if no improvement)
- **Early stop patience:** 20 epochs
- **Batch size:** 4
- **Train/val split:** 80/20
- **Data:** All frames in training ranges across all project videos

### Storage

- **Checkpoint:** `<project>/models/detector.pt` (decoder weights only)
- **Detector masks:** `<project>/masks/<video_id>.h5` in `detector_masks` dataset (same file as tracker masks)

---

## Workflow

```
1. ANNOTATE (existing)
   └─ User adds point prompts → SAM2 generates masks → propagate
   └─ Masks stored in HDF5 `masks` dataset

2. LABEL (existing)
   └─ User marks training ranges via DataTrack
   └─ Stored in DB as frame ranges per video

3. TRAIN (new)
   └─ Click "Train Detector" button in VideoPipeline.vue
   └─ Backend gathers all training frames + masks across project
   └─ Trains DINOv2 decoder (backbone frozen)
   └─ Saves checkpoint to <project>/models/detector.pt
   └─ Real-time loss chart via SSE in DetectorTraining.vue
   └─ After training completes: automatically applies detector to training frames
   └─ Stores detector predictions in HDF5 `detector_masks` dataset

4. VERIFY (new)
   └─ In VideoDetail, select "Detector" from mask view dropdown
   └─ Compare detector output to tracker masks on training frames
   └─ Assess if training was successful
```

---

## API Endpoints

### Training

- `POST /projects/{id}/detector/train` - Start training (returns immediately)
- `GET /projects/{id}/detector/training/status` - Poll progress
- `GET /projects/{id}/detector/training/stream` - SSE real-time updates
- `POST /projects/{id}/detector/stop` - Stop training

### Masks

- `GET /projects/{pid}/videos/{vid}/detector-mask/{frame_idx}` - Fetch detector mask
- `GET /projects/{pid}/videos/{vid}/detector-masks/exists` - Check if detector masks exist

### Progress Schema

```python
@dataclass
class DetectorTrainingProgress:
    is_training: bool
    status: str  # idle, training, completed, stopped, failed
    current_epoch: int
    max_epochs: int
    current_train_loss: float
    current_val_loss: float
    train_loss_history: list[float]
    val_loss_history: list[float]
    best_val_loss: float
    best_epoch: int
    current_lr: float
    epochs_without_improvement: int
```

---

## File Structure

### New Files

```
vidseq/
├── services/
│   ├── detector_service.py      # Training orchestration, progress tracking
│   └── detector_model.py        # DINOv2 + decoder architecture
├── api/routes/
│   └── detector.py              # Train/status/stream endpoints
└── schemas/
    └── detector.py              # DetectorTrainingProgress, request models

frontend/src/
├── components/
│   └── DetectorTraining.vue     # Training progress UI (loss chart)
├── composables/
│   └── useDetector.ts           # Training API calls
└── services/
    └── api.ts                   # Add detector endpoints
```

### Modified Files

- `vidseq/services/sam2/streaming_segmentor.py` - Add `train_detector()`, `apply_detector()`, `load_detector()` methods
- `frontend/src/components/VideoPipeline.vue` - Add "Train Detector" button
- `frontend/src/components/VideoDetail.vue` - Add mask view dropdown
- `frontend/src/composables/useSegmentation.ts` - Handle view mode when fetching masks
- `frontend/src/router.ts` - Add `/project/:id/detector` route

---

## UI Changes

### VideoPipeline.vue

Add "Train Detector" button in the project-wide operations section (alongside "Train YOLO").

### DetectorTraining.vue (new page)

- Chart.js line chart showing train/val loss (log scale Y-axis)
- Status display: current epoch, best epoch, learning rate, patience counters
- Matches `Alignment.vue` layout

### VideoDetail.vue

Add dropdown in toolbar:
```
[Tracker ▼]
  ├─ Tracker (default)
  ├─ Detector
  └─ Final (disabled - future)
```

Switching views fetches from appropriate HDF5 dataset. Non-training frames with no detector mask simply show no overlay (empty/zero mask).

---

## Key Design Decisions

1. **Frozen backbone** - Only train decoder (~3M params vs 1.1B), faster training, less overfitting risk
2. **Binary mask targets** - Clean supervision, standard segmentation approach
3. **Same HDF5 file** - Keep all mask data for a video together
4. **Apply only to training frames** - Verification only; full video inference deferred to detector-tracker mode
5. **Auto-apply after training** - No separate button, streamlines workflow
6. **Match alignment training UX** - Familiar patterns for loss visualization and progress tracking
