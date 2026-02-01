# Detector-Tracker Pipeline Refactor

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Combine detection and tracking into a single pass with on-the-fly detection and backtracking on drift.

**Architecture:** The command handler loads the detector model and passes a callable to the segmentor. The segmentor runs detection every N frames, compares IoU, and backtracks to re-propagate when drift is detected. Three H5 outputs capture tracker, detector, and final masks.

**Tech Stack:** PyTorch, SAM2, SegFormer detector, H5PY

---

## Phase 1: Memory Preparation Bug Fix ✅ COMPLETE (2026-02-01)

- Implemented `_set_memory_frame()` in streaming_segmentor.py
- Updated `propagate()`, `add_point_prompt()`, `refine_mask()` to accept indexable frame/mask sources
- Command handlers now pass `VideoResources.frame_source` and `VideoResources.mask_dataset`
- Note: Detector mask corrections use `mask_inputs` with `use_mask_input_as_output_without_sam=True`,
  meaning SAM2 takes detector masks as gospel (no refinement). This is intentional.

---

## Phase 2: Combined Detector-Tracker Workflow

### Current vs New Workflow

**Current (two-step):**
1. Train detector → `models/detector.pt`
2. Apply detector to all videos → `{video_id}_detector.h5`
3. Segment all videos → reads detector H5 for IoU comparison

**New (single-step):**
1. Train detector → `models/detector.pt`
2. Segment all videos → runs detector on-the-fly every N frames

### Algorithm States

**Normal mode** (detect every N frames):
- Propagate tracker each frame → write to tracker_masks AND final_masks
- Every `check_interval` frames: run detector, compare IoU
- IoU >= threshold: continue (tracker is good)
- IoU < threshold, detector non-empty: correct with detector, backtrack, re-propagate
- Detector empty: correct to empty, enter searching mode

**Searching mode** (detect every frame, no tracking):
- Run detector only (tracker is idle)
- Write empty masks to tracker_masks and final_masks
- When detector finds object: add as conditioning frame, backtrack, re-propagate, return to normal mode

### Interface Design

```python
def propagate_with_detector(
    self,
    video_id: str,
    num_frames: int,
    frames,                    # Sequence - read frames[idx]
    get_detector_mask,         # Callable[[int, np.ndarray], np.ndarray]
    tracker_masks,             # MutableSequence - write tracker output
    detector_masks,            # MutableSequence - write detector output
    final_masks,               # MutableSequence - write final output
    on_progress=None,          # Optional[Callable[[int], None]]
    check_interval: int = 10,
    iou_threshold: float = 0.5,
) -> None:
```

### Detailed Algorithm

```
Initialization:
  - frame_idx = 0
  - Run detector until non-empty mask found (skip empty frames at start)
  - Init tracker from that mask (conditioning frame)
  - Write to tracker_masks[frame_idx], detector_masks[frame_idx], final_masks[frame_idx]
  - last_successful_check = frame_idx
  - Enter normal mode from frame_idx + 1

Normal mode loop (frame N):
  1. Propagate tracker → write to tracker_masks[N] AND final_masks[N]

  2. If N % check_interval == 0:
     - Run detector → write to detector_masks[N]
     - Compute IoU(tracker, detector)

     - If IoU >= threshold:
         - (final_masks[N] already has tracker output, keep it)
         - last_successful_check = N

     - If IoU < threshold AND detector non-empty:
         - Correct with detector mask (new conditioning frame)
         - Overwrite final_masks[N] with detector mask
         - Backtrack: re-propagate (last_successful_check+1) to (N-1) → overwrite final_masks only
         - last_successful_check = N

     - If detector empty:
         - Correct with empty mask (new conditioning frame)
         - Overwrite final_masks[N] with empty
         - Backtrack: re-propagate (last_successful_check+1) to (N-1) → overwrite final_masks only
         - last_successful_check = N
         - Enter searching mode

Searching mode loop (frame N):
  1. Run detector → write to detector_masks[N]
  2. Write empty to tracker_masks[N] and final_masks[N]

  3. If detector non-empty:
     - Add detector mask as conditioning frame
     - Write detector mask to final_masks[N]
     - Backtrack: re-propagate (last_successful_check+1) to (N-1) → overwrite final_masks only
     - last_successful_check = N
     - Return to normal mode, continue from N+1
```

### Edge Cases

- **Object never appears:** Detector empty for all frames → write all empty, complete successfully
- **Empty-vs-empty at check frame:** IoU = 1.0 (agreement), no correction needed
- **Detector inference fails:** Let exception propagate, job fails

---

## Implementation Tasks

### Task 1: Update segmentor method signature

**Files:**
- Modify: `vidseq/services/sam2/streaming_segmentor.py:1178-1330`

**Step 1: Update method signature**

Change `propagate_with_detector` to accept the new parameters:

```python
def propagate_with_detector(
    self,
    video_id: str,
    num_frames: int,
    frames,                    # Sequence - read frames[idx]
    get_detector_mask,         # Callable[[int, np.ndarray], np.ndarray]
    tracker_masks,             # MutableSequence - write tracker output
    detector_masks,            # MutableSequence - write detector output
    final_masks,               # MutableSequence - write final output
    on_progress: Callable[[int], None] | None = None,
    check_interval: int = 10,
    iou_threshold: float = 0.5,
) -> None:
```

**Step 2: Commit**

```bash
git add vidseq/services/sam2/streaming_segmentor.py
git commit -m "refactor: update propagate_with_detector signature for on-the-fly detection"
```

---

### Task 2: Implement initialization logic

**Files:**
- Modify: `vidseq/services/sam2/streaming_segmentor.py`

**Step 1: Implement initialization that skips empty frames**

Replace the existing initialization block with:

```python
def propagate_with_detector(self, ...):
    """Propagate tracking with on-the-fly detection and backtracking."""
    session = self.sessions[video_id]
    output_dict = session["output_dict"]

    # Find first frame with detector mask
    start_frame = 0
    for idx in range(num_frames):
        frame = frames[idx]
        detector_mask = get_detector_mask(idx, frame)
        detector_masks[idx] = detector_mask

        if (detector_mask > 127).any():
            start_frame = idx
            break
        else:
            # Write empty to tracker and final for skipped frames
            empty_mask = np.zeros_like(detector_mask)
            tracker_masks[idx] = empty_mask
            final_masks[idx] = empty_mask
            if on_progress:
                on_progress(idx)
    else:
        # No object found in entire video
        print(f"  No object detected in any frame")
        return

    # Initialize tracker from first detected frame
    frame = frames[start_frame]
    detector_mask = detector_masks[start_frame]  # Already written above
    mask, _ = self._propagate_single_frame(
        video_id, start_frame, frame, mask_prompt=detector_mask
    )
    tracker_masks[start_frame] = mask
    final_masks[start_frame] = mask

    last_successful_check = start_frame
    searching = False

    if on_progress:
        on_progress(start_frame)
```

**Step 2: Commit**

```bash
git add vidseq/services/sam2/streaming_segmentor.py
git commit -m "feat: add initialization logic that skips empty detector frames"
```

---

### Task 3: Implement normal mode loop

**Files:**
- Modify: `vidseq/services/sam2/streaming_segmentor.py`

**Step 1: Implement normal mode processing**

Add the main loop after initialization:

```python
    # Main loop
    for frame_idx in range(start_frame + 1, num_frames):
        frame = frames[frame_idx]

        if searching:
            # Searching mode: detect every frame, no tracking
            detector_mask = get_detector_mask(frame_idx, frame)
            detector_masks[frame_idx] = detector_mask

            # Write empty to tracker and final while searching
            empty_mask = np.zeros((frames[0].shape[0], frames[0].shape[1]), dtype=np.uint8)
            tracker_masks[frame_idx] = empty_mask
            final_masks[frame_idx] = empty_mask

            if (detector_mask > 127).any():
                # Object reappeared - add as conditioning frame
                mask, _ = self._propagate_single_frame(
                    video_id, frame_idx, frame, mask_prompt=detector_mask
                )
                final_masks[frame_idx] = mask

                # Backtrack and re-propagate
                self._backtrack_reprop(
                    video_id, frames, final_masks,
                    last_successful_check + 1, frame_idx - 1
                )

                last_successful_check = frame_idx
                searching = False
        else:
            # Normal mode: propagate tracker
            tracker_mask, _ = self._propagate_single_frame(video_id, frame_idx, frame)
            tracker_masks[frame_idx] = tracker_mask
            final_masks[frame_idx] = tracker_mask

            # Check frame?
            if frame_idx % check_interval == 0:
                detector_mask = get_detector_mask(frame_idx, frame)
                detector_masks[frame_idx] = detector_mask

                iou = self._compute_iou(tracker_mask, detector_mask)

                if iou >= iou_threshold:
                    # Tracker is good
                    last_successful_check = frame_idx
                elif (detector_mask > 127).any():
                    # Drift detected, detector has mask - correct
                    mask, _ = self._propagate_single_frame(
                        video_id, frame_idx, frame, mask_prompt=detector_mask
                    )
                    final_masks[frame_idx] = mask

                    # Backtrack and re-propagate
                    self._backtrack_reprop(
                        video_id, frames, final_masks,
                        last_successful_check + 1, frame_idx - 1
                    )

                    last_successful_check = frame_idx
                else:
                    # Detector empty - object disappeared
                    empty_mask = np.zeros_like(tracker_mask)
                    self._propagate_single_frame(
                        video_id, frame_idx, frame, mask_prompt=empty_mask
                    )
                    final_masks[frame_idx] = empty_mask

                    # Backtrack and re-propagate
                    self._backtrack_reprop(
                        video_id, frames, final_masks,
                        last_successful_check + 1, frame_idx - 1
                    )

                    last_successful_check = frame_idx
                    searching = True

        if on_progress:
            on_progress(frame_idx)
```

**Step 2: Commit**

```bash
git add vidseq/services/sam2/streaming_segmentor.py
git commit -m "feat: implement normal and searching mode loop"
```

---

### Task 4: Implement backtrack helper

**Files:**
- Modify: `vidseq/services/sam2/streaming_segmentor.py`

**Step 1: Add backtrack re-propagation helper**

Add this method before `propagate_with_detector`:

```python
def _backtrack_reprop(
    self,
    video_id: str,
    frames,
    final_masks,
    start_idx: int,
    end_idx: int,
) -> None:
    """Re-propagate frames after a correction, writing only to final_masks.

    Called after adding a new conditioning frame. Uses _set_memory_frame
    to prepare memory context, then propagates forward.
    """
    if start_idx > end_idx:
        return

    for idx in range(start_idx, end_idx + 1):
        # Prepare memory for this frame (uses new conditioning frame)
        self._set_memory_frame(video_id, idx, frames, final_masks)

        frame = frames[idx]
        mask, _ = self._propagate_single_frame(video_id, idx, frame)
        final_masks[idx] = mask
```

**Step 2: Commit**

```bash
git add vidseq/services/sam2/streaming_segmentor.py
git commit -m "feat: add _backtrack_reprop helper for re-propagation after correction"
```

---

### Task 5: Add IoU computation helper

**Files:**
- Modify: `vidseq/services/sam2/streaming_segmentor.py`

**Step 1: Add IoU helper that handles empty masks**

```python
def _compute_iou(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute IoU between two binary masks.

    Handles empty masks: empty-vs-empty = 1.0, non-empty-vs-empty = 0.0
    """
    bin1 = mask1 > 127
    bin2 = mask2 > 127

    # Handle empty masks
    if not bin1.any() and not bin2.any():
        return 1.0  # Both empty = agreement
    if not bin1.any() or not bin2.any():
        return 0.0  # One empty = no overlap

    intersection = np.logical_and(bin1, bin2).sum()
    union = np.logical_or(bin1, bin2).sum()

    return intersection / union if union > 0 else 0.0
```

Note: This may replace the existing `_compute_bbox_iou` method or exist alongside it.

**Step 2: Commit**

```bash
git add vidseq/services/sam2/streaming_segmentor.py
git commit -m "feat: add _compute_iou helper with empty mask handling"
```

---

### Task 6: Update command handler

**Files:**
- Modify: `vidseq/services/segmentation_commands.py:547-619`

**Step 1: Update handle_propagate_with_detector**

```python
def handle_propagate_with_detector(
    params: dict,
    segmentor: StreamingSegmentor,
    response_callback: Callable[[dict], None],
) -> dict:
    """Propagate tracking with on-the-fly detection.

    Loads detector model and runs detection every check_interval frames.
    """
    import torch
    from vidseq.services.detector_model import SegFormerDetector
    from vidseq.services.h5_storage import open_video_h5

    video_id = params["video_id"]
    num_frames = params["num_frames"]
    project_path = Path(params["project_path"])
    check_interval = params.get("check_interval", 10)
    iou_threshold = params.get("iou_threshold", 0.5)

    if segmentor is None:
        raise RuntimeError("Model not loaded")

    if video_id not in _video_resources:
        raise RuntimeError(f"No session for video {video_id}")

    resources = _video_resources[video_id]

    # Check detector model exists
    model_path = project_path / "models" / "detector.pt"
    if not model_path.exists():
        raise RuntimeError("No trained detector model found. Train first.")

    # Load detector model
    print(f"[SAM Worker] Loading detector model from {model_path}")
    detector = SegFormerDetector(device="cuda")
    detector.load_decoder(str(model_path))
    detector.eval()
    detector = torch.compile(detector, mode="max-autotune", fullgraph=True)

    # GPU preprocessing constants
    IMG_MEAN = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
    IMG_STD = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)

    def get_detector_mask(frame_idx: int, frame: np.ndarray) -> np.ndarray:
        """Run detector on a single frame."""
        # GPU preprocessing
        frame_gpu = torch.from_numpy(frame).to("cuda")
        pixel_values = frame_gpu[..., [2, 1, 0]].permute(2, 0, 1).float().div_(255.0)
        pixel_values = pixel_values.unsqueeze(0)
        pixel_values = (pixel_values - IMG_MEAN) / IMG_STD

        # Inference
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            logits = detector(pixel_values)

        # Post-process: argmax, resize, to numpy
        pred = logits.argmax(dim=1)[0]  # (H/4, W/4)
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(0).unsqueeze(0).float(),
            size=(frame.shape[0], frame.shape[1]),
            mode="nearest"
        )[0, 0]
        mask = (pred * 255).to(torch.uint8).cpu().numpy()
        return mask

    def on_progress(frame_idx: int) -> None:
        if frame_idx % 50 == 0 or frame_idx == num_frames - 1:
            response_callback({
                "type": "progress",
                "frame_idx": frame_idx,
                "total": num_frames,
            })

    # Open H5 files with locking
    with open_video_h5(project_path, video_id, "a") as tracker_h5, \
         open_video_h5(project_path, video_id, "a", suffix="_detector") as detector_h5, \
         open_video_h5(project_path, video_id, "a", suffix="_final") as final_h5:

        segmentor.propagate_with_detector(
            video_id=str(video_id),
            num_frames=num_frames,
            frames=resources.frame_source,
            get_detector_mask=get_detector_mask,
            tracker_masks=tracker_h5["masks"],
            detector_masks=detector_h5["masks"],
            final_masks=final_h5["masks"],
            on_progress=on_progress,
            check_interval=check_interval,
            iou_threshold=iou_threshold,
        )

    return {
        "type": "propagate_with_detector_result",
        "status": "ok",
    }
```

**Step 2: Commit**

```bash
git add vidseq/services/segmentation_commands.py
git commit -m "refactor: update command handler for on-the-fly detection"
```

---

### Task 7: Remove apply-all endpoint

**Files:**
- Modify: `vidseq/api/routes/detector.py:115-132`

**Step 1: Remove the endpoint**

Delete the `apply_detector_to_all` function and its route decorator (lines 115-132).

**Step 2: Commit**

```bash
git add vidseq/api/routes/detector.py
git commit -m "remove: delete apply-all endpoint (detection now on-the-fly)"
```

---

### Task 8: Remove apply_to_all from detector service

**Files:**
- Modify: `vidseq/services/detector_service.py`

**Step 1: Remove methods**

Delete `apply_to_all` and `_apply_to_all_sync` methods from `DetectorService`.

**Step 2: Commit**

```bash
git add vidseq/services/detector_service.py
git commit -m "remove: delete apply_to_all methods from detector service"
```

---

### Task 9: Remove frontend apply-all code

**Files:**
- Modify: `frontend/src/services/api.ts:1275-1282`
- Modify: `frontend/src/components/VideoPipeline.vue`

**Step 1: Remove API function**

Delete `applyDetectorToAll` function from `api.ts`.

**Step 2: Remove button and handler from VideoPipeline.vue**

- Remove import of `applyDetectorToAll`
- Remove `handleApplyDetectorToAll` function
- Remove the "Apply to All" button from template

**Step 3: Commit**

```bash
git add frontend/src/services/api.ts frontend/src/components/VideoPipeline.vue
git commit -m "remove: delete apply-all UI and API function"
```

---

### Task 10: Update validation in sessions route

**Files:**
- Modify: `vidseq/api/routes/segmentation/sessions.py`

**Step 1: Update validation**

Change validation to check for detector model existence instead of pre-computed detector masks.

**Step 2: Commit**

```bash
git add vidseq/api/routes/segmentation/sessions.py
git commit -m "refactor: validate detector model exists instead of detector masks"
```

---

## Files Summary

### Modified:
- `vidseq/services/sam2/streaming_segmentor.py` - Rewrite `propagate_with_detector()`, add helpers
- `vidseq/services/segmentation_commands.py` - Load detector in handler, update call
- `vidseq/api/routes/detector.py` - Remove apply-all endpoint
- `vidseq/services/detector_service.py` - Remove apply_to_all methods
- `vidseq/api/routes/segmentation/sessions.py` - Update validation
- `frontend/src/services/api.ts` - Remove applyDetectorToAll
- `frontend/src/components/VideoPipeline.vue` - Remove apply-all button
