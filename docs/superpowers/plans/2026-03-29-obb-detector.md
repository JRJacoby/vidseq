# OBB Detector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a parallel OBB (Oriented Bounding Box) detector system that runs YOLO-OBB on video frames, producing tightly-fitted rotated bboxes stored in their own DB columns with dedicated training, apply, and visualization flows.

**Architecture:** Parallel to the existing axis-aligned detector — own DB columns (8 corner coords + score), own training label export via `cv2.minAreaRect`, own apply flow via the generalized `apply_detector` TCP command, own frontend sidebar section and overlay rendering. Shares `DetectorService` singleton for training infrastructure and the GPU worker for inference.

**Tech Stack:** FastAPI, SQLAlchemy, Ultralytics YOLO-OBB, Vue 3, Canvas API

**Spec:** `docs/superpowers/specs/2026-03-29-obb-detector-design.md`

---

### Task 1: Database Model + Migration Script

**Files:**
- Modify: `vidseq/models/frame_data.py`
- Create: `scripts/migrate_add_obb_columns.py`

- [ ] **Step 1: Add 9 OBB columns to FrameData model**

In `vidseq/models/frame_data.py`, add after the `has_final_mask` column (line 45), before `__table_args__`:

```python
    # OBB (Oriented Bounding Box) corner coordinates (NULL = no OBB detection)
    # 4 corners in clockwise order, absolute pixel coordinates
    obb_x1: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
    obb_y1: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
    obb_x2: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
    obb_y2: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
    obb_x3: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
    obb_y3: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
    obb_x4: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
    obb_y4: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)

    # OBB confidence score (-1.0 = not computed)
    obb_score: Mapped[float] = mapped_column(Float, nullable=False, default=-1.0)
```

- [ ] **Step 2: Create migration script**

Create `scripts/migrate_add_obb_columns.py`:

```python
"""Add OBB columns to existing project databases.

Usage: uv run python scripts/migrate_add_obb_columns.py /path/to/project
"""
import sqlite3
import sys
from pathlib import Path


OBB_COLUMNS = {
    "obb_x1": "FLOAT",
    "obb_y1": "FLOAT",
    "obb_x2": "FLOAT",
    "obb_y2": "FLOAT",
    "obb_x3": "FLOAT",
    "obb_y3": "FLOAT",
    "obb_x4": "FLOAT",
    "obb_y4": "FLOAT",
    "obb_score": "FLOAT NOT NULL DEFAULT -1.0",
}


def find_db(project_path: Path) -> Path | None:
    """Find vidseq.db under the project path."""
    db = project_path / "vidseq.db"
    if db.exists():
        return db
    # Walk subdirectories
    for db in project_path.rglob("vidseq.db"):
        return db
    return None


def migrate(db_path: Path) -> None:
    """Add OBB columns to frame_data table."""
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Get existing columns
    cursor.execute("PRAGMA table_info(frame_data)")
    existing = {row[1] for row in cursor.fetchall()}

    added = 0
    for col_name, col_type in OBB_COLUMNS.items():
        if col_name not in existing:
            cursor.execute(f"ALTER TABLE frame_data ADD COLUMN {col_name} {col_type}")
            added += 1
            print(f"  Added column: {col_name}")

    conn.commit()
    conn.close()

    if added == 0:
        print("  All OBB columns already exist, nothing to do.")
    else:
        print(f"  Added {added} columns.")


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} /path/to/project")
        sys.exit(1)

    project_path = Path(sys.argv[1])
    if not project_path.exists():
        print(f"Error: {project_path} does not exist")
        sys.exit(1)

    db_path = find_db(project_path)
    if db_path is None:
        print(f"Error: No vidseq.db found under {project_path}")
        sys.exit(1)

    print(f"Migrating: {db_path}")
    migrate(db_path)
    print("Done.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Commit**

```bash
git add vidseq/models/frame_data.py scripts/migrate_add_obb_columns.py
git commit -m "feat: add OBB columns to FrameData model and migration script"
```

---

### Task 2: Detector Model Changes

**Files:**
- Modify: `vidseq/services/detector_model.py`

- [ ] **Step 1: Add `"obb"` to PRETRAINED_MODELS**

In `detector_model.py`, update the `PRETRAINED_MODELS` dict (line 16):

```python
PRETRAINED_MODELS = {
    "rtdetr": "rtdetr-x.pt",
    "yolo": "yolo11n.pt",
    "obb": "yolo11n-obb.pt",
}
```

- [ ] **Step 2: Add `detect_obb()` function**

Add after the `detect()` function (after line 76):

```python
def detect_obb(
    model,
    frame: np.ndarray,
    conf: float = DETECTION_CONF_THRESHOLD,
) -> list[dict]:
    """Run OBB detection on a single BGR frame.

    Args:
        model: Ultralytics OBB model instance.
        frame: BGR uint8 numpy array (H, W, 3).
        conf: Confidence threshold.

    Returns:
        List of detections sorted by confidence (descending).
        Each detection: {"corners": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]], "conf": float}
        Corner coordinates are in pixel space of the original frame.
    """
    results = model(frame, conf=conf, verbose=False)
    detections = []
    if len(results) > 0 and results[0].obb is not None:
        obb = results[0].obb
        for i in range(len(obb)):
            corners = obb.xyxyxyxy[i].cpu().tolist()
            detections.append({
                "corners": corners,
                "conf": obb.conf[i].item(),
            })
    detections.sort(key=lambda d: d["conf"], reverse=True)
    return detections
```

- [ ] **Step 3: Commit**

```bash
git add vidseq/services/detector_model.py
git commit -m "feat: add OBB model variant and detect_obb function"
```

---

### Task 3: Frame Data Service (OBB CRUD)

**Files:**
- Modify: `vidseq/services/frame_data_service.py`

- [ ] **Step 1: Add `save_obb_bboxes_batch()`**

Add after `save_detector_bboxes_batch` (after line 734). OBB bboxes have 9 values: frame_idx + 8 corner coords:

```python
async def save_obb_bboxes_batch(
    session: AsyncSession,
    video_id: int,
    bboxes: list[list],
) -> None:
    """Batch insert/update OBB bounding boxes.

    Args:
        session: Async database session.
        video_id: Video ID.
        bboxes: List of [frame_idx, x1, y1, x2, y2, x3, y3, x4, y4] lists.
    """
    if not bboxes:
        return
    rows = [
        {
            "video_id": video_id,
            "frame_idx": int(b[0]),
            "obb_x1": float(b[1]), "obb_y1": float(b[2]),
            "obb_x2": float(b[3]), "obb_y2": float(b[4]),
            "obb_x3": float(b[5]), "obb_y3": float(b[6]),
            "obb_x4": float(b[7]), "obb_y4": float(b[8]),
        }
        for b in bboxes
    ]
    await _chunked_upsert(session, rows, ["video_id", "frame_idx"],
        ["obb_x1", "obb_y1", "obb_x2", "obb_y2", "obb_x3", "obb_y3", "obb_x4", "obb_y4"])
```

- [ ] **Step 2: Add `save_obb_scores_batch()`**

Add after the function above:

```python
async def save_obb_scores_batch(
    session: AsyncSession,
    video_id: int,
    scores: list[list],
) -> None:
    """Batch insert/update OBB confidence scores."""
    if not scores:
        return
    values = [
        {"video_id": video_id, "frame_idx": int(frame_idx), "obb_score": float(score)}
        for frame_idx, score in scores
    ]
    await _chunked_upsert(session, values, ["video_id", "frame_idx"], ["obb_score"])
```

- [ ] **Step 3: Add `get_obb_bbox()` and `get_obb_bboxes_batch()`**

Add after the functions above:

```python
async def get_obb_bbox(
    session: AsyncSession,
    video_id: int,
    frame_idx: int,
) -> dict | None:
    """Get OBB bbox for a single frame.

    Returns:
        Dict with corners key containing [[x1,y1],[x2,y2],[x3,y3],[x4,y4]], or None.
    """
    result = await session.execute(
        select(
            FrameData.obb_x1, FrameData.obb_y1,
            FrameData.obb_x2, FrameData.obb_y2,
            FrameData.obb_x3, FrameData.obb_y3,
            FrameData.obb_x4, FrameData.obb_y4,
        ).where(
            FrameData.video_id == video_id,
            FrameData.frame_idx == frame_idx,
            FrameData.obb_x1.isnot(None),
        )
    )
    row = result.first()
    if row is None:
        return None
    return {
        "corners": [
            [row[0], row[1]], [row[2], row[3]],
            [row[4], row[5]], [row[6], row[7]],
        ]
    }


async def get_obb_bboxes_batch(
    session: AsyncSession,
    video_id: int,
    start_frame: int,
    count: int = 100,
) -> list[dict]:
    """Get OBB bboxes for a range of frames."""
    result = await session.execute(
        select(
            FrameData.frame_idx,
            FrameData.obb_x1, FrameData.obb_y1,
            FrameData.obb_x2, FrameData.obb_y2,
            FrameData.obb_x3, FrameData.obb_y3,
            FrameData.obb_x4, FrameData.obb_y4,
        ).where(
            FrameData.video_id == video_id,
            FrameData.frame_idx >= start_frame,
            FrameData.frame_idx < start_frame + count,
            FrameData.obb_x1.isnot(None),
        ).order_by(FrameData.frame_idx)
    )
    return [
        {
            "frame_idx": row[0],
            "corners": [
                [row[1], row[2]], [row[3], row[4]],
                [row[5], row[6]], [row[7], row[8]],
            ],
        }
        for row in result.all()
    ]
```

- [ ] **Step 4: Add `get_obb_scores_downsampled()`**

Add a `load_obb_scores_in_range` helper and the downsampled function. Find the existing `load_detector_scores_in_range` function and add the OBB equivalent after it:

```python
async def load_obb_scores_in_range(
    session: AsyncSession,
    video_id: int,
    start_frame: int,
    end_frame: int,
) -> list[dict]:
    """Load OBB scores for a frame range."""
    result = await session.execute(
        select(FrameData.frame_idx, FrameData.obb_score)
        .where(
            FrameData.video_id == video_id,
            FrameData.frame_idx >= start_frame,
            FrameData.frame_idx <= end_frame,
            FrameData.obb_score > -1.0,
        )
        .order_by(FrameData.frame_idx)
    )
    return [{"frame_idx": row[0], "score": row[1]} for row in result.all()]


async def get_obb_scores_downsampled(
    session: AsyncSession,
    video_id: int,
    num_frames: int,
    max_samples: int = 800,
    start_frame: int = 0,
    end_frame: int | None = None,
) -> dict:
    """Get LTTB-downsampled OBB confidence scores for visualization."""
    from vidseq.services import lttb

    if end_frame is None:
        end_frame = num_frames - 1

    scores = await load_obb_scores_in_range(session, video_id, start_frame, end_frame)
    downsampled = lttb.downsample_scores(scores, max_samples)
    return {"scores": downsampled, "total_count": len(scores)}
```

- [ ] **Step 5: Add `obb_bboxes_exist()`**

Add after the functions above:

```python
async def obb_bboxes_exist(
    session: AsyncSession,
    video_id: int,
) -> bool:
    """Check if any OBB bbox data exists for a video."""
    result = await session.execute(
        select(FrameData.id)
        .where(
            FrameData.video_id == video_id,
            FrameData.obb_x1.isnot(None),
        )
        .limit(1)
    )
    return result.first() is not None
```

- [ ] **Step 6: Commit**

```bash
git add vidseq/services/frame_data_service.py
git commit -m "feat: add OBB bbox and score CRUD functions to frame_data_service"
```

---

### Task 4: Detector Service (OBB Training Flow)

**Files:**
- Modify: `vidseq/services/detector_service.py`

- [ ] **Step 1: Add `_mask_to_obb_label()` function**

Add after `_mask_to_yolo_bbox` (after line 63):

```python
def _mask_to_obb_label(mask: np.ndarray, img_h: int, img_w: int) -> str | None:
    """Convert a binary mask to YOLO OBB label format.

    Uses cv2.minAreaRect to find the minimum-area rotated rectangle,
    then outputs 4 normalized corner points.

    Args:
        mask: Binary mask (H, W) with values 0 or 255.
        img_h: Image height in pixels.
        img_w: Image width in pixels.

    Returns:
        YOLO OBB format string "class x1 y1 x2 y2 x3 y3 x4 y4" (normalized),
        or None if mask is empty.
    """
    contours, _ = cv2.findContours(
        (mask > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None

    # Merge all contours into one point set
    all_points = np.concatenate(contours)
    rect = cv2.minAreaRect(all_points)
    corners = cv2.boxPoints(rect)  # 4 corner points

    # Normalize to [0, 1]
    parts = []
    for cx, cy in corners:
        parts.append(f"{cx / img_w:.6f}")
        parts.append(f"{cy / img_h:.6f}")

    return f"0 {' '.join(parts)}"
```

- [ ] **Step 2: Add `training_type` parameter to `train()` and `_training_type` field**

In the `__init__` method (around line 107), add after `self._stop_requested = False`:

```python
        self._training_type: str | None = None
```

Update the `train()` method signature (line 142) to accept `training_type`:

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
    ) -> bool:
```

In the `_train_thread` closure inside `train()`, pass `training_type` through and set `self._training_type`:

```python
        self._training_type = training_type

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
                )
            except Exception as e:
                logger.exception("Training failed")
                self._training_progress.status = "failed"
                self._training_progress.error_message = str(e)
            finally:
                self._is_training = False
                self._training_type = None
```

- [ ] **Step 3: Add `obb_model_exists()` method**

Add after `model_exists` (after line 140):

```python
    def obb_model_exists(self, project_path: Path) -> bool:
        """Check if trained OBB model exists."""
        return (project_path / "models" / "obb_detector.pt").exists()
```

- [ ] **Step 4: Update `_train_sync()` to support OBB**

Add `training_type` parameter to `_train_sync` signature (line 182):

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
    ) -> None:
```

Replace the detector type reading block (lines 199-205) with:

```python
        # Determine model type
        if training_type == "obb":
            detector_type = "obb"
        else:
            detector_type = read_detector_config(project_path)

        logger.info(
            f"Starting {detector_type.upper()} training: max_epochs={max_epochs}, "
            f"batch_size={batch_size}, lr={lr}"
        )
```

In `_write_yolo_dataset`, update the label writing (around line 395) to branch on detector type. The simplest approach: pass `detector_type` to `_write_yolo_dataset`. Add it as a parameter:

```python
    def _write_yolo_dataset(
        self,
        frames: list[tuple[Path, int, int]],
        split: str,
        tmp_path: Path,
        project_path: Path,
        detector_type: str = "detector",
    ) -> None:
```

Then replace the label conversion line (around line 396):

```python
        # Convert mask to label
        if detector_type == "obb":
            yolo_line = _mask_to_obb_label(mask, img_h, img_w)
        else:
            yolo_line = _mask_to_yolo_bbox(mask, img_h, img_w)
```

Update the two calls to `_write_yolo_dataset` in `_train_sync` to pass `detector_type`:

```python
        self._write_yolo_dataset(train_frames, "train", tmp_path, project_path, detector_type)
        self._write_yolo_dataset(val_frames, "val", tmp_path, project_path, detector_type)
```

In the YOLO/RT-DETR branching block (lines 304-311), add OBB case:

```python
        if detector_type == "obb":
            train_workers = 4
            train_batch = max(batch_size, 8)
            train_name = "obb_train"
        elif detector_type == "yolo":
            train_workers = 4
            train_batch = max(batch_size, 8)
            train_name = "yolo_train"
        else:
            train_workers = 0
            train_batch = batch_size
            train_name = "rtdetr_train"
```

Add `task="obb"` to the training call for OBB. Find the `model.train(...)` call and add a conditional:

```python
        train_kwargs = dict(
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
        if detector_type == "obb":
            train_kwargs["task"] = "obb"

        model.train(**train_kwargs)
```

Update the best weights copy (lines 331-336) to use the right destination:

```python
        weights_dest = "obb_detector.pt" if training_type == "obb" else "detector.pt"
        best_pt = model_save_dir / train_name / "weights" / "best.pt"
        if best_pt.exists():
            shutil.copy2(best_pt, model_save_dir / weights_dest)
            logger.info(f"Best weights saved to {model_save_dir / weights_dest}")
        else:
            logger.warning("best.pt not found after training")
```

Pass `training_type` to `_apply_to_training_data`:

```python
            self._apply_to_training_data(project_path, video_ids, training_type)
```

- [ ] **Step 5: Update `_apply_to_training_data()` for OBB**

Add `training_type` parameter:

```python
    def _apply_to_training_data(
        self,
        project_path: Path,
        video_ids: list[int],
        training_type: str = "detector",
    ) -> None:
```

Update the imports and weights path:

```python
        from vidseq.services.detector_model import detect, detect_obb, load_finetuned
        from vidseq.services.frame_data_service import _chunked_upsert_sync

        weights_name = "obb_detector.pt" if training_type == "obb" else "detector.pt"
        weights_path = project_path / "models" / weights_name
        detector = load_finetuned(weights_path)
```

Update the detection and save logic to branch on training_type. In the per-frame detection loop, replace the detection block:

```python
            if training_type == "obb":
                detections = detect_obb(detector, frame)
                if detections:
                    best = detections[0]
                    corners = best["corners"]  # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
                    conf = best["conf"]
                    bboxes_to_save[video_id].append(
                        (frame_idx, *corners[0], *corners[1], *corners[2], *corners[3])
                    )
                    scores_to_save[video_id].append((frame_idx, conf))
                else:
                    scores_to_save[video_id].append((frame_idx, 0.0))
            else:
                detections = detect(detector, frame)
                if detections:
                    best = detections[0]
                    x1, y1, x2, y2 = best["bbox"]
                    conf = best["conf"]
                    bboxes_to_save[video_id].append(
                        (frame_idx, x1, y1, x2, y2)
                    )
                    scores_to_save[video_id].append((frame_idx, conf))
                else:
                    scores_to_save[video_id].append((frame_idx, 0.0))
```

Update the DB upsert to use OBB columns when appropriate:

```python
        with Session(engine) as session:
            for video_id, bbox_list in bboxes_to_save.items():
                if not bbox_list:
                    continue
                if training_type == "obb":
                    rows = [
                        {
                            "video_id": video_id,
                            "frame_idx": int(fi),
                            "obb_x1": float(vals[0]), "obb_y1": float(vals[1]),
                            "obb_x2": float(vals[2]), "obb_y2": float(vals[3]),
                            "obb_x3": float(vals[4]), "obb_y3": float(vals[5]),
                            "obb_x4": float(vals[6]), "obb_y4": float(vals[7]),
                        }
                        for fi, *vals in bbox_list
                    ]
                    _chunked_upsert_sync(session, rows, ["video_id", "frame_idx"],
                        ["obb_x1", "obb_y1", "obb_x2", "obb_y2",
                         "obb_x3", "obb_y3", "obb_x4", "obb_y4"])
                else:
                    rows = [
                        {
                            "video_id": video_id,
                            "frame_idx": int(fi),
                            "detector_bbox_x1": float(x1),
                            "detector_bbox_y1": float(y1),
                            "detector_bbox_x2": float(x2),
                            "detector_bbox_y2": float(y2),
                        }
                        for fi, x1, y1, x2, y2 in bbox_list
                    ]
                    _chunked_upsert_sync(session, rows, ["video_id", "frame_idx"],
                        ["detector_bbox_x1", "detector_bbox_y1",
                         "detector_bbox_x2", "detector_bbox_y2"])

            score_col = "obb_score" if training_type == "obb" else "detector_score"
            for video_id, score_list in scores_to_save.items():
                if not score_list:
                    continue
                rows = [
                    {
                        "video_id": video_id,
                        "frame_idx": int(fi),
                        score_col: float(score),
                    }
                    for fi, score in score_list
                ]
                _chunked_upsert_sync(session, rows, ["video_id", "frame_idx"], [score_col])

            session.commit()
```

- [ ] **Step 6: Commit**

```bash
git add vidseq/services/detector_service.py
git commit -m "feat: add OBB training flow with oriented label export and post-training apply"
```

---

### Task 5: Generalize Apply Detector for OBB

**Files:**
- Modify: `vidseq/services/segmentation_commands.py`
- Modify: `vidseq/services/segmentation_tcp_client.py`
- Modify: `vidseq/services/segmentation_service.py`

- [ ] **Step 1: Generalize `handle_apply_detector` for OBB**

In `segmentation_commands.py`, update `handle_apply_detector` to accept `detector_type` from params and branch on it. Add after `video_paths = params["video_paths"]`:

```python
    detector_type = params.get("detector_type", "detector")
```

Update model path:

```python
    weights_name = "obb_detector.pt" if detector_type == "obb" else "detector.pt"
    model_path = project_path / "models" / weights_name
    if not model_path.exists():
        raise RuntimeError(f"No trained {'OBB ' if detector_type == 'obb' else ''}detector model found. Train first.")
```

In the inner result extraction loop (where `result.boxes` is accessed), branch on detector_type:

```python
                for idx, result in zip(batch_indices, results):
                    if detector_type == "obb":
                        if result.obb is not None and len(result.obb) > 0:
                            best_i = result.obb.conf.argmax()
                            corners = result.obb.xyxyxyxy[best_i].cpu().tolist()
                            conf = result.obb.conf[best_i].item()
                            scores.append([idx, conf])
                            # Flatten corners: [[x1,y1],[x2,y2],...] -> [idx, x1, y1, x2, y2, ...]
                            bboxes.append([idx, *corners[0], *corners[1], *corners[2], *corners[3]])
                        else:
                            scores.append([idx, 0.0])
                    else:
                        if result.boxes is not None and len(result.boxes) > 0:
                            best_i = result.boxes.conf.argmax()
                            x1, y1, x2, y2 = result.boxes.xyxy[best_i].cpu().tolist()
                            conf = result.boxes.conf[best_i].item()
                            scores.append([idx, conf])
                            bboxes.append([idx, x1, y1, x2, y2])
                        else:
                            scores.append([idx, 0.0])
```

Update return keys to distinguish OBB:

```python
        score_key = "obb_scores" if detector_type == "obb" else "detector_scores"
        bbox_key = "obb_bboxes" if detector_type == "obb" else "detector_bboxes"

        return {
            "type": "apply_detector_result",
            "status": "ok",
            score_key: all_scores,
            bbox_key: all_bboxes,
        }
```

- [ ] **Step 2: Update TCP client to pass `detector_type`**

In `segmentation_tcp_client.py`, update the `apply_detector` method on `SegmentationService` to accept `detector_type`:

```python
    def apply_detector(
        self,
        project_path: Path,
        videos: list,
        detector_type: str = "detector",
    ) -> tuple[dict[int, list[list]], dict[int, list[list]]]:
```

Add `detector_type` to the command dict:

```python
        result = self._send_streaming({
            "type": "apply_detector",
            "video_ids": [v.id for v in videos],
            "video_paths": [v.path for v in videos],
            "project_path": str(project_path),
            "detector_type": detector_type,
        }, timeout=3600.0)
```

Update result key extraction:

```python
        score_key = "obb_scores" if detector_type == "obb" else "detector_scores"
        bbox_key = "obb_bboxes" if detector_type == "obb" else "detector_bboxes"
        raw_scores = result.get(score_key, {})
        raw_bboxes = result.get(bbox_key, {})
```

Add module-level wrapper for OBB at the end of the file (near the other wrappers):

```python
def apply_obb_detector(
    project_path: Path,
    videos: list,
) -> tuple[dict[int, list[list]], dict[int, list[list]]]:
    """Run OBB detector on all frames of given videos."""
    return SegmentationService.get_instance().apply_detector(project_path, videos, detector_type="obb")
```

- [ ] **Step 3: Add `apply_obb_detector` service function**

In `segmentation_service.py`, add after `apply_detector`:

```python
async def apply_obb_detector(
    session: "AsyncSession",
    project_id: int,
    project_path: Path,
    video_ids: list[int],
) -> int:
    """Run OBB detector on all frames of selected videos.

    Returns the number of videos processed.
    """
    from vidseq.models.video import Video

    result = await session.execute(
        select(Video).where(Video.id.in_(video_ids))
    )
    videos = list(result.scalars().all())
    if not videos:
        raise RuntimeError("No videos found")

    model_path = project_path / "models" / "obb_detector.pt"
    if not model_path.exists():
        raise RuntimeError("No trained OBB detector model found. Train first.")

    scores_by_video, bboxes_by_video = segmentation_tcp_client.apply_obb_detector(
        project_path=project_path,
        videos=videos,
    )

    for video in videos:
        vid_scores = scores_by_video.get(video.id, [])
        vid_bboxes = bboxes_by_video.get(video.id, [])
        if vid_scores:
            await frame_data_service.save_obb_scores_batch(
                session, video.id, vid_scores
            )
        if vid_bboxes:
            await frame_data_service.save_obb_bboxes_batch(
                session, video.id, vid_bboxes
            )

    await session.commit()
    return len(videos)
```

- [ ] **Step 4: Commit**

```bash
git add vidseq/services/segmentation_commands.py vidseq/services/segmentation_tcp_client.py vidseq/services/segmentation_service.py
git commit -m "feat: generalize apply_detector for OBB with oriented bbox extraction"
```

---

### Task 6: API Endpoints

**Files:**
- Modify: `vidseq/api/routes/detector.py`
- Modify: `vidseq/api/routes/segmentation/frames.py`

- [ ] **Step 1: Add OBB status endpoint**

In `detector.py`, add after the existing status endpoint (after line 88):

```python
class ObbStatusResponse(BaseModel):
    """Response for OBB detector status."""
    model_exists: bool
    is_training: bool


@router.get(
    "/projects/{project_id}/detection/obb/status",
    response_model=ObbStatusResponse,
)
async def get_obb_detection_status(
    project_path: Path = Depends(get_project_folder),
):
    """Get OBB detector model status."""
    service = DetectorService.get_instance()
    return ObbStatusResponse(
        model_exists=service.obb_model_exists(project_path),
        is_training=service.is_training() and service._training_type == "obb",
    )
```

- [ ] **Step 2: Add OBB training endpoints**

Add after the OBB status endpoint:

```python
@router.post("/projects/{project_id}/detection/obb/training")
async def create_obb_training(
    request: TrainRequest,
    project_path: Path = Depends(get_project_folder),
):
    """Start OBB detector training."""
    service = DetectorService.get_instance()
    if service.is_training():
        raise HTTPException(status_code=409, detail="Training already in progress")

    try:
        service.train(
            project_path=project_path,
            video_ids=request.video_ids,
            max_epochs=request.max_epochs,
            batch_size=request.batch_size,
            lr=request.lr,
            early_stop_patience=request.early_stop_patience,
            training_type="obb",
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"status": "started"}


@router.delete("/projects/{project_id}/detection/obb/training", status_code=204)
async def delete_obb_training():
    """Stop OBB detector training."""
    service = DetectorService.get_instance()
    if not service.is_training():
        raise HTTPException(status_code=400, detail="No training in progress")
    service.stop_training()
    return None


@router.get("/projects/{project_id}/detection/obb/training")
async def get_obb_training_progress():
    """Get OBB training progress."""
    service = DetectorService.get_instance()
    return service.get_training_progress().to_dict()


@router.get("/projects/{project_id}/detection/obb/training/stream")
async def stream_obb_training():
    """SSE stream for OBB training updates."""
    async def event_generator():
        service = DetectorService.get_instance()
        last_progress_str = None
        while True:
            progress = service.get_training_progress()
            progress_dict = progress.to_dict()
            progress_str = json.dumps(progress_dict)
            if progress_str != last_progress_str:
                yield f"data: {progress_str}\n\n"
                last_progress_str = progress_str
            if progress.status in ("completed", "failed", "stopped", "idle"):
                if not progress.is_training:
                    break
            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )
```

- [ ] **Step 3: Add OBB apply and bbox endpoints**

Add at the end of `detector.py`:

```python
@router.post("/projects/{project_id}/videos/obb-detection")
async def apply_obb_detector(
    project_id: int,
    request: VideoSelectionRequest,
    project_path: Path = Depends(get_project_folder),
    session: AsyncSession = Depends(get_project_session),
):
    """Run OBB detector on every frame of selected videos."""
    try:
        videos_processed = await segmentation_service.apply_obb_detector(
            session=session,
            project_id=project_id,
            project_path=project_path,
            video_ids=request.video_ids,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"videos_processed": videos_processed}


@router.get("/projects/{project_id}/videos/{video_id}/obb-bboxes/{frame_idx}")
async def get_obb_bbox_endpoint(
    frame_idx: int,
    video: Video = Depends(get_video),
    session: AsyncSession = Depends(get_project_session),
):
    """Get OBB bbox for a single frame."""
    bbox = await frame_data_service.get_obb_bbox(session, video.id, frame_idx)
    return {"bbox": bbox}


@router.get("/projects/{project_id}/videos/{video_id}/obb-bboxes")
async def get_obb_bboxes_endpoint(
    start_frame: int,
    count: int = 100,
    video: Video = Depends(get_video),
    session: AsyncSession = Depends(get_project_session),
):
    """Get OBB bboxes for a range of frames."""
    bboxes = await frame_data_service.get_obb_bboxes_batch(
        session, video.id, start_frame, count
    )
    return {"bboxes": bboxes}


@router.get("/projects/{project_id}/videos/{video_id}/obb-bboxes/exists")
async def obb_bboxes_exist_endpoint(
    video: Video = Depends(get_video),
    session: AsyncSession = Depends(get_project_session),
):
    """Check if OBB bbox data exists for a video."""
    exists = await frame_data_service.obb_bboxes_exist(session, video.id)
    return {"exists": exists}
```

- [ ] **Step 4: Add OBB scores downsampled endpoint**

In `vidseq/api/routes/segmentation/frames.py`, add after the existing `get_detector_scores_downsampled` endpoint (after line 171):

```python
@router.get(
    "/projects/{project_id}/videos/{video_id}/segmentation/obb-scores-downsampled",
)
async def get_obb_scores_downsampled(
    max_samples: int = 800,
    start_frame: int = 0,
    end_frame: int | None = None,
    video: Video = Depends(get_video),
    session: AsyncSession = Depends(get_project_session),
):
    """Get LTTB-downsampled OBB confidence scores for visualization."""
    return await frame_data_service.get_obb_scores_downsampled(
        session, video.id, video.num_frames, max_samples, start_frame, end_frame
    )
```

- [ ] **Step 5: Commit**

```bash
git add vidseq/api/routes/detector.py vidseq/api/routes/segmentation/frames.py
git commit -m "feat: add OBB detector API endpoints for status, training, apply, bboxes, and scores"
```

---

### Task 7: Frontend API Functions + Composable

**Files:**
- Modify: `frontend/src/services/api.ts`
- Create: `frontend/src/composables/useObbDetector.ts`

- [ ] **Step 1: Add OBB types and API functions to api.ts**

Add near the other detector types/functions:

```typescript
export interface ObbBbox {
    corners: [number, number][]
}

export interface ObbDetectorStatus {
    model_exists: boolean
    is_training: boolean
}

export async function getObbDetectionStatus(projectId: number): Promise<ObbDetectorStatus> {
    const response = await fetch(`${API_BASE}/projects/${projectId}/detection/obb/status`)
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to get OBB status'))
    }
    return response.json()
}

export async function createObbTraining(
    projectId: number,
    maxEpochs: number,
    videoIds: number[],
    lrPatience: number = 10,
    earlyStopPatience: number = 20,
): Promise<void> {
    const response = await fetch(`${API_BASE}/projects/${projectId}/detection/obb/training`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            video_ids: videoIds,
            max_epochs: maxEpochs,
            lr_patience: lrPatience,
            early_stop_patience: earlyStopPatience,
        }),
    })
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to start OBB training'))
    }
}

export async function deleteObbTraining(projectId: number): Promise<void> {
    const response = await fetch(`${API_BASE}/projects/${projectId}/detection/obb/training`, {
        method: 'DELETE',
    })
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to stop OBB training'))
    }
}

export async function applyObbDetector(projectId: number, videoIds: number[]): Promise<{ videos_processed: number }> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/obb-detection`,
        {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ video_ids: videoIds }),
        }
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to apply OBB detector'))
    }
    return response.json()
}

export async function getObbBbox(projectId: number, videoId: number, frameIdx: number): Promise<{ bbox: ObbBbox | null }> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/${videoId}/obb-bboxes/${frameIdx}`
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to get OBB bbox'))
    }
    return response.json()
}

export async function obbBboxesExist(projectId: number, videoId: number): Promise<{ exists: boolean }> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/${videoId}/obb-bboxes/exists`
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to check OBB bboxes'))
    }
    return response.json()
}

export async function getObbScoresDownsampled(
    projectId: number,
    videoId: number,
    maxSamples: number = 800,
    startFrame: number = 0,
    endFrame?: number,
): Promise<{ scores: { frame_idx: number; score: number }[]; total_count: number }> {
    let url = `${API_BASE}/projects/${projectId}/videos/${videoId}/segmentation/obb-scores-downsampled?max_samples=${maxSamples}&start_frame=${startFrame}`
    if (endFrame !== undefined) url += `&end_frame=${endFrame}`
    const response = await fetch(url)
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to get OBB scores'))
    }
    return response.json()
}
```

- [ ] **Step 2: Create `useObbDetector.ts` composable**

Create `frontend/src/composables/useObbDetector.ts`:

```typescript
import { ref, watch, onUnmounted, type Ref } from 'vue'
import {
    getObbDetectionStatus,
    createObbTraining,
    deleteObbTraining,
} from '@/services/api'

export function useObbDetector(projectId: Ref<number | null>) {
    const isTraining = ref(false)
    const modelExists = ref(false)

    const checkStatus = async () => {
        if (!projectId.value) return
        try {
            const status = await getObbDetectionStatus(projectId.value)
            modelExists.value = status.model_exists
            isTraining.value = status.is_training
        } catch (e) {
            console.error('Failed to check OBB detector status:', e)
        }
    }

    const startTraining = async (maxEpochs: number = 1000, videoIds: number[] = []) => {
        if (!projectId.value || isTraining.value) return
        isTraining.value = true
        try {
            await createObbTraining(projectId.value, maxEpochs, videoIds)
        } catch (e) {
            console.error('Failed to start OBB training:', e)
            isTraining.value = false
            throw e
        }
    }

    const stopTraining = async () => {
        if (!projectId.value) return
        try {
            await deleteObbTraining(projectId.value)
        } catch (e) {
            console.error('Failed to stop OBB training:', e)
            throw e
        }
    }

    watch(projectId, async (newId) => {
        if (newId !== null) {
            await checkStatus()
        }
    }, { immediate: true })

    let pollInterval: number | null = null

    watch(isTraining, (training) => {
        if (training) {
            if (pollInterval === null && projectId.value !== null) {
                pollInterval = window.setInterval(async () => {
                    await checkStatus()
                    if (!isTraining.value && pollInterval !== null) {
                        clearInterval(pollInterval)
                        pollInterval = null
                    }
                }, 2000)
            }
        } else {
            if (pollInterval !== null) {
                clearInterval(pollInterval)
                pollInterval = null
            }
        }
    })

    onUnmounted(() => {
        if (pollInterval !== null) {
            clearInterval(pollInterval)
            pollInterval = null
        }
    })

    return {
        isTraining,
        modelExists,
        startTraining,
        stopTraining,
        checkStatus,
    }
}
```

- [ ] **Step 3: Commit**

```bash
git add frontend/src/services/api.ts frontend/src/composables/useObbDetector.ts
git commit -m "feat: add OBB detector API functions and useObbDetector composable"
```

---

### Task 8: Frontend VideoPipeline Sidebar

**Files:**
- Modify: `frontend/src/components/VideoPipeline.vue`

- [ ] **Step 1: Add imports and composable**

Add `applyObbDetector` to the imports from `@/services/api`. Add import for the composable:

```typescript
import { useObbDetector } from '@/composables/useObbDetector'
```

Initialize the composable near the `useDetector` call (around line 124):

```typescript
const {
  isTraining: isObbTraining,
  modelExists: obbModelExists,
  startTraining: startObbTraining,
  checkStatus: checkObbStatus,
} = useObbDetector(projectId)
```

- [ ] **Step 2: Add loading state and handler**

Add ref:

```typescript
const isApplyingObb = ref(false)
```

Add handlers:

```typescript
const handleTrainObb = async () => {
  if (!projectId.value || isObbTraining.value) return
  try {
    await startObbTraining(1000, selectedVideoIdsList.value)
    router.push(`/project/${projectId.value}/detector`)
  } catch (e: any) {
    alert(e.message || 'Failed to start OBB training')
  }
}

const handleApplyObb = async () => {
  if (!projectId.value || isApplyingObb.value) return
  isApplyingObb.value = true
  try {
    await applyObbDetector(projectId.value, selectedVideoIdsList.value)
    await loadVideos()
  } catch (e: any) {
    console.error('Failed to apply OBB detector:', e)
    alert(e.message || 'Failed to apply OBB detector')
  } finally {
    isApplyingObb.value = false
  }
}
```

- [ ] **Step 3: Add OBB sidebar section to template**

Add after the "Apply Detector" button and before the "Associated Videos" section header:

```vue
          <h4 class="sidebar-section-title">OBB Detector</h4>
          <button
            class="sidebar-button"
            @click="handleTrainObb"
            :disabled="isObbTraining || isDetectorTraining || selectedCount === 0"
          >
            <span class="button-label">{{ isObbTraining ? 'Training...' : 'Train OBB Detector' }}</span>
          </button>
          <button
            v-if="obbModelExists"
            class="sidebar-button"
            @click="handleApplyObb"
            :disabled="isApplyingObb || isObbTraining || selectedCount === 0"
          >
            <span class="button-label">{{ isApplyingObb ? 'Applying...' : 'Apply OBB Detector' }}</span>
          </button>
```

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/VideoPipeline.vue
git commit -m "feat: add OBB Detector section to VideoPipeline sidebar"
```

---

### Task 9: Frontend VideoDetail (OBB View Mode)

**Files:**
- Modify: `frontend/src/components/VideoDetail.vue`

- [ ] **Step 1: Add OBB imports and state**

Add `obbBboxesExist`, `getObbBbox`, `getObbScoresDownsampled`, and `type ObbBbox` to the imports from `@/services/api`.

Update the MaskViewMode type (line 37):

```typescript
type MaskViewMode = 'tracker' | 'detector' | 'final' | 'obb'
```

Add state refs near the other detector state:

```typescript
const hasObbBboxes = ref(false)
const obbBbox = ref<ObbBbox | null>(null)
const obbScores = ref<{ frame_idx: number; score: number }[]>([])
const showObbConfidence = ref(true)
```

- [ ] **Step 2: Add OBB availability check**

Add after `checkFinalMasks`:

```typescript
const checkObbBboxes = async () => {
  if (!projectId.value || !videoId.value) return
  try {
    const result = await obbBboxesExist(projectId.value, videoId.value)
    hasObbBboxes.value = result.exists
  } catch {
    hasObbBboxes.value = false
  }
}
```

Call it in `onMounted` alongside the other checks:

```typescript
  await checkObbBboxes()
```

- [ ] **Step 3: Add OBB score fetching**

Add near the detector scores fetch:

```typescript
const fetchObbScoresForView = async () => {
  if (!projectId.value || !videoId.value || !video.value) return
  try {
    const startFrame = Math.floor(viewStart.value * video.value.fps)
    const endFrame = Math.ceil(viewEnd.value * video.value.fps)
    const response = await getObbScoresDownsampled(
      projectId.value,
      videoId.value,
      800,
      startFrame,
      endFrame
    )
    obbScores.value = response.scores
  } catch (e) {
    console.error('Failed to fetch OBB scores:', e)
  }
}

const fetchObbScores = useDebounceFn(fetchObbScoresForView, 150)
```

Call `fetchObbScoresForView()` in onMounted and `fetchObbScores()` in the view range watcher.

- [ ] **Step 4: Add OBB bbox fetching on frame change**

In the frame change handler (where detector bbox is fetched when in detector mode), add OBB handling:

```typescript
if (maskViewMode.value === 'obb') {
  try {
    const result = await getObbBbox(projectId.value, videoId.value, frameIdx)
    obbBbox.value = result.bbox
  } catch {
    obbBbox.value = null
  }
}
```

In the `maskViewMode` watcher, clear OBB state:

```typescript
obbBbox.value = null
```

- [ ] **Step 5: Add OBB option to dropdown**

In the mask view select (around line 430):

```vue
    <option value="obb" :disabled="!hasObbBboxes">
      OBB {{ hasObbBboxes ? '' : '(not available)' }}
    </option>
```

- [ ] **Step 6: Pass OBB data to child components**

Pass `obbBbox` to VideoOverlay:

```vue
:obb-bbox="obbBbox"
```

Pass OBB scores to DataTrack:

```vue
:show-obb-confidence="showObbConfidence"
:obb-scores="obbScores"
```

- [ ] **Step 7: Commit**

```bash
git add frontend/src/components/VideoDetail.vue
git commit -m "feat: add OBB view mode to VideoDetail with bbox fetching and score display"
```

---

### Task 10: Frontend VideoOverlay (Polygon Rendering)

**Files:**
- Modify: `frontend/src/components/VideoOverlay.vue`

- [ ] **Step 1: Add `obbBbox` prop**

Update the props interface to add OBB:

```typescript
  obbBbox?: { corners: [number, number][] } | null
```

- [ ] **Step 2: Add OBB polygon rendering**

In the draw function, after the detector bbox rendering block (after line 82), add:

```typescript
// Draw OBB if provided
if (props.obbBbox && props.obbBbox.corners.length === 4) {
  ctx.strokeStyle = 'rgba(0, 188, 212, 0.8)'  // cyan
  ctx.lineWidth = 3
  const scaleX = canvas.width / props.videoWidth
  const scaleY = canvas.height / props.videoHeight
  const corners = props.obbBbox.corners

  ctx.beginPath()
  ctx.moveTo(corners[0][0] * scaleX, corners[0][1] * scaleY)
  ctx.lineTo(corners[1][0] * scaleX, corners[1][1] * scaleY)
  ctx.lineTo(corners[2][0] * scaleX, corners[2][1] * scaleY)
  ctx.lineTo(corners[3][0] * scaleX, corners[3][1] * scaleY)
  ctx.closePath()
  ctx.stroke()
}
```

- [ ] **Step 3: Add `obbBbox` to the watch list**

Update the watch (line 147) to include `props.obbBbox`:

```typescript
watch(() => [props.mask, props.prompts, props.detectorBbox, props.obbBbox, props.showMask, props.showPrompts], () => {
```

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/VideoOverlay.vue
git commit -m "feat: add OBB polygon rendering to VideoOverlay"
```

---

### Task 11: Frontend DataTrack (OBB Scores)

**Files:**
- Modify: `frontend/src/components/DataTrack.vue`

- [ ] **Step 1: Add OBB score props**

Add to the props definition (near lines 17-18):

```typescript
showObbConfidence?: boolean
obbScores?: { frame_idx: number; score: number }[]
```

- [ ] **Step 2: Add OBB score drawing**

After the detector scores drawing block (after line 426), add:

```typescript
// Draw OBB confidence scores (cyan line)
let obbRange: { min: number; max: number } | null = null
if (props.showObbConfidence && props.obbScores && props.obbScores.length > 0) {
  const validScores = props.obbScores.filter(s => s.score >= 0)
  if (validScores.length > 0) {
    obbRange = drawScoreLine(ctx, validScores, 'rgba(0, 188, 212, 0.8)', width, height)
    if (obbRange && !activeRange) activeRange = obbRange
  }
}
```

- [ ] **Step 3: Add OBB to hover tooltip**

After the detector score tooltip block (after line 254), add:

```typescript
// Find nearest OBB confidence score
if (props.showObbConfidence && props.obbScores && props.obbScores.length > 0) {
  const nearest = findNearestScore(props.obbScores, frame)
  if (nearest !== null) {
    results.push({ label: 'OBB', value: nearest.score, color: 'rgba(0, 188, 212, 0.8)' })
  }
}
```

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/DataTrack.vue
git commit -m "feat: add OBB confidence score track to DataTrack"
```
