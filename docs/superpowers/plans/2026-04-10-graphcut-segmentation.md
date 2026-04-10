# Graph Cut Segmentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add spatio-temporal graph cuts as a parallel method for producing training masks, using PyMaxflow on 150-frame chunks with brush-painted foreground/background seeds.

**Architecture:** New backend service (`graphcut_service.py`) builds a 3D PyMaxflow grid graph (H x W x T, 6-connected) from video frames, sets edge weights from pixel intensity similarity, and solves maxflow using user-painted seeds as terminal constraints. Frontend adds brush painting on VideoOverlay, chunk placement on DataTrack, and a new composable for state management. Results write to existing `tracker_masks.h5`.

**Tech Stack:** PyMaxflow (Python/C++ maxflow solver), FastAPI, Vue 3 Composition API, TypeScript, OpenCV (frame reading), h5py (mask storage).

---

### Task 1: Add PyMaxflow dependency

**Files:**
- Modify: `pyproject.toml:7-34`

- [ ] **Step 1: Add PyMaxflow to dependencies**

Add `"PyMaxflow>=1.3.0"` to the dependencies list in `pyproject.toml`:

```toml
dependencies = [
    "aiosqlite>=0.21.0",
    "fastapi>=0.121.1",
    ...
    "pycocotools",
    "hydra-core>=1.3.2",
    "transformers>=5.0.0",
    "pandas>=3.0.2",
    "PyMaxflow>=1.3.0",
]
```

- [ ] **Step 2: Install**

Run: `cd /n/groups/datta/john/projects/vidseq && uv sync`
Expected: PyMaxflow installs successfully.

- [ ] **Step 3: Verify import**

Run: `uv run python -c "import maxflow; print(maxflow.__version__)"`
Expected: Prints version (e.g., `1.3.2`).

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "deps: add PyMaxflow for graph cut segmentation"
```

---

### Task 2: Extract VideoFrameSource to shared module

**Files:**
- Create: `vidseq/services/video_io.py`
- Modify: `vidseq/services/segmentation_commands.py:31-83`

VideoFrameSource currently lives in `segmentation_commands.py` which imports SAM2/PyTorch at module level (line 23). The FastAPI side cannot import from that module without pulling in GPU dependencies. Extract to a shared module.

- [ ] **Step 1: Create `video_io.py` with VideoFrameSource**

Create `vidseq/services/video_io.py` with the class copied verbatim from `segmentation_commands.py` lines 31-83:

```python
"""Shared video I/O utilities.

Extracted from segmentation_commands.py so that both the GPU worker
and the FastAPI process can use VideoFrameSource without cross-importing
GPU-dependent modules.
"""

from pathlib import Path

import cv2
import numpy as np


class VideoFrameSource:
    """Wrapper around cv2.VideoCapture with position tracking.

    Implements __getitem__ for random access to video frames, but tracks
    current position to avoid unnecessary seeks during sequential reads.
    """

    def __init__(self, path: str | Path):
        """Open a video file."""
        self._path = str(path)
        self._cap = cv2.VideoCapture(self._path)
        if not self._cap.isOpened():
            raise ValueError(f"Failed to open video: {path}")

        self._pos = 0
        self.frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def __getitem__(self, idx: int) -> np.ndarray:
        """Read a frame by index. Returns BGR numpy array."""
        if idx < 0 or idx >= self.frame_count:
            raise IndexError(f"Frame index {idx} out of range [0, {self.frame_count})")

        if idx != self._pos:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            self._pos = idx

        success, frame = self._cap.read()
        if not success:
            raise IndexError(f"Failed to read frame {idx}")

        self._pos += 1
        return frame

    def __len__(self) -> int:
        return self.frame_count

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
```

- [ ] **Step 2: Update segmentation_commands.py to import from video_io**

In `vidseq/services/segmentation_commands.py`, replace lines 26-83 (the comment block + full VideoFrameSource class) with an import:

```python
from vidseq.services.video_io import VideoFrameSource
```

Keep the rest of the file unchanged.

- [ ] **Step 3: Verify nothing is broken**

Run: `uv run python -c "from vidseq.services.segmentation_commands import VideoFrameSource; print('OK')"`
Expected: `OK` (import still works via re-export).

- [ ] **Step 4: Commit**

```bash
git add vidseq/services/video_io.py vidseq/services/segmentation_commands.py
git commit -m "refactor: extract VideoFrameSource to shared video_io module"
```

---

### Task 3: Backend — Graph cut schema and service

**Files:**
- Create: `vidseq/schemas/graphcut.py`
- Create: `vidseq/services/graphcut_service.py`

- [ ] **Step 1: Create Pydantic schemas**

Create `vidseq/schemas/graphcut.py`:

```python
"""Pydantic schemas for graph cut segmentation API."""

from pydantic import BaseModel, field_validator


class SeedPoint(BaseModel):
    x: int
    y: int
    label: int  # 1 = foreground, 2 = background


class GraphCutRequest(BaseModel):
    start_frame: int
    end_frame: int
    seeds: dict[str, list[SeedPoint]]  # frame_idx (as string) -> list of seed points

    @field_validator("seeds")
    @classmethod
    def validate_seeds(cls, v: dict[str, list[SeedPoint]]) -> dict[str, list[SeedPoint]]:
        all_labels = set()
        total_points = 0
        for points in v.values():
            for p in points:
                if p.label not in (1, 2):
                    raise ValueError("Seed label must be 1 (foreground) or 2 (background)")
                all_labels.add(p.label)
                total_points += 1
        if total_points == 0:
            raise ValueError("At least one seed point is required")
        if len(all_labels) < 2:
            raise ValueError("Both foreground (1) and background (2) seeds are required")
        return v


class GraphCutResponse(BaseModel):
    frames_processed: int
```

- [ ] **Step 2: Create the graph cut service**

Create `vidseq/services/graphcut_service.py`:

```python
"""Graph cut segmentation service.

Builds a 3D spatio-temporal graph over a chunk of video frames and solves
maxflow to produce binary masks. Runs on CPU — no GPU or TCP worker needed.
"""

import logging
from pathlib import Path

import cv2
import maxflow
import numpy as np

from vidseq.services.array_storage import tracker_masks
from vidseq.services.video_io import VideoFrameSource

logger = logging.getLogger(__name__)

GRAPHCUT_CHUNK_SIZE = 150


def run_graphcut(
    project_path: Path,
    video_id: int,
    video_path: str,
    start_frame: int,
    end_frame: int,
    seeds: dict[str, list[dict]],
) -> int:
    """Run spatio-temporal graph cut segmentation on a video chunk.

    Args:
        project_path: Path to the project directory.
        video_id: Video ID for H5 storage.
        video_path: Path to the video file.
        start_frame: First frame index (inclusive).
        end_frame: Last frame index (inclusive).
        seeds: Sparse seed dict {frame_idx_str: [{x, y, label}, ...]}.

    Returns:
        Number of frames processed.
    """
    num_frames = end_frame - start_frame + 1

    # 1. Read frames and convert to grayscale
    frames = _read_frames(video_path, start_frame, end_frame)
    T, H, W = frames.shape

    # 2. Build seed volume
    seed_vol = _build_seed_volume(seeds, start_frame, T, H, W)

    # 3. Compute beta from neighbor intensity differences
    beta = _compute_beta(frames)

    # 4. Build graph, solve, extract masks
    masks = _solve_graphcut(frames, seed_vol, beta)

    # 5. Write masks to H5
    with tracker_masks(project_path, video_id, mode="a") as h5:
        for t in range(T):
            h5[start_frame + t] = masks[t]

    logger.info("Graph cut: wrote %d masks for video %d (frames %d-%d)", T, video_id, start_frame, end_frame)
    return T


def _read_frames(video_path: str, start_frame: int, end_frame: int) -> np.ndarray:
    """Read frames and convert to single-channel grayscale float32.

    Returns:
        Array of shape (T, H, W) with values in [0, 255].
    """
    with VideoFrameSource(video_path) as src:
        frame_list = []
        for idx in range(start_frame, end_frame + 1):
            bgr = src[idx]
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
            frame_list.append(gray)
    return np.stack(frame_list)  # (T, H, W)


def _build_seed_volume(
    seeds: dict[str, list[dict]],
    start_frame: int,
    T: int,
    H: int,
    W: int,
) -> np.ndarray:
    """Build a 3D seed volume from sparse seed dict.

    Returns:
        Array of shape (T, H, W) with 0=no seed, 1=foreground, 2=background.
    """
    vol = np.zeros((T, H, W), dtype=np.uint8)
    for frame_str, points in seeds.items():
        frame_idx = int(frame_str)
        t = frame_idx - start_frame
        if t < 0 or t >= T:
            continue
        for p in points:
            x, y, label = p["x"], p["y"], p["label"]
            if 0 <= x < W and 0 <= y < H:
                vol[t, y, x] = label
    return vol


def _compute_beta(frames: np.ndarray) -> float:
    """Compute beta = 1 / (2 * mean(squared differences)) over all neighbor pairs.

    Samples spatial (horizontal + vertical) and temporal differences.
    """
    diffs_sq = []

    # Spatial horizontal: pixel vs right neighbor
    dh = (frames[:, :, :-1] - frames[:, :, 1:]) ** 2
    diffs_sq.append(dh.ravel())

    # Spatial vertical: pixel vs bottom neighbor
    dv = (frames[:, :-1, :] - frames[:, 1:, :]) ** 2
    diffs_sq.append(dv.ravel())

    # Temporal: same pixel in adjacent frames
    if frames.shape[0] > 1:
        dt = (frames[:-1] - frames[1:]) ** 2
        diffs_sq.append(dt.ravel())

    mean_sq = np.concatenate(diffs_sq).mean()
    if mean_sq < 1e-10:
        return 0.0  # Constant image — all edges get weight 1
    return 1.0 / (2.0 * mean_sq)


def _solve_graphcut(
    frames: np.ndarray,
    seed_vol: np.ndarray,
    beta: float,
) -> np.ndarray:
    """Build 3D graph, solve maxflow, return binary masks.

    Args:
        frames: (T, H, W) float32 grayscale.
        seed_vol: (T, H, W) uint8, 0=none, 1=fg, 2=bg.
        beta: Edge weight parameter.

    Returns:
        (T, H, W) uint8 binary masks (0 or 1).
    """
    T, H, W = frames.shape
    num_nodes = T * H * W

    g = maxflow.Graph[float](num_nodes, num_nodes * 6)
    node_ids = g.add_grid_nodes((T, H, W))

    # Spatial horizontal edges (axis=2: W dimension)
    w_horiz = np.exp(-beta * (frames[:, :, :-1] - frames[:, :, 1:]) ** 2)
    struct_horiz = np.zeros((3, 3, 3), dtype=int)
    struct_horiz[1, 1, 2] = 1  # center to right neighbor
    g.add_grid_edges(node_ids, weights=w_horiz, structure=struct_horiz, symmetric=True)

    # Spatial vertical edges (axis=1: H dimension)
    w_vert = np.exp(-beta * (frames[:, :-1, :] - frames[:, 1:, :]) ** 2)
    struct_vert = np.zeros((3, 3, 3), dtype=int)
    struct_vert[1, 2, 1] = 1  # center to bottom neighbor
    g.add_grid_edges(node_ids, weights=w_vert, structure=struct_vert, symmetric=True)

    # Temporal edges (axis=0: T dimension)
    if T > 1:
        w_temp = np.exp(-beta * (frames[:-1] - frames[1:]) ** 2)
        struct_temp = np.zeros((3, 3, 3), dtype=int)
        struct_temp[2, 1, 1] = 1  # center to next frame
        g.add_grid_edges(node_ids, weights=w_temp, structure=struct_temp, symmetric=True)

    # Terminal edges (seeds)
    # K = guaranteed-cut capacity: 1 + max_possible_edge_weight * max_degree
    # max edge weight is exp(0) = 1.0 (when pixels are identical), degree = 6
    K = 1.0 + 1.0 * 6.0

    source_cap = np.zeros((T, H, W), dtype=np.float64)
    sink_cap = np.zeros((T, H, W), dtype=np.float64)
    source_cap[seed_vol == 1] = K
    sink_cap[seed_vol == 2] = K

    g.add_grid_tedges(node_ids, source_cap, sink_cap)

    g.maxflow()

    segments = g.get_grid_segments(node_ids)  # True = source (foreground)
    return segments.astype(np.uint8)
```

Use this corrected version when creating the file.

- [ ] **Step 3: Verify the service imports cleanly**

Run: `uv run python -c "from vidseq.services.graphcut_service import run_graphcut; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add vidseq/schemas/graphcut.py vidseq/services/graphcut_service.py
git commit -m "feat: add graph cut service and schemas"
```

---

### Task 4: Backend — REST endpoint and router registration

**Files:**
- Create: `vidseq/api/routes/graphcut.py`
- Modify: `vidseq/server.py:22,49-58`

- [ ] **Step 1: Create the route**

Create `vidseq/api/routes/graphcut.py`:

```python
"""Graph cut segmentation endpoint."""

import asyncio
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from vidseq.api.dependencies import get_project_folder, get_project_session, get_video
from vidseq.models.video import Video
from vidseq.schemas.graphcut import GraphCutRequest, GraphCutResponse
from vidseq.services import graphcut_service

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/projects/{project_id}/videos/{video_id}/graphcut-masks",
    response_model=GraphCutResponse,
)
async def create_graphcut_masks(
    project_id: int,
    request: GraphCutRequest,
    video: Video = Depends(get_video),
    project_path: Path = Depends(get_project_folder),
):
    """Run graph cut segmentation on a video chunk."""
    # Clamp end_frame to video length
    end_frame = min(request.end_frame, video.num_frames - 1)
    if request.start_frame > end_frame:
        raise HTTPException(status_code=400, detail="start_frame is past end of video")

    # Convert seed points to plain dicts for the service
    seeds = {
        k: [{"x": p.x, "y": p.y, "label": p.label} for p in v]
        for k, v in request.seeds.items()
    }

    try:
        frames_processed = await asyncio.to_thread(
            graphcut_service.run_graphcut,
            project_path,
            video.id,
            video.filepath,
            request.start_frame,
            end_frame,
            seeds,
        )
    except MemoryError:
        raise HTTPException(
            status_code=500,
            detail="Out of memory building graph cut. Try a smaller chunk or lower-resolution video.",
        )

    return GraphCutResponse(frames_processed=frames_processed)
```

- [ ] **Step 2: Register the router in server.py**

In `vidseq/server.py`, add the import at line 22:

```python
from vidseq.api.routes import arhmm, cropped_videos, detector, exports, filesystem, graphcut, pca, pose, projects, segmentation, videos
```

And add the router registration after line 58:

```python
app.include_router(graphcut.router, prefix="/api", tags=["graphcut"])
```

- [ ] **Step 3: Verify the server starts**

Run: `uv run python -c "from vidseq.server import app; print('routes:', len(app.routes))"`
Expected: Prints route count (one more than before).

- [ ] **Step 4: Commit**

```bash
git add vidseq/api/routes/graphcut.py vidseq/server.py
git commit -m "feat: add graph cut REST endpoint"
```

---

### Task 5: Frontend — API function and composable

**Files:**
- Modify: `frontend/src/services/api.ts`
- Create: `frontend/src/composables/useGraphCut.ts`

- [ ] **Step 1: Add API function to api.ts**

Add to `frontend/src/services/api.ts` (near the bottom, alongside other POST functions):

```typescript
// Graph Cut Segmentation
export interface GraphCutSeed {
    x: number
    y: number
    label: number  // 1 = foreground, 2 = background
}

export async function createGraphcutMasks(
    projectId: number,
    videoId: number,
    startFrame: number,
    endFrame: number,
    seeds: Record<string, GraphCutSeed[]>,
): Promise<{ frames_processed: number }> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/${videoId}/graphcut-masks`,
        {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                start_frame: startFrame,
                end_frame: endFrame,
                seeds,
            }),
        }
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to run graph cut'))
    }
    return response.json()
}
```

- [ ] **Step 2: Create useGraphCut composable**

Create `frontend/src/composables/useGraphCut.ts`:

```typescript
import { ref, computed, type Ref } from 'vue'
import { createGraphcutMasks, type GraphCutSeed } from '@/services/api'

export const GRAPHCUT_CHUNK_SIZE = 150

export function useGraphCut(
    projectId: Ref<number>,
    videoId: Ref<number>,
) {
    const graphcutRegion = ref<{ start: number; end: number } | null>(null)
    const seeds = ref<Map<number, GraphCutSeed[]>>(new Map())
    const brushSize = ref(5)
    const seedsVisible = ref(true)
    const isRunning = ref(false)

    const isGraphCutMode = computed(() => graphcutRegion.value !== null)

    function placeRegion(frameIdx: number, numFrames: number) {
        const end = Math.min(frameIdx + GRAPHCUT_CHUNK_SIZE - 1, numFrames - 1)
        graphcutRegion.value = { start: frameIdx, end }
        // Clear seeds when region moves
        seeds.value = new Map()
    }

    function paintSeed(frameIdx: number, x: number, y: number, label: number) {
        if (!graphcutRegion.value) return
        if (frameIdx < graphcutRegion.value.start || frameIdx > graphcutRegion.value.end) return

        const existing = seeds.value.get(frameIdx) ?? []
        existing.push({ x, y, label })
        seeds.value.set(frameIdx, existing)
    }

    function getSeedsForFrame(frameIdx: number): GraphCutSeed[] {
        return seeds.value.get(frameIdx) ?? []
    }

    async function runGraphCut(clearMaskCache: (start: number, end: number) => void) {
        if (!graphcutRegion.value) return
        if (seeds.value.size === 0) return

        // Convert Map to Record<string, GraphCutSeed[]> for the API
        const seedsObj: Record<string, GraphCutSeed[]> = {}
        for (const [frameIdx, points] of seeds.value) {
            seedsObj[String(frameIdx)] = points
        }

        isRunning.value = true
        try {
            await createGraphcutMasks(
                projectId.value,
                videoId.value,
                graphcutRegion.value.start,
                graphcutRegion.value.end,
                seedsObj,
            )
            clearMaskCache(graphcutRegion.value.start, graphcutRegion.value.end)
        } finally {
            isRunning.value = false
        }
    }

    function clearSeeds() {
        seeds.value = new Map()
    }

    function exitGraphCutMode() {
        graphcutRegion.value = null
        seeds.value = new Map()
    }

    return {
        graphcutRegion,
        seeds,
        brushSize,
        seedsVisible,
        isRunning,
        isGraphCutMode,
        placeRegion,
        paintSeed,
        getSeedsForFrame,
        runGraphCut,
        clearSeeds,
        exitGraphCutMode,
    }
}
```

- [ ] **Step 3: Verify TypeScript compiles**

Run: `cd /n/groups/datta/john/projects/vidseq/frontend && npx vue-tsc --noEmit 2>&1 | tail -5`
Expected: No errors related to `useGraphCut` or `createGraphcutMasks`.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/services/api.ts frontend/src/composables/useGraphCut.ts
git commit -m "feat: add graph cut API function and composable"
```

---

### Task 6: Frontend — DataTrack graph cut region

**Files:**
- Modify: `frontend/src/components/DataTrack.vue`

Add a new prop for the graph cut region, render it as a teal range bar, and handle click-to-place when in graph cut mode.

- [ ] **Step 1: Add props and emit**

In `DataTrack.vue`, add to the props definition (around line 31):

```typescript
// Graph cut region (single fixed-size region)
graphcutRegion?: { start: number; end: number } | null
isGraphCutMode?: boolean
```

Add defaults (around line 39):

```typescript
graphcutRegion: null,
isGraphCutMode: false,
```

Add to emit definition (around line 55):

```typescript
'place-graphcut-region': [frameIdx: number]
'clear-graphcut-region': []
```

- [ ] **Step 2: Add computed style**

After `workingRangeStyle` (around line 156), add:

```typescript
const graphcutRegionStyle = computed(() => {
  if (!props.graphcutRegion) return null
  return rangeToStyle([props.graphcutRegion.start, props.graphcutRegion.end])
})
```

- [ ] **Step 3: Add click handling for graph cut mode**

In `onMouseDown` (around line 220), add a check at the top of the function, before the existing click handlers:

```typescript
// Graph cut mode: single click to place region
if (props.isGraphCutMode) {
  const frame = getFrameFromEvent(event)
  emit('place-graphcut-region', frame)
  return
}
```

In `onKeyDown` (around line 288), add before the existing `selectedWorkingRange` check:

```typescript
if (props.graphcutRegion) {
  event.preventDefault()
  emit('clear-graphcut-region')
  return
}
```

- [ ] **Step 4: Add template rendering**

In the template, after the working range div (after line 640), add:

```html
<!-- Graph cut region -->
<div
  v-if="graphcutRegionStyle"
  class="range-overlay graphcut-region"
  :style="graphcutRegionStyle"
/>
```

- [ ] **Step 5: Add CSS**

After the `.range-overlay.working-range-drag` styles (around line 823), add:

```css
.range-overlay.graphcut-region {
  background-color: rgba(6, 182, 212, 0.4);
  pointer-events: none;
}
```

- [ ] **Step 6: Verify TypeScript compiles**

Run: `cd /n/groups/datta/john/projects/vidseq/frontend && npx vue-tsc --noEmit 2>&1 | tail -5`
Expected: No new errors.

- [ ] **Step 7: Commit**

```bash
git add frontend/src/components/DataTrack.vue
git commit -m "feat: add graph cut region rendering to DataTrack"
```

---

### Task 7: Frontend — VideoOverlay brush painting

**Files:**
- Modify: `frontend/src/components/VideoOverlay.vue`

Add brush painting mode: left-drag paints foreground seeds, right-drag paints background seeds. Render seeds as colored dots. Prevent context menu on right-click.

- [ ] **Step 1: Extend ToolType**

In `frontend/src/composables/useSegmentation.ts` line 22, extend the type:

```typescript
export type ToolType = 'none' | 'positive_point' | 'negative_point' | 'bounding_box' | 'graphcut_brush'
```

- [ ] **Step 2: Add brush props to VideoOverlay**

In `VideoOverlay.vue`, add to props (around line 5):

```typescript
brushSize?: number
graphcutSeeds?: { x: number; y: number; label: number }[]
showSeeds?: boolean
```

Add to emits (around line 23):

```typescript
(e: 'brush-stroke', point: { x: number; y: number; label: number }): void
```

- [ ] **Step 3: Add coordinate mapping function**

After `getNormalizedCoords` (around line 48), add a function that returns native pixel coordinates:

```typescript
function getNativeCoords(event: MouseEvent): { x: number; y: number } | null {
  const canvas = canvasRef.value
  if (!canvas) return null
  const rect = canvas.getBoundingClientRect()
  const scaleX = props.videoWidth / rect.width
  const scaleY = props.videoHeight / rect.height
  const x = Math.floor((event.clientX - rect.left) * scaleX)
  const y = Math.floor((event.clientY - rect.top) * scaleY)
  if (x < 0 || x >= props.videoWidth || y < 0 || y >= props.videoHeight) return null
  return { x, y }
}
```

- [ ] **Step 4: Add brush stroke handling to mouse events**

Modify `onMouseDown` (line 50) to handle brush mode:

```typescript
function onMouseDown(event: MouseEvent) {
  // Brush painting mode
  if (props.activeTool === 'graphcut_brush') {
    const coords = getNativeCoords(event)
    if (!coords) return
    isBrushPainting.value = true
    brushButton.value = event.button  // 0 = left (fg), 2 = right (bg)
    const label = event.button === 2 ? 2 : 1
    emitBrushStroke(coords.x, coords.y, label)
    return
  }
  // ... existing point/box handling unchanged ...
}
```

Add state variables near the top of the script:

```typescript
const isBrushPainting = ref(false)
const brushButton = ref(0)
```

Add a helper to emit brush strokes for all pixels in the brush radius:

```typescript
function emitBrushStroke(cx: number, cy: number, label: number) {
  const canvas = canvasRef.value
  if (!canvas) return
  const rect = canvas.getBoundingClientRect()
  const scaleX = props.videoWidth / rect.width
  const r = Math.max(1, Math.round((props.brushSize ?? 5) * scaleX))
  for (let dy = -r; dy <= r; dy++) {
    for (let dx = -r; dx <= r; dx++) {
      if (dx * dx + dy * dy <= r * r) {
        const x = cx + dx
        const y = cy + dy
        if (x >= 0 && x < props.videoWidth && y >= 0 && y < props.videoHeight) {
          emit('brush-stroke', { x, y, label })
        }
      }
    }
  }
  render()
}
```

Modify `onMouseMove` (line 71) to handle brush:

```typescript
function onMouseMove(event: MouseEvent) {
  if (isBrushPainting.value) {
    const coords = getNativeCoords(event)
    if (!coords) return
    const label = brushButton.value === 2 ? 2 : 1
    emitBrushStroke(coords.x, coords.y, label)
    return
  }
  // ... existing box drag handling unchanged ...
}
```

Modify `onMouseUp` (line 86) to handle brush:

```typescript
function onMouseUp(_event: MouseEvent) {
  if (isBrushPainting.value) {
    isBrushPainting.value = false
    return
  }
  // ... existing box handling unchanged ...
}
```

- [ ] **Step 5: Add context menu prevention and template update**

In the template (line 306), add `@contextmenu.prevent` to the canvas:

```html
<canvas
  ref="canvasRef"
  class="video-overlay"
  :class="{ 'tool-active': activeTool !== 'none' || isLabelingKeypoints }"
  @mousedown="onMouseDown"
  @mousemove="onMouseMove"
  @mouseup="onMouseUp"
  @contextmenu.prevent
/>
```

- [ ] **Step 6: Add seed rendering to render()**

In the `render()` function (around line 103), add seed rendering after the existing mask/prompt drawing. Add before the closing of the function:

```typescript
// Draw graph cut seeds
if (props.showSeeds !== false && props.graphcutSeeds) {
  const canvas = canvasRef.value!
  const rect = canvas.getBoundingClientRect()
  const scaleX = props.videoWidth / rect.width
  const dotSize = Math.max(1, Math.round(1 / scaleX * 2))
  for (const seed of props.graphcutSeeds) {
    ctx.fillStyle = seed.label === 1
      ? 'rgba(34, 197, 94, 0.7)'  // green for foreground
      : 'rgba(239, 68, 68, 0.7)'  // red for background
    ctx.fillRect(seed.x - dotSize / 2, seed.y - dotSize / 2, dotSize, dotSize)
  }
}
```

- [ ] **Step 7: Verify TypeScript compiles**

Run: `cd /n/groups/datta/john/projects/vidseq/frontend && npx vue-tsc --noEmit 2>&1 | tail -5`
Expected: No new errors.

- [ ] **Step 8: Commit**

```bash
git add frontend/src/components/VideoOverlay.vue frontend/src/composables/useSegmentation.ts
git commit -m "feat: add brush painting mode to VideoOverlay"
```

---

### Task 8: Frontend — VideoDetail wiring

**Files:**
- Modify: `frontend/src/components/VideoDetail.vue`

Wire up the graph cut composable, add tool button, brush size slider, run button, seed visibility toggle, and connect DataTrack + VideoOverlay.

- [ ] **Step 1: Import and initialize composable**

Add import at the top of the script section:

```typescript
import { useGraphCut } from '@/composables/useGraphCut'
```

After the existing composable destructuring (around line 157), add:

```typescript
const {
  graphcutRegion,
  brushSize,
  seedsVisible,
  isRunning: isRunningGraphCut,
  isGraphCutMode,
  placeRegion,
  paintSeed,
  getSeedsForFrame,
  runGraphCut,
  exitGraphCutMode,
} = useGraphCut(projectId, videoId)
```

- [ ] **Step 2: Add graph cut tool toggle**

Add a ref and toggle function:

```typescript
const isGraphCutToolActive = ref(false)

const toggleGraphCutTool = () => {
  isGraphCutToolActive.value = !isGraphCutToolActive.value
  if (isGraphCutToolActive.value) {
    // Deactivate all other tools
    activeTool.value = 'graphcut_brush'
    isMarkingMode.value = false
    isWorkingRangeMode.value = false
  } else {
    activeTool.value = 'none'
    exitGraphCutMode()
  }
}
```

Also ensure that toggling other tools deactivates graph cut. In the existing tool toggle watchers or handlers, when `activeTool` changes to a SAM2 tool, set `isGraphCutToolActive.value = false` and call `exitGraphCutMode()`:

```typescript
watch(activeTool, (newTool) => {
  if (newTool !== 'graphcut_brush' && newTool !== 'none') {
    isGraphCutToolActive.value = false
    exitGraphCutMode()
  }
})
```

- [ ] **Step 3: Add computed for current frame seeds**

```typescript
const currentFrameSeeds = computed(() => getSeedsForFrame(currentFrameIdx.value))
```

- [ ] **Step 4: Add handler functions**

```typescript
const handlePlaceGraphcutRegion = (frameIdx: number) => {
  if (!video.value) return
  placeRegion(frameIdx, video.value.num_frames)
}

const handleRunGraphCut = async () => {
  await runGraphCut(clearMaskCache)
  await refreshFrameRanges()
}

const handleBrushStroke = (point: { x: number; y: number; label: number }) => {
  paintSeed(currentFrameIdx.value, point.x, point.y, point.label)
}

const handleClearGraphcutRegion = () => {
  exitGraphCutMode()
  isGraphCutToolActive.value = false
  activeTool.value = 'none'
}
```

- [ ] **Step 5: Add UI controls to template**

After the existing tool buttons div (after line 626), add:

```html
<h4 class="action-bar-title" style="margin-top: 16px;">Graph Cut</h4>
<div class="tool-buttons">
  <button
    class="tool-button graphcut"
    :class="{ active: isGraphCutToolActive }"
    @click="toggleGraphCutTool"
  >
    <span class="tool-icon">◈</span>
    <span class="tool-label">Graph Cut Brush</span>
  </button>
  <button
    v-if="isGraphCutMode"
    class="tool-button"
    :disabled="isRunningGraphCut"
    @click="handleRunGraphCut"
  >
    <span class="tool-icon">▶</span>
    <span class="tool-label">{{ isRunningGraphCut ? 'Running...' : 'Run Graph Cut' }}</span>
  </button>
</div>
<div v-if="isGraphCutMode" class="memory-options">
  <label class="memory-toggle">
    <span>Brush Size: {{ brushSize }}px</span>
    <input type="range" v-model.number="brushSize" min="1" max="50" />
  </label>
  <label class="memory-toggle">
    <input type="checkbox" v-model="seedsVisible" />
    Show Seeds
  </label>
</div>
```

- [ ] **Step 6: Wire DataTrack props**

On the `<DataTrack>` component (around line 545), add:

```
:graphcut-region="graphcutRegion"
:is-graph-cut-mode="isGraphCutToolActive && !graphcutRegion"
@place-graphcut-region="handlePlaceGraphcutRegion"
@clear-graphcut-region="handleClearGraphcutRegion"
```

Note: `isGraphCutMode` is `true` only when the tool is active AND no region is placed yet (so clicks place the region). Once placed, clicks on the data track don't re-place — the user must delete first.

Wait — actually, per the spec, "clicking again moves it." So the mode should stay active:

```
:is-graph-cut-mode="isGraphCutToolActive"
```

- [ ] **Step 7: Wire VideoOverlay props**

On the `<VideoOverlay>` component, add the new props:

```
:brush-size="brushSize"
:graphcut-seeds="currentFrameSeeds"
:show-seeds="seedsVisible"
@brush-stroke="handleBrushStroke"
```

- [ ] **Step 8: Verify TypeScript compiles**

Run: `cd /n/groups/datta/john/projects/vidseq/frontend && npx vue-tsc --noEmit 2>&1 | tail -5`
Expected: No new errors.

- [ ] **Step 9: Commit**

```bash
git add frontend/src/components/VideoDetail.vue
git commit -m "feat: wire graph cut UI controls in VideoDetail"
```

---

### Task 9: End-to-end verification

**Files:** None (testing only)

- [ ] **Step 1: Start the backend**

Run: `cd /n/groups/datta/john/projects/vidseq && vidseq`
Expected: Server starts on port 8000 without import errors.

- [ ] **Step 2: Start the frontend**

Run: `cd /n/groups/datta/john/projects/vidseq/frontend && npm run dev`
Expected: Dev server starts on port 5173.

- [ ] **Step 3: Test in browser**

1. Open a project with a video in the browser
2. Click the "Graph Cut Brush" button in the sidebar
3. Click on the data track — a teal region should appear
4. Left-click drag on the video to paint green foreground seeds
5. Right-click drag to paint red background seeds
6. Adjust brush size with the slider
7. Toggle "Show Seeds" to hide/show seed overlay
8. Click "Run Graph Cut" — should complete and masks appear
9. Scrub through the chunk to verify masks on multiple frames
10. Delete the region (Delete/Backspace on data track) — seeds and region disappear

- [ ] **Step 4: Verify overwrite behavior**

1. Run SAM2 propagation on some frames within the region
2. Run graph cut on the same frames
3. Verify the masks are overwritten (graph cut result, not SAM2)

- [ ] **Step 5: Final commit (if any fixes needed)**

Fix any issues found during testing and commit.
