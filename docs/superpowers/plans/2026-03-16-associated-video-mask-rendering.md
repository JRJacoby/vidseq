# Associated Video Mask Rendering Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire up mask overlay rendering in AssociatedVideoDetail.vue with three view modes: tracker_mask, final_mask (via existing useSegmentation composable), and main_bbox (via new backend bbox endpoint + lightweight rAF sync loop).

**Architecture:** Backend adds two bbox endpoints (routed through segmentation_service) that read tracker masks from H5 and return computed bounding boxes. Frontend reuses `useSegmentation` for mask modes and adds a custom bbox fetch/cache/sync loop for main_bbox mode, both rendering through the existing `VideoOverlay` component.

**Tech Stack:** Python/FastAPI, h5py, Vue 3/TypeScript, HTML5 Canvas

**Spec:** `docs/superpowers/specs/2026-03-16-associated-video-mask-rendering-design.md`

---

## Chunk 1: Schema Fix + Backend Bbox Endpoints

### Task 0: Add Missing width/height/num_frames to VideoResponse and Video Interface

The `VideoResponse` schema and frontend `Video` interface are missing `width`, `height`, and `num_frames` fields. These are needed for bbox coordinate rescaling and exist on the `Video` ORM model (so `from_attributes = True` will pick them up).

**Files:**
- Modify: `vidseq/schemas/video.py`
- Modify: `frontend/src/services/api.ts`

- [ ] **Step 1: Add fields to VideoResponse**

In `vidseq/schemas/video.py`, add three fields to `VideoResponse` before `segmentation_status`:

```python
class VideoResponse(BaseModel):
    id: int
    name: str
    path: str
    fps: float
    width: int = 0
    height: int = 0
    num_frames: int = 0
    segmentation_status: str | None = None
    # Associated video fields
    is_associated: bool = False
    associated_with_id: int | None = None
    associated_video_id: int | None = None

    class Config:
        from_attributes = True
```

- [ ] **Step 2: Add fields to frontend Video interface**

In `frontend/src/services/api.ts`, add to the `Video` interface after `fps`:

```typescript
export interface Video {
    id: number
    name: string
    path: string
    fps: number
    width: number
    height: number
    num_frames: number
    segmentation_status: 'in_progress' | 'segmented' | null
    // ... existing fields ...
}
```

- [ ] **Step 3: Commit**

```bash
git add vidseq/schemas/video.py frontend/src/services/api.ts
git commit -m "feat: add width, height, num_frames to VideoResponse and Video interface"
```

---

### Task 1: Backend — Bbox Service Functions and Endpoints

**Files:**
- Modify: `vidseq/services/segmentation_service.py`
- Modify: `vidseq/api/routes/segmentation/masks.py`

- [ ] **Step 1: Add service functions to `segmentation_service.py`**

Add two functions:

```python
def get_tracker_mask_bbox(
    project_path: Path,
    video_id: int,
    frame_idx: int,
) -> dict | None:
    """Get bounding box of tracker mask for a single frame.

    Returns {x1, y1, x2, y2} dict or None if mask is empty.
    Coordinates are in native video pixel space.
    """
    from vidseq.services.array_storage import tracker_masks, compute_bbox_from_mask

    try:
        with tracker_masks(project_path, video_id, "r") as masks:
            mask = np.asarray(masks[frame_idx])
            bbox = compute_bbox_from_mask(mask)
    except FileNotFoundError:
        return None

    if bbox is None:
        return None
    return {"x1": float(bbox[0]), "y1": float(bbox[1]), "x2": float(bbox[2]), "y2": float(bbox[3])}


def get_tracker_mask_bboxes_batch(
    project_path: Path,
    video_id: int,
    start_frame: int,
    count: int,
    num_frames: int,
) -> list[dict]:
    """Get bounding boxes for a range of tracker mask frames.

    Returns list of {frame_idx, x1, y1, x2, y2} dicts, only for non-empty masks.
    Uses batch H5 read for performance.
    """
    from vidseq.services.array_storage import tracker_masks, compute_bbox_from_mask

    end_frame = min(start_frame + count, num_frames)
    bboxes = []

    try:
        with tracker_masks(project_path, video_id, "r") as masks:
            masks_arr = np.asarray(masks[start_frame:end_frame])
            for i, mask in enumerate(masks_arr):
                bbox = compute_bbox_from_mask(mask)
                if bbox is not None:
                    bboxes.append({
                        "frame_idx": start_frame + i,
                        "x1": float(bbox[0]),
                        "y1": float(bbox[1]),
                        "x2": float(bbox[2]),
                        "y2": float(bbox[3]),
                    })
    except FileNotFoundError:
        pass

    return bboxes
```

Make sure `import numpy as np` is at the top of the file (check existing imports — it may already be there).

- [ ] **Step 2: Add route endpoints to `masks.py`**

Add after the existing tracker mask endpoints:

```python
@router.get(
    "/projects/{project_id}/videos/{video_id}/segmentation/tracker-mask-bboxes/{frame_idx}",
)
async def get_tracker_mask_bbox(
    frame_idx: int,
    video: Video = Depends(get_video),
    project_path: Path = Depends(get_project_folder),
):
    """Get bounding box of the tracker mask for a specific frame."""
    bbox = segmentation_service.get_tracker_mask_bbox(
        project_path=project_path,
        video_id=video.id,
        frame_idx=frame_idx,
    )
    return {"bbox": bbox}


@router.get(
    "/projects/{project_id}/videos/{video_id}/segmentation/tracker-mask-bboxes",
)
async def get_tracker_mask_bboxes(
    start_frame: int,
    count: int = 100,
    video: Video = Depends(get_video),
    project_path: Path = Depends(get_project_folder),
):
    """Get bounding boxes for a range of tracker mask frames."""
    bboxes = segmentation_service.get_tracker_mask_bboxes_batch(
        project_path=project_path,
        video_id=video.id,
        start_frame=start_frame,
        count=count,
        num_frames=video.num_frames,
    )
    return {"bboxes": bboxes}
```

- [ ] **Step 3: Commit**

```bash
git add vidseq/services/segmentation_service.py vidseq/api/routes/segmentation/masks.py
git commit -m "feat: add tracker mask bbox service functions and endpoints"
```

---

## Chunk 2: Frontend API + AssociatedVideoDetail Wiring

### Task 2: Frontend API — Bbox Functions

**Files:**
- Modify: `frontend/src/services/api.ts`

- [ ] **Step 1: Add bbox API functions**

Add after the existing `getTrackerMasks` function:

```typescript
export interface BboxResult {
    x1: number
    y1: number
    x2: number
    y2: number
}

export async function getTrackerMaskBbox(
    projectId: number,
    videoId: number,
    frameIdx: number,
): Promise<BboxResult | null> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/${videoId}/segmentation/tracker-mask-bboxes/${frameIdx}`,
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to fetch tracker mask bbox'))
    }
    const data = await response.json()
    return data.bbox
}

export interface BboxBatchItem {
    frame_idx: number
    x1: number
    y1: number
    x2: number
    y2: number
}

export async function getTrackerMaskBboxes(
    projectId: number,
    videoId: number,
    startFrame: number,
    count: number = 100,
): Promise<BboxBatchItem[]> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/${videoId}/segmentation/tracker-mask-bboxes?start_frame=${startFrame}&count=${count}`,
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to fetch tracker mask bboxes'))
    }
    const data = await response.json()
    return data.bboxes
}
```

- [ ] **Step 2: Commit**

```bash
cd /n/groups/datta/john/projects/vidseq && git add frontend/src/services/api.ts
git commit -m "feat: add tracker mask bbox API functions"
```

---

### Task 3: AssociatedVideoDetail — Wire Up Mask Overlay

**Files:**
- Modify: `frontend/src/components/AssociatedVideoDetail.vue`

This is the main task. Read the full file first, then apply changes.

- [ ] **Step 1: Add imports**

Add to the existing imports at the top of `<script setup>`:

```typescript
import { onUnmounted } from 'vue'
import {
    getTrackerMaskBbox,
    getTrackerMaskBboxes,
    type BboxResult,
} from '@/services/api'
import { useSegmentation } from '@/composables/useSegmentation'
import VideoOverlay from './VideoOverlay.vue'
```

Note: `onUnmounted` may need to be added to the existing `import { ref, computed, watch } from 'vue'` line instead.

- [ ] **Step 2: Destructure `videoWidth`, `videoHeight`, `setMetadataCallback` from `useVideoPlayback`**

Update the existing `useVideoPlayback()` destructuring to include these three additional values:

```typescript
const {
    videoRef,
    currentTime,
    duration,
    isPlaying,
    videoWidth,
    videoHeight,
    onTimeUpdate,
    onLoadedMetadata,
    onPlay,
    onPause,
    seek,
    togglePlay: handleTogglePlay,
    setMetadataCallback,
} = useVideoPlayback()
```

- [ ] **Step 3: Add computed `maskViewMode` and wire `useSegmentation`**

After the existing `viewMode` ref, add:

```typescript
// Map local view modes to useSegmentation's maskViewMode
// In main_bbox mode, useSegmentation stays on 'tracker' — it will fetch masks from
// the associated video but we won't display them. This is a minor waste but avoids
// complexity of conditionally initializing the composable.
const segMaskViewMode = computed<'tracker' | 'detector' | 'final'>(() => {
    if (viewMode.value === 'final_mask') return 'final'
    return 'tracker'
})

// Associated video ID as a ref for useSegmentation
const assocVideoId = computed(() => associatedVideo.value?.id ?? null)
const assocFps = computed(() => associatedVideo.value?.fps ?? 30)

const {
    currentMask,
    seekToFrame,
    loadFrameData,
} = useSegmentation(
    projectId,
    assocVideoId,
    currentFrameIdx,
    isPlaying,
    videoRef,
    assocFps,
    segMaskViewMode,
)

// Trigger first mask load when video metadata is ready
setMetadataCallback(() => {
    if (viewMode.value !== 'main_bbox') {
        loadFrameData(0)
    }
})
```

- [ ] **Step 4: Add main_bbox mode — bbox cache, fetch, and rAF sync loop**

Add after the useSegmentation setup:

```typescript
// ---- Main bbox mode state ----
const bboxCache = new Map<number, BboxResult | null>()
const currentBbox = ref<BboxResult | null>(null)
let bboxPrefetchedUpTo = -1
let bboxAnimFrameId: number | null = null
let bboxPrefetching = false

const BBOX_PREFETCH_BATCH = 100
const BBOX_PREFETCH_THRESHOLD = 100

// Rescale bbox from main video native coords to associated video native coords
const rescaledBbox = computed(() => {
    if (!currentBbox.value || !mainVideo.value || !associatedVideo.value) return null
    const xScale = associatedVideo.value.width / mainVideo.value.width
    const yScale = associatedVideo.value.height / mainVideo.value.height
    if (xScale === 1 && yScale === 1) return currentBbox.value
    return {
        x1: currentBbox.value.x1 * xScale,
        y1: currentBbox.value.y1 * yScale,
        x2: currentBbox.value.x2 * xScale,
        y2: currentBbox.value.y2 * yScale,
    }
})

// Effective overlay props — mask or bbox depending on mode
const overlayMask = computed(() =>
    viewMode.value !== 'main_bbox' ? currentMask.value : null
)
// Reuse detectorBbox prop to render main video bbox on associated video
const overlayBbox = computed(() =>
    viewMode.value === 'main_bbox' ? rescaledBbox.value : null
)

async function prefetchBboxes(startFrame: number) {
    if (bboxPrefetching || !projectId.value || !mainVideoId.value) return
    bboxPrefetching = true
    try {
        const items = await getTrackerMaskBboxes(
            projectId.value, mainVideoId.value, startFrame, BBOX_PREFETCH_BATCH,
        )
        // Fill cache — frames not in response have empty masks (null)
        const endFrame = startFrame + BBOX_PREFETCH_BATCH
        for (let i = startFrame; i < endFrame; i++) {
            if (!bboxCache.has(i)) bboxCache.set(i, null)
        }
        for (const item of items) {
            bboxCache.set(item.frame_idx, {
                x1: item.x1, y1: item.y1, x2: item.x2, y2: item.y2,
            })
        }
        if (endFrame - 1 > bboxPrefetchedUpTo) {
            bboxPrefetchedUpTo = endFrame - 1
        }
    } catch (e) {
        console.error('Failed to prefetch bboxes:', e)
    } finally {
        bboxPrefetching = false
    }
}

async function loadBboxForFrame(frameIdx: number) {
    if (bboxCache.has(frameIdx)) {
        currentBbox.value = bboxCache.get(frameIdx) ?? null
        return
    }
    if (!projectId.value || !mainVideoId.value) return
    try {
        const bbox = await getTrackerMaskBbox(projectId.value, mainVideoId.value, frameIdx)
        bboxCache.set(frameIdx, bbox)
        currentBbox.value = bbox
    } catch {
        currentBbox.value = null
    }
}

let lastBboxFrame = -1

function syncBboxToVideo() {
    if (!isPlaying.value || !videoRef.value || viewMode.value !== 'main_bbox') return
    const fps = associatedVideo.value?.fps ?? 30
    const frameIdx = Math.floor(videoRef.value.currentTime * fps)
    if (frameIdx !== lastBboxFrame) {
        const cached = bboxCache.get(frameIdx)
        if (cached !== undefined) {
            currentBbox.value = cached
            lastBboxFrame = frameIdx
        }
        const framesAhead = bboxPrefetchedUpTo - frameIdx
        if (framesAhead < BBOX_PREFETCH_THRESHOLD) {
            prefetchBboxes(bboxPrefetchedUpTo + 1)
        }
    }
    bboxAnimFrameId = requestAnimationFrame(syncBboxToVideo)
}

// Cleanup rAF on unmount
onUnmounted(() => {
    if (bboxAnimFrameId !== null) {
        cancelAnimationFrame(bboxAnimFrameId)
        bboxAnimFrameId = null
    }
})
```

- [ ] **Step 5: Add watchers for view mode switching and bbox playback**

```typescript
// View mode switch handler
watch(viewMode, async (mode) => {
    if (mode === 'main_bbox') {
        // Entering bbox mode: clear stale cache, load bbox for current frame
        bboxCache.clear()
        bboxPrefetchedUpTo = -1
        currentBbox.value = null
        lastBboxFrame = -1
        await loadBboxForFrame(currentFrameIdx.value)
        if (isPlaying.value) {
            await prefetchBboxes(currentFrameIdx.value)
            syncBboxToVideo()
        }
    } else {
        // Leaving bbox mode: stop bbox sync loop
        if (bboxAnimFrameId !== null) {
            cancelAnimationFrame(bboxAnimFrameId)
            bboxAnimFrameId = null
        }
        currentBbox.value = null
    }
})

// Bbox playback start/stop
watch(isPlaying, async (playing) => {
    if (viewMode.value !== 'main_bbox') return
    if (playing) {
        await prefetchBboxes(currentFrameIdx.value)
        prefetchBboxes(currentFrameIdx.value + BBOX_PREFETCH_BATCH)
        lastBboxFrame = -1
        syncBboxToVideo()
    } else {
        if (bboxAnimFrameId !== null) {
            cancelAnimationFrame(bboxAnimFrameId)
            bboxAnimFrameId = null
        }
    }
})

// Bbox scrub sync (when paused)
watch(currentFrameIdx, (frameIdx) => {
    if (viewMode.value === 'main_bbox' && !isPlaying.value) {
        loadBboxForFrame(frameIdx)
    }
})
```

- [ ] **Step 6: Update the `handleSeek` function**

Replace the existing `handleSeek`:

```typescript
const handleSeek = (time: number) => {
    seek(time)
    if (viewMode.value !== 'main_bbox') {
        const frameIdx = Math.floor(time * (associatedVideo.value?.fps ?? 30))
        seekToFrame(frameIdx)
    }
}
```

- [ ] **Step 7: Update the template — add `.video-wrapper` and `VideoOverlay`**

Replace the `.video-container` div content with:

```html
<div class="video-container">
    <div class="video-wrapper">
        <video
            ref="videoRef"
            class="video-player"
            :src="videoStreamUrl"
            @timeupdate="onTimeUpdate"
            @loadedmetadata="onLoadedMetadata"
            @play="onPlay"
            @pause="onPause"
        >
            Your browser does not support the video tag.
        </video>
        <VideoOverlay
            v-if="videoWidth > 0 && videoHeight > 0"
            :video-width="videoWidth"
            :video-height="videoHeight"
            :active-tool="'none'"
            :mask="overlayMask"
            :prompts="[]"
            :show-mask="viewMode !== 'main_bbox'"
            :detector-bbox="overlayBbox"
        />
    </div>
</div>
```

Note: `:active-tool="'none'"` is a string literal, not a ref. No point handlers wired. `:prompts="[]"` is an empty array.

- [ ] **Step 8: Update CSS — add `.video-wrapper`**

Add `.video-wrapper` styles to the `<style scoped>` section. The existing `.video-container` styles stay, just add:

```css
.video-wrapper {
    position: relative;
    display: inline-block;
    max-width: 100%;
    max-height: 100%;
}
```

- [ ] **Step 9: Commit**

```bash
cd /n/groups/datta/john/projects/vidseq && git add frontend/src/components/AssociatedVideoDetail.vue
git commit -m "feat: wire up mask and bbox overlay rendering in AssociatedVideoDetail"
```

---

## Task Dependencies

```
Independent:
  Task 0: Schema fix (backend + frontend)
  Task 1: Backend bbox endpoints
  Task 2: Frontend API functions (depends on Task 0 for Video interface changes)

Sequential:
  Task 0 → Task 2 (Task 2 modifies api.ts which Task 0 also touches)
  Task 1 + Task 2 → Task 3 (Task 3 uses both the endpoints and API functions)
```

Tasks 0 and 1 can run in parallel (different files). Task 2 depends on Task 0. Task 3 depends on Tasks 1 and 2.
