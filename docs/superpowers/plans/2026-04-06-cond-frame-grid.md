# Conditioning Frame Mask Grid Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Display a 2-column grid of conditioning frame masks below the video timeline so the user can visually inspect which masks are influencing SAM2 propagation.

**Architecture:** New `CondFrameGrid.vue` component fetches conditioning frame indices and their tracker mask PNGs, displays them in a scrollable 2-column CSS grid. VideoDetail.vue mounts it below the timeline and passes a refresh key that increments after prompt actions.

**Tech Stack:** Vue 3 (Composition API), TypeScript

**Spec:** `docs/superpowers/specs/2026-04-06-cond-frame-grid-design.md`

---

### Task 1: Create CondFrameGrid.vue component

**Files:**
- Create: `frontend/src/components/CondFrameGrid.vue`

- [ ] **Step 1: Create the component file**

Create `frontend/src/components/CondFrameGrid.vue` with this content:

```vue
<script setup lang="ts">
import { ref, watch, onUnmounted } from 'vue'
import { getConditioningFrames, getTrackerMask } from '@/services/api'

const props = defineProps<{
  projectId: number
  videoId: number
  refreshKey: number
}>()

interface CondFrame {
  frameIdx: number
  objectUrl: string
}

const condFrames = ref<CondFrame[]>([])
const isLoading = ref(false)

const revokeUrls = () => {
  for (const frame of condFrames.value) {
    URL.revokeObjectURL(frame.objectUrl)
  }
}

const fetchCondFrames = async () => {
  if (!props.projectId || !props.videoId) return
  isLoading.value = true
  try {
    const frameIndices = await getConditioningFrames(props.projectId, props.videoId)
    revokeUrls()

    const frames: CondFrame[] = []
    for (const idx of frameIndices) {
      try {
        const blob = await getTrackerMask(props.projectId, props.videoId, idx)
        frames.push({ frameIdx: idx, objectUrl: URL.createObjectURL(blob) })
      } catch {
        // Skip frames where mask fetch fails
      }
    }
    condFrames.value = frames
  } catch {
    condFrames.value = []
  } finally {
    isLoading.value = false
  }
}

watch(
  () => [props.projectId, props.videoId, props.refreshKey],
  () => fetchCondFrames(),
  { immediate: true },
)

onUnmounted(() => revokeUrls())
</script>

<template>
  <div v-if="condFrames.length > 0" class="cond-frame-grid-section">
    <h4 class="cond-frame-title">Conditioning Frames ({{ condFrames.length }})</h4>
    <div class="cond-frame-grid">
      <div v-for="frame in condFrames" :key="frame.frameIdx" class="cond-frame-cell">
        <img :src="frame.objectUrl" class="cond-frame-img" />
        <span class="cond-frame-label">Frame {{ frame.frameIdx }}</span>
      </div>
    </div>
  </div>
</template>

<style scoped>
.cond-frame-grid-section {
  width: 100%;
  max-height: 400px;
  overflow-y: auto;
}

.cond-frame-title {
  margin: 0 0 6px 0;
  font-size: 0.85rem;
  font-weight: 600;
  color: #444;
}

.cond-frame-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
}

.cond-frame-cell {
  display: flex;
  flex-direction: column;
  align-items: center;
}

.cond-frame-img {
  width: 100%;
  background: #000;
  display: block;
}

.cond-frame-label {
  font-size: 0.75rem;
  color: #666;
  margin-top: 2px;
}
</style>
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/components/CondFrameGrid.vue
git commit -m "feat: add CondFrameGrid component for conditioning frame mask display"
```

---

### Task 2: Wire CondFrameGrid into VideoDetail.vue

**Files:**
- Modify: `frontend/src/components/VideoDetail.vue`

- [ ] **Step 1: Add the import**

In the `<script setup>` section, after the existing component imports (after line 14 where `TimelineSystem` is imported), add:

```typescript
import CondFrameGrid from './CondFrameGrid.vue'
```

- [ ] **Step 2: Add the refresh key ref**

After the existing `showObbConfidence` ref (line 45), add:

```typescript
const condFrameRefreshKey = ref(0)
```

- [ ] **Step 3: Increment refresh key in prompt handlers**

Modify `handlePointCompleteWithRefresh` (line 318) to increment the key:

Change from:
```typescript
const handlePointCompleteWithRefresh = async (point: { x: number; y: number; type: 'positive_point' | 'negative_point' }) => {
  await handlePointComplete(point)
  await refreshFrameRanges()
}
```

To:
```typescript
const handlePointCompleteWithRefresh = async (point: { x: number; y: number; type: 'positive_point' | 'negative_point' }) => {
  await handlePointComplete(point)
  await refreshFrameRanges()
  condFrameRefreshKey.value++
}
```

Modify `handleBoxCompleteWithRefresh` (line 323) the same way:

Change from:
```typescript
const handleBoxCompleteWithRefresh = async (box: { x1: number; y1: number; x2: number; y2: number }) => {
  await handleBoxComplete(box)
  await refreshFrameRanges()
}
```

To:
```typescript
const handleBoxCompleteWithRefresh = async (box: { x1: number; y1: number; x2: number; y2: number }) => {
  await handleBoxComplete(box)
  await refreshFrameRanges()
  condFrameRefreshKey.value++
}
```

- [ ] **Step 4: Add the component to the template**

After the closing `</TimelineSystem>` tag (line 444), add:

```vue
          <CondFrameGrid
            v-if="video && segmentationIsReady"
            :project-id="projectId"
            :video-id="videoId"
            :refresh-key="condFrameRefreshKey"
          />
```

- [ ] **Step 5: Verify frontend compiles**

Run: `cd /n/groups/datta/john/projects/vidseq/frontend && npm run type-check`

Expected: No new errors

- [ ] **Step 6: Commit**

```bash
git add frontend/src/components/VideoDetail.vue
git commit -m "feat: wire CondFrameGrid into VideoDetail below timeline"
```
