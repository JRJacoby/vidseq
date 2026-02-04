# Frontend Refactor Phase 1: Extract Reusable Composables

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create `useVideo` and `useSSEStream` composables to eliminate duplicate logic across multiple components.

**Architecture:** Extract video metadata fetching into `useVideo` (used by 3 components) and SSE connection management into `useSSEStream` (used by 3 components). Add `connect*Stream()` functions to api.ts to keep web communication in the API layer.

**Tech Stack:** Vue 3 Composition API, TypeScript, VueUse

---

## Task 1: Create `useVideo` composable

**Files:**
- Create: `frontend/src/composables/useVideo.ts`

**Step 1: Create the composable**

```typescript
import { ref, watch, onMounted, type Ref, type MaybeRef, toValue } from 'vue'
import { getVideo, type Video } from '@/services/api'

export function useVideo(videoId: MaybeRef<number>, projectId: MaybeRef<number>) {
  const video = ref<Video | null>(null)
  const isLoading = ref(false)
  const error = ref<string | null>(null)

  const load = async () => {
    const vid = toValue(videoId)
    const pid = toValue(projectId)
    if (!vid || !pid) return

    isLoading.value = true
    error.value = null

    try {
      video.value = await getVideo(pid, vid)
    } catch (e) {
      error.value = e instanceof Error ? e.message : 'Failed to load video'
    } finally {
      isLoading.value = false
    }
  }

  // Auto-fetch on mount
  onMounted(load)

  // Re-fetch when IDs change
  watch([() => toValue(videoId), () => toValue(projectId)], load)

  return {
    video,
    isLoading,
    error,
    refresh: load,
  }
}
```

**Step 2: Verify it compiles**

Run: `cd frontend && npm run type-check`
Expected: No errors related to useVideo.ts

**Step 3: Commit**

```bash
git add frontend/src/composables/useVideo.ts
git commit -m "$(cat <<'EOF'
feat(frontend): add useVideo composable

Extracts video metadata fetching logic into reusable composable.
Auto-fetches on mount and when videoId/projectId change.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Refactor VideoDetail.vue to use `useVideo`

**Files:**
- Modify: `frontend/src/components/VideoDetail.vue`

**Step 1: Update imports and remove inline logic**

Replace lines 1-79 (the script setup opening through loadVideo function) with updated version that:
1. Adds import for `useVideo`
2. Removes: `const video = ref<Video | null>(null)`
3. Removes: `const isLoading = ref(true)`
4. Removes: `const error = ref<string | null>(null)`
5. Removes: the entire `loadVideo` function (lines 68-79)
6. Adds: `const { video, isLoading, error, refresh: refreshVideo } = useVideo(videoId, projectId)`

The imports should become:
```typescript
import { ref, onMounted, computed, watch } from 'vue'
import { useDebounceFn } from '@vueuse/core'
import { useRoute, useRouter } from 'vue-router'
import { getVideoStreamUrl, createPropagation, getScoresDownsampled, detectorMasksExist, finalMasksExist, type Video, type MaskScore } from '@/services/api'
import { useSegmentationSession } from '@/composables/useSegmentationSession'
import { useVideoPlayback } from '@/composables/useVideoPlayback'
import { useSegmentation } from '@/composables/useSegmentation'
import { useFrameRanges } from '@/composables/useFrameRanges'
import { useVideo } from '@/composables/useVideo'
```

Add after projectId/videoId computeds:
```typescript
const { video, isLoading, error, refresh: refreshVideo } = useVideo(videoId, projectId)
```

**Step 2: Update onMounted**

The `onMounted` block should no longer call `loadVideo()`. Find the onMounted and remove the `await loadVideo()` line. Keep other calls like `checkDetectorMasks()`, `checkFinalMasks()`.

**Step 3: Verify it compiles**

Run: `cd frontend && npm run type-check`
Expected: No errors

**Step 4: Commit**

```bash
git add frontend/src/components/VideoDetail.vue
git commit -m "$(cat <<'EOF'
refactor(frontend): use useVideo in VideoDetail

Replaces inline video fetching with useVideo composable.
Removes ~15 lines of duplicated logic.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Refactor AlignedVideoDetail.vue to use `useVideo`

**Files:**
- Modify: `frontend/src/components/AlignedVideoDetail.vue`

**Step 1: Update imports and remove inline logic**

1. Add `useVideo` import
2. Remove: `const video = ref<Video | null>(null)` (line 25)
3. Remove: `const isLoading = ref(true)` (line 26)
4. Remove: `const error = ref<string | null>(null)` (line 27)
5. Remove: the entire `loadVideo` function (lines 37-48)
6. Add: `const { video, isLoading, error } = useVideo(videoId, projectId)`

Update imports:
```typescript
import { ref, onMounted, computed, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useDebounceFn } from '@vueuse/core'
import {
  getAlignedVideoStreamUrl,
  getPCAStatus,
  getPCAScoresDownsampled,
  checkPCAScoresExist,
  type PCAScorePoint,
} from '@/services/api'
import { useVideoPlayback } from '@/composables/useVideoPlayback'
import { useVideo } from '@/composables/useVideo'
```

Note: Remove `getVideo` and `type Video` from api imports since useVideo handles that.

**Step 2: Update onMounted**

Remove `await loadVideo()` from onMounted. The useVideo composable handles loading automatically.

**Step 3: Verify it compiles**

Run: `cd frontend && npm run type-check`
Expected: No errors

**Step 4: Commit**

```bash
git add frontend/src/components/AlignedVideoDetail.vue
git commit -m "$(cat <<'EOF'
refactor(frontend): use useVideo in AlignedVideoDetail

Replaces inline video fetching with useVideo composable.
Removes ~15 lines of duplicated logic.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Refactor CroppedVideoDetail.vue to use `useVideo`

**Files:**
- Modify: `frontend/src/components/CroppedVideoDetail.vue`

**Step 1: Analyze current loadVideo**

CroppedVideoDetail.vue has extra logic in loadVideo (lines 59-81):
- Fetches video
- Checks `hasAlignmentPredictions`
- Loads `getVideoAlignmentLabels`

This is more than just video metadata. We have two options:
1. Keep a local `loadVideo` that calls `useVideo.refresh()` plus the extra calls
2. Add an `onLoad` callback to useVideo

For simplicity, we'll use option 1: use `useVideo` for the video state, but keep a local function for the extra initialization.

**Step 2: Update imports and use composable**

1. Add `useVideo` import
2. Remove: `const video = ref<Video | null>(null)` (line 26)
3. Remove: `const isLoading = ref(true)` (line 27)
4. Remove: `const error = ref<string | null>(null)` (line 28)
5. Add: `const { video, isLoading, error, refresh: refreshVideo } = useVideo(videoId, projectId)`

Update loadVideo to only do the extra work:
```typescript
const loadExtraData = async () => {
  if (!video.value) return
  try {
    // Check if stored predictions exist
    hasStoredPredictions.value = await hasAlignmentPredictions(
      projectId.value,
      videoId.value
    )
    // Load alignment labels for this video
    const labelsResponse = await getVideoAlignmentLabels(
      projectId.value,
      videoId.value
    )
    alignmentLabelFrames.value = labelsResponse.frame_indices
  } catch (e) {
    // Extra data loading failed, but video loaded fine
    console.error('Failed to load extra data:', e)
  }
}

// Load extra data when video loads
watch(video, (v) => {
  if (v) loadExtraData()
})
```

Remove the old `loadVideo` function entirely.

**Step 3: Update onMounted**

Remove `await loadVideo()` from onMounted.

**Step 4: Update imports to remove getVideo**

```typescript
import {
  getCroppedVideoStreamUrl,
  hasAlignmentPredictions,
  getStoredAlignmentPredictionUrl,
  getVideoAlignmentLabels,
  saveAlignmentLabel,
  deleteAlignmentLabel,
  deleteVideoAlignmentLabels,
} from '@/services/api'
```

**Step 5: Verify it compiles**

Run: `cd frontend && npm run type-check`
Expected: No errors

**Step 6: Commit**

```bash
git add frontend/src/components/CroppedVideoDetail.vue
git commit -m "$(cat <<'EOF'
refactor(frontend): use useVideo in CroppedVideoDetail

Replaces inline video fetching with useVideo composable.
Keeps extra data loading (predictions, labels) in component.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Add SSE connect functions to api.ts

**Files:**
- Modify: `frontend/src/services/api.ts`

**Step 1: Add connect functions after existing URL functions**

Find `getDetectionTrainingStreamUrl` (line 1164) and add after it:

```typescript
export function connectDetectorTrainingStream(projectId: number): EventSource {
    return new EventSource(getDetectionTrainingStreamUrl(projectId))
}
```

Find `getAlignmentTrainingStreamUrl` (line 697) and add after it:

```typescript
export function connectAlignmentTrainingStream(projectId: number): EventSource {
    return new EventSource(getAlignmentTrainingStreamUrl(projectId))
}
```

Find `getVideosAlignmentStreamUrl` (line 734) and add after it:

```typescript
export function connectAlignmentApplyStream(projectId: number): EventSource {
    return new EventSource(getVideosAlignmentStreamUrl(projectId))
}
```

Find `getARHMMStreamUrl` (line 992) and add after it:

```typescript
export function connectARHMMStream(projectId: number): EventSource {
    return new EventSource(getARHMMStreamUrl(projectId))
}
```

Find `getCrowdMovieStreamUrl` (line 1079) and add after it:

```typescript
export function connectCrowdMovieStream(projectId: number): EventSource {
    return new EventSource(getCrowdMovieStreamUrl(projectId))
}
```

**Step 2: Verify it compiles**

Run: `cd frontend && npm run type-check`
Expected: No errors

**Step 3: Commit**

```bash
git add frontend/src/services/api.ts
git commit -m "$(cat <<'EOF'
feat(frontend): add SSE connect functions to api.ts

Adds connect*Stream() functions that return EventSource objects.
Keeps web communication in the API layer.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Create `useSSEStream` composable

**Files:**
- Create: `frontend/src/composables/useSSEStream.ts`

**Step 1: Create the composable**

```typescript
import { ref, onUnmounted, type Ref } from 'vue'

export function useSSEStream<T>(
  connect: () => EventSource
): {
  data: Ref<T | null>
  isConnected: Ref<boolean>
  error: Ref<string | null>
  start: () => void
  stop: () => void
} {
  const data = ref<T | null>(null) as Ref<T | null>
  const isConnected = ref(false)
  const error = ref<string | null>(null)

  let eventSource: EventSource | null = null

  const stop = () => {
    if (eventSource) {
      eventSource.close()
      eventSource = null
    }
    isConnected.value = false
  }

  const start = () => {
    // Close existing connection if any
    stop()
    error.value = null

    try {
      eventSource = connect()
      isConnected.value = true

      eventSource.onmessage = (event) => {
        try {
          data.value = JSON.parse(event.data) as T
        } catch (e) {
          error.value = 'Failed to parse SSE data'
        }
      }

      eventSource.onerror = () => {
        error.value = 'SSE connection error'
        stop()
      }
    } catch (e) {
      error.value = e instanceof Error ? e.message : 'Failed to connect'
      isConnected.value = false
    }
  }

  // Cleanup on unmount
  onUnmounted(stop)

  return {
    data,
    isConnected,
    error,
    start,
    stop,
  }
}
```

**Step 2: Verify it compiles**

Run: `cd frontend && npm run type-check`
Expected: No errors

**Step 3: Commit**

```bash
git add frontend/src/composables/useSSEStream.ts
git commit -m "$(cat <<'EOF'
feat(frontend): add useSSEStream composable

Generic composable for SSE connections with:
- Reactive data, connection status, and error state
- start/stop controls
- Automatic cleanup on unmount
- JSON parsing of messages

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Refactor DetectorTraining.vue to use `useSSEStream`

**Files:**
- Modify: `frontend/src/components/DetectorTraining.vue`

**Step 1: Update imports**

Replace api imports:
```typescript
import {
    connectDetectorTrainingStream,
    getDetectionTraining,
    deleteDetectionTraining,
    type DetectorTrainingProgress,
} from '@/services/api'
import { useSSEStream } from '@/composables/useSSEStream'
```

Remove `getDetectionTrainingStreamUrl` from imports.

**Step 2: Replace EventSource logic with composable**

Remove:
- `let eventSource: EventSource | null = null` (line 21)
- The entire `connectToStream` function (lines 83-111)

Add after projectId computed:
```typescript
const {
    data: streamData,
    isConnected,
    start: startStream,
    stop: stopStream,
} = useSSEStream<DetectorTrainingProgress>(() =>
    connectDetectorTrainingStream(projectId.value!)
)
```

**Step 3: Update chart when stream data changes**

Add a watch for streamData:
```typescript
watch(streamData, (data) => {
    if (data) {
        progress.value = data
        updateChart(data.train_loss_history, data.val_loss_history)

        // Stop stream if training finished
        if (['completed', 'stopped', 'failed', 'idle'].includes(data.status)) {
            stopStream()
        }
    }
})
```

**Step 4: Update loadTrainingStatus**

Replace `connectToStream()` with `startStream()`:
```typescript
async function loadTrainingStatus() {
    if (!projectId.value) return

    try {
        progress.value = await getDetectionTraining(projectId.value)
        if (progress.value.train_loss_history.length > 0 || progress.value.val_loss_history.length > 0) {
            updateChart(progress.value.train_loss_history, progress.value.val_loss_history)
        }

        // If training in progress, connect to stream
        if (progress.value.is_training || progress.value.status === 'applying') {
            startStream()
        }
    } catch (e) {
        console.error('Failed to load training status:', e)
    }
}
```

**Step 5: Update onUnmounted**

Remove `eventSource?.close()` - the composable handles cleanup automatically:
```typescript
onUnmounted(() => {
    chart?.destroy()
})
```

**Step 6: Update project change watcher**

Replace:
```typescript
watch(() => projectId.value, () => {
    stopStream()
    loadTrainingStatus()
})
```

**Step 7: Update training start watcher**

Replace `connectToStream()` with `startStream()`:
```typescript
watch(() => progress.value?.is_training, (isTraining) => {
    if (isTraining && !isConnected.value) {
        startStream()
    }
})
```

**Step 8: Verify it compiles**

Run: `cd frontend && npm run type-check`
Expected: No errors

**Step 9: Commit**

```bash
git add frontend/src/components/DetectorTraining.vue
git commit -m "$(cat <<'EOF'
refactor(frontend): use useSSEStream in DetectorTraining

Replaces inline EventSource logic with useSSEStream composable.
Removes ~30 lines of duplicated SSE handling code.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Refactor Alignment.vue to use `useSSEStream`

**Files:**
- Modify: `frontend/src/components/Alignment.vue`

**Step 1: Update imports**

```typescript
import {
    connectAlignmentTrainingStream,
    connectAlignmentApplyStream,
    getAlignmentTraining,
    getVideosAlignmentStatus,
    type TrainingProgress,
    type AlignmentApplyProgress
} from '@/services/api'
import { useSSEStream } from '@/composables/useSSEStream'
```

Remove `getAlignmentTrainingStreamUrl` and `getVideosAlignmentStreamUrl`.

**Step 2: Replace EventSource logic with two composables**

Remove:
- `let eventSource: EventSource | null = null` (line 23)
- `let alignEventSource: EventSource | null = null` (line 27)
- `connectToStream` function (lines 89-117)
- `connectToAlignStream` function (lines 139-166)

Add after projectId:
```typescript
// Training stream
const {
    data: trainingStreamData,
    isConnected: trainingConnected,
    start: startTrainingStream,
    stop: stopTrainingStream,
} = useSSEStream<TrainingProgress>(() =>
    connectAlignmentTrainingStream(projectId.value!)
)

// Alignment apply stream
const {
    data: alignStreamData,
    isConnected: alignConnected,
    start: startAlignStream,
    stop: stopAlignStream,
} = useSSEStream<AlignmentApplyProgress>(() =>
    connectAlignmentApplyStream(projectId.value!)
)
```

**Step 3: Add watchers for stream data**

```typescript
// Handle training stream data
watch(trainingStreamData, (data) => {
    if (data) {
        progress.value = data
        updateChart(data.train_loss_history, data.val_loss_history)

        if (['completed', 'stopped', 'failed'].includes(data.status)) {
            stopTrainingStream()
        }
    }
})

// Handle alignment apply stream data
watch(alignStreamData, (data) => {
    if (data) {
        alignProgress.value = data

        if (['completed', 'failed'].includes(data.status)) {
            stopAlignStream()
        }
    }
})
```

**Step 4: Update load functions**

```typescript
async function loadTrainingStatus() {
    if (!projectId.value) return

    try {
        progress.value = await getAlignmentTraining(projectId.value)
        if (progress.value.train_loss_history.length > 0 || progress.value.val_loss_history.length > 0) {
            updateChart(progress.value.train_loss_history, progress.value.val_loss_history)
        }

        if (progress.value.is_training) {
            startTrainingStream()
        }
    } catch (e) {
        console.error('Failed to load training status:', e)
    }
}

async function loadAlignmentStatus() {
    if (!projectId.value) return

    try {
        alignProgress.value = await getVideosAlignmentStatus(projectId.value)

        if (alignProgress.value.is_aligning) {
            startAlignStream()
        }
    } catch (e) {
        console.error('Failed to load alignment status:', e)
    }
}
```

**Step 5: Update onUnmounted**

```typescript
onUnmounted(() => {
    chart?.destroy()
})
```

**Step 6: Update watchers**

```typescript
watch(() => projectId.value, () => {
    stopTrainingStream()
    stopAlignStream()
    loadTrainingStatus()
    loadAlignmentStatus()
})

watch(() => progress.value?.is_training, (isTraining) => {
    if (isTraining && !trainingConnected.value) {
        startTrainingStream()
    }
})

watch(() => alignProgress.value?.is_aligning, (isAligning) => {
    if (isAligning && !alignConnected.value) {
        startAlignStream()
    }
})
```

**Step 7: Verify it compiles**

Run: `cd frontend && npm run type-check`
Expected: No errors

**Step 8: Commit**

```bash
git add frontend/src/components/Alignment.vue
git commit -m "$(cat <<'EOF'
refactor(frontend): use useSSEStream in Alignment

Replaces inline EventSource logic for both training and apply streams.
Removes ~60 lines of duplicated SSE handling code.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Refactor ARHMMView.vue to use `useSSEStream`

**Files:**
- Modify: `frontend/src/components/ARHMMView.vue`

**Step 1: Update imports**

```typescript
import {
    getARHMMStatus,
    createARHMMTraining,
    deleteARHMMTraining,
    connectARHMMStream,
    getARHMMAnalysis,
    createCrowdMoviesGeneration,
    deleteCrowdMoviesGeneration,
    getCrowdMovieStatus,
    getCrowdMovieList,
    connectCrowdMovieStream,
    getCrowdMovieVideoUrl,
    type ARHMMProgress,
    type ARHMMAnalysis,
    type CrowdMovieProgress,
    type CrowdMovieEntry,
} from '@/services/api'
import { useSSEStream } from '@/composables/useSSEStream'
```

Remove `getARHMMStreamUrl` and `getCrowdMovieStreamUrl`.

**Step 2: Replace EventSource logic with composables**

Remove:
- `let eventSource: EventSource | null = null` (line 32)
- `let crowdMovieEventSource: EventSource | null = null` (line 47)
- `connectToStream` function (lines 70-99)
- The crowd movie stream connection code (around lines 272-297)

Add after projectId:
```typescript
// ARHMM stream
const {
    data: arhmmStreamData,
    isConnected: arhmmConnected,
    start: startARHMMStream,
    stop: stopARHMMStream,
} = useSSEStream<ARHMMProgress>(() =>
    connectARHMMStream(projectId.value!)
)

// Crowd movie stream
const {
    data: crowdMovieStreamData,
    isConnected: crowdMovieConnected,
    start: startCrowdMovieStream,
    stop: stopCrowdMovieStream,
} = useSSEStream<CrowdMovieProgress>(() =>
    connectCrowdMovieStream(projectId.value!)
)
```

**Step 3: Add watchers for stream data**

```typescript
// Handle ARHMM stream data
watch(arhmmStreamData, (data) => {
    if (data) {
        progress.value = data

        if (['completed', 'failed'].includes(data.status)) {
            stopARHMMStream()
            if (data.status === 'completed') {
                loadAnalysis()
                loadCrowdMovies()
            }
        }
    }
})

// Handle crowd movie stream data
watch(crowdMovieStreamData, (data) => {
    if (data) {
        crowdMovieProgress.value = data

        if (['completed', 'failed'].includes(data.status)) {
            stopCrowdMovieStream()
            if (data.status === 'completed') {
                loadCrowdMovies()
            }
        }
    }
})
```

**Step 4: Update loadStatus**

```typescript
async function loadStatus() {
    if (!projectId.value) return
    isLoading.value = true
    try {
        progress.value = await getARHMMStatus(projectId.value)
        if (progress.value.is_running) {
            startARHMMStream()
        }
        if (progress.value.status === 'completed') {
            loadAnalysis()
            loadCrowdMovies()
        }
    } catch (e) {
        console.error('Failed to load ARHMM status:', e)
    } finally {
        isLoading.value = false
    }
}
```

**Step 5: Update handleStart**

Replace `connectToStream()` with `startARHMMStream()`:
```typescript
async function handleStart() {
    if (!projectId.value) return
    actionError.value = null
    try {
        await createARHMMTraining(projectId.value)
        await new Promise(r => setTimeout(r, 300))
        await loadStatus()
        startARHMMStream()
    } catch (e: any) {
        actionError.value = e.message || 'Failed to start ARHMM'
    }
}
```

**Step 6: Update handleStartCrowdMovies**

Find the function that starts crowd movie generation and use `startCrowdMovieStream()` instead of inline EventSource creation.

**Step 7: Update onUnmounted**

```typescript
onUnmounted(() => {
    durationChart?.destroy()
    frequencyChart?.destroy()
})
```

**Step 8: Update project change watcher**

```typescript
watch(() => projectId.value, () => {
    stopARHMMStream()
    stopCrowdMovieStream()
    loadStatus()
})
```

**Step 9: Verify it compiles**

Run: `cd frontend && npm run type-check`
Expected: No errors

**Step 10: Commit**

```bash
git add frontend/src/components/ARHMMView.vue
git commit -m "$(cat <<'EOF'
refactor(frontend): use useSSEStream in ARHMMView

Replaces inline EventSource logic for both ARHMM and crowd movie streams.
Removes ~60 lines of duplicated SSE handling code.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Final verification

**Step 1: Run full type check**

Run: `cd frontend && npm run type-check`
Expected: No errors

**Step 2: Run dev server and smoke test**

Run: `cd frontend && npm run dev`

Manual verification:
1. Open a project with videos
2. Click on a video - verify VideoDetail loads
3. Navigate to cropped video view - verify CroppedVideoDetail loads
4. Navigate to aligned video view - verify AlignedVideoDetail loads
5. Go to detector training - verify SSE stream works when training
6. Go to alignment view - verify both training and apply SSE streams work

**Step 3: Update overview doc**

Mark Phase 1 as complete in `docs/plans/2026-02-04-frontend-refactor-overview.md`.

**Step 4: Final commit**

```bash
git add docs/plans/
git commit -m "$(cat <<'EOF'
docs: mark Phase 1 frontend refactor complete

Created composables:
- useVideo: video metadata fetching (used by 3 components)
- useSSEStream: SSE connection management (used by 3 components)

Added to api.ts:
- 5 connect*Stream() functions

Refactored components:
- VideoDetail.vue
- AlignedVideoDetail.vue
- CroppedVideoDetail.vue
- DetectorTraining.vue
- Alignment.vue
- ARHMMView.vue

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Summary

| Task | Description | Files |
|------|-------------|-------|
| 1 | Create useVideo composable | +useVideo.ts |
| 2 | Refactor VideoDetail | ~VideoDetail.vue |
| 3 | Refactor AlignedVideoDetail | ~AlignedVideoDetail.vue |
| 4 | Refactor CroppedVideoDetail | ~CroppedVideoDetail.vue |
| 5 | Add SSE connect functions | ~api.ts |
| 6 | Create useSSEStream composable | +useSSEStream.ts |
| 7 | Refactor DetectorTraining | ~DetectorTraining.vue |
| 8 | Refactor Alignment | ~Alignment.vue |
| 9 | Refactor ARHMMView | ~ARHMMView.vue |
| 10 | Final verification | docs |

**Net result:** ~120 lines added (composables + api), ~250 lines removed from components.
