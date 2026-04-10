<script setup lang="ts">
import { ref, computed, watch, onUnmounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import {
  getAssociatedVideo,
  getVideoStreamUrl,
  getTrackerMaskBbox,
  getTrackerMaskBboxes,
  resetAssociatedSegmentation,
  type BboxResult,
  type Video,
} from '@/services/api'
import { useVideoPlayback } from '@/composables/useVideoPlayback'
import { useSegmentation } from '@/composables/useSegmentation'
import { useVideo } from '@/composables/useVideo'
import VideoTimeline from './VideoTimeline.vue'
import TimelineSystem from './TimelineSystem.vue'
import DataTrack from './DataTrack.vue'
import VideoOverlay from './VideoOverlay.vue'

const route = useRoute()
const router = useRouter()

const projectId = computed(() => Number(route.params.id))
const mainVideoId = computed(() => Number(route.params.videoId))

// Load the main video metadata (for the header name)
const { video: mainVideo } = useVideo(mainVideoId, projectId)

const associatedVideo = ref<Video | null>(null)
const isLoading = ref(true)
const error = ref<string | null>(null)

// View modes - spec requires three: main_bbox, tracker_mask, final_mask
type ViewMode = 'main_bbox' | 'tracker_mask' | 'final_mask'
const viewMode = ref<ViewMode>('final_mask')

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

const viewStart = ref(0)
const viewEnd = ref(0)

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

watch(duration, (d) => {
  viewStart.value = 0
  viewEnd.value = d
})

// Calculate current frame index from currentTime and fps
const currentFrameIdx = computed(() => {
  if (!associatedVideo.value?.fps) return 0
  return Math.floor(currentTime.value * associatedVideo.value.fps)
})

const videoStreamUrl = computed(() => {
  if (!projectId.value || !associatedVideo.value) return ''
  return getVideoStreamUrl(projectId.value, associatedVideo.value.id)
})

const {
    currentMask,
    seekToFrame,
    loadFrameData,
    clearMaskCache,
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

const handleSeek = (time: number) => {
  seek(time)
  if (viewMode.value !== 'main_bbox') {
      const frameIdx = Math.floor(time * (associatedVideo.value?.fps ?? 30))
      seekToFrame(frameIdx)
  }
}

const handleViewChange = (start: number, end: number) => {
  viewStart.value = start
  viewEnd.value = end
}

const handleBack = () => {
  router.push(`/project/${projectId.value}`)
}

const isResetting = ref(false)

const handleReset = async () => {
    if (!projectId.value || !associatedVideo.value || isResetting.value) return
    isResetting.value = true
    try {
        await resetAssociatedSegmentation(projectId.value, associatedVideo.value.id)
        // Reload associated video metadata to get updated segmentation_status
        await loadAssociatedVideo()
        // Clear all caches so stale masks disappear
        clearMaskCache()
        currentMask.value = null
        bboxCache.clear()
        bboxPrefetchedUpTo = -1
        currentBbox.value = null
    } catch (e: any) {
        alert(e.message || 'Reset failed')
    } finally {
        isResetting.value = false
    }
}

// Load associated video on mount via watcher (mainVideoId may resolve asynchronously)
const loadAssociatedVideo = async () => {
  if (!projectId.value || !mainVideoId.value) return
  isLoading.value = true
  error.value = null
  try {
    associatedVideo.value = await getAssociatedVideo(
      projectId.value,
      mainVideoId.value,
    )
  } catch (e: any) {
    error.value = e.message || 'Failed to load associated video'
  } finally {
    isLoading.value = false
  }
}

// Trigger load when IDs are available
watch([projectId, mainVideoId], loadAssociatedVideo, { immediate: true })
</script>

<template>
  <div class="associated-video-detail-container">
    <div class="video-content">
      <div class="video-header">
        <button class="back-button" @click="handleBack">← Back to Videos</button>
        <h3 v-if="associatedVideo" class="video-title">
          {{ associatedVideo.name }} (Associated)
        </h3>
      </div>

      <div class="video-area">
        <div v-if="isLoading" class="loading-state">
          Loading video...
        </div>
        <div v-else-if="error" class="error-state">
          {{ error }}
        </div>
        <div v-else-if="associatedVideo" class="video-with-timeline">
          <div class="video-container">
            <div class="video-wrapper">
              <video
                ref="videoRef"
                class="video-player"
                :src="videoStreamUrl"
                preload="metadata"
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
          <TimelineSystem>
            <VideoTimeline
              :current-time="currentTime"
              :duration="duration"
              :is-playing="isPlaying"
              :fps="associatedVideo.fps"
              :external-view-start="viewStart"
              :external-view-end="viewEnd"
              @seek="handleSeek"
              @toggle-play="handleTogglePlay"
              @view-change="handleViewChange"
            />
            <DataTrack
              :duration="duration"
              :fps="associatedVideo.fps"
              :current-time="currentTime"
              :view-start="viewStart"
              :view-end="viewEnd"
              :tracker-masked-ranges="[]"
              :training-ranges="[]"
              :show-tracker-masked-frames="false"
              :show-training-frames="false"
              :show-confidence-plot="false"
              :is-marking-mode="false"
              :confidence-scores="[]"
              @view-change="handleViewChange"
            />
          </TimelineSystem>
        </div>
      </div>
    </div>

    <aside class="action-bar">
      <div class="action-bar-content">
        <p class="action-bar-info">
          This is the associated video view. It shows a secondary camera angle
          linked to the main video.
        </p>

        <h4 class="action-bar-title">View Mode</h4>
        <div class="view-mode-selector">
          <label class="view-mode-option">
            <input type="radio" v-model="viewMode" value="main_bbox" />
            <span>Main BBox</span>
          </label>
          <label class="view-mode-option">
            <input type="radio" v-model="viewMode" value="tracker_mask" />
            <span>Tracker Mask</span>
          </label>
          <label class="view-mode-option">
            <input type="radio" v-model="viewMode" value="final_mask" />
            <span>Final Mask</span>
          </label>
        </div>

        <template v-if="associatedVideo">
          <h4 class="action-bar-title">Info</h4>
          <p class="frame-indicator">Frame: {{ currentFrameIdx }}</p>
          <p class="video-info">FPS: {{ associatedVideo.fps }}</p>
          <p v-if="mainVideo" class="video-info">
            Main video: {{ mainVideo.name }}
          </p>
        </template>

        <template v-if="associatedVideo">
          <h4 class="action-bar-title">Actions</h4>
          <button
            class="reset-button"
            :disabled="!associatedVideo.segmentation_status || isResetting"
            @click="handleReset"
          >
            {{ isResetting ? 'Resetting...' : 'Reset Segmentation' }}
          </button>
        </template>
      </div>
    </aside>
  </div>
</template>

<style scoped>
.associated-video-detail-container {
  display: flex;
  flex: 1;
  min-height: 0;
}

.video-content {
  display: flex;
  flex-direction: column;
  flex: 1;
  min-height: 0;
  min-width: 0;
}

.video-header {
  display: flex;
  align-items: center;
  gap: 1rem;
  padding: 1rem;
  border-bottom: 1px solid #e0e0e0;
  flex-shrink: 0;
}

.back-button {
  padding: 0.5rem 1rem;
  border: 1px solid #ccc;
  border-radius: 4px;
  background-color: #f8f8f8;
  cursor: pointer;
  font-size: 0.9rem;
}

.back-button:hover {
  background-color: #e8e8e8;
}

.video-title {
  margin: 0;
  font-size: 1.1rem;
  font-weight: 600;
}

.video-area {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  background-color: #c2c2c2;
  min-height: 0;
  padding: 1rem;
}

.video-with-timeline {
  display: flex;
  flex-direction: column;
  align-items: center;
  width: 100%;
  max-height: 100%;
  min-height: 0;
  gap: 8px;
}

.video-container {
  position: relative;
  max-width: 100%;
  max-height: calc(100% - 80px);
  flex-shrink: 1;
  flex: 1;
  min-height: 0;
  display: flex;
  align-items: center;
  justify-content: center;
}

.video-wrapper {
  position: relative;
  display: inline-block;
  max-width: 100%;
  max-height: 100%;
}

.video-player {
  display: block;
  max-width: 100%;
  max-height: 100%;
  width: auto;
  height: auto;
}

.loading-state,
.error-state {
  color: #999;
  font-size: 1.1rem;
}

.error-state {
  color: #e57373;
}

.action-bar {
  width: 250px;
  flex-shrink: 0;
  background-color: #f8f8f8;
  border-left: 1px solid #e0e0e0;
}

.action-bar-content {
  padding: 1rem;
}

.action-bar-info {
  margin: 0;
  font-size: 0.9rem;
  color: #666;
  line-height: 1.5;
}

.action-bar-title {
  margin: 1.5rem 0 0.5rem 0;
  font-size: 0.85rem;
  font-weight: 700;
  color: #888;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.view-mode-selector {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.view-mode-option {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  cursor: pointer;
  font-size: 0.85rem;
  color: #444;
}

.view-mode-option input[type="radio"] {
  cursor: pointer;
}

.frame-indicator {
  margin: 0 0 0.25rem 0;
  font-size: 0.85rem;
  color: #666;
}

.video-info {
  margin: 0 0 0.25rem 0;
  font-size: 0.85rem;
  color: #666;
}

.reset-button {
  width: 100%;
  padding: 0.5rem;
  border: 1px solid #e57373;
  border-radius: 4px;
  background-color: #fff;
  color: #e57373;
  cursor: pointer;
  font-size: 0.85rem;
}

.reset-button:hover:not(:disabled) {
  background-color: #ffebee;
}

.reset-button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
</style>
