<script setup lang="ts">
import { ref, onMounted, computed, watch, nextTick } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import {
  getVideo,
  getCroppedVideoStreamUrl,
  hasAlignmentPredictions,
  getStoredAlignmentPredictionUrl,
  type Video,
} from '@/services/api'
import { useVideoPlayback } from '@/composables/useVideoPlayback'
import VideoTimeline from './VideoTimeline.vue'

const route = useRoute()
const router = useRouter()

const projectId = computed(() => Number(route.params.id))
const videoId = computed(() => Number(route.params.videoId))

const video = ref<Video | null>(null)
const isLoading = ref(true)
const error = ref<string | null>(null)

const viewStart = ref(0)
const viewEnd = ref(0)

// Heatmap overlay state
const hasStoredPredictions = ref(false)
const showHeatmap = ref(false)
const heatmapImage = ref<HTMLImageElement | null>(null)
const overlayCanvasRef = ref<HTMLCanvasElement | null>(null)
const lastLoadedFrameIdx = ref<number | null>(null)
const isLoadingHeatmap = ref(false)

const croppedVideoStreamUrl = computed(() => {
  if (!projectId.value || !videoId.value) return ''
  return getCroppedVideoStreamUrl(projectId.value, videoId.value)
})

// Calculate current frame index from currentTime and fps
const currentFrameIdx = computed(() => {
  if (!video.value?.fps) return 0
  return Math.floor(currentTime.value * video.value.fps)
})

const loadVideo = async () => {
  isLoading.value = true
  error.value = null

  try {
    video.value = await getVideo(projectId.value, videoId.value)
    // Check if stored predictions exist
    hasStoredPredictions.value = await hasAlignmentPredictions(
      projectId.value,
      videoId.value
    )
  } catch (e) {
    error.value = e instanceof Error ? e.message : 'Failed to load video'
  } finally {
    isLoading.value = false
  }
}

const handleBack = () => {
  router.push(`/project/${projectId.value}`)
}

const {
  videoRef,
  currentTime,
  duration,
  isPlaying,
  onTimeUpdate,
  onLoadedMetadata,
  onPlay,
  onPause,
  seek,
  togglePlay: handleTogglePlay,
} = useVideoPlayback()

watch(duration, (d) => {
  viewStart.value = 0
  viewEnd.value = d
})

const handleSeek = (time: number) => {
  seek(time)
}

const handleViewChange = (start: number, end: number) => {
  viewStart.value = start
  viewEnd.value = end
}

// Load heatmap for current frame
async function loadHeatmap(frameIdx: number) {
  if (!projectId.value || !videoId.value) return
  if (isLoadingHeatmap.value) return
  if (lastLoadedFrameIdx.value === frameIdx) return

  isLoadingHeatmap.value = true
  lastLoadedFrameIdx.value = frameIdx

  const url = getStoredAlignmentPredictionUrl(projectId.value, videoId.value, frameIdx)
  const img = new Image()
  img.crossOrigin = 'anonymous'
  img.onload = () => {
    heatmapImage.value = img
    isLoadingHeatmap.value = false
    renderOverlay()
  }
  img.onerror = () => {
    heatmapImage.value = null
    isLoadingHeatmap.value = false
  }
  img.src = url
}

// Render heatmap overlay
function renderOverlay() {
  const canvas = overlayCanvasRef.value
  const videoEl = videoRef.value
  if (!canvas || !videoEl) return

  // Match canvas size to video display size
  const rect = videoEl.getBoundingClientRect()
  canvas.width = rect.width
  canvas.height = rect.height

  const ctx = canvas.getContext('2d')
  if (!ctx) return

  ctx.clearRect(0, 0, canvas.width, canvas.height)

  if (!heatmapImage.value || !showHeatmap.value) return

  // Create a temporary canvas to read heatmap pixels
  const tempCanvas = document.createElement('canvas')
  tempCanvas.width = heatmapImage.value.width
  tempCanvas.height = heatmapImage.value.height
  const tempCtx = tempCanvas.getContext('2d')
  if (!tempCtx) return

  tempCtx.drawImage(heatmapImage.value, 0, 0)
  const imageData = tempCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height)
  const data = imageData.data

  // Create overlay image data
  const overlayData = ctx.createImageData(canvas.width, canvas.height)
  const scaleX = tempCanvas.width / canvas.width
  const scaleY = tempCanvas.height / canvas.height

  for (let y = 0; y < canvas.height; y++) {
    for (let x = 0; x < canvas.width; x++) {
      const srcX = Math.floor(x * scaleX)
      const srcY = Math.floor(y * scaleY)
      const srcIdx = (srcY * tempCanvas.width + srcX) * 4
      const dstIdx = (y * canvas.width + x) * 4

      const frontProb = (data[srcIdx] ?? 0) / 255     // R channel = front
      const rearProb = (data[srcIdx + 1] ?? 0) / 255  // G channel = rear

      // Blend: green for front, red for rear
      overlayData.data[dstIdx] = Math.floor(rearProb * 255)      // R
      overlayData.data[dstIdx + 1] = Math.floor(frontProb * 255) // G
      overlayData.data[dstIdx + 2] = 0                            // B
      overlayData.data[dstIdx + 3] = Math.floor(Math.max(frontProb, rearProb) * 150) // A
    }
  }
  ctx.putImageData(overlayData, 0, 0)
}

// Toggle heatmap display
function toggleHeatmap() {
  showHeatmap.value = !showHeatmap.value
  if (showHeatmap.value) {
    loadHeatmap(currentFrameIdx.value)
  } else {
    renderOverlay() // Clear overlay
  }
}

// Watch for frame changes when heatmap is enabled
watch(currentFrameIdx, (frameIdx) => {
  if (showHeatmap.value && hasStoredPredictions.value) {
    loadHeatmap(frameIdx)
  }
})

// Re-render overlay when video resizes
watch([() => videoRef.value?.videoWidth, () => videoRef.value?.videoHeight], () => {
  if (showHeatmap.value) {
    nextTick(() => renderOverlay())
  }
})

onMounted(async () => {
  await loadVideo()
})
</script>

<template>
  <div class="cropped-video-detail-container">
    <div class="video-content">
      <div class="video-header">
        <button class="back-button" @click="handleBack">← Back to Videos</button>
        <h3 v-if="video" class="video-title">{{ video.name }} (Cropped)</h3>
      </div>

      <div class="video-area">
        <div v-if="isLoading" class="loading-state">
          Loading video...
        </div>
        <div v-else-if="error" class="error-state">
          {{ error }}
        </div>
        <div v-else class="video-with-timeline">
          <div class="video-container">
            <video
              ref="videoRef"
              class="video-player"
              :src="croppedVideoStreamUrl"
              @timeupdate="onTimeUpdate"
              @loadedmetadata="onLoadedMetadata"
              @play="onPlay"
              @pause="onPause"
            >
              Your browser does not support the video tag.
            </video>
            <canvas
              v-if="showHeatmap"
              ref="overlayCanvasRef"
              class="overlay-canvas"
            />
          </div>
          <VideoTimeline
            :current-time="currentTime"
            :duration="duration"
            :is-playing="isPlaying"
            :fps="video!.fps"
            :external-view-start="viewStart"
            :external-view-end="viewEnd"
            @seek="handleSeek"
            @toggle-play="handleTogglePlay"
            @view-change="handleViewChange"
          />
        </div>
      </div>
    </div>

    <aside class="action-bar">
      <div class="action-bar-content">
        <p class="action-bar-info">This is the cropped video view. Use the timeline to navigate through frames.</p>

        <template v-if="hasStoredPredictions">
          <h4 class="action-bar-title">Alignment Debug</h4>
          <button
            class="heatmap-toggle-button"
            :class="{ active: showHeatmap }"
            @click="toggleHeatmap"
          >
            {{ showHeatmap ? 'Hide' : 'Show' }} Prediction Heatmap
          </button>
          <p v-if="showHeatmap" class="heatmap-legend">
            <span class="legend-front">Green = Front</span>
            <span class="legend-rear">Red = Rear</span>
          </p>
          <p v-if="showHeatmap" class="frame-indicator">
            Frame: {{ currentFrameIdx }}
          </p>
        </template>
      </div>
    </aside>
  </div>
</template>

<style scoped>
.cropped-video-detail-container {
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

.overlay-canvas {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  pointer-events: none;
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

.heatmap-toggle-button {
  width: 100%;
  padding: 0.75rem 1rem;
  border: 1px solid #ccc;
  border-radius: 4px;
  background-color: #f8f8f8;
  cursor: pointer;
  font-size: 0.9rem;
  font-weight: 500;
  transition: all 0.2s;
}

.heatmap-toggle-button:hover {
  background-color: #e8e8e8;
  border-color: #999;
}

.heatmap-toggle-button.active {
  background-color: #e3f2fd;
  border-color: #2196f3;
  color: #1976d2;
}

.heatmap-legend {
  margin: 0.75rem 0 0 0;
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
  font-size: 0.85rem;
}

.legend-front {
  color: #22c55e;
  font-weight: 500;
}

.legend-rear {
  color: #ef4444;
  font-weight: 500;
}

.frame-indicator {
  margin: 0.5rem 0 0 0;
  font-size: 0.85rem;
  color: #666;
}
</style>
