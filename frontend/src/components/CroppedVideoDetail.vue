<script setup lang="ts">
import { ref, onMounted, computed, watch, nextTick } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import {
  getCroppedVideoStreamUrl,
  getVideoAlignmentLabels,
  saveAlignmentLabel,
  deleteAlignmentLabel,
  deleteVideoAlignmentLabels,
  propagateKeypoints,
  getKeypointLabeledRanges,
  type KeypointLabeledRangesResponse,
} from '@/services/api'
import { useVideoPlayback } from '@/composables/useVideoPlayback'
import { useVideo } from '@/composables/useVideo'
import { useKeypointSession } from '@/composables/useKeypointSession'
import { useKeypointTracking } from '@/composables/useKeypointTracking'
import TimelineSystem from './TimelineSystem.vue'
import VideoTimeline from './VideoTimeline.vue'
import DataTrack from './DataTrack.vue'

const route = useRoute()
const router = useRouter()

const projectId = computed(() => Number(route.params.id))
const videoId = computed(() => Number(route.params.videoId))

const { video, isLoading, error, refresh: refreshVideo } = useVideo(videoId, projectId)

const viewStart = ref(0)
const viewEnd = ref(0)

const overlayCanvasRef = ref<HTMLCanvasElement | null>(null)

// Training mode state (alignment labels)
const isTrainingMode = ref(false)
const frontPoint = ref<{ x: number; y: number } | null>(null)
const rearPoint = ref<{ x: number; y: number } | null>(null)
const alignmentLabelFrames = ref<number[]>([])
const isSaving = ref(false)

// Keypoint tracking mode state
const isKeypointMode = ref(false)
const isPropagating = ref(false)
const propagationProgress = ref('')
const keypointLabeledRanges = ref<[number, number][]>([])

const croppedVideoStreamUrl = computed(() => {
  if (!projectId.value || !videoId.value) return ''
  return getCroppedVideoStreamUrl(projectId.value, videoId.value)
})

// Calculate current frame index from currentTime and fps
const currentFrameIdx = computed(() => {
  if (!video.value?.fps) return 0
  return Math.floor(currentTime.value * video.value.fps)
})

// Keypoint session and tracking composables
const { keypointStatus, isReady, statusText, sessionInitialized } = useKeypointSession(projectId, videoId)
const {
  activeTool,
  currentKeypoints,
  isPrompting,
  handlePointComplete,
  handleResetFrame: handleKeypointResetFrame,
  handleResetVideo: handleKeypointResetVideo,
  loadFrameKeypoints,
  prefetchKeypoints,
  clearCache: clearKeypointCache,
  toggleFrontTool,
  toggleRearTool,
} = useKeypointTracking(projectId, videoId, currentFrameIdx)

const loadExtraData = async () => {
  if (!video.value) return
  try {
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

// Load keypoint labeled ranges
async function loadKeypointLabeledRanges() {
  if (!projectId.value || !videoId.value) return
  try {
    const resp = await getKeypointLabeledRanges(projectId.value, videoId.value)
    keypointLabeledRanges.value = resp.labeled_ranges
  } catch (e) {
    console.error('Failed to load keypoint labeled ranges:', e)
  }
}

// Load extra data when video loads
watch(video, (v) => {
  if (v) {
    loadExtraData()
    loadKeypointLabeledRanges()
  }
})

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

// Render overlay: training points AND/OR keypoint tracking points
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

  const radius = 10

  // Draw alignment training points
  if (isTrainingMode.value) {
    if (frontPoint.value) {
      const px = frontPoint.value.x * canvas.width
      const py = frontPoint.value.y * canvas.height
      ctx.beginPath()
      ctx.arc(px, py, radius, 0, 2 * Math.PI)
      ctx.fillStyle = 'rgba(34, 197, 94, 0.8)' // Green
      ctx.fill()
      ctx.strokeStyle = 'white'
      ctx.lineWidth = 2
      ctx.stroke()
    }

    if (rearPoint.value) {
      const px = rearPoint.value.x * canvas.width
      const py = rearPoint.value.y * canvas.height
      ctx.beginPath()
      ctx.arc(px, py, radius, 0, 2 * Math.PI)
      ctx.fillStyle = 'rgba(239, 68, 68, 0.8)' // Red
      ctx.fill()
      ctx.strokeStyle = 'white'
      ctx.lineWidth = 2
      ctx.stroke()
    }
  }

  // Draw keypoint tracking points
  if (isKeypointMode.value && currentKeypoints.value) {
    const kp = currentKeypoints.value
    if (kp.front_x !== null && kp.front_y !== null) {
      const px = kp.front_x * canvas.width
      const py = kp.front_y * canvas.height
      // Crosshair
      ctx.strokeStyle = '#22c55e'
      ctx.lineWidth = 1.5
      ctx.beginPath()
      ctx.moveTo(px - 14, py)
      ctx.lineTo(px + 14, py)
      ctx.moveTo(px, py - 14)
      ctx.lineTo(px, py + 14)
      ctx.stroke()
      // Circle
      ctx.beginPath()
      ctx.arc(px, py, radius, 0, 2 * Math.PI)
      ctx.fillStyle = 'rgba(34, 197, 94, 0.7)'
      ctx.fill()
      ctx.strokeStyle = 'white'
      ctx.lineWidth = 2
      ctx.stroke()
    }
    if (kp.rear_x !== null && kp.rear_y !== null) {
      const px = kp.rear_x * canvas.width
      const py = kp.rear_y * canvas.height
      // Crosshair
      ctx.strokeStyle = '#ef4444'
      ctx.lineWidth = 1.5
      ctx.beginPath()
      ctx.moveTo(px - 14, py)
      ctx.lineTo(px + 14, py)
      ctx.moveTo(px, py - 14)
      ctx.lineTo(px, py + 14)
      ctx.stroke()
      // Circle
      ctx.beginPath()
      ctx.arc(px, py, radius, 0, 2 * Math.PI)
      ctx.fillStyle = 'rgba(239, 68, 68, 0.7)'
      ctx.fill()
      ctx.strokeStyle = 'white'
      ctx.lineWidth = 2
      ctx.stroke()
    }
  }
}

// Re-render overlay when keypoints change
watch(currentKeypoints, () => renderOverlay(), { deep: true })

// Toggle training mode (alignment labels)
async function toggleTrainingMode() {
  isTrainingMode.value = !isTrainingMode.value
  if (isTrainingMode.value) isKeypointMode.value = false
  frontPoint.value = null
  rearPoint.value = null
  // Wait for canvas to be rendered in DOM before setting dimensions
  await nextTick()
  renderOverlay()
}

// Toggle keypoint tracking mode
async function toggleKeypointMode() {
  isKeypointMode.value = !isKeypointMode.value
  if (isKeypointMode.value) {
    isTrainingMode.value = false
    frontPoint.value = null
    rearPoint.value = null
    // Load keypoints for current frame
    await loadFrameKeypoints(currentFrameIdx.value)
  }
  await nextTick()
  renderOverlay()
}

// Handle overlay click — routes to appropriate mode handler
function handleOverlayClick(event: MouseEvent) {
  const canvas = overlayCanvasRef.value
  const videoEl = videoRef.value
  if (!canvas || !videoEl) return

  const rect = canvas.getBoundingClientRect()
  const x = (event.clientX - rect.left) / rect.width
  const y = (event.clientY - rect.top) / rect.height

  if (isTrainingMode.value && !isSaving.value) {
    // Alignment training click handler
    if (!frontPoint.value) {
      frontPoint.value = { x, y }
      renderOverlay()
    } else if (!rearPoint.value) {
      rearPoint.value = { x, y }
      renderOverlay()
      saveAndAdvance()
    }
  } else if (isKeypointMode.value && activeTool.value !== 'none' && !isPrompting.value) {
    // Keypoint tracking click handler
    handlePointComplete({ x, y })
  }
}

// Save alignment label and advance to next frame
async function saveAndAdvance() {
  if (!frontPoint.value || !rearPoint.value) return
  if (!video.value) return

  isSaving.value = true
  try {
    await saveAlignmentLabel(
      projectId.value,
      videoId.value,
      currentFrameIdx.value,
      frontPoint.value.x,
      frontPoint.value.y,
      rearPoint.value.x,
      rearPoint.value.y
    )

    // Add to local list if not already present
    if (!alignmentLabelFrames.value.includes(currentFrameIdx.value)) {
      alignmentLabelFrames.value = [...alignmentLabelFrames.value, currentFrameIdx.value].sort(
        (a, b) => a - b
      )
    }

    // Clear points
    frontPoint.value = null
    rearPoint.value = null

    // Advance to next frame
    const nextFrame = currentFrameIdx.value + 1
    const nextTime = (nextFrame + 0.5) / video.value.fps
    seek(nextTime)
  } catch (e) {
    console.error('Failed to save alignment label:', e)
  } finally {
    isSaving.value = false
    renderOverlay()
  }
}

// Reset current frame's alignment label
async function handleResetFrame() {
  try {
    await deleteAlignmentLabel(projectId.value, videoId.value, currentFrameIdx.value)
    alignmentLabelFrames.value = alignmentLabelFrames.value.filter(
      (f) => f !== currentFrameIdx.value
    )
  } catch (e) {
    console.error('Failed to reset frame:', e)
  }
}

// Reset all alignment labels for this video
async function handleResetVideo() {
  if (!confirm('Delete all alignment labels for this video?')) return
  try {
    await deleteVideoAlignmentLabels(projectId.value, videoId.value)
    alignmentLabelFrames.value = []
  } catch (e) {
    console.error('Failed to reset video:', e)
  }
}

// Propagate keypoints
async function handlePropagate() {
  if (!video.value || isPropagating.value) return
  isPropagating.value = true
  propagationProgress.value = 'Starting...'
  try {
    const result = await propagateKeypoints(
      projectId.value,
      videoId.value,
      currentFrameIdx.value,
      video.value.num_frames
    )
    propagationProgress.value = `Done: ${result.frames_processed} frames`
    // Refresh keypoints and labeled ranges
    clearKeypointCache()
    await loadFrameKeypoints(currentFrameIdx.value)
    await loadKeypointLabeledRanges()
  } catch (e) {
    console.error('Failed to propagate keypoints:', e)
    propagationProgress.value = 'Error'
  } finally {
    isPropagating.value = false
  }
}

// Reset keypoint frame wrapper (also refreshes labeled ranges)
async function onKeypointResetFrame() {
  await handleKeypointResetFrame()
  await loadKeypointLabeledRanges()
  renderOverlay()
}

// Reset keypoint video wrapper (also refreshes labeled ranges)
async function onKeypointResetVideo() {
  if (!confirm('Delete all keypoint tracking data for this video?')) return
  await handleKeypointResetVideo()
  await loadKeypointLabeledRanges()
  renderOverlay()
}

// Discard partial alignment label (front without rear) when navigating away
watch(currentFrameIdx, (newFrame, oldFrame) => {
  if (isTrainingMode.value && frontPoint.value && !rearPoint.value && newFrame !== oldFrame) {
    frontPoint.value = null
    renderOverlay()
  }
})

// Whether the canvas overlay should be visible
const showOverlay = computed(() => isTrainingMode.value || isKeypointMode.value)

// Whether the overlay should accept pointer events (crosshair cursor)
const overlayInteractive = computed(() => {
  if (isTrainingMode.value) return true
  if (isKeypointMode.value && activeTool.value !== 'none') return true
  return false
})

// Has any conditioning frames (for propagation button)
const hasConditioningFrames = computed(() => {
  const kp = currentKeypoints.value
  if (!kp) return false
  // Check if any frame in cache has non-null keypoints (simple heuristic)
  // More accurately: user has placed at least one keypoint somewhere
  return keypointLabeledRanges.value.length > 0 ||
    (kp.front_x !== null || kp.rear_x !== null)
})

onMounted(() => {
  // Handle query params from AlignedVideoDetail navigation
  if (route.query.training === 'true') {
    isTrainingMode.value = true
  }
  if (route.query.frame !== undefined && video.value?.fps) {
    const frameIdx = parseInt(route.query.frame as string, 10)
    if (!isNaN(frameIdx)) {
      const targetTime = (frameIdx + 0.5) / video.value.fps
      // Wait for video to be ready before seeking
      setTimeout(() => seek(targetTime), 100)
    }
  }
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
        <div v-else-if="video" class="video-with-timeline">
          <div class="video-container">
            <div class="video-wrapper">
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
                v-if="showOverlay"
                ref="overlayCanvasRef"
                class="overlay-canvas"
                :class="{ interactive: overlayInteractive }"
                @click="handleOverlayClick"
              />
            </div>
          </div>
          <TimelineSystem>
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
            <DataTrack
              :duration="duration"
              :fps="video!.fps"
              :current-time="currentTime"
              :view-start="viewStart"
              :view-end="viewEnd"
              :tracker-masked-ranges="keypointLabeledRanges"
              :training-ranges="[]"
              :show-tracker-masked-frames="isKeypointMode"
              :show-training-frames="false"
              :show-confidence-plot="false"
              :is-marking-mode="false"
              :confidence-scores="[]"
              :alignment-label-frames="alignmentLabelFrames"
              :show-alignment-labels="true"
              @view-change="handleViewChange"
            />
          </TimelineSystem>
        </div>
      </div>
    </div>

    <aside class="action-bar">
      <div class="action-bar-content">
        <h4 class="action-bar-title">Alignment Training</h4>
        <button
          class="training-toggle-button"
          :class="{ active: isTrainingMode }"
          @click="toggleTrainingMode"
        >
          {{ isTrainingMode ? 'Stop Training' : 'Start Training' }}
        </button>

        <template v-if="isTrainingMode">
          <div class="training-instructions">
            <p class="instruction-step">
              <span class="step-number">1</span>
              Click <span class="front-text">front</span> of animal
            </p>
            <p class="instruction-step">
              <span class="step-number">2</span>
              Click <span class="rear-text">rear</span> of animal
            </p>
            <p class="instruction-note">Auto-saves and advances</p>
          </div>

          <div class="current-state">
            <p class="frame-indicator">Frame: {{ currentFrameIdx }}</p>
            <p v-if="frontPoint && !rearPoint" class="waiting-indicator">
              Waiting for rear click...
            </p>
            <p v-if="isSaving" class="saving-indicator">Saving...</p>
          </div>

          <div class="training-actions">
            <button
              class="reset-button"
              :disabled="!alignmentLabelFrames.includes(currentFrameIdx)"
              @click="handleResetFrame"
            >
              Reset Frame
            </button>
            <button
              class="reset-button danger"
              :disabled="alignmentLabelFrames.length === 0"
              @click="handleResetVideo"
            >
              Reset Video
            </button>
          </div>
        </template>

        <div class="label-count">
          <span class="count-label">Labels (this video):</span>
          <span class="count-value">{{ alignmentLabelFrames.length }}</span>
        </div>

        <h4 class="action-bar-title">Keypoint Tracking</h4>
        <button
          class="training-toggle-button"
          :class="{ active: isKeypointMode }"
          :disabled="!isReady && !isKeypointMode"
          @click="toggleKeypointMode"
        >
          {{ isKeypointMode ? 'Stop Tracking' : 'Start Tracking' }}
        </button>

        <template v-if="isKeypointMode">
          <div class="keypoint-tools">
            <button
              class="tool-button front-tool"
              :class="{ active: activeTool === 'front' }"
              :disabled="isPrompting"
              @click="toggleFrontTool"
            >
              Place Front
            </button>
            <button
              class="tool-button rear-tool"
              :class="{ active: activeTool === 'rear' }"
              :disabled="isPrompting"
              @click="toggleRearTool"
            >
              Place Rear
            </button>
          </div>

          <div class="current-state">
            <p class="frame-indicator">Frame: {{ currentFrameIdx }}</p>
            <p v-if="isPrompting" class="saving-indicator">Processing...</p>
            <p v-if="currentKeypoints" class="keypoint-info">
              <span v-if="currentKeypoints.front_x !== null" class="front-text">F </span>
              <span v-if="currentKeypoints.rear_x !== null" class="rear-text">R</span>
              <span v-if="currentKeypoints.front_x === null && currentKeypoints.rear_x === null" class="no-data-text">No keypoints</span>
            </p>
          </div>

          <div class="propagation-section">
            <button
              class="propagate-button"
              :disabled="isPropagating || !hasConditioningFrames"
              @click="handlePropagate"
            >
              {{ isPropagating ? 'Propagating...' : 'Propagate' }}
            </button>
            <p v-if="propagationProgress" class="propagation-progress">
              {{ propagationProgress }}
            </p>
          </div>

          <div class="training-actions">
            <button
              class="reset-button"
              :disabled="!currentKeypoints || (currentKeypoints.front_x === null && currentKeypoints.rear_x === null)"
              @click="onKeypointResetFrame"
            >
              Reset Frame
            </button>
            <button
              class="reset-button danger"
              @click="onKeypointResetVideo"
            >
              Reset Video
            </button>
          </div>
        </template>

        <div class="keypoint-status" :class="keypointStatus.status">
          {{ statusText }}
        </div>
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

.overlay-canvas.interactive {
  pointer-events: auto;
  cursor: crosshair;
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
  position: relative;
}

.action-bar-content {
  padding: 1rem;
  padding-bottom: 3rem;
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

.frame-indicator {
  margin: 0.5rem 0 0 0;
  font-size: 0.85rem;
  color: #666;
}

.action-bar-title:first-child {
  margin-top: 0;
}

.training-toggle-button {
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

.training-toggle-button:hover:not(:disabled) {
  background-color: #e8e8e8;
  border-color: #999;
}

.training-toggle-button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.training-toggle-button.active {
  background-color: #fef3c7;
  border-color: #f59e0b;
  color: #b45309;
}

.training-instructions {
  margin-top: 1rem;
  padding: 0.75rem;
  background-color: #f0f0f0;
  border-radius: 4px;
}

.instruction-step {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin: 0 0 0.5rem 0;
  font-size: 0.85rem;
}

.instruction-step:last-of-type {
  margin-bottom: 0;
}

.step-number {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 20px;
  height: 20px;
  border-radius: 50%;
  background-color: #666;
  color: white;
  font-size: 0.75rem;
  font-weight: 600;
}

.front-text {
  color: #22c55e;
  font-weight: 600;
}

.rear-text {
  color: #ef4444;
  font-weight: 600;
}

.no-data-text {
  color: #999;
  font-style: italic;
}

.instruction-note {
  margin: 0.5rem 0 0 0;
  font-size: 0.8rem;
  color: #888;
  font-style: italic;
}

.current-state {
  margin-top: 1rem;
  padding: 0.5rem 0;
  border-top: 1px solid #e0e0e0;
}

.waiting-indicator {
  margin: 0.25rem 0 0 0;
  font-size: 0.85rem;
  color: #f59e0b;
  font-weight: 500;
}

.saving-indicator {
  margin: 0.25rem 0 0 0;
  font-size: 0.85rem;
  color: #2196f3;
  font-weight: 500;
}

.training-actions {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  margin-top: 1rem;
}

.reset-button {
  padding: 0.5rem 0.75rem;
  border: 1px solid #ccc;
  border-radius: 4px;
  background-color: #f8f8f8;
  cursor: pointer;
  font-size: 0.85rem;
  transition: all 0.2s;
}

.reset-button:hover:not(:disabled) {
  background-color: #e8e8e8;
  border-color: #999;
}

.reset-button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.reset-button.danger {
  color: #dc2626;
  border-color: #fca5a5;
}

.reset-button.danger:hover:not(:disabled) {
  background-color: #fee2e2;
  border-color: #dc2626;
}

.label-count {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-top: 1rem;
  padding: 0.5rem 0;
  border-top: 1px solid #e0e0e0;
}

.count-label {
  font-size: 0.85rem;
  color: #666;
}

.count-value {
  font-size: 0.9rem;
  font-weight: 600;
  color: #a855f7;
}

/* Keypoint tracking specific styles */

.keypoint-tools {
  display: flex;
  gap: 0.5rem;
  margin-top: 1rem;
}

.tool-button {
  flex: 1;
  padding: 0.5rem;
  border: 2px solid #ccc;
  border-radius: 4px;
  background-color: #f8f8f8;
  cursor: pointer;
  font-size: 0.8rem;
  font-weight: 600;
  transition: all 0.2s;
}

.tool-button:hover:not(:disabled) {
  background-color: #e8e8e8;
}

.tool-button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.tool-button.front-tool.active {
  background-color: #dcfce7;
  border-color: #22c55e;
  color: #166534;
}

.tool-button.rear-tool.active {
  background-color: #fee2e2;
  border-color: #ef4444;
  color: #991b1b;
}

.keypoint-info {
  margin: 0.25rem 0 0 0;
  font-size: 0.85rem;
}

.propagation-section {
  margin-top: 1rem;
}

.propagate-button {
  width: 100%;
  padding: 0.75rem 1rem;
  border: 1px solid #3b82f6;
  border-radius: 4px;
  background-color: #eff6ff;
  color: #1d4ed8;
  cursor: pointer;
  font-size: 0.9rem;
  font-weight: 500;
  transition: all 0.2s;
}

.propagate-button:hover:not(:disabled) {
  background-color: #dbeafe;
  border-color: #2563eb;
}

.propagate-button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.propagation-progress {
  margin: 0.5rem 0 0 0;
  font-size: 0.8rem;
  color: #666;
}

.keypoint-status {
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  padding: 0.5rem 1rem;
  font-size: 0.75rem;
  text-align: center;
  border-top: 1px solid #e0e0e0;
}

.keypoint-status.not_loaded {
  background-color: #f5f5f5;
  color: #666;
}

.keypoint-status.loading_model {
  background-color: #fff3cd;
  color: #856404;
}

.keypoint-status.ready {
  background-color: #d4edda;
  color: #155724;
}

.keypoint-status.error {
  background-color: #f8d7da;
  color: #721c24;
}
</style>
