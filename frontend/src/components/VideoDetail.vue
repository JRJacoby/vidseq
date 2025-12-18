<script setup lang="ts">
import { ref, onMounted, computed, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { getVideo, getVideoStreamUrl, propagateMask, type Video } from '@/services/api'
import { useSegmentationSession } from '@/composables/useSegmentationSession'
import { useVideoPlayback } from '@/composables/useVideoPlayback'
import { useSegmentation } from '@/composables/useSegmentation'
import { useFrameRanges } from '@/composables/useFrameRanges'
import VideoTimeline from './VideoTimeline.vue'
import VideoOverlay from './VideoOverlay.vue'
import DataTrack from './DataTrack.vue'

const route = useRoute()
const router = useRouter()

const projectId = computed(() => Number(route.params.id))
const videoId = computed(() => Number(route.params.videoId))

const video = ref<Video | null>(null)
const isLoading = ref(true)
const error = ref<string | null>(null)
const showMask = ref(true)
const showPrompts = ref(true)

const showMaskedFrames = ref(true)
const showTrainingFrames = ref(true)
const isMarkingMode = ref(false)
const maxFrames = ref(1000)

const viewStart = ref(0)
const viewEnd = ref(0)

const videoStreamUrl = computed(() => {
  if (!projectId.value || !videoId.value) return ''
  return getVideoStreamUrl(projectId.value, videoId.value)
})

const loadVideo = async () => {
  isLoading.value = true
  error.value = null

  try {
    video.value = await getVideo(projectId.value, videoId.value)
  } catch (e) {
    error.value = e instanceof Error ? e.message : 'Failed to load video'
  } finally {
    isLoading.value = false
  }
}

const handleBack = () => {
  router.push(`/project/${projectId.value}`)
}

const { segmentationStatus, isReady: segmentationIsReady, statusText: segmentationStatusText } = useSegmentationSession(projectId, videoId)

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

const onSeeked = () => {
}

const currentFrameIdx = computed(() => {
  if (!video.value) return 0
  return Math.floor(currentTime.value * video.value.fps)
})

const fps = computed(() => video.value?.fps ?? 30)

const {
  activeTool,
  currentMask,
  currentBbox,
  currentPrompts,
  isSegmenting,
  loadFrameData,
  seekToFrame,
  togglePositivePointTool,
  toggleNegativePointTool,
  handlePointComplete,
  handleResetFrame,
  handleResetVideo,
  clearMaskCache,
} = useSegmentation(projectId, videoId, currentFrameIdx, isPlaying, videoRef, fps)

const {
  maskedRanges,
  trainingRanges,
  refresh: refreshFrameRanges,
  markTraining,
  unmarkTraining,
  validateRange,
} = useFrameRanges(projectId, videoId)

const isPropagating = ref(false)

const handlePropagateMask = async () => {
  if (!projectId.value || !videoId.value) return
  
  isPropagating.value = true
  try {
    await propagateMask(
      projectId.value,
      videoId.value,
      currentFrameIdx.value,
      maxFrames.value
    )
    clearMaskCache()
    await loadFrameData(currentFrameIdx.value)
    await refreshFrameRanges()
  } catch (e) {
    console.error('Failed to propagate mask:', e)
    alert(e instanceof Error ? e.message : 'Failed to propagate mask')
  } finally {
    isPropagating.value = false
  }
}

const handleSeek = (time: number) => {
  seek(time)
  if (video.value) {
    const targetFrame = Math.floor(time * video.value.fps)
    seekToFrame(targetFrame)
  }
}

const handleViewChange = (start: number, end: number) => {
  viewStart.value = start
  viewEnd.value = end
}

const handleMarkTraining = async (startFrame: number, endFrame: number) => {
  const validation = await validateRange(startFrame, endFrame)
  if (!validation.valid) {
    const missingCount = validation.missingFrames?.length ?? 0
    alert(`Cannot mark as training: ${missingCount} frame(s) are missing masks.\nMissing frames: ${validation.missingFrames?.slice(0, 10).join(', ')}${missingCount > 10 ? '...' : ''}`)
    return
  }
  
  const result = await markTraining(startFrame, endFrame)
  if (!result.success) {
    alert(`Failed to mark training: ${result.error}`)
  }
}

const handleUnmarkTraining = async (startFrame: number, endFrame: number) => {
  await unmarkTraining(startFrame, endFrame)
}

const handleResetFrameWithRefresh = async () => {
  await handleResetFrame()
  await refreshFrameRanges()
}

const handleResetVideoWithRefresh = async () => {
  await handleResetVideo()
  await refreshFrameRanges()
}

setMetadataCallback(() => {
  loadFrameData(0)
})

onMounted(() => {
  loadVideo()
})
</script>

<template>
  <div class="video-detail-container">
    <div class="video-content">
      <div class="video-header">
        <button class="back-button" @click="handleBack">← Back to Videos</button>
        <h3 v-if="video" class="video-title">{{ video.name }}</h3>
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
              :src="videoStreamUrl"
              @timeupdate="onTimeUpdate"
              @loadedmetadata="onLoadedMetadata"
              @play="onPlay"
              @pause="onPause"
              @seeked="onSeeked"
            >
              Your browser does not support the video tag.
            </video>
            <VideoOverlay
              v-if="videoWidth > 0 && videoHeight > 0"
              :video-width="videoWidth"
              :video-height="videoHeight"
              :active-tool="activeTool"
              :mask="currentMask"
              :bbox="currentBbox"
              :prompts="currentPrompts"
              :show-mask="showMask"
              :show-bbox="true"
              :show-prompts="showPrompts"
              @point-complete="handlePointComplete"
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
          <DataTrack
            :duration="duration"
            :fps="video!.fps"
            :current-time="currentTime"
            :view-start="viewStart"
            :view-end="viewEnd"
            :masked-ranges="maskedRanges"
            :training-ranges="trainingRanges"
            :show-masked-frames="showMaskedFrames"
            :show-training-frames="showTrainingFrames"
            :is-marking-mode="isMarkingMode"
            @mark-training="handleMarkTraining"
            @unmark-training="handleUnmarkTraining"
          />
        </div>
      </div>
    </div>

    <aside class="action-bar">
      <div class="action-bar-content">
        <h4 class="action-bar-title">Segmentation Tools</h4>
        <div class="tool-buttons">
          <button
            class="tool-button positive-point"
            :class="{ active: activeTool === 'positive_point' }"
            @click="togglePositivePointTool"
            :disabled="isSegmenting || !segmentationIsReady"
          >
            <span class="tool-icon">⊕</span>
            <span class="tool-label">Positive Point</span>
          </button>
          <button
            class="tool-button negative-point"
            :class="{ active: activeTool === 'negative_point' }"
            @click="toggleNegativePointTool"
            :disabled="isSegmenting || !segmentationIsReady"
          >
            <span class="tool-icon">⊖</span>
            <span class="tool-label">Negative Point</span>
          </button>
          <button
            class="tool-button reset-button"
            @click="handleResetFrameWithRefresh"
            :disabled="isSegmenting || isPropagating"
          >
            <span class="tool-icon">↺</span>
            <span class="tool-label">Reset Frame</span>
          </button>
          <button
            class="tool-button reset-video-button"
            @click="handleResetVideoWithRefresh"
            :disabled="isSegmenting || isPropagating"
          >
            <span class="tool-icon">⟲</span>
            <span class="tool-label">Reset Video</span>
          </button>
        </div>
        
        <h4 class="action-bar-title">Propagation</h4>
        <div class="propagate-section">
          <div class="max-frames-input">
            <label for="maxFrames">Max Frames:</label>
            <input
              id="maxFrames"
              type="number"
              v-model.number="maxFrames"
              min="1"
              max="10000"
              :disabled="isPropagating"
            />
          </div>
          <button
            class="tool-button propagate-button"
            @click="handlePropagateMask"
            :disabled="isSegmenting || isPropagating || !segmentationIsReady"
          >
            <span class="tool-icon">▶▶</span>
            <span class="tool-label">{{ isPropagating ? 'Propagating...' : 'Propagate Mask' }}</span>
          </button>
        </div>
        
        <h4 class="action-bar-title">Training Data</h4>
        <div class="tool-buttons">
          <button
            class="tool-button mark-training-button"
            :class="{ active: isMarkingMode }"
            @click="isMarkingMode = !isMarkingMode"
          >
            <span class="tool-icon">✓</span>
            <span class="tool-label">{{ isMarkingMode ? 'Exit Marking Mode' : 'Mark Training Frames' }}</span>
          </button>
        </div>
        <p v-if="isMarkingMode" class="marking-hint">
          Drag on the data track to mark frames as training data. Press Delete to remove.
        </p>
        
        <h4 class="action-bar-title">Video Visibility</h4>
        <div class="tool-buttons">
          <button
            class="tool-button toggle-button"
            :class="{ active: showMask }"
            @click="showMask = !showMask"
          >
            <span class="tool-icon">{{ showMask ? '👁' : '👁‍🗨' }}</span>
            <span class="tool-label">{{ showMask ? 'Mask On' : 'Mask Off' }}</span>
          </button>
          <button
            class="tool-button toggle-button"
            :class="{ active: showPrompts }"
            @click="showPrompts = !showPrompts"
          >
            <span class="tool-icon">{{ showPrompts ? '📍' : '📍' }}</span>
            <span class="tool-label">{{ showPrompts ? 'Prompts On' : 'Prompts Off' }}</span>
          </button>
        </div>
        
        <h4 class="action-bar-title">Data Track</h4>
        <div class="tool-buttons">
          <button
            class="tool-button toggle-button masked-toggle"
            :class="{ active: showMaskedFrames }"
            @click="showMaskedFrames = !showMaskedFrames"
          >
            <span class="tool-icon">◼</span>
            <span class="tool-label">{{ showMaskedFrames ? 'Masked Frames' : 'Masked Frames Off' }}</span>
          </button>
          <button
            class="tool-button toggle-button training-toggle"
            :class="{ active: showTrainingFrames }"
            @click="showTrainingFrames = !showTrainingFrames"
          >
            <span class="tool-icon">◼</span>
            <span class="tool-label">{{ showTrainingFrames ? 'Training Frames' : 'Training Frames Off' }}</span>
          </button>
        </div>
        
        <div v-if="isSegmenting" class="segmenting-indicator">
          Segmenting...
        </div>
        <div v-if="isPropagating" class="propagating-indicator">
          Propagating masks...
        </div>
        <div v-if="currentPrompts.length > 0" class="prompts-info">
          <h5>Frame {{ currentFrameIdx }}</h5>
          <p>{{ currentPrompts.length }} prompt(s)</p>
        </div>
      </div>
      <div class="segmentation-status" :class="segmentationStatus.status">
        {{ segmentationStatusText }}
      </div>
    </aside>
  </div>
</template>

<style scoped>
.video-detail-container {
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
  max-height: calc(100% - 140px);
  flex-shrink: 1;
  flex: 1;
  min-height: 0;
  display: flex;
  align-items: center;
  justify-content: center;
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
  animation: slideIn 0.2s ease-out;
  position: relative;
}

@keyframes slideIn {
  from {
    transform: translateX(100%);
  }
  to {
    transform: translateX(0);
  }
}

.action-bar-content {
  padding: 1rem;
}

.action-bar-title {
  margin: 0 0 1rem 0;
  font-size: 0.9rem;
  font-weight: 600;
  text-transform: uppercase;
  color: #666;
}

.tool-buttons {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.tool-button {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.75rem 1rem;
  border: 1px solid #ddd;
  border-radius: 6px;
  background-color: white;
  cursor: pointer;
  font-size: 0.9rem;
  transition: all 0.15s ease;
}

.tool-button:hover:not(:disabled) {
  background-color: #f0f0f0;
  border-color: #ccc;
}

.tool-button.active {
  background-color: #e3f2fd;
  border-color: #2196f3;
  color: #1976d2;
}

.tool-button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.tool-button.reset-button:not(:disabled):hover {
  background-color: #ffebee;
  border-color: #ef5350;
  color: #c62828;
}

.tool-button.reset-video-button:not(:disabled):hover {
  background-color: #fff3e0;
  border-color: #ff9800;
  color: #e65100;
}

.tool-button.positive-point.active {
  background-color: #dcfce7;
  border-color: #22c55e;
  color: #15803d;
}

.tool-button.negative-point.active {
  background-color: #fee2e2;
  border-color: #ef4444;
  color: #b91c1c;
}

.tool-button.toggle-button {
  background-color: #fafafa;
  border-color: #ccc;
  color: #666;
}

.tool-button.toggle-button.active {
  background-color: #e8f5e9;
  border-color: #4caf50;
  color: #2e7d32;
}

.tool-icon {
  font-size: 1.2rem;
}

.tool-label {
  flex: 1;
}

.segmenting-indicator,
.propagating-indicator {
  margin-top: 1rem;
  padding: 0.5rem;
  background-color: #fff3cd;
  border-radius: 4px;
  font-size: 0.85rem;
  color: #856404;
  text-align: center;
}

.propagating-indicator {
  background-color: #e3f2fd;
  color: #1565c0;
}

.tool-button.propagate-button:not(:disabled):hover {
  background-color: #e3f2fd;
  border-color: #2196f3;
  color: #1565c0;
}

.prompts-info {
  margin-top: 1.5rem;
  padding-top: 1rem;
  border-top: 1px solid #e0e0e0;
}

.prompts-info h5 {
  margin: 0 0 0.25rem 0;
  font-size: 0.85rem;
  color: #666;
}

.prompts-info p {
  margin: 0;
  font-size: 0.9rem;
}

.segmentation-status {
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  padding: 0.5rem 1rem;
  font-size: 0.8rem;
  text-align: center;
  border-top: 1px solid #e0e0e0;
}

.segmentation-status.not_loaded {
  background-color: #f5f5f5;
  color: #666;
}

.segmentation-status.loading_model {
  background-color: #fff3cd;
  color: #856404;
}

.segmentation-status.ready {
  background-color: #d4edda;
  color: #155724;
}

.segmentation-status.error {
  background-color: #f8d7da;
  color: #721c24;
}

.propagate-section {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.max-frames-input {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.max-frames-input label {
  font-size: 0.85rem;
  color: #666;
}

.max-frames-input input {
  width: 80px;
  padding: 0.25rem 0.5rem;
  border: 1px solid #ddd;
  border-radius: 4px;
  font-size: 0.85rem;
}

.max-frames-input input:disabled {
  background-color: #f5f5f5;
  color: #999;
}

.tool-button.mark-training-button {
  background-color: #fafafa;
  border-color: #ccc;
}

.tool-button.mark-training-button.active {
  background-color: #dcfce7;
  border-color: #22c55e;
  color: #15803d;
}

.tool-button.mark-training-button:not(:disabled):hover {
  background-color: #dcfce7;
  border-color: #22c55e;
  color: #15803d;
}

.marking-hint {
  margin: 0.5rem 0 0 0;
  padding: 0.5rem;
  background-color: #dcfce7;
  border-radius: 4px;
  font-size: 0.75rem;
  color: #15803d;
}

.tool-button.masked-toggle.active {
  background-color: #dbeafe;
  border-color: #3b82f6;
  color: #1d4ed8;
}

.tool-button.training-toggle.active {
  background-color: #dcfce7;
  border-color: #22c55e;
  color: #15803d;
}

.action-bar-title {
  margin: 1rem 0 0.5rem 0;
}

.action-bar-title:first-child {
  margin-top: 0;
}
</style>
