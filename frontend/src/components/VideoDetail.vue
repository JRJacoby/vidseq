<script setup lang="ts">
import { ref, onMounted, onUnmounted, computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { getVideo, getVideoStreamUrl, type Video } from '@/services/api'
import { useSegmentationSession } from '@/composables/useSegmentationSession'
import { useVideoPlayback } from '@/composables/useVideoPlayback'
import { useSegmentation } from '@/composables/useSegmentation'
import VideoTimeline from './VideoTimeline.vue'
import VideoOverlay from './VideoOverlay.vue'

const route = useRoute()
const router = useRouter()

const projectId = computed(() => Number(route.params.id))
const videoId = computed(() => Number(route.params.videoId))

const video = ref<Video | null>(null)
const isLoading = ref(true)
const error = ref<string | null>(null)
const showMask = ref(true)
const showPrompts = ref(true)

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
  currentPrompts,
  isSegmenting,
  isPropagating,
  loadFrameData,
  seekToFrame,
  togglePositivePointTool,
  toggleNegativePointTool,
  handlePointComplete,
  handleResetFrame,
  handleResetVideo,
  handlePropagateForward,
  handlePropagateBackward,
} = useSegmentation(projectId, videoId, currentFrameIdx, isPlaying, videoRef, fps)

const handleSeek = (time: number) => {
  seek(time)
  if (video.value) {
    const targetFrame = Math.floor(time * video.value.fps)
    seekToFrame(targetFrame)
  }
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
              :prompts="currentPrompts"
              :show-mask="showMask"
              :show-prompts="showPrompts"
              @point-complete="handlePointComplete"
            />
          </div>
          <VideoTimeline
            :current-time="currentTime"
            :duration="duration"
            :is-playing="isPlaying"
            :fps="video!.fps"
            @seek="handleSeek"
            @toggle-play="handleTogglePlay"
          />
        </div>
      </div>
    </div>

    <aside class="action-bar">
      <div class="action-bar-content">
        <h4 class="action-bar-title">Tools</h4>
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
            @click="handleResetFrame"
            :disabled="isSegmenting || isPropagating"
          >
            <span class="tool-icon">↺</span>
            <span class="tool-label">Reset Frame</span>
          </button>
          <button
            class="tool-button reset-video-button"
            @click="handleResetVideo"
            :disabled="isSegmenting || isPropagating"
          >
            <span class="tool-icon">⟲</span>
            <span class="tool-label">Reset Video</span>
          </button>
          <button
            class="tool-button propagate-button"
            @click="handlePropagateBackward"
            :disabled="isSegmenting || isPropagating || !segmentationIsReady"
          >
            <span class="tool-icon">◀◀</span>
            <span class="tool-label">{{ isPropagating ? 'Propagating...' : 'Propagate Backward' }}</span>
          </button>
          <button
            class="tool-button propagate-button"
            @click="handlePropagateForward"
            :disabled="isSegmenting || isPropagating || !segmentationIsReady"
          >
            <span class="tool-icon">▶▶</span>
            <span class="tool-label">{{ isPropagating ? 'Propagating...' : 'Propagate Forward' }}</span>
          </button>
        </div>
        <h4 class="action-bar-title">Visibility</h4>
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
  max-height: calc(100% - 104px);
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
</style>
