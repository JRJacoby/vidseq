<script setup lang="ts">
import { ref, onMounted, computed, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import {
  getCroppedVideoStreamUrl,
} from '@/services/api'
import { useVideoPlayback } from '@/composables/useVideoPlayback'
import { useVideo } from '@/composables/useVideo'
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

const croppedVideoStreamUrl = computed(() => {
  if (!projectId.value || !videoId.value) return ''
  return getCroppedVideoStreamUrl(projectId.value, videoId.value)
})

// Calculate current frame index from currentTime and fps
const currentFrameIdx = computed(() => {
  if (!video.value?.fps) return 0
  return Math.floor(currentTime.value * video.value.fps)
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

onMounted(() => {
  if (route.query.frame !== undefined && video.value?.fps) {
    const frameIdx = parseInt(route.query.frame as string, 10)
    if (!isNaN(frameIdx)) {
      const targetTime = (frameIdx + 0.5) / video.value.fps
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
        <h4 class="action-bar-title">Cropped Video</h4>
        <p class="marking-hint">Keypoint labeling has moved to the original video view.</p>
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

.overlay-canvas.training-mode {
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

.training-toggle-button:hover {
  background-color: #e8e8e8;
  border-color: #999;
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
</style>
