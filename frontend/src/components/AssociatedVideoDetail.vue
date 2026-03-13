<script setup lang="ts">
import { ref, computed, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import {
  getAssociatedVideo,
  getVideoStreamUrl,
  type Video,
} from '@/services/api'
import { useVideoPlayback } from '@/composables/useVideoPlayback'
import { useVideo } from '@/composables/useVideo'
import VideoTimeline from './VideoTimeline.vue'
import TimelineSystem from './TimelineSystem.vue'
import DataTrack from './DataTrack.vue'

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

const viewStart = ref(0)
const viewEnd = ref(0)

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

// Calculate current frame index from currentTime and fps
const currentFrameIdx = computed(() => {
  if (!associatedVideo.value?.fps) return 0
  return Math.floor(currentTime.value * associatedVideo.value.fps)
})

const videoStreamUrl = computed(() => {
  if (!projectId.value || !associatedVideo.value) return ''
  return getVideoStreamUrl(projectId.value, associatedVideo.value.id)
})

const handleSeek = (time: number) => {
  seek(time)
}

const handleViewChange = (start: number, end: number) => {
  viewStart.value = start
  viewEnd.value = end
}

const handleBack = () => {
  router.push(`/project/${projectId.value}`)
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
</style>
