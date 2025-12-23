<script setup lang="ts">
import { ref, onMounted, computed, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { getVideo, getCroppedVideoStreamUrl, type Video } from '@/services/api'
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

const croppedVideoStreamUrl = computed(() => {
  if (!projectId.value || !videoId.value) return ''
  return getCroppedVideoStreamUrl(projectId.value, videoId.value)
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
</style>
