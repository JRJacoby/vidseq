<script setup lang="ts">
import { ref, onMounted, computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { getVideo, getVideoStreamUrl, type Video } from '@/services/api'
import VideoTimeline from './VideoTimeline.vue'

const route = useRoute()
const router = useRouter()

const video = ref<Video | null>(null)
const isLoading = ref(true)
const error = ref<string | null>(null)

const videoRef = ref<HTMLVideoElement | null>(null)
const currentTime = ref(0)
const duration = ref(0)
const isPlaying = ref(false)

const projectId = computed(() => Number(route.params.id))
const videoId = computed(() => Number(route.params.videoId))

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

const onTimeUpdate = () => {
  if (videoRef.value) {
    currentTime.value = videoRef.value.currentTime
  }
}

const onLoadedMetadata = () => {
  if (videoRef.value) {
    duration.value = videoRef.value.duration
  }
}

const onPlay = () => {
  isPlaying.value = true
}

const onPause = () => {
  isPlaying.value = false
}

const handleSeek = (time: number) => {
  if (videoRef.value) {
    videoRef.value.currentTime = time
  }
}

const handleTogglePlay = () => {
  if (videoRef.value) {
    if (isPlaying.value) {
      videoRef.value.pause()
    } else {
      videoRef.value.play()
    }
  }
}

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
          <VideoTimeline
            :current-time="currentTime"
            :duration="duration"
            :is-playing="isPlaying"
            @seek="handleSeek"
            @toggle-play="handleTogglePlay"
          />
        </div>
      </div>
    </div>
    
    <aside class="action-bar">
      <div class="action-bar-content">
        <!-- Empty for now - future segmentation tools will go here -->
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
  max-width: 100%;
  max-height: 100%;
  min-height: 0;
  gap: 8px;
}

.video-player {
  max-width: 100%;
  max-height: calc(100% - 80px);
  width: auto;
  height: auto;
  flex-shrink: 1;
  min-height: 0;
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
</style>
