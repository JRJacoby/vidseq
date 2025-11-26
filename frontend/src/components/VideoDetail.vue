<script setup lang="ts">
import { ref, onMounted, computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { getVideo, getVideoStreamUrl, type Video } from '@/services/api'

const route = useRoute()
const router = useRouter()

const video = ref<Video | null>(null)
const isLoading = ref(true)
const error = ref<string | null>(null)

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
        <video 
          v-else
          class="video-player"
          controls
          :src="videoStreamUrl"
        >
          Your browser does not support the video tag.
        </video>
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

.video-player {
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

