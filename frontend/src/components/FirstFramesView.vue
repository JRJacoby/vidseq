<script setup lang="ts">
import { ref, onMounted, onUnmounted, computed, watch, nextTick } from 'vue'
import { useRoute } from 'vue-router'
import { getVideos, getFrameImage, getBbox, type Video, type Bbox } from '@/services/api'
import { useProjectStore } from '@/stores/project'

const route = useRoute()
const projectStore = useProjectStore()

const projectId = computed(() => Number(route.params.id))

const videos = ref<Video[]>([])
const currentIndex = ref(0)
const currentFrameImage = ref<ImageBitmap | null>(null)
const currentBbox = ref<Bbox | null>(null)
const isLoading = ref(false)
const error = ref<string | null>(null)
const frameCanvas = ref<HTMLCanvasElement | null>(null)

const currentVideo = computed(() => {
  if (videos.value.length === 0 || currentIndex.value < 0 || currentIndex.value >= videos.value.length) {
    return null
  }
  return videos.value[currentIndex.value]
})

const videoInfo = computed(() => {
  if (videos.value.length === 0) return ''
  return `Video ${currentIndex.value + 1} of ${videos.value.length}`
})

const loadVideos = async () => {
  if (!projectId.value) return
  
  isLoading.value = true
  error.value = null
  
  try {
    videos.value = await getVideos(projectId.value)
    if (videos.value.length > 0 && videos.value[0]) {
      currentIndex.value = 0
      await loadFrameData(videos.value[0].id)
    }
  } catch (e) {
    error.value = e instanceof Error ? e.message : 'Failed to load videos'
    console.error('Failed to load videos:', e)
  } finally {
    isLoading.value = false
  }
}

const loadFrameData = async (videoId: number) => {
  if (!projectId.value) return
  
  isLoading.value = true
  error.value = null
  
  try {
    const [frameBlob, bbox] = await Promise.all([
      getFrameImage(projectId.value, videoId, 0),
      getBbox(projectId.value, videoId, 0)
    ])
    
    if (currentFrameImage.value) {
      currentFrameImage.value.close()
    }
    
    currentFrameImage.value = await createImageBitmap(frameBlob)
    currentBbox.value = bbox
    
    nextTick(() => {
      drawFrame()
    })
  } catch (e) {
    error.value = e instanceof Error ? e.message : 'Failed to load frame data'
    console.error('Failed to load frame data:', e)
    currentFrameImage.value = null
    currentBbox.value = null
  } finally {
    isLoading.value = false
  }
}

const drawFrame = () => {
  if (!frameCanvas.value || !currentFrameImage.value || !currentVideo.value) return
  
  const canvas = frameCanvas.value
  const ctx = canvas.getContext('2d')
  if (!ctx) return
  
  // Set canvas size to match image dimensions (ImageBitmap has width/height)
  canvas.width = currentFrameImage.value.width
  canvas.height = currentFrameImage.value.height
  
  ctx.clearRect(0, 0, canvas.width, canvas.height)
  
  // Draw the frame image
  ctx.drawImage(currentFrameImage.value, 0, 0)
  
  // Draw bounding box if available
  if (currentBbox.value) {
    const { x1, y1, x2, y2 } = currentBbox.value
    const clampedX1 = Math.max(0, Math.min(x1, canvas.width))
    const clampedY1 = Math.max(0, Math.min(y1, canvas.height))
    const clampedX2 = Math.max(0, Math.min(x2, canvas.width))
    const clampedY2 = Math.max(0, Math.min(y2, canvas.height))
    
    ctx.strokeStyle = '#22c55e'
    ctx.lineWidth = 3
    ctx.setLineDash([])
    ctx.strokeRect(clampedX1, clampedY1, clampedX2 - clampedX1, clampedY2 - clampedY1)
  }
}

watch([currentFrameImage, currentBbox, currentVideo], () => {
  nextTick(() => {
    drawFrame()
  })
}, { immediate: true })

const nextVideo = () => {
  if (videos.value.length === 0) return
  currentIndex.value = (currentIndex.value + 1) % videos.value.length
  if (currentVideo.value) {
    loadFrameData(currentVideo.value.id)
  }
}

const prevVideo = () => {
  if (videos.value.length === 0) return
  currentIndex.value = currentIndex.value === 0 ? videos.value.length - 1 : currentIndex.value - 1
  if (currentVideo.value) {
    loadFrameData(currentVideo.value.id)
  }
}

const handleKeyPress = (event: KeyboardEvent) => {
  if (event.key === 'ArrowLeft') {
    event.preventDefault()
    prevVideo()
  } else if (event.key === 'ArrowRight') {
    event.preventDefault()
    nextVideo()
  }
}

onMounted(() => {
  loadVideos()
  window.addEventListener('keydown', handleKeyPress)
})

onUnmounted(() => {
  window.removeEventListener('keydown', handleKeyPress)
  if (currentFrameImage.value) {
    currentFrameImage.value.close()
  }
})
</script>

<template>
  <div class="first-frames-container">
    <div v-if="isLoading && videos.length === 0" class="loading-state">
      Loading videos...
    </div>
    <div v-else-if="error && videos.length === 0" class="error-state">
      {{ error }}
    </div>
    <div v-else-if="videos.length === 0" class="empty-state">
      No videos found in this project.
    </div>
    <div v-else class="frame-viewer">
      <div class="viewer-header">
        <div class="video-info">
          <h3 v-if="currentVideo">{{ currentVideo.name }}</h3>
          <p class="video-meta">ID: {{ currentVideo?.id }} | {{ videoInfo }}</p>
        </div>
        <div class="navigation-controls">
          <button 
            class="nav-button"
            @click="prevVideo"
            :disabled="isLoading || videos.length === 0"
            title="Previous video (←)"
          >
            ←
          </button>
          <button 
            class="nav-button"
            @click="nextVideo"
            :disabled="isLoading || videos.length === 0"
            title="Next video (→)"
          >
            →
          </button>
        </div>
      </div>
      
      <div class="frame-container">
        <div v-if="isLoading" class="loading-overlay">
          Loading frame...
        </div>
        <div v-else-if="error" class="error-overlay">
          {{ error }}
        </div>
        <div v-else-if="currentFrameImage && currentVideo" class="frame-wrapper">
          <canvas
            ref="frameCanvas"
            class="frame-canvas"
          />
        </div>
        <div v-else class="no-frame">
          No frame available
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.first-frames-container {
  display: flex;
  flex-direction: column;
  flex: 1;
  min-height: 0;
  padding: 1rem;
}

.loading-state,
.error-state,
.empty-state {
  display: flex;
  align-items: center;
  justify-content: center;
  flex: 1;
  font-size: 1.1rem;
  color: #666;
}

.error-state {
  color: #d32f2f;
}

.frame-viewer {
  display: flex;
  flex-direction: column;
  flex: 1;
  min-height: 0;
}

.viewer-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1rem;
  border-bottom: 1px solid #e0e0e0;
  margin-bottom: 1rem;
}

.video-info h3 {
  margin: 0 0 0.25rem 0;
  font-size: 1.25rem;
}

.video-meta {
  margin: 0;
  font-size: 0.9rem;
  color: #666;
}

.navigation-controls {
  display: flex;
  gap: 0.5rem;
}

.nav-button {
  padding: 0.5rem 1rem;
  font-size: 1.5rem;
  border: 1px solid #ccc;
  border-radius: 4px;
  background-color: #f8f8f8;
  cursor: pointer;
  transition: background-color 0.2s;
  min-width: 50px;
}

.nav-button:hover:not(:disabled) {
  background-color: #e8e8e8;
}

.nav-button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.frame-container {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  position: relative;
  min-height: 0;
  overflow: auto;
}

.loading-overlay,
.error-overlay {
  padding: 2rem;
  font-size: 1.1rem;
  color: #666;
}

.error-overlay {
  color: #d32f2f;
}

.frame-wrapper {
  position: relative;
  display: inline-block;
  max-width: 100%;
  max-height: 100%;
}

.frame-canvas {
  max-width: 100%;
  max-height: 100%;
  display: block;
}

.no-frame {
  padding: 2rem;
  font-size: 1.1rem;
  color: #666;
}
</style>
