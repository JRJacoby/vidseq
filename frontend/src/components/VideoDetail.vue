<script setup lang="ts">
import { ref, onMounted, onUnmounted, computed, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { 
  getVideo, 
  getVideoStreamUrl, 
  getMask, 
  getPrompts, 
  addPrompt,
  resetFrame,
  getSAM3Status,
  preloadSAM3,
  initVideoSession,
  closeVideoSession,
  type Video, 
  type Prompt,
  type SAM3Status 
} from '@/services/api'
import VideoTimeline from './VideoTimeline.vue'
import VideoOverlay from './VideoOverlay.vue'

const route = useRoute()
const router = useRouter()

const video = ref<Video | null>(null)
const isLoading = ref(true)
const error = ref<string | null>(null)

const videoRef = ref<HTMLVideoElement | null>(null)
const currentTime = ref(0)
const duration = ref(0)
const isPlaying = ref(false)

const activeTool = ref<'none' | 'bbox'>('none')
const currentMask = ref<ImageBitmap | null>(null)
const currentPrompts = ref<Prompt[]>([])
const videoWidth = ref(0)
const videoHeight = ref(0)
const isSegmenting = ref(false)

const sam3Status = ref<SAM3Status>({ status: 'not_loaded', error: null })
let statusPollInterval: number | null = null

const projectId = computed(() => Number(route.params.id))
const videoId = computed(() => Number(route.params.videoId))

const videoStreamUrl = computed(() => {
  if (!projectId.value || !videoId.value) return ''
  return getVideoStreamUrl(projectId.value, videoId.value)
})

const currentFrameIdx = computed(() => {
  if (!video.value) return 0
  return Math.floor(currentTime.value * video.value.fps)
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

const loadFrameData = async (frameIdx: number) => {
  if (!projectId.value || !videoId.value) return
  
  try {
    const [maskBlob, prompts] = await Promise.all([
      getMask(projectId.value, videoId.value, frameIdx),
      getPrompts(projectId.value, videoId.value, frameIdx),
    ])
    
    currentMask.value = await createImageBitmap(maskBlob)
    currentPrompts.value = prompts
  } catch (e) {
    console.error('Failed to load frame data:', e)
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
    videoWidth.value = videoRef.value.videoWidth
    videoHeight.value = videoRef.value.videoHeight
    loadFrameData(0)
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

const toggleBboxTool = () => {
  activeTool.value = activeTool.value === 'bbox' ? 'none' : 'bbox'
}

const handleBboxComplete = async (bbox: { x1: number; y1: number; x2: number; y2: number }) => {
  if (!projectId.value || !videoId.value) return
  
  isSegmenting.value = true
  activeTool.value = 'none'
  
  try {
    await addPrompt(
      projectId.value,
      videoId.value,
      currentFrameIdx.value,
      'bbox',
      bbox
    )
    
    await loadFrameData(currentFrameIdx.value)
  } catch (e) {
    console.error('Failed to segment:', e)
  } finally {
    isSegmenting.value = false
  }
}

const handleResetFrame = async () => {
  if (!projectId.value || !videoId.value) return
  
  try {
    await resetFrame(projectId.value, videoId.value, currentFrameIdx.value)
    await loadFrameData(currentFrameIdx.value)
  } catch (e) {
    console.error('Failed to reset frame:', e)
  }
}

let debounceTimeout: number | null = null
watch(currentFrameIdx, (newFrameIdx) => {
  if (debounceTimeout) {
    clearTimeout(debounceTimeout)
  }
  debounceTimeout = window.setTimeout(() => {
    loadFrameData(newFrameIdx)
  }, 100)
})

const sam3StatusText = computed(() => {
  switch (sam3Status.value.status) {
    case 'not_loaded': return 'SAM3: Not loaded'
    case 'loading_model': return 'SAM3: Loading model...'
    case 'ready': return 'SAM3: Ready'
    case 'error': return `SAM3: Error - ${sam3Status.value.error}`
    default: return 'SAM3: Unknown'
  }
})

const sam3IsReady = computed(() => sam3Status.value.status === 'ready')
const sessionInitialized = ref(false)

const pollSAM3Status = async () => {
  try {
    sam3Status.value = await getSAM3Status()
    
    if (sam3Status.value.status === 'ready' || sam3Status.value.status === 'error') {
      if (statusPollInterval) {
        clearInterval(statusPollInterval)
        statusPollInterval = null
      }
    }
  } catch (e) {
    console.error('Failed to poll SAM3 status:', e)
  }
}

// Initialize video session when SAM3 becomes ready
watch(sam3IsReady, async (isReady) => {
  if (isReady && !sessionInitialized.value && projectId.value && videoId.value) {
    try {
      await initVideoSession(projectId.value, videoId.value)
      sessionInitialized.value = true
    } catch (e) {
      console.error('Failed to init video session:', e)
    }
  }
})

onMounted(async () => {
  loadVideo()
  
  await pollSAM3Status()
  if (sam3Status.value.status === 'not_loaded') {
    preloadSAM3()
  }
  
  statusPollInterval = window.setInterval(pollSAM3Status, 1000)
})

onUnmounted(async () => {
  if (statusPollInterval) {
    clearInterval(statusPollInterval)
  }
  
  if (projectId.value && videoId.value) {
    try {
      await closeVideoSession(projectId.value, videoId.value)
    } catch (e) {
      console.error('Failed to close video session:', e)
    }
  }
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
              @bbox-complete="handleBboxComplete"
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
            class="tool-button"
            :class="{ active: activeTool === 'bbox' }"
            @click="toggleBboxTool"
            :disabled="isSegmenting || !sam3IsReady"
          >
            <span class="tool-icon">▢</span>
            <span class="tool-label">Bounding Box</span>
          </button>
          <button
            class="tool-button reset-button"
            @click="handleResetFrame"
            :disabled="isSegmenting || currentPrompts.length === 0"
          >
            <span class="tool-icon">↺</span>
            <span class="tool-label">Reset Frame</span>
          </button>
        </div>
        <div v-if="isSegmenting" class="segmenting-indicator">
          Segmenting...
        </div>
        <div v-if="currentPrompts.length > 0" class="prompts-info">
          <h5>Frame {{ currentFrameIdx }}</h5>
          <p>{{ currentPrompts.length }} prompt(s)</p>
        </div>
      </div>
      <div class="sam3-status" :class="sam3Status.status">
        {{ sam3StatusText }}
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

.tool-icon {
  font-size: 1.2rem;
}

.tool-label {
  flex: 1;
}

.segmenting-indicator {
  margin-top: 1rem;
  padding: 0.5rem;
  background-color: #fff3cd;
  border-radius: 4px;
  font-size: 0.85rem;
  color: #856404;
  text-align: center;
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

.sam3-status {
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  padding: 0.5rem 1rem;
  font-size: 0.8rem;
  text-align: center;
  border-top: 1px solid #e0e0e0;
}

.sam3-status.not_loaded {
  background-color: #f5f5f5;
  color: #666;
}

.sam3-status.loading_model {
  background-color: #fff3cd;
  color: #856404;
}

.sam3-status.ready {
  background-color: #d4edda;
  color: #155724;
}

.sam3-status.error {
  background-color: #f8d7da;
  color: #721c24;
}
</style>
