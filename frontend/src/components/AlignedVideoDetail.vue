<script setup lang="ts">
import { ref, onMounted, computed, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useDebounceFn } from '@vueuse/core'
import {
  getVideo,
  getAlignedVideoStreamUrl,
  getPCAStatus,
  getPCAScoresDownsampled,
  checkPCAScoresExist,
  type Video,
  type PCAScorePoint,
} from '@/services/api'
import { useVideoPlayback } from '@/composables/useVideoPlayback'
import VideoTimeline from './VideoTimeline.vue'
import TimelineSystem from './TimelineSystem.vue'
import DataTrack from './DataTrack.vue'

const route = useRoute()
const router = useRouter()

const projectId = computed(() => Number(route.params.id))
const videoId = computed(() => Number(route.params.videoId))

const video = ref<Video | null>(null)
const isLoading = ref(true)
const error = ref<string | null>(null)

const viewStart = ref(0)
const viewEnd = ref(0)

const alignedVideoStreamUrl = computed(() => {
  if (!projectId.value || !videoId.value) return ''
  return getAlignedVideoStreamUrl(projectId.value, videoId.value)
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

// Calculate current frame index from currentTime and fps
const currentFrameIdx = computed(() => {
  if (!video.value?.fps) return 0
  return Math.floor(currentTime.value * video.value.fps)
})

const handleLabelFrame = () => {
  router.push({
    path: `/project/${projectId.value}/video/${videoId.value}/cropped`,
    query: { frame: currentFrameIdx.value.toString(), training: 'true' },
  })
}

// PCA state
const hasPCAScores = ref(false)
const nComponents = ref(0)
const visiblePCs = ref<number[]>([0, 1, 2])  // Default: show first 3 PCs
const showPCAPlot = ref(true)
const pcaScores = ref<Record<string, PCAScorePoint[]>>({})

// Color palette for PC checkboxes
const PC_COLORS = [
  '#3b82f6', '#ef4444', '#22c55e', '#a855f7',
  '#f97316', '#ec4899', '#14b8a6', '#eab308',
]

const getPCColor = (pcIdx: number): string => {
  return PC_COLORS[pcIdx % PC_COLORS.length] ?? '#888888'
}

const fetchPCAScores = async () => {
  if (!projectId.value || !videoId.value || !video.value || !hasPCAScores.value) return
  if (visiblePCs.value.length === 0) {
    pcaScores.value = {}
    return
  }

  try {
    const startFrame = Math.floor(viewStart.value * video.value.fps)
    const endFrame = Math.ceil(viewEnd.value * video.value.fps)

    const response = await getPCAScoresDownsampled(
      projectId.value,
      videoId.value,
      visiblePCs.value,
      800,
      startFrame,
      endFrame
    )
    pcaScores.value = response.scores
    nComponents.value = response.n_components
  } catch (e) {
    console.error('Failed to fetch PCA scores:', e)
  }
}

const debouncedFetchPCAScores = useDebounceFn(fetchPCAScores, 150)

// Watch for view changes
watch([viewStart, viewEnd], () => {
  if (hasPCAScores.value) {
    debouncedFetchPCAScores()
  }
})

// Watch for PC selection changes (immediate fetch)
watch(visiblePCs, () => {
  if (hasPCAScores.value) {
    fetchPCAScores()
  }
}, { deep: true })

const loadPCAStatus = async () => {
  try {
    hasPCAScores.value = await checkPCAScoresExist(projectId.value, videoId.value)
    if (hasPCAScores.value) {
      const status = await getPCAStatus(projectId.value)
      if (status.has_pca && status.n_components) {
        nComponents.value = status.n_components
      }
    }
  } catch (e) {
    console.error('Failed to check PCA scores:', e)
  }
}

onMounted(async () => {
  await loadVideo()
  await loadPCAStatus()
  if (hasPCAScores.value) {
    await fetchPCAScores()
  }
})
</script>

<template>
  <div class="aligned-video-detail-container">
    <div class="video-content">
      <div class="video-header">
        <button class="back-button" @click="handleBack">← Back to Videos</button>
        <h3 v-if="video" class="video-title">{{ video.name }} (Aligned)</h3>
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
              :src="alignedVideoStreamUrl"
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
              :fps="video!.fps"
              :external-view-start="viewStart"
              :external-view-end="viewEnd"
              @seek="handleSeek"
              @toggle-play="handleTogglePlay"
              @view-change="handleViewChange"
            />
            <DataTrack
              v-if="hasPCAScores"
              :duration="duration"
              :fps="video!.fps"
              :current-time="currentTime"
              :view-start="viewStart"
              :view-end="viewEnd"
              :masked-ranges="[]"
              :training-ranges="[]"
              :show-masked-frames="false"
              :show-training-frames="false"
              :show-confidence-plot="false"
              :is-marking-mode="false"
              :confidence-scores="[]"
              :pca-scores="pcaScores"
              :visible-p-cs="visiblePCs"
              :show-p-c-a-plot="showPCAPlot"
              @view-change="handleViewChange"
            />
          </TimelineSystem>
        </div>
      </div>
    </div>

    <aside class="action-bar">
      <div class="action-bar-content">
        <p class="action-bar-info">This is the aligned video view. The animal has been rotated to face right in each frame.</p>

        <h4 class="action-bar-title">Training</h4>
        <p class="frame-indicator">Frame: {{ currentFrameIdx }}</p>
        <button class="label-frame-button" @click="handleLabelFrame">
          Label This Frame
        </button>
        <p class="label-hint">Jump to cropped video to add this frame to alignment training data.</p>

        <template v-if="hasPCAScores">
          <h4 class="action-bar-title">PCA Visualization</h4>
          <button
            class="toggle-button"
            :class="{ active: showPCAPlot }"
            @click="showPCAPlot = !showPCAPlot"
          >
            {{ showPCAPlot ? 'PCA Plot On' : 'PCA Plot Off' }}
          </button>

          <div v-if="showPCAPlot" class="pc-selector">
            <label class="pc-selector-label">Visible PCs:</label>
            <div class="pc-checkboxes">
              <label
                v-for="pc in nComponents"
                :key="pc - 1"
                class="pc-checkbox"
              >
                <input
                  type="checkbox"
                  :value="pc - 1"
                  v-model="visiblePCs"
                />
                <span
                  class="pc-color-dot"
                  :style="{ backgroundColor: getPCColor(pc - 1) }"
                ></span>
                <span class="pc-label">PC{{ pc }}</span>
              </label>
            </div>
          </div>
        </template>
        <p v-else class="no-pca-hint">Run PCA to visualize PC scores.</p>
      </div>
    </aside>
  </div>
</template>

<style scoped>
.aligned-video-detail-container {
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

.frame-indicator {
  margin: 0 0 0.75rem 0;
  font-size: 0.85rem;
  color: #666;
}

.label-frame-button {
  width: 100%;
  padding: 0.75rem 1rem;
  border: 1px solid #f59e0b;
  border-radius: 4px;
  background-color: #fffbeb;
  color: #b45309;
  cursor: pointer;
  font-size: 0.9rem;
  font-weight: 500;
  transition: all 0.2s;
}

.label-frame-button:hover {
  background-color: #fef3c7;
  border-color: #d97706;
}

.label-hint {
  margin: 0.5rem 0 0 0;
  font-size: 0.8rem;
  color: #888;
  line-height: 1.4;
}

/* PCA Controls */
.toggle-button {
  width: 100%;
  padding: 0.5rem 0.75rem;
  border: 1px solid #ccc;
  border-radius: 4px;
  background-color: #f8f8f8;
  cursor: pointer;
  font-size: 0.85rem;
  transition: all 0.2s;
}

.toggle-button:hover {
  background-color: #e8e8e8;
}

.toggle-button.active {
  background-color: #dbeafe;
  border-color: #3b82f6;
  color: #1d4ed8;
}

.pc-selector {
  margin-top: 0.75rem;
}

.pc-selector-label {
  display: block;
  font-size: 0.8rem;
  color: #666;
  margin-bottom: 0.5rem;
}

.pc-checkboxes {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
  max-height: 200px;
  overflow-y: auto;
}

.pc-checkbox {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.25rem 0;
  cursor: pointer;
  font-size: 0.85rem;
}

.pc-checkbox input[type="checkbox"] {
  cursor: pointer;
}

.pc-color-dot {
  width: 12px;
  height: 12px;
  border-radius: 50%;
  flex-shrink: 0;
}

.pc-label {
  color: #444;
}

.no-pca-hint {
  margin: 1rem 0 0 0;
  font-size: 0.85rem;
  color: #888;
  font-style: italic;
}
</style>
