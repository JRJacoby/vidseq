<script setup lang="ts">
import { ref, onMounted, onUnmounted, computed, watch } from 'vue'
import { useDebounceFn } from '@vueuse/core'
import { useRoute, useRouter } from 'vue-router'
import { getVideoStreamUrl, createPropagation, createPropagationWithoutMemory, getScoresDownsampled, getDetectorScoresDownsampled, deleteSegmentationRange, detectorMasksExist, finalMasksExist, obbBboxesExist, getObbBbox, getObbScoresDownsampled, getPoseScoresDownsampled, segDetectorMasksExist, type Video, type MaskScore, type ObbBbox } from '@/services/api'
import { useSegmentationSession } from '@/composables/useSegmentationSession'
import { useVideoPlayback } from '@/composables/useVideoPlayback'
import { useSegmentation } from '@/composables/useSegmentation'
import { useFrameRanges } from '@/composables/useFrameRanges'
import { useVideo } from '@/composables/useVideo'
import { usePoseLabels } from '@/composables/usePoseLabels'
import { getPosePrediction, type PosePrediction } from '@/services/api'
import VideoTimeline from './VideoTimeline.vue'
import VideoOverlay from './VideoOverlay.vue'
import DataTrack from './DataTrack.vue'
import TimelineSystem from './TimelineSystem.vue'
import CondFrameGrid from './CondFrameGrid.vue'

const route = useRoute()
const router = useRouter()

const projectId = computed(() => Number(route.params.id))
const videoId = computed(() => Number(route.params.videoId))

const { video, isLoading, error, refresh: refreshVideo } = useVideo(videoId, projectId)

const showMask = ref(true)
const showPrompts = ref(true)

const showTrackerMaskedFrames = ref(true)
const showTrainingFrames = ref(true)
const showConfidencePlot = ref(true)
const isMarkingMode = ref(false)
const workingRange = ref<[number, number] | null>(null)
const isWorkingRangeMode = ref(false)
const maxFrames = ref(1000)
const confidenceScores = ref<MaskScore[]>([])
const showDetectorConfidence = ref(true)
const detectorScores = ref<MaskScore[]>([])

// Mask view mode
type MaskViewMode = 'tracker' | 'detector' | 'final' | 'obb' | 'seg'
const maskViewMode = ref<MaskViewMode>('tracker')
const hasDetectorMasks = ref(false)
const hasFinalMasks = ref(false)
const hasObbBboxes = ref(false)
const hasSegMasks = ref(false)
const obbBbox = ref<ObbBbox | null>(null)
const obbScores = ref<{ frame_idx: number; score: number }[]>([])
const showObbConfidence = ref(true)
const poseScores = ref<MaskScore[]>([])
const showPoseConfidence = ref(true)
const condFrameRefreshKey = ref(0)

const checkDetectorMasks = async () => {
  if (!projectId.value || !videoId.value) return
  try {
    const result = await detectorMasksExist(projectId.value, videoId.value)
    hasDetectorMasks.value = result.exists
  } catch {
    hasDetectorMasks.value = false
  }
}

const checkFinalMasks = async () => {
  if (!projectId.value || !videoId.value) return
  try {
    const result = await finalMasksExist(projectId.value, videoId.value)
    hasFinalMasks.value = result.exists
  } catch {
    hasFinalMasks.value = false
  }
}

const checkObbBboxes = async () => {
  if (!projectId.value || !videoId.value) return
  try {
    const result = await obbBboxesExist(projectId.value, videoId.value)
    hasObbBboxes.value = result.exists
  } catch {
    hasObbBboxes.value = false
  }
}

const checkSegMasks = async () => {
  if (!projectId.value || !videoId.value) return
  try {
    const result = await segDetectorMasksExist(projectId.value, videoId.value)
    hasSegMasks.value = result.exists
  } catch {
    hasSegMasks.value = false
  }
}

const viewStart = ref(0)
const viewEnd = ref(0)

const videoStreamUrl = computed(() => {
  if (!projectId.value || !videoId.value) return ''
  return getVideoStreamUrl(projectId.value, videoId.value)
})

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

const currentFrameIdx = computed(() => {
  if (!video.value) return 0
  return Math.floor(currentTime.value * video.value.fps)
})

const fps = computed(() => video.value?.fps ?? 30)

const {
  activeTool,
  currentMask,
  detectorBbox,
  currentPrompts,
  isSegmenting,
  useCondMemory,
  useNonCondMemory,
  lastCondUsed,
  lastNonCondUsed,
  loadFrameData,
  seekToFrame,
  togglePositivePointTool,
  toggleNegativePointTool,
  toggleBoundingBoxTool,
  handlePointComplete,
  handleBoxComplete,
  handleResetFrame,
  handleResetVideo,
  clearMaskCache,
} = useSegmentation(projectId, videoId, currentFrameIdx, isPlaying, videoRef, fps, maskViewMode, workingRange)

const {
  trackerMaskedRanges,
  trainingRanges,
  refresh: refreshFrameRanges,
  markTraining,
  unmarkTraining,
  validateRange,
} = useFrameRanges(projectId, videoId)

const {
  currentLabel: poseLabel,
  labeledFrameCount: poseLabelCount,
  labelingState,
  pendingFront,
  loadLabel: loadPoseLabel,
  deleteLabel: deletePoseLabel,
  refresh: refreshPoseLabels,
  startLabeling,
  stopLabeling,
  handleClick: handleKeypointClick,
} = usePoseLabels(projectId, videoId)

const isLabelingKeypoints = ref(false)
const showPoseKeypoints = ref(true)
const posePrediction = ref<PosePrediction | null>(null)

const toggleLabelingMode = () => {
  isLabelingKeypoints.value = !isLabelingKeypoints.value
  if (isLabelingKeypoints.value) {
    startLabeling()
  } else {
    stopLabeling()
  }
}

const advanceFrame = () => {
  if (!video.value) return
  const nextFrame = currentFrameIdx.value + 1
  if (nextFrame < video.value.num_frames) {
    seekToFrame(nextFrame)
  }
}

const onKeypointClick = async (point: { x: number; y: number }) => {
  await handleKeypointClick(point.x, point.y, currentFrameIdx.value, advanceFrame)
}

const fetchScoresForView = async () => {
  if (!projectId.value || !videoId.value || !video.value) return
  try {
    // Convert view time range to frame indices
    const startFrame = Math.floor(viewStart.value * video.value.fps)
    const endFrame = Math.ceil(viewEnd.value * video.value.fps)

    // Fetch LTTB-downsampled scores for the current view range
    // Returns max ~800 points optimized for the visible canvas width
    const response = await getScoresDownsampled(
      projectId.value,
      videoId.value,
      800,  // max samples for canvas width
      startFrame,
      endFrame
    )
    confidenceScores.value = response.scores
  } catch (e) {
    console.error('Failed to fetch confidence scores:', e)
  }
}

// Debounced version for view changes (150ms delay)
const fetchScores = useDebounceFn(fetchScoresForView, 150)

const fetchDetectorScoresForView = async () => {
  if (!projectId.value || !videoId.value || !video.value) return
  try {
    const startFrame = Math.floor(viewStart.value * video.value.fps)
    const endFrame = Math.ceil(viewEnd.value * video.value.fps)
    const response = await getDetectorScoresDownsampled(
      projectId.value,
      videoId.value,
      800,
      startFrame,
      endFrame
    )
    detectorScores.value = response.scores
  } catch (e) {
    console.error('Failed to fetch detector scores:', e)
  }
}

const fetchDetectorScores = useDebounceFn(fetchDetectorScoresForView, 150)

const fetchObbScoresForView = async () => {
  if (!projectId.value || !videoId.value || !video.value) return
  try {
    const startFrame = Math.floor(viewStart.value * video.value.fps)
    const endFrame = Math.ceil(viewEnd.value * video.value.fps)
    const response = await getObbScoresDownsampled(
      projectId.value,
      videoId.value,
      800,
      startFrame,
      endFrame
    )
    obbScores.value = response.scores
  } catch (e) {
    console.error('Failed to fetch OBB scores:', e)
  }
}

const fetchObbScores = useDebounceFn(fetchObbScoresForView, 150)

const fetchPoseScores = useDebounceFn(async () => {
  if (!projectId.value || !videoId.value || !video.value) return
  try {
    const response = await getPoseScoresDownsampled(
      projectId.value, videoId.value, 800,
      Math.floor(viewStart.value * video.value.fps),
      Math.floor(viewEnd.value * video.value.fps),
    )
    poseScores.value = response.scores
  } catch {
    poseScores.value = []
  }
}, 150)

// Re-fetch scores when view range changes
watch([viewStart, viewEnd], () => {
  fetchScores()
  fetchDetectorScores()
  fetchObbScores()
  fetchPoseScores()
}, { flush: 'post' })

const isPropagating = ref(false)

const handlePropagateMask = async () => {
  if (!projectId.value || !videoId.value) return

  isPropagating.value = true
  try {
    await createPropagation(
      projectId.value,
      videoId.value,
      currentFrameIdx.value,
      maxFrames.value,
      workingRange.value?.[0] ?? null,
      workingRange.value?.[1] ?? null,
    )
    clearMaskCache()
    await loadFrameData(currentFrameIdx.value)
    await refreshFrameRanges()
    // Refresh scores immediately (no debounce)
    await fetchScoresForView()
    await fetchDetectorScoresForView()
    await fetchObbScoresForView()
    fetchPoseScores()
  } catch (e) {
    console.error('Failed to propagate mask:', e)
    alert(e instanceof Error ? e.message : 'Failed to propagate mask')
  } finally {
    isPropagating.value = false
  }
}

const handlePropagateWithoutMemory = async () => {
  if (!projectId.value || !videoId.value) return

  isPropagating.value = true
  try {
    await createPropagationWithoutMemory(
      projectId.value,
      videoId.value,
      currentFrameIdx.value,
      maxFrames.value,
      workingRange.value?.[0] ?? null,
      workingRange.value?.[1] ?? null,
    )
    clearMaskCache()
    await loadFrameData(currentFrameIdx.value)
    await refreshFrameRanges()
    // Refresh scores immediately (no debounce)
    await fetchScoresForView()
    await fetchDetectorScoresForView()
    await fetchObbScoresForView()
    fetchPoseScores()
  } catch (e) {
    console.error('Failed to propagate without memory:', e)
    alert(e instanceof Error ? e.message : 'Failed to propagate without memory')
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

const handleUnmarkMasked = async (startFrame: number, endFrame: number) => {
  try {
    await deleteSegmentationRange(projectId.value, videoId.value, startFrame, endFrame)
    clearMaskCache(startFrame, endFrame)
    await refreshFrameRanges()
    condFrameRefreshKey.value++
  } catch (e) {
    console.error('Failed to reset segmentation range:', e)
  }
}

const handleSetWorkingRange = (startFrame: number, endFrame: number) => {
  workingRange.value = [startFrame, endFrame]
  isWorkingRangeMode.value = false
}

const handleResetFrameWithRefresh = async () => {
  await handleResetFrame()
  await refreshFrameRanges()
}

const handlePointCompleteWithRefresh = async (point: { x: number; y: number; type: 'positive_point' | 'negative_point' }) => {
  await handlePointComplete(point)
  await refreshFrameRanges()
  condFrameRefreshKey.value++
}

const handleBoxCompleteWithRefresh = async (box: { x1: number; y1: number; x2: number; y2: number }) => {
  await handleBoxComplete(box)
  await refreshFrameRanges()
  condFrameRefreshKey.value++
}

const handleResetVideoWithRefresh = async () => {
  await handleResetVideo()
  await refreshFrameRanges()
  confidenceScores.value = []
}

setMetadataCallback(() => {
  loadFrameData(0)
})

// Load pose label and prediction when frame changes
watch(currentFrameIdx, async (frameIdx) => {
  if (frameIdx !== undefined && frameIdx !== null) {
    await loadPoseLabel(frameIdx)
    try {
      posePrediction.value = await getPosePrediction(projectId.value, videoId.value, frameIdx)
    } catch {
      posePrediction.value = null
    }
  }
})

// Fetch OBB bbox when frame changes in OBB mode
watch([currentFrameIdx, maskViewMode], async ([frameIdx, mode]) => {
  if (mode !== 'obb' || !projectId.value || !videoId.value) {
    if (mode !== 'obb') obbBbox.value = null
    return
  }
  try {
    const result = await getObbBbox(projectId.value, videoId.value, frameIdx)
    obbBbox.value = result.bbox
  } catch {
    obbBbox.value = null
  }
})

const handlePoseKeyDown = (e: KeyboardEvent) => {
  if ((e.key === 'Delete' || e.key === 'Backspace') && isLabelingKeypoints.value && poseLabel.value) {
    e.preventDefault()
    deletePoseLabel(currentFrameIdx.value)
  }
}

onMounted(async () => {
  window.addEventListener('keydown', handlePoseKeyDown)
  await refreshFrameRanges()
  await checkDetectorMasks()
  await checkFinalMasks()
  await checkObbBboxes()
  await checkSegMasks()
  // Initial fetch without debounce
  await fetchScoresForView()
  await fetchDetectorScoresForView()
  await fetchObbScoresForView()
  fetchPoseScores()
  refreshPoseLabels()
})

onUnmounted(() => {
  window.removeEventListener('keydown', handlePoseKeyDown)
})
</script>

<template>
  <div class="video-detail-page">
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
            <div class="video-wrapper">
              <video
                ref="videoRef"
                class="video-player"
                :src="videoStreamUrl"
                preload="metadata"
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
                :show-mask="showMask"
                :show-prompts="showPrompts"
                :detector-bbox="detectorBbox"
                :obb-bbox="obbBbox"
                :pose-label="poseLabel"
                :pose-prediction="posePrediction"
                :pending-front="pendingFront"
                :show-pose-keypoints="showPoseKeypoints"
                :is-labeling-keypoints="isLabelingKeypoints"
                @point-complete="handlePointCompleteWithRefresh"
                @box-complete="handleBoxCompleteWithRefresh"
                @keypoint-click="onKeypointClick"
              />
            </div>
          </div>
          <TimelineSystem v-if="video">
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
              :tracker-masked-ranges="trackerMaskedRanges"
              :training-ranges="trainingRanges"
              :show-tracker-masked-frames="showTrackerMaskedFrames"
              :show-training-frames="showTrainingFrames"
              :show-confidence-plot="showConfidencePlot"
              :is-marking-mode="isMarkingMode"
              :confidence-scores="confidenceScores"
              :show-detector-confidence="showDetectorConfidence"
              :detector-scores="detectorScores"
              :show-obb-confidence="showObbConfidence"
              :obb-scores="obbScores"
              :show-pose-confidence="showPoseConfidence"
              :pose-scores="poseScores"
              :working-range="workingRange"
              :is-working-range-mode="isWorkingRangeMode"
              @mark-training="handleMarkTraining"
              @unmark-training="handleUnmarkTraining"
              @unmark-masked="handleUnmarkMasked"
              @set-working-range="handleSetWorkingRange"
              @clear-working-range="workingRange = null"
              @view-change="handleViewChange"
            />
          </TimelineSystem>
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
            class="tool-button bounding-box"
            :class="{ active: activeTool === 'bounding_box' }"
            @click="toggleBoundingBoxTool"
            :disabled="isSegmenting || !segmentationIsReady"
          >
            <span class="tool-icon">▢</span>
            <span class="tool-label">Bounding Box</span>
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
        <div class="memory-options">
          <label class="memory-toggle">
            <input type="checkbox" v-model="useCondMemory" />
            Cond Memory
          </label>
          <label class="memory-toggle">
            <input type="checkbox" v-model="useNonCondMemory" />
            Non-Cond Memory
          </label>
        </div>
        <div class="memory-counts" v-if="lastCondUsed > 0 || lastNonCondUsed > 0">
          used {{ lastCondUsed }} cond, {{ lastNonCondUsed }} non-cond frames
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
          <button
            class="tool-button propagate-button"
            @click="handlePropagateWithoutMemory"
            :disabled="isSegmenting || isPropagating || !segmentationIsReady"
          >
            <span class="tool-icon">▶▷</span>
            <span class="tool-label">{{ isPropagating ? 'Propagating...' : 'Propagate Without Memory' }}</span>
          </button>
        </div>

        <h4 class="action-bar-title">Training Data</h4>
        <div class="tool-buttons">
          <button
            class="tool-button mark-training-button"
            :class="{ active: isMarkingMode }"
            @click="isMarkingMode = !isMarkingMode; if (isMarkingMode) isWorkingRangeMode = false"
          >
            <span class="tool-icon">✓</span>
            <span class="tool-label">{{ isMarkingMode ? 'Exit Marking Mode' : 'Mark Training Frames' }}</span>
          </button>
          <button
            class="tool-button working-range-button"
            :class="{ active: isWorkingRangeMode }"
            @click="isWorkingRangeMode = !isWorkingRangeMode; if (isWorkingRangeMode) isMarkingMode = false"
          >
            <span class="tool-icon">⊞</span>
            <span class="tool-label">{{ isWorkingRangeMode ? 'Exit Working Range' : 'Working Range' }}</span>
          </button>
        </div>
        <p v-if="isMarkingMode" class="marking-hint">
          Drag on the data track to mark frames as training data. Press Delete to remove.
        </p>

        <h4 class="action-bar-title">Pose Keypoints</h4>
        <div class="tool-buttons">
          <button
            class="tool-button"
            :class="{ active: isLabelingKeypoints }"
            @click="toggleLabelingMode"
            :disabled="isSegmenting || isPropagating"
          >
            <span class="tool-icon">+</span>
            <span class="tool-label">{{ isLabelingKeypoints ? 'Exit Labeling' : 'Label Keypoints' }}</span>
          </button>
        </div>
        <p v-if="isLabelingKeypoints" class="marking-hint">
          {{ labelingState === 'awaiting_front' ? 'Click front (nose)' : 'Click rear (tail)' }}
        </p>
        <p v-if="poseLabelCount > 0" class="marking-hint">
          {{ poseLabelCount }} frame(s) labeled
        </p>

        <h4 class="action-bar-title">Video Visibility</h4>
        <div class="mask-view-section">
          <label class="mask-view-label">Mask Source:</label>
          <select v-model="maskViewMode" class="mask-view-select">
            <option value="tracker">Tracker</option>
            <option value="detector" :disabled="!hasDetectorMasks">
              Detector {{ hasDetectorMasks ? '' : '(not available)' }}
            </option>
            <option value="final" :disabled="!hasFinalMasks">
              Final {{ hasFinalMasks ? '' : '(not available)' }}
            </option>
            <option value="obb" :disabled="!hasObbBboxes">
              OBB {{ hasObbBboxes ? '' : '(not available)' }}
            </option>
            <option value="seg" :disabled="!hasSegMasks">
              Seg {{ hasSegMasks ? '' : '(not available)' }}
            </option>
          </select>
        </div>
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
            :class="{ active: showTrackerMaskedFrames }"
            @click="showTrackerMaskedFrames = !showTrackerMaskedFrames"
          >
            <span class="tool-icon">◼</span>
            <span class="tool-label">{{ showTrackerMaskedFrames ? 'Tracker Masks' : 'Tracker Masks Off' }}</span>
          </button>
          <button
            class="tool-button toggle-button training-toggle"
            :class="{ active: showTrainingFrames }"
            @click="showTrainingFrames = !showTrainingFrames"
          >
            <span class="tool-icon">◼</span>
            <span class="tool-label">{{ showTrainingFrames ? 'Training Frames' : 'Training Frames Off' }}</span>
          </button>
          <button
            class="tool-button toggle-button confidence-toggle"
            :class="{ active: showConfidencePlot }"
            @click="showConfidencePlot = !showConfidencePlot"
          >
            <span class="tool-icon">📈</span>
            <span class="tool-label">{{ showConfidencePlot ? 'Confidence Plot' : 'Confidence Plot Off' }}</span>
          </button>
          <button
            class="tool-button toggle-button detector-confidence-toggle"
            :class="{ active: showDetectorConfidence }"
            @click="showDetectorConfidence = !showDetectorConfidence"
          >
            <span class="tool-icon">🔍</span>
            <span class="tool-label">{{ showDetectorConfidence ? 'Detector Confidence' : 'Detector Confidence Off' }}</span>
          </button>
          <button
            class="tool-button toggle-button"
            :class="{ active: showPoseConfidence }"
            @click="showPoseConfidence = !showPoseConfidence"
          >
            <span class="tool-icon">*</span>
            <span class="tool-label">{{ showPoseConfidence ? 'Pose Confidence' : 'Pose Confidence Off' }}</span>
          </button>
          <button
            class="tool-button toggle-button"
            :class="{ active: showPoseKeypoints }"
            @click="showPoseKeypoints = !showPoseKeypoints"
          >
            <span class="tool-icon">*</span>
            <span class="tool-label">{{ showPoseKeypoints ? 'Pose Keypoints' : 'Pose Keypoints Off' }}</span>
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

  <CondFrameGrid
    v-if="video && segmentationIsReady"
    :project-id="projectId"
    :video-id="videoId"
    :refresh-key="condFrameRefreshKey"
  />
  </div>
</template>

<style scoped>
.video-detail-page {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
}

.video-detail-container {
  display: flex;
  height: 100%;
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
  max-height: calc(100% - 240px);
  flex-shrink: 1;
  flex: 1;
  min-height: 0;
  display: flex;
  align-items: center;
  justify-content: center;
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

.tool-button.bounding-box.active {
  background-color: #dbeafe;
  border-color: #3b82f6;
  color: #1d4ed8;
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

.tool-button.confidence-toggle.active {
  background-color: #fef9c3;
  border-color: #f59e0b;
  color: #b45309;
}

.tool-button.detector-confidence-toggle.active {
  background-color: #fee2e2;
  border-color: #ef4444;
  color: #b91c1c;
}

.memory-options {
    display: flex;
    gap: 12px;
    align-items: center;
}

.memory-counts {
    font-size: 11px;
    color: #888;
}

.memory-toggle {
    display: flex;
    align-items: center;
    gap: 4px;
    font-size: 12px;
    color: #ccc;
    cursor: pointer;
    user-select: none;
}

.action-bar-title {
  margin: 1rem 0 0.5rem 0;
}

.action-bar-title:first-child {
  margin-top: 0;
}

.mask-view-section {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 0.75rem;
}

.mask-view-label {
  font-size: 0.85rem;
  color: #666;
}

.mask-view-select {
  flex: 1;
  padding: 0.4rem 0.5rem;
  border: 1px solid #ddd;
  border-radius: 4px;
  font-size: 0.85rem;
  background-color: white;
  cursor: pointer;
}

.mask-view-select:focus {
  outline: none;
  border-color: #2196f3;
}

.mask-view-select option:disabled {
  color: #999;
}
</style>
