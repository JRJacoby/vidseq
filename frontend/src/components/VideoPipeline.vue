<script setup lang="ts">
import { ref, onMounted, onUnmounted, computed } from 'vue'
import { useRouter } from 'vue-router'
import {
  getProject,
  getVideos,
  addVideos,
  deleteVideos,
  deleteVideosSegmentation,
  createVideosSegmentation,
  createSimpleSegmentation,
  createVideosExtraction,
  getCroppedVideoExists,
  getAlignedVideoExists,
  getAlignmentStatus,
  createAlignmentTraining,
  createVideosAlignment,
  clearAlignmentModel,
  clearAllAlignmentLabels,
  getPCAStatus,
  createPCA,
  addAssociatedVideos,
  coSegmentVideos,
  applyDetector,
  applyObbDetector,
  applySegDetector,
  exportDetectorBboxes,
  type Video,
  type Project,
  type AlignmentStatus,
  type PCAStatus,
} from '@/services/api'
import FilePickerModal from '@/components/FilePickerModal.vue'
import { useProjectStore } from '@/stores/project'
import { useDetector } from '@/composables/useDetector'
import { useObbDetector } from '@/composables/useObbDetector'
import { useSegDetector } from '@/composables/useSegDetector'

const router = useRouter()
const projectStore = useProjectStore()

const project = ref<Project | null>(null)
const videos = ref<Video[]>([])
const isLoading = ref(false)
const showFilePicker = ref(false)
const isSegmenting = ref(false)
const isExtracting = ref(false)
const isDeleting = ref(false)
const croppedVideoExists = ref<Record<number, boolean>>({})
const alignedVideoExists = ref<Record<number, boolean>>({})

// Video selection state
const selectedVideoIds = ref(new Set<number>())

const selectedCount = computed(() => selectedVideoIds.value.size)
const allSelected = computed(() => selectedVideoIds.value.size === videos.value.length && videos.value.length > 0)
const someSelected = computed(() => selectedVideoIds.value.size > 0 && !allSelected.value)
const selectedVideoIdsList = computed(() => [...selectedVideoIds.value])

// Associated videos
const expandedVideoIds = ref(new Set<number>())
const isCoSegmenting = ref(false)
const confidenceThreshold = ref(0.9)
const showAssociatedFilePicker = ref(false)
const isAddingAssociated = ref(false)

const toggleExpand = (videoId: number) => {
  const next = new Set(expandedVideoIds.value)
  if (next.has(videoId)) {
    next.delete(videoId)
  } else {
    next.add(videoId)
  }
  expandedVideoIds.value = next
}

const selectedWithAssociated = computed(() =>
  videos.value.filter(
    v => selectedVideoIds.value.has(v.id) && v.associated_video_id != null
  )
)

const toggleVideo = (id: number) => {
  const next = new Set(selectedVideoIds.value)
  if (next.has(id)) {
    next.delete(id)
  } else {
    next.add(id)
  }
  selectedVideoIds.value = next
}

const toggleAll = () => {
  if (allSelected.value) {
    selectedVideoIds.value = new Set()
  } else {
    selectedVideoIds.value = new Set(videos.value.map(v => v.id))
  }
}

// Alignment state
const alignmentStatus = ref<AlignmentStatus | null>(null)
const alignmentEpochs = ref(100)

// PCA state
const pcaStatus = ref<PCAStatus | null>(null)
const isRunningPCA = ref(false)
const pcaComponents = ref(20)
const isApplyingDetector = ref(false)
const isApplyingObb = ref(false)
const isApplyingSeg = ref(false)
const isExportingBboxes = ref(false)

const projectId = computed(() => projectStore.currentProjectId)

const getStatusDisplay = (video: Video) => {
  if (video.segmentation_status === 'in_progress') return 'In Progress'
  if (video.segmentation_status === 'segmented') return 'Segmented'
  return null
}

const getStatusClass = (video: Video) => {
  if (video.segmentation_status === 'in_progress') return 'in-progress'
  if (video.segmentation_status === 'segmented') return 'segmented'
  return null
}

const {
  isTraining: isDetectorTraining,
  modelExists: detectorModelExists,
  detectorType,
  startTraining: startDetectorTraining,
  checkStatus: checkDetectorStatus,
  setDetectorType,
} = useDetector(projectId)

const {
  isTraining: isObbTraining,
  modelExists: obbModelExists,
  startTraining: startObbTraining,
  checkStatus: checkObbStatus,
} = useObbDetector(projectId)

const {
  isTraining: isSegTraining,
  modelExists: segModelExists,
  startTraining: startSegTraining,
} = useSegDetector(projectId)

const loadProject = async () => {
  if (!projectStore.currentProjectId) return
  try {
    project.value = await getProject(projectStore.currentProjectId)
  } catch (error) {
    console.error('Error loading project:', error)
  }
}

const loadVideos = async () => {
  if (!projectStore.currentProjectId) return
  isLoading.value = true
  try {
    selectedVideoIds.value = new Set()
    videos.value = await getVideos(projectStore.currentProjectId)
  } catch (error) {
    console.error('Error loading videos:', error)
  } finally {
    isLoading.value = false
  }
}

// Poll alignment status while training is in progress
let alignmentPollInterval: ReturnType<typeof setInterval> | null = null
const pollAlignmentStatus = () => {
  // Clear any existing interval
  if (alignmentPollInterval) {
    clearInterval(alignmentPollInterval)
  }
  // Poll every 2 seconds
  alignmentPollInterval = setInterval(async () => {
    await loadAlignmentStatus()
    // Stop polling when training is done
    if (!alignmentStatus.value?.is_training) {
      if (alignmentPollInterval) {
        clearInterval(alignmentPollInterval)
        alignmentPollInterval = null
      }
    }
  }, 2000)
}

onMounted(async () => {
  await loadProject()
  await loadVideos()
  await loadCroppedVideoStatus()
  await loadAlignedVideoStatus()
  await loadAlignmentStatus()
  await loadPCAStatus()
  // If training is already in progress (e.g., user refreshed), start polling
  if (alignmentStatus.value?.is_training) {
    pollAlignmentStatus()
  }
})

onUnmounted(() => {
  // Clean up polling interval
  if (alignmentPollInterval) {
    clearInterval(alignmentPollInterval)
    alignmentPollInterval = null
  }
})

const handleAddVideos = () => {
  showFilePicker.value = true
}

const handleDeleteVideos = async () => {
  if (!projectStore.currentProjectId || isDeleting.value || selectedCount.value === 0) return
  const count = selectedCount.value
  if (!confirm(`Delete ${count} video(s)? This will remove all annotations, masks, and generated videos. Original video files will not be affected.`)) return
  isDeleting.value = true
  try {
    await deleteVideos(projectStore.currentProjectId, selectedVideoIdsList.value)
    await loadVideos()
    await loadCroppedVideoStatus()
    await loadAlignedVideoStatus()
  } catch (e: any) {
    console.error('Failed to delete videos:', e)
    alert(e.message || 'Failed to delete videos')
  } finally {
    isDeleting.value = false
  }
}

const handleFilesSelected = async (selectedPaths: string[]) => {
  if (!projectStore.currentProjectId) return
  showFilePicker.value = false
  
  try {
    await addVideos(projectStore.currentProjectId, selectedPaths)
    await loadVideos()
  } catch (error) {
    console.error('Error adding videos:', error)
  }
}

const handleFilePickerCancel = () => {
  showFilePicker.value = false
}

const handleVideoDoubleClick = (videoId: number) => {
  if (projectStore.currentProjectId) {
    router.push(`/project/${projectStore.currentProjectId}/video/${videoId}`)
  }
}

const handleSegmentAll = async () => {
  if (!projectId.value || isSegmenting.value) return
  isSegmenting.value = true
  try {
    await createVideosSegmentation(projectId.value, selectedVideoIdsList.value)
    await loadVideos()
  } catch (e: any) {
    console.error('Failed to segment all videos:', e)
    alert(e.message || 'Failed to start segmentation')
  } finally {
    isSegmenting.value = false
  }
}

const handleSimpleSegmentAll = async () => {
  if (!projectId.value || isSegmenting.value) return
  isSegmenting.value = true
  try {
    await createSimpleSegmentation(projectId.value, selectedVideoIdsList.value)
    await loadVideos()
  } catch (e: any) {
    console.error('Failed to start simple segmentation:', e)
    alert(e.message || 'Failed to start simple segmentation')
  } finally {
    isSegmenting.value = false
  }
}

const isDeletingSegmentations = ref(false)

const handleDeleteSegmentations = async () => {
  if (!projectStore.currentProjectId || isDeletingSegmentations.value || selectedCount.value === 0) return
  const count = selectedCount.value
  if (!confirm(`Delete segmentations for ${count} video(s)? This will remove all masks, conditioning frames, and frame data.`)) return
  isDeletingSegmentations.value = true
  try {
    await deleteVideosSegmentation(projectStore.currentProjectId, selectedVideoIdsList.value)
    await loadVideos()
  } catch (e: any) {
    console.error('Failed to delete segmentations:', e)
    alert(e.message || 'Failed to delete segmentations')
  } finally {
    isDeletingSegmentations.value = false
  }
}

const loadCroppedVideoStatus = async () => {
  if (!projectStore.currentProjectId) return
  const status: Record<number, boolean> = {}
  for (const video of videos.value) {
    try {
      const result = await getCroppedVideoExists(projectStore.currentProjectId, video.id)
      status[video.id] = result.exists
    } catch {
      status[video.id] = false
    }
  }
  croppedVideoExists.value = status
}

const handleExtractCroppedVideos = async () => {
  if (!projectId.value || isExtracting.value) return
  isExtracting.value = true
  try {
    await createVideosExtraction(projectId.value, selectedVideoIdsList.value)
    // Refresh status after starting extraction
    setTimeout(() => loadCroppedVideoStatus(), 1000)
  } catch (e: any) {
    console.error('Failed to extract cropped videos:', e)
    alert(e.message || 'Failed to start cropped video extraction')
  } finally {
    isExtracting.value = false
  }
}

const handleViewCropped = (videoId: number) => {
  if (projectStore.currentProjectId) {
    router.push(`/project/${projectStore.currentProjectId}/video/${videoId}/cropped`)
  }
}

const loadAlignedVideoStatus = async () => {
  if (!projectStore.currentProjectId) return
  const status: Record<number, boolean> = {}
  for (const video of videos.value) {
    try {
      const result = await getAlignedVideoExists(projectStore.currentProjectId, video.id)
      status[video.id] = result.exists
    } catch {
      status[video.id] = false
    }
  }
  alignedVideoExists.value = status
}

const handleViewAligned = (videoId: number) => {
  if (projectStore.currentProjectId) {
    router.push(`/project/${projectStore.currentProjectId}/video/${videoId}/aligned`)
  }
}

const loadAlignmentStatus = async () => {
  if (!projectStore.currentProjectId) return
  try {
    alignmentStatus.value = await getAlignmentStatus(projectStore.currentProjectId)
  } catch (e) {
    console.error('Failed to load alignment status:', e)
  }
}

const handleTrainAlignment = async () => {
  if (!projectId.value || alignmentStatus.value?.is_training) return
  try {
    // Fire-and-forget: endpoint returns immediately after starting training
    await createAlignmentTraining(projectId.value, alignmentEpochs.value, true, 100, 50, selectedVideoIdsList.value)
    // Refresh status - will now show is_training=true
    await loadAlignmentStatus()
    // Start polling to detect when training completes
    pollAlignmentStatus()
  } catch (e: any) {
    console.error('Failed to start alignment training:', e)
    alert(e.message || 'Failed to start alignment training')
  }
}

const handleApplyAlignment = async () => {
  if (!projectId.value) return

  // If already aligning, just navigate to view progress
  if (alignmentStatus.value?.is_applying) {
    router.push(`/project/${projectId.value}/alignment`)
    return
  }

  try {
    // Start alignment in background (returns immediately)
    await createVideosAlignment(projectId.value, selectedVideoIdsList.value)
    // Navigate to Alignment screen to monitor progress
    router.push(`/project/${projectId.value}/alignment`)
  } catch (e: any) {
    console.error('Failed to start alignment:', e)
    alert(e.message || 'Failed to start alignment')
  }
}

const handleClearAlignmentModel = async () => {
  if (!projectId.value) return
  if (!confirm('Delete the alignment model? You will need to retrain.')) return
  try {
    await clearAlignmentModel(projectId.value)
    await loadAlignmentStatus()
  } catch (e: any) {
    console.error('Failed to clear alignment model:', e)
    alert(e.message || 'Failed to clear alignment model')
  }
}

const handleClearAlignmentLabels = async () => {
  if (!projectId.value) return
  if (!confirm('Delete ALL alignment labels across ALL videos?')) return
  try {
    await clearAllAlignmentLabels(projectId.value)
    await loadAlignmentStatus()
  } catch (e: any) {
    console.error('Failed to clear alignment labels:', e)
    alert(e.message || 'Failed to clear alignment labels')
  }
}

// PCA handlers
const loadPCAStatus = async () => {
  if (!projectStore.currentProjectId) return
  try {
    pcaStatus.value = await getPCAStatus(projectStore.currentProjectId)
  } catch (e) {
    console.error('Failed to load PCA status:', e)
  }
}

const handleRunPCA = async () => {
  if (!projectId.value || isRunningPCA.value) return
  isRunningPCA.value = true
  try {
    await createPCA(projectId.value, pcaComponents.value, selectedVideoIdsList.value)
    await loadPCAStatus()
  } catch (e: any) {
    console.error('Failed to run PCA:', e)
    alert(e.message || 'Failed to run PCA')
  } finally {
    isRunningPCA.value = false
  }
}

const handleTrainDetector = async () => {
  if (!projectId.value || isDetectorTraining.value) return
  try {
    await startDetectorTraining(1000, selectedVideoIdsList.value)
    // Navigate to detector training page to see progress
    router.push(`/project/${projectId.value}/detector`)
  } catch (e: any) {
    console.error('Failed to start detector training:', e)
    alert(e.message || 'Failed to start detector training')
  }
}

const handleApplyDetector = async () => {
  if (!projectId.value || isApplyingDetector.value) return
  isApplyingDetector.value = true
  try {
    await applyDetector(projectId.value, selectedVideoIdsList.value)
    await loadVideos()
  } catch (e: any) {
    console.error('Failed to apply detector:', e)
    alert(e.message || 'Failed to apply detector')
  } finally {
    isApplyingDetector.value = false
  }
}

const handleTrainObb = async () => {
  if (!projectId.value || isObbTraining.value) return
  try {
    await startObbTraining(1000, selectedVideoIdsList.value)
    router.push(`/project/${projectId.value}/detector`)
  } catch (e: any) {
    alert(e.message || 'Failed to start OBB training')
  }
}

const handleApplyObb = async () => {
  if (!projectId.value || isApplyingObb.value) return
  isApplyingObb.value = true
  try {
    await applyObbDetector(projectId.value, selectedVideoIdsList.value)
    await loadVideos()
  } catch (e: any) {
    console.error('Failed to apply OBB detector:', e)
    alert(e.message || 'Failed to apply OBB detector')
  } finally {
    isApplyingObb.value = false
  }
}

const handleTrainSeg = async () => {
  if (!projectId.value || isSegTraining.value) return
  try {
    await startSegTraining(1000, selectedVideoIdsList.value)
    router.push(`/project/${projectId.value}/detector`)
  } catch (e: any) {
    alert(e.message || 'Failed to start seg training')
  }
}

const handleExportBboxes = async () => {
  if (!projectId.value || isExportingBboxes.value) return
  isExportingBboxes.value = true
  try {
    const result = await exportDetectorBboxes(projectId.value, selectedVideoIdsList.value)
    alert(`Exported ${result.row_count} rows to:\n${result.path}`)
  } catch (e: any) {
    console.error('Failed to export detector bboxes:', e)
    alert(e.message || 'Failed to export detector bboxes')
  } finally {
    isExportingBboxes.value = false
  }
}

const handleApplySeg = async () => {
  if (!projectId.value || isApplyingSeg.value) return
  isApplyingSeg.value = true
  try {
    await applySegDetector(projectId.value, selectedVideoIdsList.value)
    await loadVideos()
  } catch (e: any) {
    console.error('Failed to apply seg detector:', e)
    alert(e.message || 'Failed to apply seg detector')
  } finally {
    isApplyingSeg.value = false
  }
}

const handleCoSegment = async () => {
  if (!projectId.value || isCoSegmenting.value) return
  isCoSegmenting.value = true
  try {
    const ids = selectedWithAssociated.value.map(v => v.id)
    await coSegmentVideos(projectId.value, ids, confidenceThreshold.value)
    await loadVideos()
  } catch (e: any) {
    alert(e.message || 'Co-segmentation failed')
  } finally {
    isCoSegmenting.value = false
  }
}

const handleAssociatedFileSelected = async (paths: string[]) => {
  showAssociatedFilePicker.value = false
  if (!projectStore.currentProjectId || paths.length === 0) return

  isAddingAssociated.value = true
  try {
    await addAssociatedVideos(projectStore.currentProjectId, paths[0])
    await loadVideos()
  } catch (e: any) {
    alert(e.message || 'Failed to add associated videos')
  } finally {
    isAddingAssociated.value = false
  }
}

const hasAlignedVideos = computed(() => {
  return Object.values(alignedVideoExists.value).some(exists => exists)
})

// Sorting logic
type SortOption = 'name' | 'min_confidence' | 'p50_confidence' | 'p95_confidence'
const sortBy = ref<SortOption>('name')
const sortOrder = ref<'asc' | 'desc'>('asc')

const toggleSortOrder = () => {
  sortOrder.value = sortOrder.value === 'asc' ? 'desc' : 'asc'
}

const sortedVideos = computed(() => {
  return [...videos.value].sort((a, b) => {
    let result = 0
    
    if (sortBy.value === 'name') {
      result = a.name.localeCompare(b.name)
    } else {
      // Sort by confidence stats
      // Handle missing stats: push to bottom regardless of sort order (effectively)
      // or treat as -1. Let's treat undefined as -infinity for asc, +infinity for desc?
      // Better: always put unsegmented/no-stats videos at the bottom.
      
      const valA = a[sortBy.value]
      const valB = b[sortBy.value]
      
      if (valA === undefined && valB === undefined) return 0
      if (valA === undefined) return 1 // A goes to bottom
      if (valB === undefined) return -1 // B goes to bottom
      
      result = valA - valB
    }
    
    return sortOrder.value === 'asc' ? result : -result
  })
})

const hasStats = (video: Video) => {
  return video.min_confidence !== undefined && video.min_confidence !== null
}

const formatScore = (score: number | undefined) => {
  if (score === undefined || score === null) return 'N/A'
  return (score * 100).toFixed(1) + '%'
}
</script>

<template>
  <div class="pipeline-container">
    <div class="pipeline-content">
      <h3 class="screen-title">Video Pipeline</h3>
      <div class="content-area">
        <div class="screen-content">
          <div v-if="isLoading" class="loading-state">
            Loading videos...
          </div>
          <div v-else-if="videos.length === 0" class="empty-state">
            Add videos to get started.
          </div>
            <div v-else class="videos-list-container">
              <div class="sort-controls">
                <input
                  type="checkbox"
                  class="select-all-checkbox"
                  :checked="allSelected"
                  :indeterminate="someSelected"
                  @change="toggleAll"
                />
                <span class="sort-label">Sort by:</span>
                <select v-model="sortBy" class="sort-select">
                  <option value="name">Name</option>
                  <option value="min_confidence">Min Confidence</option>
                  <option value="p50_confidence">Median Confidence</option>
                  <option value="p95_confidence">95% Confidence</option>
                </select>
                <button class="sort-order-btn" @click="toggleSortOrder" title="Toggle Sort Order">
                  {{ sortOrder === 'asc' ? '↑' : '↓' }}
                </button>
              </div>

              <div class="videos-list">
                <div
                  v-for="video in sortedVideos"
                  :key="video.id"
                  class="video-item"
                  :class="[getStatusClass(video), { selected: selectedVideoIds.has(video.id) }]"
                >
                  <div class="video-item-row" @dblclick="handleVideoDoubleClick(video.id)">
                    <input
                      type="checkbox"
                      class="video-checkbox"
                      :checked="selectedVideoIds.has(video.id)"
                      @change="toggleVideo(video.id)"
                      @click.stop
                    />
                    <div class="video-info">
                      <p class="video-name">{{ video.name }}</p>
                      <p class="video-path">{{ video.path }}</p>
                      <div v-if="video.training_frame_count" class="video-stats">
                        <span class="stat-item" title="Training Frames">
                          Training frames: <strong>{{ video.training_frame_count.toLocaleString() }}</strong>
                        </span>
                      </div>
                      <div v-if="hasStats(video)" class="video-stats">
                        <span class="stat-item" title="Minimum Confidence">
                          Min: <strong>{{ formatScore(video.min_confidence) }}</strong>
                        </span>
                        <span class="stat-item" title="Median Confidence">
                          Med: <strong>{{ formatScore(video.p50_confidence) }}</strong>
                        </span>
                        <span class="stat-item" title="95th Percentile Confidence">
                          P95: <strong>{{ formatScore(video.p95_confidence) }}</strong>
                        </span>
                      </div>
                    </div>
                    <div class="video-actions">
                      <button
                        v-if="croppedVideoExists[video.id]"
                        class="view-cropped-button"
                        @click.stop="handleViewCropped(video.id)"
                      >
                        View Cropped
                      </button>
                      <button
                        v-if="alignedVideoExists[video.id]"
                        class="view-aligned-button"
                        @click.stop="handleViewAligned(video.id)"
                      >
                        View Aligned
                      </button>
                      <span v-if="getStatusDisplay(video)" class="status-badge" :class="getStatusClass(video)">
                        {{ getStatusDisplay(video) }}
                      </span>
                      <button
                        v-if="video.associated_video_id"
                        class="expand-toggle"
                        @click.stop="toggleExpand(video.id)"
                      >
                        {{ expandedVideoIds.has(video.id) ? '&#9662; 1 associated' : '&#9656; 1 associated' }}
                      </button>
                    </div>
                  </div>

                  <div
                    v-if="video.associated_video_id && expandedVideoIds.has(video.id)"
                    class="associated-video-expanded"
                  >
                    <router-link
                      :to="`/project/${projectId}/video/${video.id}/associated`"
                      class="associated-link"
                    >
                      View Associated Video &rarr;
                    </router-link>
                  </div>
                </div>
              </div>
            </div>
          </div>
        <aside class="sidebar">
          <button class="sidebar-button" @click="handleAddVideos">Add Videos</button>
          <button
            class="sidebar-button delete-button"
            @click="handleDeleteVideos"
            :disabled="isDeleting || selectedCount === 0"
          >
            <span class="button-label">{{ isDeleting ? 'Deleting...' : `Delete ${selectedCount} Videos` }}</span>
          </button>

          <h4 class="sidebar-section-title">Segmentation</h4>
          <button
            class="sidebar-button segment-button"
            @click="handleSegmentAll"
            :disabled="isSegmenting || selectedCount === 0"
          >
            <span class="button-label">{{ isSegmenting ? 'Starting...' : `Segment ${selectedCount} Videos` }}</span>
          </button>
          <button
            class="sidebar-button"
            @click="handleSimpleSegmentAll"
            :disabled="isSegmenting || selectedCount === 0"
          >
            <span class="button-label">{{ isSegmenting ? 'Segmenting...' : 'Simple Segment All' }}</span>
          </button>
          <button
            class="sidebar-button delete-button"
            @click="handleDeleteSegmentations"
            :disabled="isDeletingSegmentations || selectedCount === 0"
          >
            <span class="button-label">{{ isDeletingSegmentations ? 'Deleting...' : 'Delete Segmentations' }}</span>
          </button>

          <h4 class="sidebar-section-title">Cropped Videos</h4>
          <button
            class="sidebar-button extract-button"
            @click="handleExtractCroppedVideos"
            :disabled="isExtracting || selectedCount === 0"
          >
            <span class="button-label">{{ isExtracting ? 'Starting...' : `Extract ${selectedCount} Cropped Videos` }}</span>
          </button>

          <h4 class="sidebar-section-title">Egocentric Alignment</h4>
          <div class="alignment-info">
            <span class="info-label">Labels:</span>
            <span class="info-value">{{ alignmentStatus?.label_count ?? 0 }}</span>
          </div>
          <div class="epochs-input">
            <label for="epochs">Max Epochs:</label>
            <input
              id="epochs"
              v-model.number="alignmentEpochs"
              type="number"
              min="1"
              max="500"
              class="epochs-field"
            />
          </div>
          <button
            class="sidebar-button train-alignment-button"
            @click="handleTrainAlignment"
            :disabled="alignmentStatus?.is_training || alignmentStatus?.is_applying || (alignmentStatus?.label_count ?? 0) === 0 || selectedCount === 0"
          >
            <span class="button-label">{{ alignmentStatus?.is_training ? 'Training...' : 'Train Alignment Model' }}</span>
          </button>
          <button
            class="sidebar-button apply-alignment-button"
            @click="handleApplyAlignment"
            :disabled="alignmentStatus?.is_training || (!alignmentStatus?.model_trained && !alignmentStatus?.is_applying) || selectedCount === 0"
          >
            <span class="button-label">{{ alignmentStatus?.is_applying ? 'View Progress' : `Align ${selectedCount} Videos` }}</span>
          </button>

          <div class="alignment-clear-buttons">
            <button
              class="clear-button"
              @click="handleClearAlignmentModel"
              :disabled="!alignmentStatus?.model_trained"
              title="Delete trained model"
            >
              Clear Model
            </button>
            <button
              class="clear-button"
              @click="handleClearAlignmentLabels"
              :disabled="(alignmentStatus?.label_count ?? 0) === 0"
              title="Delete all training labels"
            >
              Clear Labels
            </button>
          </div>

          <h4 class="sidebar-section-title">PCA</h4>
          <div v-if="pcaStatus?.has_pca" class="pca-info">
            <div class="info-row">
              <span class="info-label">Components:</span>
              <span class="info-value">{{ pcaStatus.n_components }}</span>
            </div>
            <div class="info-row">
              <span class="info-label">Frames:</span>
              <span class="info-value">{{ pcaStatus.total_frames?.toLocaleString() }}</span>
            </div>
          </div>
          <div v-else class="pca-info">
            <span class="info-label">No PCA computed yet</span>
          </div>
          <div class="pca-input">
            <label for="pca-components">Components:</label>
            <input
              id="pca-components"
              v-model.number="pcaComponents"
              type="number"
              min="1"
              max="100"
              class="pca-field"
            />
          </div>
          <button
            class="sidebar-button run-pca-button"
            @click="handleRunPCA"
            :disabled="isRunningPCA || !hasAlignedVideos || selectedCount === 0"
          >
            <span class="button-label">{{ isRunningPCA ? 'Running PCA...' : `Run PCA (${selectedCount} Videos)` }}</span>
          </button>

          <h4 class="sidebar-section-title">Detector</h4>
          <div class="detector-type-selector">
            <label class="detector-radio">
              <input type="radio" value="rtdetr" :checked="detectorType === 'rtdetr'" @change="setDetectorType('rtdetr')" :disabled="isDetectorTraining" />
              RT-DETR
            </label>
            <label class="detector-radio">
              <input type="radio" value="yolo" :checked="detectorType === 'yolo'" @change="setDetectorType('yolo')" :disabled="isDetectorTraining" />
              YOLO
            </label>
          </div>
          <button
            class="sidebar-button train-detector-button"
            @click="handleTrainDetector"
            :disabled="isDetectorTraining || selectedCount === 0"
          >
            <span class="button-label">{{ isDetectorTraining ? 'Training...' : 'Train Detector' }}</span>
          </button>
          <button
            v-if="detectorModelExists || isDetectorTraining"
            class="sidebar-button view-detector-button"
            @click="router.push(`/project/${projectId}/detector`)"
          >
            <span class="button-label">View Detector Training</span>
          </button>
          <button
            v-if="detectorModelExists"
            class="sidebar-button"
            @click="handleApplyDetector"
            :disabled="isApplyingDetector || isDetectorTraining || selectedCount === 0"
          >
            <span class="button-label">{{ isApplyingDetector ? 'Applying...' : 'Apply Detector' }}</span>
          </button>
          <button
            class="sidebar-button"
            @click="handleExportBboxes"
            :disabled="isExportingBboxes || selectedCount === 0"
          >
            <span class="button-label">{{ isExportingBboxes ? 'Exporting...' : 'Export Bboxes' }}</span>
          </button>

          <h4 class="sidebar-section-title">OBB Detector</h4>
          <button
            class="sidebar-button"
            @click="handleTrainObb"
            :disabled="isObbTraining || isDetectorTraining || selectedCount === 0"
          >
            <span class="button-label">{{ isObbTraining ? 'Training...' : 'Train OBB Detector' }}</span>
          </button>
          <button
            v-if="obbModelExists"
            class="sidebar-button"
            @click="handleApplyObb"
            :disabled="isApplyingObb || isObbTraining || selectedCount === 0"
          >
            <span class="button-label">{{ isApplyingObb ? 'Applying...' : 'Apply OBB Detector' }}</span>
          </button>

          <h4 class="sidebar-section-title">Seg Detector</h4>
          <button
            class="sidebar-button"
            @click="handleTrainSeg"
            :disabled="isSegTraining || isDetectorTraining || isObbTraining || selectedCount === 0"
          >
            <span class="button-label">{{ isSegTraining ? 'Training...' : 'Train Seg Detector' }}</span>
          </button>
          <button
            v-if="segModelExists"
            class="sidebar-button"
            @click="handleApplySeg"
            :disabled="isApplyingSeg || isSegTraining || selectedCount === 0"
          >
            <span class="button-label">{{ isApplyingSeg ? 'Applying...' : 'Apply Seg Detector' }}</span>
          </button>

          <h4 class="sidebar-section-title">Associated Videos</h4>
          <button
            class="sidebar-button"
            @click="showAssociatedFilePicker = true"
            :disabled="isAddingAssociated"
          >
            <span class="button-label">{{ isAddingAssociated ? 'Adding...' : 'Add Associated Videos' }}</span>
          </button>
          <div class="co-segment-controls">
            <button
              class="sidebar-button"
              @click="handleCoSegment"
              :disabled="isCoSegmenting || selectedWithAssociated.length === 0"
            >
              <span class="button-label">{{ isCoSegmenting ? 'Co-Segmenting...' : `Co-Segment ${selectedWithAssociated.length} Videos` }}</span>
            </button>
            <label class="threshold-label">
              Confidence:
              <input
                type="number"
                v-model.number="confidenceThreshold"
                min="0"
                max="1"
                step="0.05"
                class="threshold-input"
              />
            </label>
          </div>

          <div v-if="isDeleting || isDeletingSegmentations || isSegmenting || isExtracting || alignmentStatus?.is_training || alignmentStatus?.is_applying || isRunningPCA || isDetectorTraining || isCoSegmenting || isAddingAssociated" class="status-indicator">
            {{ isDeleting ? 'Deleting videos...' : isDeletingSegmentations ? 'Deleting segmentations...' : isSegmenting ? 'Starting segmentation batch...' : isExtracting ? 'Starting cropped video extraction...' : alignmentStatus?.is_training ? 'Training alignment model...' : alignmentStatus?.is_applying ? 'Applying alignment...' : isRunningPCA ? 'Running PCA...' : isDetectorTraining ? 'Training detector...' : isCoSegmenting ? 'Co-segmenting videos...' : 'Adding associated videos...' }}
          </div>
        </aside>
      </div>
    </div>
    <FilePickerModal
      v-if="showFilePicker"
      :initial-path="project?.path"
      @files-selected="handleFilesSelected"
      @cancel="handleFilePickerCancel"
    />
    <FilePickerModal
      v-if="showAssociatedFilePicker"
      :initial-path="project?.path"
      :accept="['.json']"
      :single-select="true"
      @files-selected="handleAssociatedFileSelected"
      @cancel="showAssociatedFilePicker = false"
    />
  </div>
</template>

<style scoped>
.pipeline-container {
  display: flex;
  flex-direction: column;
  flex: 1;
  min-height: 0;
}

.pipeline-content {
  display: flex;
  flex-direction: column;
  flex: 1;
  min-height: 0;
}

.screen-title {
  flex-shrink: 0;
  padding: 1rem;
  margin: 0;
  border-bottom: 1px solid #e0e0e0;
}

.content-area {
  display: flex;
  flex: 1;
  min-height: 0;
}

.screen-content {
  display: flex;
  flex-direction: column;
  flex: 1;
  min-height: 0;
  overflow: auto;
  padding: 1rem;
}

.sidebar {
  flex-shrink: 0;
  width: 250px;
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  padding: 1rem;
  border-left: 1px solid #e0e0e0;
  background-color: #fcfcfc;
}

.sidebar-button {
  padding: 0.75rem 1rem;
  border: 1px solid #ccc;
  border-radius: 4px;
  background-color: #f8f8f8;
  cursor: pointer;
  text-align: left;
  font-weight: 500;
  transition: all 0.2s;
}

.sidebar-button:hover:not(:disabled) {
  background-color: #e8e8e8;
  border-color: #999;
}

.sidebar-button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.delete-button {
  background-color: #fef2f2;
  border-color: #fca5a5;
  color: #dc2626;
}

.delete-button:hover:not(:disabled) {
  background-color: #fee2e2;
  border-color: #dc2626;
}

.sidebar-section-title {
  margin: 1.5rem 0 0.5rem 0;
  font-size: 0.85rem;
  font-weight: 700;
  color: #888;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.button-label {
  display: block;
}

.status-indicator {
  margin-top: 1rem;
  padding: 0.75rem;
  font-size: 0.85rem;
  color: #856404;
  background-color: #fff3cd;
  border-radius: 4px;
  font-style: italic;
}

.loading-state,
.empty-state {
  padding: 3rem;
  text-align: center;
  color: #888;
}

.videos-list {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.video-item {
  display: flex;
  flex-direction: column;
  border: 1px solid #e0e0e0;
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.2s;
  background-color: white;
}

.video-item-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1.25rem;
}

.video-item:hover {
  border-color: #aaa;
  box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}

.video-item.in-progress {
  border-left: 4px solid #ffc107;
}

.video-item.segmented {
  border-left: 4px solid #28a745;
}

.video-info {
  flex: 1;
}

.video-name {
  margin: 0;
  font-weight: 600;
  font-size: 1.1rem;
}

.video-path {
  margin: 0.25rem 0 0 0;
  font-size: 0.85rem;
  color: #666;
  font-family: monospace;
}

.video-actions {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  margin-left: 1.5rem;
}

.view-cropped-button {
  padding: 0.4rem 0.8rem;
  border: 1px solid #0366d6;
  border-radius: 4px;
  background-color: #f1f8ff;
  color: #0366d6;
  font-size: 0.85rem;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s;
}

.view-cropped-button:hover {
  background-color: #0366d6;
  color: white;
}

.view-aligned-button {
  padding: 0.4rem 0.8rem;
  border: 1px solid #28a745;
  border-radius: 4px;
  background-color: #f0fff4;
  color: #28a745;
  font-size: 0.85rem;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s;
}

.view-aligned-button:hover {
  background-color: #28a745;
  color: white;
}

.status-badge {
  padding: 0.4rem 0.8rem;
  border-radius: 20px;
  font-size: 0.85rem;
  font-weight: 600;
}

.status-badge.in-progress {
  background-color: #fff3cd;
  color: #856404;
}

.status-badge.segmented {
  background-color: #d4edda;
  color: #155724;
}

/* Sorting Controls */
.videos-list-container {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.sort-controls {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding-bottom: 0.5rem;
  border-bottom: 1px solid #eee;
}

.sort-label {
  font-size: 0.85rem;
  color: #666;
  font-weight: 500;
}

.sort-select {
  padding: 0.3rem 0.5rem;
  border: 1px solid #ccc;
  border-radius: 4px;
  font-size: 0.85rem;
  background-color: white;
  cursor: pointer;
}

.sort-order-btn {
  padding: 0.3rem 0.6rem;
  border: 1px solid #ccc;
  border-radius: 4px;
  background-color: #f8f8f8;
  cursor: pointer;
  font-size: 0.85rem;
  transition: all 0.2s;
}

.sort-order-btn:hover {
  background-color: #e8e8e8;
}

/* Video Stats */
.video-stats {
  display: flex;
  gap: 1rem;
  margin-top: 0.5rem;
  font-size: 0.8rem;
  color: #555;
}

.stat-item {
  background-color: #f0f4f8;
  padding: 0.2rem 0.6rem;
  border-radius: 12px;
  border: 1px solid #e1e4e8;
}

.stat-item strong {
  color: #0366d6;
}

/* Alignment Controls */
.alignment-info {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.5rem 0;
}

.info-label {
  font-size: 0.85rem;
  color: #666;
}

.info-value {
  font-size: 0.9rem;
  font-weight: 600;
  color: #a855f7;
}

.epochs-input {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 0.5rem;
}

.epochs-input label {
  font-size: 0.85rem;
  color: #666;
}

.epochs-field {
  width: 60px;
  padding: 0.3rem 0.5rem;
  border: 1px solid #ccc;
  border-radius: 4px;
  font-size: 0.85rem;
}

.train-alignment-button,
.apply-alignment-button {
  margin-top: 0.25rem;
}

.alignment-clear-buttons {
  display: flex;
  gap: 0.5rem;
  margin-top: 0.75rem;
}

.clear-button {
  flex: 1;
  padding: 0.4rem 0.5rem;
  border: 1px solid #fca5a5;
  border-radius: 4px;
  background-color: #fef2f2;
  color: #dc2626;
  cursor: pointer;
  font-size: 0.8rem;
  transition: all 0.2s;
}

.clear-button:hover:not(:disabled) {
  background-color: #fee2e2;
  border-color: #dc2626;
}

.clear-button:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

/* PCA Controls */
.pca-info {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
  padding: 0.5rem 0;
}

.info-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.pca-input {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 0.5rem;
}

.pca-input label {
  font-size: 0.85rem;
  color: #666;
}

.pca-field {
  width: 60px;
  padding: 0.3rem 0.5rem;
  border: 1px solid #ccc;
  border-radius: 4px;
  font-size: 0.85rem;
}

.run-pca-button {
  margin-top: 0.25rem;
}

.select-all-checkbox {
  cursor: pointer;
  width: 16px;
  height: 16px;
  margin-right: 0.25rem;
}

.video-checkbox {
  cursor: pointer;
  width: 16px;
  height: 16px;
  flex-shrink: 0;
  margin-right: 1rem;
}

.video-item.selected {
  background-color: #f0f7ff;
  border-color: #a8d1ff;
}

.detector-type-selector {
  display: flex;
  gap: 1rem;
  padding: 0.25rem 0;
}

.detector-radio {
  display: flex;
  align-items: center;
  gap: 0.3rem;
  font-size: 0.85rem;
  color: #555;
  cursor: pointer;
}

.detector-radio input[type="radio"] {
  cursor: pointer;
}

.detector-radio input[type="radio"]:disabled {
  cursor: not-allowed;
}

/* Associated Videos */
.associated-video-expanded {
  padding: 8px 16px 8px 40px;
  background: #f5f5f5;
  border-top: 1px solid #e0e0e0;
}

.associated-link {
  color: #0366d6;
  text-decoration: none;
  font-size: 0.9rem;
  font-weight: 500;
}

.associated-link:hover {
  text-decoration: underline;
}

.expand-toggle {
  background: none;
  border: none;
  cursor: pointer;
  font-size: 0.85em;
  color: #666;
  padding: 2px 6px;
  white-space: nowrap;
}

.expand-toggle:hover {
  color: #333;
}

.co-segment-controls {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.threshold-label {
  font-size: 0.85em;
  color: #666;
  display: flex;
  align-items: center;
  gap: 6px;
}

.threshold-input {
  width: 60px;
  padding: 2px 4px;
  border: 1px solid #ccc;
  border-radius: 4px;
  font-size: 0.85rem;
}
</style>
