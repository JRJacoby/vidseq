<script setup lang="ts">
import { ref, onMounted, computed } from 'vue'
import { useRouter } from 'vue-router'
import { getProject, getVideos, addVideos, segmentAllVideos, extractCroppedVideos, getCroppedVideoExists, type Video, type Project } from '@/services/api'
import FilePickerModal from '@/components/FilePickerModal.vue'
import { useProjectStore } from '@/stores/project'
import { useYOLO } from '@/composables/useYOLO'

const router = useRouter()
const projectStore = useProjectStore()

const project = ref<Project | null>(null)
const videos = ref<Video[]>([])
const isLoading = ref(false)
const showFilePicker = ref(false)
const isSegmenting = ref(false)
const isExtracting = ref(false)
const croppedVideoExists = ref<Record<number, boolean>>({})

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
  isTraining,
  isApplying,
  modelExists,
  trainModel,
  runInitialDetection,
  checkModelStatus,
} = useYOLO(projectId)

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
    videos.value = await getVideos(projectStore.currentProjectId)
  } catch (error) {
    console.error('Error loading videos:', error)
  } finally {
    isLoading.value = false
  }
}

onMounted(async () => {
  await loadProject()
  await loadVideos()
  await loadCroppedVideoStatus()
})

const handleAddVideos = () => {
  showFilePicker.value = true
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

const handleTrainModel = async () => {
  if (!projectId.value || isTraining.value) return
  try {
    await trainModel(projectId.value)
  } catch (e) {
    console.error('Failed to train model:', e)
  }
}

const handleRunInitialDetection = async () => {
  if (!projectId.value || isApplying.value || !modelExists.value) return
  try {
    await runInitialDetection(projectId.value)
  } catch (e) {
    console.error('Failed to run initial detection:', e)
  }
}

const handleSegmentAll = async () => {
  if (!projectId.value || isSegmenting.value) return
  isSegmenting.value = true
  try {
    await segmentAllVideos(projectId.value)
    await loadVideos()
  } catch (e: any) {
    console.error('Failed to segment all videos:', e)
    alert(e.message || 'Failed to start segmentation')
  } finally {
    isSegmenting.value = false
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
    await extractCroppedVideos(projectId.value)
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
                  :class="getStatusClass(video)"
                  @dblclick="handleVideoDoubleClick(video.id)"
                >
                  <div class="video-info">
                    <p class="video-name">{{ video.name }}</p>
                    <p class="video-path">{{ video.path }}</p>
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
                    <span v-if="getStatusDisplay(video)" class="status-badge" :class="getStatusClass(video)">
                      {{ getStatusDisplay(video) }}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        <aside class="sidebar">
          <button class="sidebar-button" @click="handleAddVideos">Add Videos</button>
          
          <h4 class="sidebar-section-title">Initial Detection</h4>
          <button 
            class="sidebar-button train-button"
            @click="handleTrainModel"
            :disabled="isTraining || isApplying"
          >
            <span class="button-label">{{ isTraining ? 'Training...' : 'Train Initial Detection Model' }}</span>
          </button>
          <button 
            class="sidebar-button apply-button"
            @click="handleRunInitialDetection"
            :disabled="isTraining || isApplying || !modelExists"
          >
            <span class="button-label">{{ isApplying ? 'Running...' : 'Run Initial Detection' }}</span>
          </button>
          
          <h4 class="sidebar-section-title">Segmentation</h4>
          <button
            class="sidebar-button segment-button"
            @click="handleSegmentAll"
            :disabled="isSegmenting"
          >
            <span class="button-label">{{ isSegmenting ? 'Starting...' : 'Segment All Videos' }}</span>
          </button>

          <h4 class="sidebar-section-title">Cropped Videos</h4>
          <button
            class="sidebar-button extract-button"
            @click="handleExtractCroppedVideos"
            :disabled="isExtracting"
          >
            <span class="button-label">{{ isExtracting ? 'Starting...' : 'Extract Cropped Videos' }}</span>
          </button>

          <div v-if="isTraining || isApplying || isSegmenting || isExtracting" class="status-indicator">
            {{ isTraining ? 'Training model...' : isApplying ? 'Running initial detection...' : isSegmenting ? 'Starting segmentation batch...' : 'Starting cropped video extraction...' }}
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
  justify-content: space-between;
  align-items: center;
  padding: 1.25rem;
  border: 1px solid #e0e0e0;
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.2s;
  background-color: white;
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
</style>
