<script setup lang="ts">
import { ref, onMounted, computed } from 'vue'
import { useRouter } from 'vue-router'
import { getProject, getVideos, addVideos, type Video, type Project } from '@/services/api'
import FilePickerModal from '@/components/FilePickerModal.vue'
import { useProjectStore } from '@/stores/project'
import { useYOLO } from '@/composables/useYOLO'

const router = useRouter()
const projectStore = useProjectStore()

const project = ref<Project | null>(null)
const videos = ref<Video[]>([])
const isLoading = ref(false)
const showFilePicker = ref(false)

const projectId = computed(() => projectStore.currentProjectId)

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

onMounted(() => {
  loadProject()
  loadVideos()
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
          <div v-else class="videos-list">
            <div 
              v-for="video in videos" 
              :key="video.id" 
              class="video-item"
              @dblclick="handleVideoDoubleClick(video.id)"
            >
              <p>ID: {{ video.id }}</p>
              <p>Name: {{ video.name }}</p>
              <p>Path: {{ video.path }}</p>
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
          <div v-if="isTraining || isApplying" class="status-indicator">
            {{ isTraining ? 'Training model...' : 'Running initial detection...' }}
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
  width: 200px;
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  padding: 1rem;
  border-left: 1px solid #e0e0e0;
}

.sidebar-button {
  padding: 0.75rem 1rem;
  border: 1px solid #ccc;
  border-radius: 4px;
  background-color: #f8f8f8;
  cursor: pointer;
  text-align: left;
}

.sidebar-button:hover {
  background-color: #e8e8e8;
}

.sidebar-button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.sidebar-section-title {
  margin: 1rem 0 0.5rem 0;
  font-size: 0.9rem;
  font-weight: 600;
  color: #666;
}

.button-label {
  display: block;
}

.status-indicator {
  margin-top: 0.5rem;
  padding: 0.5rem;
  font-size: 0.85rem;
  color: #666;
  font-style: italic;
}

.loading-state,
.empty-state {
  padding: 2rem;
  text-align: center;
  color: #666;
}

.videos-list {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  overflow: auto;
}

.video-item {
  padding: 1rem;
  border: 1px solid #e0e0e0;
  border-radius: 4px;
  cursor: pointer;
  transition: background-color 0.15s;
}

.video-item:hover {
  background-color: #f5f5f5;
}

.video-item:active {
  background-color: #e8e8e8;
}

.video-item p {
  margin: 0.25rem 0;
}
</style>
