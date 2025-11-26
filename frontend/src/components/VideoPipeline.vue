<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { getProject, getVideos, addVideos, runSegmentation, type Video, type Project } from '@/services/api'
import FilePickerModal from '@/components/FilePickerModal.vue'
import SegmentationPromptModal from '@/components/SegmentationPromptModal.vue'
import { useProjectStore } from '@/stores/project'

const projectStore = useProjectStore()

const project = ref<Project | null>(null)
const videos = ref<Video[]>([])
const isLoading = ref(false)
const showFilePicker = ref(false)
const selectedVideoIds = ref<number[]>([])
const showSegmentationModal = ref(false)

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

const handleVideoSelect = (videoId: number) => {
  const index = selectedVideoIds.value.indexOf(videoId)
  if (index === -1) {
    selectedVideoIds.value.push(videoId)
  } else {
    selectedVideoIds.value.splice(index, 1)
  }
}

const handleSelectAll = () => {
  if (selectedVideoIds.value.length === videos.value.length) {
    selectedVideoIds.value = []
  } else {
    selectedVideoIds.value = videos.value.map(v => v.id)
  }
}

const handleRunSegmentation = () => {
  showSegmentationModal.value = true
}

const handleSegmentationConfirm = async (prompt: string) => {
  if (!projectStore.currentProjectId) return
  showSegmentationModal.value = false
  
  try {
    await runSegmentation(projectStore.currentProjectId, selectedVideoIds.value, prompt)
    console.log('Segmentation started successfully')
  } catch (error) {
    console.error('Error starting segmentation:', error)
  }
}

const handleSegmentationCancel = () => {
  showSegmentationModal.value = false
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
          <div v-else class="videos-section">
            <div class="videos-header">
              <button class="select-all-button" @click="handleSelectAll">
                {{ selectedVideoIds.length === videos.length ? 'Deselect All' : 'Select All' }}
              </button>
            </div>
            <div class="videos-list">
              <div 
                v-for="video in videos" 
                :key="video.id" 
                class="video-item"
                :class="{ 'video-selected': selectedVideoIds.includes(video.id) }"
                @click="handleVideoSelect(video.id)"
              >
                <p>ID: {{ video.id }}</p>
                <p>Name: {{ video.name }}</p>
                <p>Path: {{ video.path }}</p>
                <p>Has Segmentation: {{ video.has_segmentation }}</p>
              </div>
            </div>
          </div>
        </div>
        <aside class="sidebar">
          <button class="sidebar-button" @click="handleAddVideos">Add Videos</button>
          <button 
            class="sidebar-button" 
            @click="handleRunSegmentation"
            :disabled="selectedVideoIds.length === 0"
          >
            Run Automatic Preliminary Segmentation
          </button>
        </aside>
      </div>
    </div>
    <FilePickerModal
      v-if="showFilePicker"
      :initial-path="project?.path"
      @files-selected="handleFilesSelected"
      @cancel="handleFilePickerCancel"
    />
    <SegmentationPromptModal
      v-if="showSegmentationModal"
      @confirm="handleSegmentationConfirm"
      @cancel="handleSegmentationCancel"
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

.sidebar-button:hover:not(:disabled) {
  background-color: #e8e8e8;
}

.sidebar-button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.loading-state,
.empty-state {
  padding: 2rem;
  text-align: center;
  color: #666;
}

.videos-section {
  display: flex;
  flex-direction: column;
  height: 100%;
}

.videos-header {
  padding: 0.5rem 0;
  flex-shrink: 0;
}

.select-all-button {
  padding: 0.5rem 1rem;
  background-color: #f0f0f0;
  border: 1px solid #ccc;
  border-radius: 4px;
  cursor: pointer;
}

.select-all-button:hover {
  background-color: #e0e0e0;
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
  transition: background-color 0.2s;
}

.video-item:hover {
  background-color: #f5f5f5;
}

.video-selected {
  background-color: #e3f2fd;
  border-color: #4a90e2;
}

.video-item p {
  margin: 0.25rem 0;
}
</style>

