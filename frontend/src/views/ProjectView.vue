<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useRoute } from 'vue-router'
import { getProject, getVideos, addVideos, runSegmentation, type Video, type Project } from '@/services/api'
import FilePickerModal from '@/components/FilePickerModal.vue'
import SegmentationPromptModal from '@/components/SegmentationPromptModal.vue'

const route = useRoute()
const projectId = route.params.id

const project = ref<Project | null>(null)
const videos = ref<Video[]>([])
const isLoading = ref(false)
const showFilePicker = ref(false)
const selectedVideoIds = ref<number[]>([])
const showSegmentationModal = ref(false)

const loadProject = async () => {
  try {
    project.value = await getProject(Number(projectId))
  } catch (error) {
    console.error('Error loading project:', error)
  }
}

const loadVideos = async () => {
  isLoading.value = true
  try {
    videos.value = await getVideos(Number(projectId))
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
  showFilePicker.value = false
  
  try {
    await addVideos(Number(projectId), selectedPaths)
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
  showSegmentationModal.value = false
  
  try {
    await runSegmentation(Number(projectId), selectedVideoIds.value, prompt)
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
  <div class="page-container">
    <header class="title-bar">
      <h1 class="title-bar-title">VidSeq - Animal Behavior Modeling from Raw Video</h1>
    </header>
    <div class="page-content">
      <nav class="navbar">
        <button class="navbar-button">Video Pipeline</button>
      </nav>
      <main class="main-screen">
        <h3 class="screen-title">Video Pipeline</h3>
        <div class="main-content-area">
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
      </main>
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
.page-container {
  display: flex;
  flex-direction: column;
  height: 100vh;
  width: 100vw;
}

.title-bar {
  flex-shrink: 0;
}

.title-bar-title {
  margin: 0;
}

.page-content {
  display: flex;
  flex: 1;
  min-height: 0;
}

.navbar {
  flex-shrink: 0;
  width: 200px;
  display: flex;
  flex-direction: column;
}

.main-screen {
  display: flex;
  flex-direction: column;
  flex: 1;
  min-height: 0;
}

.screen-title {
  flex-shrink: 0;
}

.main-content-area {
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
}

.sidebar {
  flex-shrink: 0;
  width: 200px;
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.videos-section {
  display: flex;
  flex-direction: column;
  height: 100%;
}

.videos-header {
  padding: 0.5rem;
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

.sidebar-button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
</style>