<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useRoute } from 'vue-router'
import { getProject, getVideos, addVideos, type Video, type Project } from '@/services/api'
import FilePickerModal from '@/components/FilePickerModal.vue'

const route = useRoute()
const projectId = route.params.id

const project = ref<Project | null>(null)
const videos = ref<Video[]>([])
const isLoading = ref(false)
const showFilePicker = ref(false)

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
            <div v-else class="videos-list">
              <div v-for="video in videos" :key="video.id" class="video-item">
                <p>ID: {{ video.id }}</p>
                <p>Name: {{ video.name }}</p>
                <p>Path: {{ video.path }}</p>
              </div>
            </div>
          </div>
          <aside class="sidebar">
            <button class="sidebar-button" @click="handleAddVideos">Add Videos</button>
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
}
</style>