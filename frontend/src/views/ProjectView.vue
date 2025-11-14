<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useRoute } from 'vue-router'
import { getVideos, type Video } from '@/services/api'

const route = useRoute()
const projectId = route.params.id

const videos = ref<Video[]>([])
const isLoading = ref(false)

onMounted(async () => {
  isLoading.value = true
  try {
    videos.value = await getVideos(Number(projectId))
  } catch (error) {
    console.error('Error loading videos:', error)
  } finally {
    isLoading.value = false
  }
})
</script>

<template>
  <div class="page-container">
    <header class="title-bar">
      <h1 class="title-bar-title">VidSeq - Animal Behavior Modeling from Raw Video</h1>
    </header>
    <div class="page-content">
      <aside class="sidebar">
        <button class="sidebar-button">Video Pipeline</button>
      </aside>
      <main class="main-screen">
        <h3 class="screen-title">Video Pipeline</h3>
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
      </main>
    </div>
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

.sidebar {
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

.screen-content {
  display: flex;
  flex-direction: column;
  flex: 1;
  min-height: 0;
  overflow: auto;
}
</style>