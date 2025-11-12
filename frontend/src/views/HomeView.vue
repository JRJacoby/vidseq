<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { getProjects, type Project } from '@/services/api'
import RecentProjects from '@/components/RecentProjects.vue'
import NewProject from '@/components/NewProject.vue'

const recentProjects = ref<Project[]>([])
const isLoading = ref(false)
const currentView = ref<'recent-projects' | 'new-project'>('recent-projects')

onMounted(async () => {
  isLoading.value = true
  try {
    recentProjects.value = await getProjects()
  } catch (error) {
    console.error('Error loading projects:', error)
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
        <button class="sidebar-button-primary" @click="currentView = 'new-project'">New Project</button>
        <button class="sidebar-button-secondary">Open Project</button>
      </aside>
      <main class="main-screen">
        <RecentProjects 
          v-if="currentView === 'recent-projects'"
          :projects="recentProjects"
          :is-loading="isLoading"
        />
        <NewProject v-else-if="currentView === 'new-project'" />
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
  background-color: #2c3e50;
  color: white;
  padding: 1rem 2rem;
}

.title-bar-title {
  margin: 0;
  font-size: 1.25rem;
  font-weight: 600;
}

.page-content {
  display: flex;
  flex: 1;
  min-height: 0;
}

.sidebar {
  flex-shrink: 0;
  width: 200px;
  background-color: #f5f5f5;
  padding: 1.5rem 1rem;
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.sidebar-button-primary,
.sidebar-button-secondary {
  padding: 0.75rem 1rem;
  border: none;
  border-radius: 0.25rem;
  font-size: 0.9375rem;
  cursor: pointer;
  transition: background-color 0.2s;
}

.sidebar-button-primary {
  background-color: #3498db;
  color: white;
}

.sidebar-button-primary:hover {
  background-color: #2980b9;
}

.sidebar-button-secondary {
  background-color: white;
  color: #333;
  border: 1px solid #ddd;
}

.sidebar-button-secondary:hover {
  background-color: #f8f8f8;
}

.main-screen {
  display: flex;
  flex-direction: column;
  flex: 1;
  min-height: 0;
}

.main-screen :deep(.screen-title) {
  flex-shrink: 0;
  margin: 0;
  padding: 1.5rem 2rem;
  font-size: 1.5rem;
  font-weight: 600;
  border-bottom: 1px solid #e0e0e0;
}

.main-screen :deep(.screen-content) {
  display: flex;
  flex-direction: column;
  flex: 1;
  min-height: 0;
  overflow: auto;
  padding: 2rem;
}
</style>

