<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { getProjects, type Project } from '@/services/api'

const recentProjects = ref<Project[]>([])
const isLoading = ref(false)

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
        <button class="sidebar-button-primary">New Project</button>
        <button class="sidebar-button-secondary">Open Project</button>
      </aside>
      <main class="main-screen">
        <h3 class="screen-title">Recent Projects</h3>
        <div class="screen-content">
          <!-- Loading state -->
          <div v-if="isLoading" class="loading-state">
            Loading projects...
          </div>

          <!-- Empty state -->
          <div v-else-if="recentProjects.length === 0" class="empty-state">
            No projects yet. Create a new project to get started.
          </div>

          <!-- Projects list -->
          <div v-else class="projects-list">
            <div v-for="project in recentProjects"
            :key="project.id"
            class="project-card"
            >
              {{ project.name }}
              {{ project.path }}
              {{ project.created_at }}
              {{ project.updated_at }}
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
}

.title-bar {
  flex-shrink: 0;
}

.page-content {
  display: flex;
  flex-direction: row;
  flex: 1;
  min-height: 0;
}

.sidebar {
  display: flex;
  flex-direction: column;
  flex-shrink: 0;
  width: 250px;
  padding: 1rem;
  gap: 1rem;
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