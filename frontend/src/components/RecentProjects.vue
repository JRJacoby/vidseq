<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { getProjects, deleteProject, type Project } from '@/services/api'
import { useProjectStore } from '@/stores/project'
import { useRouter } from 'vue-router'

const projects = ref<Project[]>([])
const isLoading = ref(false)

const projectStore = useProjectStore()
const router = useRouter()

const loadProjects = async () => {
  isLoading.value = true
  try {
    projects.value = await getProjects()
  } catch (error) {
    console.error('Error loading projects:', error)
  } finally {
    isLoading.value = false
  }
}

onMounted(() => {
  loadProjects()
})

const openProject = (project: Project) => {
  projectStore.setCurrentProject(project.id)
  router.push(`/project/${project.id}`)
}

const handleDeleteProject = async (event: Event, project: Project) => {
  event.stopPropagation()
  
  const confirmed = window.confirm(`Delete project "${project.name}"? This will permanently delete all project data.`)
  if (!confirmed) return
  
  try {
    await deleteProject(project.id)
    
    if (projectStore.currentProjectId === project.id) {
      projectStore.clearCurrentProject()
    }
    
    loadProjects()
  } catch (error) {
    console.error('Error deleting project:', error)
    alert('Failed to delete project')
  }
}
</script>

<template>
  <h3 class="screen-title">Recent Projects</h3>
  <div class="screen-content">
    <div v-if="isLoading" class="loading-state">
      Loading projects...
    </div>

    <div v-else-if="projects.length === 0" class="empty-state">
      No projects yet. Create a new project to get started.
    </div>

    <div v-else class="projects-list">
      <div 
        v-for="project in projects"
        :key="project.id"
        class="project-card"
        @click="openProject(project)"
      >
        <div class="project-info">
          <span class="project-name">{{ project.name }}</span>
          <span class="project-path">{{ project.path }}</span>
        </div>
        <button 
          class="delete-button" 
          @click="handleDeleteProject($event, project)"
          title="Delete project"
        >
          ×
        </button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.project-card {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1rem;
  border: 1px solid #e0e0e0;
  border-radius: 4px;
  cursor: pointer;
  transition: background-color 0.2s;
}

.project-card:hover {
  background-color: #f5f5f5;
}

.project-info {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
}

.project-name {
  font-weight: 600;
}

.project-path {
  font-size: 0.85rem;
  color: #666;
}

.delete-button {
  background: none;
  border: none;
  font-size: 1.5rem;
  color: #999;
  cursor: pointer;
  padding: 0.25rem 0.5rem;
  border-radius: 4px;
  transition: color 0.2s, background-color 0.2s;
}

.delete-button:hover {
  color: #d32f2f;
  background-color: #ffebee;
}

.projects-list {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}
</style>
