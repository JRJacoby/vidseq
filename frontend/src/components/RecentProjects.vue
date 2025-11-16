<script setup lang="ts">
import { type Project } from '@/services/api'
import { useProjectStore } from '@/stores/project'
import { useRouter } from 'vue-router'

defineProps<{
  projects: Project[]
  isLoading: boolean
}>()

const projectStore = useProjectStore()
const router = useRouter()

const openProject = (project: Project) => {
  projectStore.setCurrentProject(project.id)
  router.push(`/project/${project.id}`)
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
        {{ project.name }}
        {{ project.path }}
        {{ project.created_at }}
        {{ project.updated_at }}
      </div>
    </div>
  </div>
</template>