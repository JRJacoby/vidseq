<script setup lang="ts">
import { type Project } from '@/services/api'

defineProps<{
  projects: Project[]
  isLoading: boolean
}>()
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
      >
        {{ project.name }}
        {{ project.path }}
        {{ project.created_at }}
        {{ project.updated_at }}
      </div>
    </div>
  </div>
</template>