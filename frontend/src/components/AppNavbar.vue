<script setup lang="ts">
import { useRouter, useRoute } from 'vue-router'
import { useProjectStore } from '@/stores/project'

const router = useRouter()
const route = useRoute()
const projectStore = useProjectStore()

const navigateToVideoPipeline = () => {
  if (projectStore.currentProjectId) {
    router.push(`/project/${projectStore.currentProjectId}`)
  }
}

const navigateToJobs = () => {
  if (projectStore.currentProjectId) {
    router.push(`/project/${projectStore.currentProjectId}/jobs`)
  }
}

const isActiveRoute = (routeName: string) => {
  return route.name === routeName
}
</script>

<template>
  <nav class="navbar">
    <div class="navbar-top">
      <button 
        class="navbar-button" 
        :class="{ 'navbar-button-active': isActiveRoute('pipeline') }"
        @click="navigateToVideoPipeline"
        :disabled="!projectStore.currentProjectId"
      >
        Video Pipeline
      </button>
    </div>
    <div class="navbar-bottom">
      <button 
        class="navbar-button"
        :class="{ 'navbar-button-active': isActiveRoute('jobs') }"
        @click="navigateToJobs"
        :disabled="!projectStore.currentProjectId"
      >
        Jobs
      </button>
    </div>
  </nav>
</template>

<style scoped>
.navbar {
  flex-shrink: 0;
  width: 200px;
  display: flex;
  flex-direction: column;
  background-color: #f8f8f8;
  border-right: 1px solid #e0e0e0;
}

.navbar-top {
  flex: 1;
  display: flex;
  flex-direction: column;
  border-bottom: 1px solid #e0e0e0;
}

.navbar-bottom {
  flex-shrink: 0;
  display: flex;
  flex-direction: column;
  padding: 0.5rem 0;
}

.navbar-button {
  padding: 0.75rem 1rem;
  border: none;
  background: none;
  text-align: left;
  cursor: pointer;
  font-size: 0.95rem;
  transition: background-color 0.2s;
}

.navbar-button:hover:not(:disabled) {
  background-color: #e8e8e8;
}

.navbar-button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.navbar-button-active {
  background-color: #e3f2fd;
  border-left: 3px solid #4a90e2;
  font-weight: 600;
}
</style>
