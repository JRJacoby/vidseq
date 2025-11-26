<script setup lang="ts">
import { watch } from 'vue'
import { useRoute } from 'vue-router'
import AppNavbar from '@/components/AppNavbar.vue'
import { useProjectStore } from '@/stores/project'

const route = useRoute()
const projectStore = useProjectStore()

watch(
  () => route.params.id,
  (newId) => {
    if (newId && Number(newId) !== projectStore.currentProjectId) {
      projectStore.setCurrentProject(Number(newId))
    }
  },
  { immediate: true }
)
</script>

<template>
  <div class="page-container">
    <header class="title-bar">
      <h1 class="title-bar-title">VidSeq - Animal Behavior Modeling from Raw Video</h1>
    </header>
    <div class="page-content">
      <AppNavbar />
      <main class="main-screen">
        <RouterView />
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
  padding: 0.5rem 1rem;
  border-bottom: 1px solid #e0e0e0;
}

.title-bar-title {
  margin: 0;
  font-size: 1.25rem;
}

.page-content {
  display: flex;
  flex: 1;
  min-height: 0;
}

.main-screen {
  display: flex;
  flex-direction: column;
  flex: 1;
  min-height: 0;
}
</style>
