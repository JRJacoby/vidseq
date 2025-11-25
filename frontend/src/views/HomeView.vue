<script setup lang="ts">
import { ref } from 'vue'
import RecentProjects from '@/components/RecentProjects.vue'
import NewProject from '@/components/NewProject.vue'

const currentView = ref<'recent-projects' | 'new-project'>('recent-projects')
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
        <RecentProjects v-if="currentView === 'recent-projects'" />
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

.main-screen {
  flex-shrink: 0;
}

.main-screen {
  display: flex;
  flex-direction: column;
  flex: 1;
  min-height: 0;
  overflow: auto;
}
</style>

