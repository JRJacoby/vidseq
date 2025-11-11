<script setup lang="ts">
// HomeView - Start screen with project selection
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
          <!-- Main content -->
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
  flex: 1;
  min-height: 0;
  overflow: auto;
}

</style>