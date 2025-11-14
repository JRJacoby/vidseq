<script setup lang="ts">
import { useRoute } from 'vue-router'

const route = useRoute()
const projectId = route.params.id
</script>

<template>
  <div class="page-container">
    <header class="title-bar">
      <h1 class="title-bar-title">Project {{ projectId }}</h1>
    </header>
    <div class="page-content">
      <p>Project view placeholder - Video Pipeline will go here</p>
    </div>
  </div>
</template>