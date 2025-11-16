<script setup lang="ts">
import { RouterView } from 'vue-router'
import { onMounted } from 'vue'
import { useProjectStore } from './stores/project'

const projectStore = useProjectStore()

onMounted(async () => {
  await projectStore.validateCurrentProject()
})
</script>

<template>
  <RouterView />
</template>