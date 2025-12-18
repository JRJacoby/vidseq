<script setup lang="ts">
</script>

<template>
  <div class="timeline-system">
    <slot />
  </div>
</template>

<style scoped>
.timeline-system {
  background-color: #1a1a1a;
  border-radius: 8px;
  padding: 12px;
  width: 100%;
  box-sizing: border-box;
  display: flex;
  flex-direction: column;
  gap: 4px;
}

/* Ensure children rows align their columns */
:deep(.timeline-row) {
  display: flex;
  align-items: flex-start;
  gap: 12px;
  width: 100%;
}

:deep(.timeline-col-left) {
  flex-shrink: 0;
  width: 64px;
  display: flex;
  justify-content: center;
}

:deep(.timeline-col-center) {
  flex: 1;
  min-width: 0;
}

:deep(.timeline-col-right) {
  flex-shrink: 0;
  width: 90px;
  display: flex;
}
</style>
