<script setup lang="ts">
import { ref, watch, onUnmounted } from 'vue'
import { getConditioningFrames, getTrackerMask } from '@/services/api'

const props = defineProps<{
  projectId: number
  videoId: number
  refreshKey: number
}>()

interface CondFrame {
  frameIdx: number
  objectUrl: string
}

const condFrames = ref<CondFrame[]>([])
const isLoading = ref(false)

const revokeUrls = () => {
  for (const frame of condFrames.value) {
    URL.revokeObjectURL(frame.objectUrl)
  }
}

const fetchCondFrames = async () => {
  if (!props.projectId || !props.videoId) return
  isLoading.value = true
  try {
    const frameIndices = await getConditioningFrames(props.projectId, props.videoId)
    revokeUrls()

    const frames: CondFrame[] = []
    for (const idx of frameIndices) {
      try {
        const blob = await getTrackerMask(props.projectId, props.videoId, idx)
        frames.push({ frameIdx: idx, objectUrl: URL.createObjectURL(blob) })
      } catch {
        // Skip frames where mask fetch fails
      }
    }
    condFrames.value = frames
  } catch {
    condFrames.value = []
  } finally {
    isLoading.value = false
  }
}

watch(
  () => [props.projectId, props.videoId, props.refreshKey],
  () => fetchCondFrames(),
  { immediate: true },
)

onUnmounted(() => revokeUrls())
</script>

<template>
  <div v-if="condFrames.length > 0" class="cond-frame-grid-section">
    <h4 class="cond-frame-title">Conditioning Frames ({{ condFrames.length }})</h4>
    <div class="cond-frame-grid">
      <div v-for="frame in condFrames" :key="frame.frameIdx" class="cond-frame-cell">
        <img :src="frame.objectUrl" class="cond-frame-img" />
        <span class="cond-frame-label">Frame {{ frame.frameIdx }}</span>
      </div>
    </div>
  </div>
</template>

<style scoped>
.cond-frame-grid-section {
  width: 100%;
  border-top: 1px solid #e0e0e0;
  padding: 0.75rem 1rem;
  background-color: #f5f5f5;
  box-sizing: border-box;
}

.cond-frame-title {
  margin: 0 0 6px 0;
  font-size: 0.85rem;
  font-weight: 600;
  color: #444;
}

.cond-frame-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
}

.cond-frame-cell {
  display: flex;
  flex-direction: column;
  align-items: center;
}

.cond-frame-img {
  width: 100%;
  background: #000;
  display: block;
}

.cond-frame-label {
  font-size: 0.75rem;
  color: #666;
  margin-top: 2px;
}
</style>
