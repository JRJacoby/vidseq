<script setup lang="ts">
import { ref, onMounted, computed } from 'vue'
import { useProjectStore } from '@/stores/project'
import {
  getPCAStatus,
  getPCAScreePlotUrl,
  getPCAComponentsPlotUrl,
  type PCAStatus,
} from '@/services/api'

const projectStore = useProjectStore()
const projectId = computed(() => projectStore.currentProjectId)

const pcaStatus = ref<PCAStatus | null>(null)
const isLoading = ref(true)

const loadPCAStatus = async () => {
  if (!projectId.value) return
  isLoading.value = true
  try {
    pcaStatus.value = await getPCAStatus(projectId.value)
  } catch (e) {
    console.error('Failed to load PCA status:', e)
  } finally {
    isLoading.value = false
  }
}

const screePlotUrl = computed(() => {
  if (!projectId.value) return ''
  return getPCAScreePlotUrl(projectId.value)
})

const componentsPlotUrl = computed(() => {
  if (!projectId.value) return ''
  return getPCAComponentsPlotUrl(projectId.value)
})

onMounted(async () => {
  await loadPCAStatus()
})
</script>

<template>
  <div class="pca-view">
    <h3 class="screen-title">PCA Analysis</h3>

    <div v-if="isLoading" class="loading-state">
      Loading PCA status...
    </div>

    <div v-else-if="!pcaStatus?.has_pca" class="no-pca">
      <p>PCA has not been computed yet.</p>
      <p>Go to Video Pipeline and click "Run PCA" to compute.</p>
    </div>

    <div v-else class="pca-results">
      <div class="pca-summary">
        <div class="summary-item">
          <span class="summary-label">Components:</span>
          <span class="summary-value">{{ pcaStatus.n_components }}</span>
        </div>
        <div class="summary-item">
          <span class="summary-label">Total Frames:</span>
          <span class="summary-value">{{ pcaStatus.total_frames?.toLocaleString() }}</span>
        </div>
      </div>

      <div class="plots-container">
        <div class="plot-section">
          <h4>Scree Plot</h4>
          <img :src="screePlotUrl" alt="Scree Plot" class="plot-image" />
        </div>

        <div class="plot-section">
          <h4>PCA Components</h4>
          <img :src="componentsPlotUrl" alt="PCA Components" class="plot-image" />
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.pca-view {
  display: flex;
  flex-direction: column;
  flex: 1;
  min-height: 0;
  overflow: auto;
}

.screen-title {
  flex-shrink: 0;
  padding: 1rem;
  margin: 0;
  border-bottom: 1px solid #e0e0e0;
}

.loading-state,
.no-pca {
  padding: 3rem;
  text-align: center;
  color: #888;
}

.no-pca p {
  margin: 0.5rem 0;
}

.pca-results {
  padding: 1.5rem;
  display: flex;
  flex-direction: column;
  gap: 2rem;
}

.pca-summary {
  display: flex;
  gap: 2rem;
  padding: 1rem;
  background-color: #f8f9fa;
  border-radius: 8px;
  border: 1px solid #e0e0e0;
}

.summary-item {
  display: flex;
  gap: 0.5rem;
  align-items: baseline;
}

.summary-label {
  font-size: 0.9rem;
  color: #666;
}

.summary-value {
  font-size: 1.1rem;
  font-weight: 600;
  color: #333;
}

.plots-container {
  display: flex;
  flex-direction: column;
  gap: 2rem;
}

.plot-section {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.plot-section h4 {
  margin: 0;
  font-size: 1rem;
  color: #444;
}

.plot-image {
  max-width: 100%;
  height: auto;
  border: 1px solid #e0e0e0;
  border-radius: 4px;
  background-color: white;
}
</style>
