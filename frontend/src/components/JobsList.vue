<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'
import { type Job, API_BASE } from '@/services/api'

const jobs = ref<Job[]>([])
const isLoading = ref(true)
let eventSource: EventSource | null = null

const connectToJobsStream = () => {
  eventSource = new EventSource(`${API_BASE}/jobs/stream`)
  
  eventSource.onmessage = (event) => {
    jobs.value = JSON.parse(event.data)
    isLoading.value = false
  }
  
  eventSource.onerror = (error) => {
    console.error('SSE error:', error)
    isLoading.value = false
  }
}

onMounted(() => {
  connectToJobsStream()
})

onUnmounted(() => {
  if (eventSource) {
    eventSource.close()
  }
})

const getStatusClass = (status: string) => {
  const statusMap: Record<string, string> = {
    running: 'status-running',
    completed: 'status-completed',
    failed: 'status-failed',
    pending: 'status-pending'
  }
  return statusMap[status.toLowerCase()] || ''
}
</script>

<template>
  <div class="jobs-container">
    <h3 class="screen-title">Jobs</h3>
    <div class="jobs-content">
      <div v-if="isLoading" class="loading-state">
        Loading jobs...
      </div>
      <div v-else-if="jobs.length === 0" class="empty-state">
        No jobs found.
      </div>
      <div v-else class="jobs-list">
        <div v-for="job in jobs" :key="job.id" class="job-item">
          <div class="job-header">
            <span class="job-id">Job #{{ job.id }}</span>
            <span class="job-status" :class="getStatusClass(job.status)">
              {{ job.status }}
            </span>
          </div>
          <div class="job-details">
            <p><strong>Type:</strong> {{ job.type }}</p>
            <p><strong>Project ID:</strong> {{ job.project_id }}</p>
            <p><strong>Created:</strong> {{ new Date(job.created_at).toLocaleString() }}</p>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.jobs-container {
  display: flex;
  flex-direction: column;
  flex: 1;
  min-height: 0;
}

.screen-title {
  flex-shrink: 0;
  padding: 1rem;
  margin: 0;
  border-bottom: 1px solid #e0e0e0;
}

.jobs-content {
  flex: 1;
  min-height: 0;
  overflow: auto;
  padding: 1rem;
}

.loading-state,
.empty-state {
  padding: 2rem;
  text-align: center;
  color: #666;
}

.jobs-list {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.job-item {
  border: 1px solid #e0e0e0;
  border-radius: 4px;
  padding: 1rem;
  background-color: white;
}

.job-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.75rem;
  padding-bottom: 0.75rem;
  border-bottom: 1px solid #f0f0f0;
}

.job-id {
  font-weight: 600;
  font-size: 1.1rem;
}

.job-status {
  padding: 0.25rem 0.75rem;
  border-radius: 12px;
  font-size: 0.85rem;
  font-weight: 500;
}

.status-running {
  background-color: #fff3cd;
  color: #856404;
}

.status-completed {
  background-color: #d4edda;
  color: #155724;
}

.status-failed {
  background-color: #f8d7da;
  color: #721c24;
}

.status-pending {
  background-color: #e7f3ff;
  color: #004085;
}

.job-details {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
}

.job-details p {
  margin: 0;
  font-size: 0.9rem;
}
</style>

