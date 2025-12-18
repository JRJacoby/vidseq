import { ref, onMounted, onUnmounted } from 'vue'
import { API_BASE, type Job } from '@/services/api'

export function useJobs() {
  const jobs = ref<Job[]>([])
  const isLoading = ref(true)
  let eventSource: EventSource | null = null

  const connectToJobsStream = () => {
    if (eventSource) {
      eventSource.close()
    }

    eventSource = new EventSource(`${API_BASE}/jobs/stream`)
    
    eventSource.onmessage = (event) => {
      jobs.value = JSON.parse(event.data)
      isLoading.value = false
    }
    
    eventSource.onerror = (error) => {
      console.error('Jobs SSE error:', error)
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

  return {
    jobs,
    isLoading,
    connectToJobsStream
  }
}
