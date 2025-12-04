import { ref, computed, watch, onMounted, onUnmounted, type Ref, type ComputedRef } from 'vue'
import {
  preloadSAM3,
  initVideoSession,
  closeVideoSession,
  type SAM3Status,
} from '@/services/api'

const API_BASE = '/api'

export interface UseSAM3SessionReturn {
  sam3Status: Ref<SAM3Status>
  isReady: ComputedRef<boolean>
  statusText: ComputedRef<string>
  sessionInitialized: Ref<boolean>
}

export function useSAM3Session(
  projectId: Ref<number>,
  videoId: Ref<number>
): UseSAM3SessionReturn {
  const sam3Status = ref<SAM3Status>({ status: 'not_loaded', error: null })
  const sessionInitialized = ref(false)
  let eventSource: EventSource | null = null

  const isReady = computed(() => sam3Status.value.status === 'ready')

  const statusText = computed(() => {
    switch (sam3Status.value.status) {
      case 'not_loaded': return 'SAM3: Not loaded'
      case 'loading_model': return 'SAM3: Loading model...'
      case 'ready': return 'SAM3: Ready'
      case 'error': return `SAM3: Error - ${sam3Status.value.error}`
      default: return 'SAM3: Unknown'
    }
  })

  const connectSSE = () => {
    eventSource = new EventSource(`${API_BASE}/sam3/status/stream`)
    
    eventSource.onmessage = (event) => {
      try {
        const status = JSON.parse(event.data) as SAM3Status
        sam3Status.value = status
      } catch (e) {
        console.error('Failed to parse SAM3 status:', e)
      }
    }
    
    eventSource.onerror = () => {
      eventSource?.close()
      setTimeout(connectSSE, 1000)
    }
  }

  watch(isReady, async (ready, wasReady) => {
    if (ready && !sessionInitialized.value && projectId.value && videoId.value) {
      try {
        await initVideoSession(projectId.value, videoId.value)
        sessionInitialized.value = true
      } catch (e) {
        console.error('Failed to init video session:', e)
      }
    }
    
    // Worker died - reset session state
    if (wasReady && !ready) {
      sessionInitialized.value = false
    }
  })

  // Auto-restart loading if status goes back to not_loaded (worker died)
  watch(() => sam3Status.value.status, (status) => {
    if (status === 'not_loaded') {
      preloadSAM3()
    }
  })

  onMounted(() => {
    connectSSE()
    preloadSAM3()
  })

  onUnmounted(async () => {
    eventSource?.close()
    eventSource = null

    if (projectId.value && videoId.value) {
      try {
        await closeVideoSession(projectId.value, videoId.value)
      } catch (e) {
        console.error('Failed to close video session:', e)
      }
    }
  })

  return {
    sam3Status,
    isReady,
    statusText,
    sessionInitialized,
  }
}
