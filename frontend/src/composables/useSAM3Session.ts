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
    console.log('[useSAM3Session] Connecting to SSE...')
    eventSource = new EventSource(`${API_BASE}/sam3/status/stream`)
    
    eventSource.onmessage = (event) => {
      try {
        const status = JSON.parse(event.data) as SAM3Status
        console.log(`[useSAM3Session] SSE message: ${JSON.stringify(status)}, was: ${JSON.stringify(sam3Status.value)}`)
        sam3Status.value = status
      } catch (e) {
        console.error('[useSAM3Session] Failed to parse SAM3 status:', e)
      }
    }
    
    eventSource.onerror = () => {
      console.error('[useSAM3Session] SSE connection error, reconnecting in 1s...')
      eventSource?.close()
      setTimeout(connectSSE, 1000)
    }
  }

  watch(isReady, async (ready, wasReady) => {
    console.log(`[useSAM3Session] isReady changed: ${wasReady} -> ${ready}, sessionInitialized=${sessionInitialized.value}`)
    if (ready && !sessionInitialized.value && projectId.value && videoId.value) {
      console.log(`[useSAM3Session] SAM3 ready, initializing video session...`)
      try {
        await initVideoSession(projectId.value, videoId.value)
        sessionInitialized.value = true
        console.log(`[useSAM3Session] Video session initialized`)
      } catch (e) {
        console.error('[useSAM3Session] Failed to init video session:', e)
      }
    }
    
    // Worker died - reset session state
    if (wasReady && !ready) {
      console.log(`[useSAM3Session] SAM3 no longer ready, resetting sessionInitialized`)
      sessionInitialized.value = false
    }
  })

  // Auto-restart loading if status goes back to not_loaded (worker died)
  watch(() => sam3Status.value.status, (status, oldStatus) => {
    console.log(`[useSAM3Session] status watch: ${oldStatus} -> ${status}`)
    if (status === 'not_loaded') {
      console.log(`[useSAM3Session] Status is not_loaded, calling preloadSAM3()`)
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
