import { ref, computed, watch, onMounted, onUnmounted, type Ref, type ComputedRef } from 'vue'
import {
  createSegmentationLoadedModel,
  initVideoSession,
  closeVideoSession,
  type SegmentationStatus,
} from '@/services/api'

const API_BASE = '/api'

export interface UseSegmentationSessionReturn {
  segmentationStatus: Ref<SegmentationStatus>
  isReady: ComputedRef<boolean>
  statusText: ComputedRef<string>
  sessionInitialized: Ref<boolean>
}

export function useSegmentationSession(
  projectId: Ref<number>,
  videoId: Ref<number>
): UseSegmentationSessionReturn {
  const segmentationStatus = ref<SegmentationStatus>({ status: 'not_loaded', error: null })
  const sessionInitialized = ref(false)
  const sessionInitializing = ref(false)
  // Capture IDs at init time so teardown works after route changes
  const sessionProjectId = ref<number | null>(null)
  const sessionVideoId = ref<number | null>(null)
  let eventSource: EventSource | null = null

  const isReady = computed(() => 
    segmentationStatus.value.status === 'ready' && sessionInitialized.value
  )

  const statusText = computed(() => {
    switch (segmentationStatus.value.status) {
      case 'not_loaded': return 'SAM2: Not loaded'
      case 'loading_model': return 'SAM2: Loading model...'
      case 'ready':
        if (sessionInitializing.value) {
          return 'SAM2: Warming up (first run may take a few minutes)...'
        }
        if (!sessionInitialized.value) {
          return 'SAM2: Ready (waiting for session)'
        }
        return 'SAM2: Ready'
      case 'error': return `SAM2: Error - ${segmentationStatus.value.error}`
      default: return 'SAM2: Unknown'
    }
  })

  const connectSSE = () => {
    eventSource = new EventSource(`${API_BASE}/segmentation/status/stream`)
    
    eventSource.onmessage = (event) => {
      try {
        const status = JSON.parse(event.data) as SegmentationStatus
        segmentationStatus.value = status
      } catch (e) {
        console.error('Failed to parse segmentation status:', e)
      }
    }
    
    eventSource.onerror = () => {
      eventSource?.close()
      setTimeout(connectSSE, 1000)
    }
  }

  watch(() => segmentationStatus.value.status, async (status, oldStatus) => {
    if (status === 'ready' && !sessionInitialized.value && !sessionInitializing.value && projectId.value && videoId.value) {
      sessionInitializing.value = true
      try {
        await initVideoSession(projectId.value, videoId.value)
        sessionProjectId.value = projectId.value
        sessionVideoId.value = videoId.value
        sessionInitialized.value = true
      } catch (e) {
        console.error('Failed to init video session:', e)
      } finally {
        sessionInitializing.value = false
      }
    }
    
    if (oldStatus === 'ready' && status !== 'ready') {
      sessionInitialized.value = false
      sessionInitializing.value = false
    }
    
    if (status === 'not_loaded') {
      createSegmentationLoadedModel()
    }
  })

  onMounted(() => {
    connectSSE()
    createSegmentationLoadedModel()
  })

  onUnmounted(async () => {
    eventSource?.close()
    eventSource = null

    if (sessionProjectId.value && sessionVideoId.value) {
      try {
        await closeVideoSession(sessionProjectId.value, sessionVideoId.value)
      } catch (e) {
        console.error('Failed to close video session:', e)
      }
    }
  })

  return {
    segmentationStatus,
    isReady,
    statusText,
    sessionInitialized,
  }
}
