import { ref, computed, watch, onMounted, onUnmounted, type Ref, type ComputedRef } from 'vue'
import {
  createKeypointModel,
  initKeypointSession,
  closeKeypointSession,
  type KeypointTrackingStatus,
} from '@/services/api'

const API_BASE = '/api'

export interface UseKeypointSessionReturn {
  keypointStatus: Ref<KeypointTrackingStatus>
  isReady: ComputedRef<boolean>
  statusText: ComputedRef<string>
  sessionInitialized: Ref<boolean>
}

export function useKeypointSession(
  projectId: Ref<number>,
  videoId: Ref<number>
): UseKeypointSessionReturn {
  const keypointStatus = ref<KeypointTrackingStatus>({ status: 'not_loaded', error: null })
  const sessionInitialized = ref(false)
  const sessionInitializing = ref(false)
  // Capture IDs at init time so teardown works after route changes
  const sessionProjectId = ref<number | null>(null)
  const sessionVideoId = ref<number | null>(null)
  let eventSource: EventSource | null = null

  const isReady = computed(() =>
    keypointStatus.value.status === 'ready' && sessionInitialized.value
  )

  const statusText = computed(() => {
    switch (keypointStatus.value.status) {
      case 'not_loaded': return 'Keypoint Tracker: Not loaded'
      case 'loading_model': return 'Keypoint Tracker: Loading model...'
      case 'ready':
        if (sessionInitializing.value) {
          return 'Keypoint Tracker: Warming up (first run may take a few minutes)...'
        }
        if (!sessionInitialized.value) {
          return 'Keypoint Tracker: Ready (waiting for session)'
        }
        return 'Keypoint Tracker: Ready'
      case 'error': return `Keypoint Tracker: Error - ${keypointStatus.value.error}`
      default: return 'Keypoint Tracker: Unknown'
    }
  })

  const connectSSE = () => {
    eventSource = new EventSource(`${API_BASE}/keypoint-tracking/status/stream`)

    eventSource.onmessage = (event) => {
      try {
        const status = JSON.parse(event.data) as KeypointTrackingStatus
        keypointStatus.value = status
      } catch (e) {
        console.error('Failed to parse keypoint status:', e)
      }
    }

    eventSource.onerror = () => {
      eventSource?.close()
      setTimeout(connectSSE, 1000)
    }
  }

  watch(() => keypointStatus.value.status, async (status, oldStatus) => {
    if (status === 'ready' && !sessionInitialized.value && !sessionInitializing.value && projectId.value && videoId.value) {
      sessionInitializing.value = true
      try {
        await initKeypointSession(projectId.value, videoId.value)
        sessionProjectId.value = projectId.value
        sessionVideoId.value = videoId.value
        sessionInitialized.value = true
      } catch (e) {
        console.error('Failed to init keypoint session:', e)
      } finally {
        sessionInitializing.value = false
      }
    }

    if (oldStatus === 'ready' && status !== 'ready') {
      sessionInitialized.value = false
      sessionInitializing.value = false
    }

    if (status === 'not_loaded') {
      createKeypointModel()
    }
  })

  onMounted(() => {
    connectSSE()
    createKeypointModel()
  })

  onUnmounted(async () => {
    eventSource?.close()
    eventSource = null

    if (sessionProjectId.value && sessionVideoId.value) {
      try {
        await closeKeypointSession(sessionProjectId.value, sessionVideoId.value)
      } catch (e) {
        console.error('Failed to close keypoint session:', e)
      }
    }
  })

  return {
    keypointStatus,
    isReady,
    statusText,
    sessionInitialized,
  }
}
