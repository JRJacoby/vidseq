import { ref, computed, watch, onMounted, onUnmounted, type Ref, type ComputedRef } from 'vue'
import {
  getSAM3Status,
  preloadSAM3,
  initVideoSession,
  closeVideoSession,
  type SAM3Status,
} from '@/services/api'

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
  let statusPollInterval: number | null = null

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

  const pollStatus = async () => {
    try {
      sam3Status.value = await getSAM3Status()

      if (sam3Status.value.status === 'ready' || sam3Status.value.status === 'error') {
        if (statusPollInterval) {
          clearInterval(statusPollInterval)
          statusPollInterval = null
        }
      }
    } catch (e) {
      console.error('Failed to poll SAM3 status:', e)
    }
  }

  const stopPolling = () => {
    if (statusPollInterval) {
      clearInterval(statusPollInterval)
      statusPollInterval = null
    }
  }

  watch(isReady, async (ready) => {
    if (ready && !sessionInitialized.value && projectId.value && videoId.value) {
      try {
        await initVideoSession(projectId.value, videoId.value)
        sessionInitialized.value = true
      } catch (e) {
        console.error('Failed to init video session:', e)
      }
    }
  })

  onMounted(async () => {
    await pollStatus()
    if (sam3Status.value.status === 'not_loaded') {
      preloadSAM3()
    }
    statusPollInterval = window.setInterval(pollStatus, 1000)
  })

  onUnmounted(async () => {
    stopPolling()

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

