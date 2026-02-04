import { ref, onUnmounted, type Ref } from 'vue'

export interface UseSSEStreamReturn<T> {
  data: Ref<T | null>
  isConnected: Ref<boolean>
  error: Ref<string | null>
  start: () => void
  stop: () => void
}

/**
 * Composable for managing SSE (Server-Sent Events) connections.
 * Handles connection lifecycle, JSON parsing, and cleanup on unmount.
 */
export function useSSEStream<T>(
  connect: () => EventSource
): UseSSEStreamReturn<T> {
  const data = ref<T | null>(null) as Ref<T | null>
  const isConnected = ref(false)
  const error = ref<string | null>(null)

  let eventSource: EventSource | null = null

  const stop = () => {
    if (eventSource) {
      eventSource.close()
      eventSource = null
    }
    isConnected.value = false
  }

  const start = () => {
    // Close existing connection if any
    stop()
    error.value = null

    try {
      eventSource = connect()
      isConnected.value = true

      eventSource.onmessage = (event) => {
        try {
          data.value = JSON.parse(event.data) as T
        } catch (e) {
          error.value = `Failed to parse SSE data: ${e instanceof Error ? e.message : String(e)}`
        }
      }

      eventSource.onerror = () => {
        error.value = 'SSE connection error'
        stop()
      }
    } catch (e) {
      error.value = e instanceof Error ? e.message : 'Failed to connect'
      isConnected.value = false
    }
  }

  // Cleanup on unmount
  onUnmounted(stop)

  return {
    data,
    isConnected,
    error,
    start,
    stop,
  }
}
