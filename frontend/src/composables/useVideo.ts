import { ref, watch, onMounted, type Ref, type MaybeRef, toValue } from 'vue'
import { getVideo, type Video } from '@/services/api'

export interface UseVideoReturn {
  video: Ref<Video | null>
  isLoading: Ref<boolean>
  error: Ref<string | null>
  refresh: () => Promise<void>
}

/**
 * Composable for fetching and managing video metadata.
 * Auto-fetches on mount and re-fetches when videoId or projectId change.
 */
export function useVideo(
  videoId: MaybeRef<number>,
  projectId: MaybeRef<number>
): UseVideoReturn {
  const video = ref<Video | null>(null)
  const isLoading = ref(false)
  const error = ref<string | null>(null)

  const load = async () => {
    const vid = toValue(videoId)
    const pid = toValue(projectId)
    if (!vid || !pid) return

    isLoading.value = true
    error.value = null

    try {
      video.value = await getVideo(pid, vid)
    } catch (e) {
      error.value = e instanceof Error ? e.message : 'Failed to load video'
    } finally {
      isLoading.value = false
    }
  }

  // Auto-fetch on mount
  onMounted(load)

  // Re-fetch when IDs change
  watch([videoId, projectId], load)

  return {
    video,
    isLoading,
    error,
    refresh: load,
  }
}
