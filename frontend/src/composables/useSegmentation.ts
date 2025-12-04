import { ref, watch, onUnmounted, type Ref } from 'vue'
import {
  getMask,
  getPrompts,
  addPrompt,
  resetFrame,
  propagateForward,
  type Prompt,
} from '@/services/api'

export type ToolType = 'none' | 'bbox' | 'positive_point' | 'negative_point'

export interface UseSegmentationReturn {
  activeTool: Ref<ToolType>
  currentMask: Ref<ImageBitmap | null>
  currentPrompts: Ref<Prompt[]>
  isSegmenting: Ref<boolean>
  isPropagating: Ref<boolean>
  loadFrameData: (frameIdx: number) => Promise<void>
  seekToFrame: (frameIdx: number) => void
  toggleTool: () => void
  togglePositivePointTool: () => void
  toggleNegativePointTool: () => void
  handleBboxComplete: (bbox: { x1: number; y1: number; x2: number; y2: number }) => Promise<void>
  handlePointComplete: (point: { x: number; y: number; type: 'positive_point' | 'negative_point' }) => Promise<void>
  handleResetFrame: () => Promise<void>
  handlePropagate: () => Promise<void>
}

export function useSegmentation(
  projectId: Ref<number>,
  videoId: Ref<number>,
  currentFrameIdx: Ref<number>
): UseSegmentationReturn {
  const activeTool = ref<ToolType>('none')
  const currentMask = ref<ImageBitmap | null>(null)
  const currentPrompts = ref<Prompt[]>([])
  const isSegmenting = ref(false)
  const isPropagating = ref(false)
  const intendedFrameIdx = ref(0)  // The frame we want to be on (for parallel loading)

  let debounceTimeout: number | null = null

  const loadFrameData = async (frameIdx: number) => {
    if (!projectId.value || !videoId.value) return

    try {
      const [maskBlob, prompts] = await Promise.all([
        getMask(projectId.value, videoId.value, frameIdx),
        getPrompts(projectId.value, videoId.value, frameIdx),
      ])

      // Discard stale response if we've moved to a different intended frame
      if (frameIdx !== intendedFrameIdx.value) {
        return
      }

      currentMask.value = await createImageBitmap(maskBlob)
      currentPrompts.value = prompts
    } catch (e) {
      console.error('Failed to load frame data:', e)
    }
  }

  const toggleTool = () => {
    activeTool.value = activeTool.value === 'bbox' ? 'none' : 'bbox'
  }

  const togglePositivePointTool = () => {
    activeTool.value = activeTool.value === 'positive_point' ? 'none' : 'positive_point'
  }

  const toggleNegativePointTool = () => {
    activeTool.value = activeTool.value === 'negative_point' ? 'none' : 'negative_point'
  }

  // Called when user seeks - starts loading immediately without waiting for video
  const seekToFrame = (frameIdx: number) => {
    intendedFrameIdx.value = frameIdx
    loadFrameData(frameIdx)
  }

  const handleBboxComplete = async (bbox: { x1: number; y1: number; x2: number; y2: number }) => {
    if (!projectId.value || !videoId.value) return

    isSegmenting.value = true
    activeTool.value = 'none'

    try {
      await addPrompt(
        projectId.value,
        videoId.value,
        currentFrameIdx.value,
        'bbox',
        bbox
      )

      await loadFrameData(currentFrameIdx.value)
    } catch (e) {
      console.error('Failed to segment:', e)
    } finally {
      isSegmenting.value = false
    }
  }

  const handlePointComplete = async (point: { x: number; y: number; type: 'positive_point' | 'negative_point' }) => {
    if (!projectId.value || !videoId.value) return

    isSegmenting.value = true

    try {
      await addPrompt(
        projectId.value,
        videoId.value,
        currentFrameIdx.value,
        point.type,
        { x: point.x, y: point.y }
      )

      await loadFrameData(currentFrameIdx.value)
    } catch (e) {
      console.error('Failed to add point:', e)
    } finally {
      isSegmenting.value = false
    }
  }

  const handleResetFrame = async () => {
    if (!projectId.value || !videoId.value) return

    try {
      await resetFrame(projectId.value, videoId.value, currentFrameIdx.value)
      await loadFrameData(currentFrameIdx.value)
    } catch (e) {
      console.error('Failed to reset frame:', e)
    }
  }

  const handlePropagate = async () => {
    if (!projectId.value || !videoId.value) return

    isPropagating.value = true

    try {
      const result = await propagateForward(
        projectId.value,
        videoId.value,
        currentFrameIdx.value,
        100
      )
      console.log(`Propagated ${result.frames_processed} frames`)
      await loadFrameData(currentFrameIdx.value)
    } catch (e) {
      console.error('Failed to propagate:', e)
    } finally {
      isPropagating.value = false
    }
  }

  watch(currentFrameIdx, (newFrameIdx) => {
    // Skip if we're already loading/loaded this frame (e.g., from seekToFrame)
    if (newFrameIdx === intendedFrameIdx.value) {
      return
    }
    
    intendedFrameIdx.value = newFrameIdx
    
    if (debounceTimeout) {
      clearTimeout(debounceTimeout)
    }
    debounceTimeout = window.setTimeout(() => {
      loadFrameData(newFrameIdx)
    }, 100)
  })

  onUnmounted(() => {
    if (debounceTimeout) {
      clearTimeout(debounceTimeout)
    }
  })

  return {
    activeTool,
    currentMask,
    currentPrompts,
    isSegmenting,
    isPropagating,
    loadFrameData,
    seekToFrame,
    toggleTool,
    togglePositivePointTool,
    toggleNegativePointTool,
    handleBboxComplete,
    handlePointComplete,
    handleResetFrame,
    handlePropagate,
  }
}

