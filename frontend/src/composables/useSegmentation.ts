import { ref, watch, onUnmounted, type Ref } from 'vue'
import {
  getMask,
  getPrompts,
  addPrompt,
  resetFrame,
  type Prompt,
} from '@/services/api'

export interface UseSegmentationReturn {
  activeTool: Ref<'none' | 'bbox'>
  currentMask: Ref<ImageBitmap | null>
  currentPrompts: Ref<Prompt[]>
  isSegmenting: Ref<boolean>
  loadFrameData: (frameIdx: number) => Promise<void>
  toggleTool: () => void
  handleBboxComplete: (bbox: { x1: number; y1: number; x2: number; y2: number }) => Promise<void>
  handleResetFrame: () => Promise<void>
}

export function useSegmentation(
  projectId: Ref<number>,
  videoId: Ref<number>,
  currentFrameIdx: Ref<number>
): UseSegmentationReturn {
  const activeTool = ref<'none' | 'bbox'>('none')
  const currentMask = ref<ImageBitmap | null>(null)
  const currentPrompts = ref<Prompt[]>([])
  const isSegmenting = ref(false)

  let debounceTimeout: number | null = null

  const loadFrameData = async (frameIdx: number) => {
    if (!projectId.value || !videoId.value) return

    try {
      const [maskBlob, prompts] = await Promise.all([
        getMask(projectId.value, videoId.value, frameIdx),
        getPrompts(projectId.value, videoId.value, frameIdx),
      ])

      currentMask.value = await createImageBitmap(maskBlob)
      currentPrompts.value = prompts
    } catch (e) {
      console.error('Failed to load frame data:', e)
    }
  }

  const toggleTool = () => {
    activeTool.value = activeTool.value === 'bbox' ? 'none' : 'bbox'
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

  const handleResetFrame = async () => {
    if (!projectId.value || !videoId.value) return

    try {
      await resetFrame(projectId.value, videoId.value, currentFrameIdx.value)
      await loadFrameData(currentFrameIdx.value)
    } catch (e) {
      console.error('Failed to reset frame:', e)
    }
  }

  watch(currentFrameIdx, (newFrameIdx) => {
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
    loadFrameData,
    toggleTool,
    handleBboxComplete,
    handleResetFrame,
  }
}

