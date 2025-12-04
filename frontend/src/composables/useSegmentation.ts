import { ref, watch, onUnmounted, computed, type Ref } from 'vue'
import {
    getMask,
    runSegmentation,
    resetFrame,
    resetVideo,
    propagateForward,
} from '@/services/api'
import { usePromptStorage, type StoredPrompt } from './usePromptStorage'

export type ToolType = 'none' | 'bbox' | 'positive_point' | 'negative_point'

export interface UseSegmentationReturn {
    activeTool: Ref<ToolType>
    textPrompt: Ref<string>
    currentMask: Ref<ImageBitmap | null>
    currentPrompts: Ref<StoredPrompt[]>
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
    handleResetVideo: () => Promise<void>
    handlePropagate: () => Promise<void>
    clearPromptStorage: () => void
}

export function useSegmentation(
    projectId: Ref<number | null>,
    videoId: Ref<number | null>,
    currentFrameIdx: Ref<number>
): UseSegmentationReturn {
    const activeTool = ref<ToolType>('none')
    const textPrompt = ref('')
    const currentMask = ref<ImageBitmap | null>(null)
    const isSegmenting = ref(false)
    const isPropagating = ref(false)
    const intendedFrameIdx = ref(0)

    const promptStorage = usePromptStorage(projectId, videoId)
    
    const currentPrompts = computed(() => {
        return promptStorage.getPromptsForFrame(currentFrameIdx.value)
    })

    let debounceTimeout: number | null = null

    const loadFrameData = async (frameIdx: number) => {
        if (!projectId.value || !videoId.value) return

        try {
            const maskBlob = await getMask(projectId.value, videoId.value, frameIdx)

            if (frameIdx !== intendedFrameIdx.value) {
                return
            }

            currentMask.value = await createImageBitmap(maskBlob)
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

    const seekToFrame = (frameIdx: number) => {
        intendedFrameIdx.value = frameIdx
        loadFrameData(frameIdx)
    }

    const handleBboxComplete = async (bbox: { x1: number; y1: number; x2: number; y2: number }) => {
        if (!projectId.value || !videoId.value) return

        isSegmenting.value = true
        activeTool.value = 'none'

        try {
            const maskBlob = await runSegmentation(
                projectId.value,
                videoId.value,
                currentFrameIdx.value,
                'bbox',
                bbox,
                textPrompt.value
            )

            promptStorage.addPrompt(currentFrameIdx.value, { type: 'bbox', details: bbox })

            currentMask.value = await createImageBitmap(maskBlob)
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
            const maskBlob = await runSegmentation(
                projectId.value,
                videoId.value,
                currentFrameIdx.value,
                point.type,
                { x: point.x, y: point.y },
                textPrompt.value
            )

            promptStorage.addPrompt(currentFrameIdx.value, {
                type: point.type,
                details: { x: point.x, y: point.y }
            })

            currentMask.value = await createImageBitmap(maskBlob)
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
            promptStorage.removePromptsForFrame(currentFrameIdx.value)
            await loadFrameData(currentFrameIdx.value)
        } catch (e) {
            console.error('Failed to reset frame:', e)
        }
    }

    const handleResetVideo = async () => {
        if (!projectId.value || !videoId.value) return

        try {
            await resetVideo(projectId.value, videoId.value)
            promptStorage.clearAll()
            currentMask.value = null
        } catch (e) {
            console.error('Failed to reset video:', e)
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

    const clearPromptStorage = () => {
        promptStorage.clearAll()
    }

    watch(currentFrameIdx, (newFrameIdx) => {
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
        textPrompt,
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
        handleResetVideo,
        handlePropagate,
        clearPromptStorage,
    }
}
