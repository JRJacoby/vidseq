import { ref, watch, onUnmounted, computed, type Ref } from 'vue'
import {
    getMask,
    getMasksBatch,
    runSegmentation,
    resetFrame,
    resetVideo,
    propagateForward,
    propagateBackward,
} from '@/services/api'
import { usePromptStorage, type StoredPrompt } from './usePromptStorage'

export type ToolType = 'none' | 'positive_point' | 'negative_point'

export interface UseSegmentationReturn {
    activeTool: Ref<ToolType>
    currentMask: Ref<ImageBitmap | null>
    currentPrompts: Ref<StoredPrompt[]>
    isSegmenting: Ref<boolean>
    isPropagating: Ref<boolean>
    loadFrameData: (frameIdx: number) => Promise<void>
    seekToFrame: (frameIdx: number) => void
    togglePositivePointTool: () => void
    toggleNegativePointTool: () => void
    handlePointComplete: (point: { x: number; y: number; type: 'positive_point' | 'negative_point' }) => Promise<void>
    handleResetFrame: () => Promise<void>
    handleResetVideo: () => Promise<void>
    handlePropagateForward: () => Promise<void>
    handlePropagateBackward: () => Promise<void>
    clearPromptStorage: () => void
}

const PREFETCH_BATCH_SIZE = 100
const PREFETCH_THRESHOLD = 100

export function useSegmentation(
    projectId: Ref<number | null>,
    videoId: Ref<number | null>,
    currentFrameIdx: Ref<number>,
    isPlaying: Ref<boolean> = ref(false),
    videoRef: Ref<HTMLVideoElement | null> = ref(null),
    fps: Ref<number> = ref(30)
): UseSegmentationReturn {
    const activeTool = ref<ToolType>('none')
    const currentMask = ref<ImageBitmap | null>(null)
    const isSegmenting = ref(false)
    const isPropagating = ref(false)
    const intendedFrameIdx = ref(0)

    const promptStorage = usePromptStorage(projectId, videoId)
    
    const currentPrompts = computed(() => {
        return promptStorage.getPromptsForFrame(currentFrameIdx.value)
    })

    let debounceTimeout: number | null = null
    
    const maskCache = new Map<number, ImageBitmap>()
    let isPrefetching = false
    let prefetchedUpTo = -1
    let animationFrameId: number | null = null
    let lastDisplayedFrame = -1

    const base64ToBlob = (base64: string): Blob => {
        const binary = atob(base64)
        const bytes = new Uint8Array(binary.length)
        for (let i = 0; i < binary.length; i++) {
            bytes[i] = binary.charCodeAt(i)
        }
        return new Blob([bytes], { type: 'image/png' })
    }

    const prefetchMasks = async (startFrame: number) => {
        if (!projectId.value || !videoId.value || isPrefetching) return
        if (startFrame <= prefetchedUpTo) return
        
        isPrefetching = true
        try {
            const response = await getMasksBatch(
                projectId.value,
                videoId.value,
                startFrame,
                PREFETCH_BATCH_SIZE
            )
            
            for (const item of response.masks) {
                if (!maskCache.has(item.frame_idx)) {
                    const blob = base64ToBlob(item.png_base64)
                    const bitmap = await createImageBitmap(blob)
                    maskCache.set(item.frame_idx, bitmap)
                }
            }
            
            if (response.masks.length > 0) {
                prefetchedUpTo = response.masks[response.masks.length - 1].frame_idx
            }
        } catch (e) {
            console.error('Failed to prefetch masks:', e)
        } finally {
            isPrefetching = false
        }
    }

    const loadFrameData = async (frameIdx: number) => {
        if (!projectId.value || !videoId.value) return

        const cached = maskCache.get(frameIdx)
        if (cached) {
            if (frameIdx === intendedFrameIdx.value) {
                currentMask.value = cached
            }
            return
        }

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

    const handlePointComplete = async (point: { x: number; y: number; type: 'positive_point' | 'negative_point' }) => {
        if (!projectId.value || !videoId.value) return

        isSegmenting.value = true

        try {
            const maskBlob = await runSegmentation(
                projectId.value,
                videoId.value,
                currentFrameIdx.value,
                point.type,
                { x: point.x, y: point.y }
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

    const handlePropagateForward = async () => {
        if (!projectId.value || !videoId.value) return

        isPropagating.value = true

        try {
            await propagateForward(
                projectId.value,
                videoId.value,
                currentFrameIdx.value,
                1000
            )
            await loadFrameData(currentFrameIdx.value)
        } catch (e) {
            console.error('Failed to propagate forward:', e)
        } finally {
            isPropagating.value = false
        }
    }

    const handlePropagateBackward = async () => {
        if (!projectId.value || !videoId.value) return

        isPropagating.value = true

        try {
            await propagateBackward(
                projectId.value,
                videoId.value,
                currentFrameIdx.value,
                1000
            )
            await loadFrameData(currentFrameIdx.value)
        } catch (e) {
            console.error('Failed to propagate backward:', e)
        } finally {
            isPropagating.value = false
        }
    }

    const clearPromptStorage = () => {
        promptStorage.clearAll()
    }

    watch(currentFrameIdx, (newFrameIdx, oldFrameIdx) => {
        if (newFrameIdx === intendedFrameIdx.value) {
            return
        }
        
        intendedFrameIdx.value = newFrameIdx
        
        if (isPlaying.value) {
            return
        }
        
        if (debounceTimeout) {
            clearTimeout(debounceTimeout)
        }
        debounceTimeout = window.setTimeout(() => {
            loadFrameData(newFrameIdx)
        }, 100)
    })

    const syncMaskToVideo = () => {
        if (!isPlaying.value || !videoRef.value) return
        
        const frameIdx = Math.floor(videoRef.value.currentTime * fps.value)
        
        if (frameIdx !== lastDisplayedFrame) {
            const cached = maskCache.get(frameIdx)
            if (cached) {
                currentMask.value = cached
                lastDisplayedFrame = frameIdx
            }
            
            const framesAhead = prefetchedUpTo - frameIdx
            if (framesAhead < PREFETCH_THRESHOLD) {
                prefetchMasks(prefetchedUpTo + 1)
            }
        }
        
        animationFrameId = requestAnimationFrame(syncMaskToVideo)
    }

    watch(isPlaying, async (playing) => {
        if (playing) {
            await prefetchMasks(currentFrameIdx.value)
            prefetchMasks(currentFrameIdx.value + PREFETCH_BATCH_SIZE)
            lastDisplayedFrame = -1
            syncMaskToVideo()
        } else {
            if (animationFrameId !== null) {
                cancelAnimationFrame(animationFrameId)
                animationFrameId = null
            }
        }
    })

    watch(videoId, () => {
        maskCache.clear()
        prefetchedUpTo = -1
    })

    onUnmounted(() => {
        if (debounceTimeout) {
            clearTimeout(debounceTimeout)
        }
        if (animationFrameId !== null) {
            cancelAnimationFrame(animationFrameId)
        }
        maskCache.clear()
    })

    return {
        activeTool,
        currentMask,
        currentPrompts,
        isSegmenting,
        isPropagating,
        loadFrameData,
        seekToFrame,
        togglePositivePointTool,
        toggleNegativePointTool,
        handlePointComplete,
        handleResetFrame,
        handleResetVideo,
        handlePropagateForward,
        handlePropagateBackward,
        clearPromptStorage,
    }
}
