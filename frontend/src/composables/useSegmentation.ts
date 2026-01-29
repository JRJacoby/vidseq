import { ref, watch, onUnmounted, computed, type Ref } from 'vue'
import {
    getMask,
    getMasksBatch,
    getDetectorMasksBatch,
    runSegmentation,
    refineMaskMultiPoint,
    resetFrame,
    resetVideo,
    getDetectorMask,
} from '@/services/api'
import { LruCache } from '@/utils/LruCache'

export type ToolType = 'none' | 'positive_point' | 'negative_point'

export interface UseSegmentationReturn {
    activeTool: Ref<ToolType>
    currentMask: Ref<ImageBitmap | null>
    currentPrompts: Ref<Array<{ x: number; y: number; type: 'positive_point' | 'negative_point' }>>
    isSegmenting: Ref<boolean>
    loadFrameData: (frameIdx: number) => Promise<void>
    seekToFrame: (frameIdx: number) => void
    togglePositivePointTool: () => void
    toggleNegativePointTool: () => void
    handlePointComplete: (point: { x: number; y: number; type: 'positive_point' | 'negative_point' }) => Promise<void>
    handleResetFrame: () => Promise<void>
    handleResetVideo: () => Promise<void>
    clearMaskCache: (startFrame?: number, endFrame?: number) => void
}

const PREFETCH_BATCH_SIZE = 100
const PREFETCH_THRESHOLD = 100
const MASK_CACHE_MAX_SIZE = 10000  // ~1GB assuming ~100KB per mask

export function useSegmentation(
    projectId: Ref<number | null>,
    videoId: Ref<number | null>,
    currentFrameIdx: Ref<number>,
    isPlaying: Ref<boolean> = ref(false),
    videoRef: Ref<HTMLVideoElement | null> = ref(null),
    fps: Ref<number> = ref(30),
    maskViewMode: Ref<'tracker' | 'detector'> = ref('tracker'),
): UseSegmentationReturn {
    const activeTool = ref<ToolType>('none')
    const currentMask = ref<ImageBitmap | null>(null)
    const isSegmenting = ref(false)
    const intendedFrameIdx = ref(0)

    // Local prompts tracking (not persisted to server)
    const localPrompts = ref<Map<number, Array<{ x: number; y: number; type: 'positive_point' | 'negative_point' }>>>(new Map())

    const currentPrompts = computed(() => {
        return localPrompts.value.get(currentFrameIdx.value) || []
    })

    let debounceTimeout: number | null = null

    const maskCache = new LruCache<number, ImageBitmap>(
        MASK_CACHE_MAX_SIZE,
        (bitmap) => bitmap.close()
    )
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
            const batchFn = maskViewMode.value === 'detector' ? getDetectorMasksBatch : getMasksBatch
            const maskResponse = await batchFn(
                projectId.value,
                videoId.value,
                startFrame,
                PREFETCH_BATCH_SIZE
            )

            for (const item of maskResponse.masks) {
                if (!maskCache.has(item.frame_idx)) {
                    const blob = base64ToBlob(item.png_base64)
                    const bitmap = await createImageBitmap(blob)
                    maskCache.set(item.frame_idx, bitmap)
                }
            }

            if (maskResponse.masks.length > 0) {
                prefetchedUpTo = maskResponse.masks[maskResponse.masks.length - 1]!.frame_idx
            }
        } catch (e) {
            console.error('Failed to prefetch masks:', e)
        } finally {
            isPrefetching = false
        }
    }

    const fetchMaskForFrame = async (frameIdx: number): Promise<Blob | null> => {
        if (!projectId.value || !videoId.value) return null

        try {
            if (maskViewMode.value === 'detector') {
                return await getDetectorMask(projectId.value, videoId.value, frameIdx)
            } else {
                return await getMask(projectId.value, videoId.value, frameIdx)
            }
        } catch {
            return null
        }
    }

    const loadFrameData = async (frameIdx: number) => {
        if (!projectId.value || !videoId.value) return

        const cachedMask = maskCache.get(frameIdx)

        if (cachedMask !== undefined) {
            if (frameIdx === intendedFrameIdx.value) {
                currentMask.value = cachedMask
            }
            return
        }

        try {
            const maskBlob = await fetchMaskForFrame(frameIdx)

            if (frameIdx !== intendedFrameIdx.value) {
                return
            }

            if (maskBlob) {
                const bitmap = await createImageBitmap(maskBlob)
                maskCache.set(frameIdx, bitmap)
                currentMask.value = bitmap
            } else {
                currentMask.value = null
            }
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

        // Add to local prompts first
        const framePrompts = localPrompts.value.get(currentFrameIdx.value) || []
        framePrompts.push({ x: point.x, y: point.y, type: point.type })
        localPrompts.value.set(currentFrameIdx.value, framePrompts)

        try {
            let maskBlob: Blob

            // Check if this frame already has a mask (refinement vs new)
            const hasExistingMask = maskCache.has(currentFrameIdx.value) && maskCache.get(currentFrameIdx.value) !== null

            if (hasExistingMask && framePrompts.length > 1) {
                // Refinement with accumulated points - send ALL prompts
                maskBlob = await refineMaskMultiPoint(
                    projectId.value,
                    videoId.value,
                    currentFrameIdx.value,
                    framePrompts
                )
            } else {
                // First point on frame - use single-point API
                maskBlob = await runSegmentation(
                    projectId.value,
                    videoId.value,
                    currentFrameIdx.value,
                    point.type,
                    { x: point.x, y: point.y }
                )
            }

            const bitmap = await createImageBitmap(maskBlob)
            currentMask.value = bitmap
            maskCache.set(currentFrameIdx.value, bitmap)
        } catch (e) {
            console.error('Failed to add point:', e)
            // Rollback: remove the point we just added
            framePrompts.pop()
            if (framePrompts.length === 0) {
                localPrompts.value.delete(currentFrameIdx.value)
            }
        } finally {
            isSegmenting.value = false
        }
    }

    const handleResetFrame = async () => {
        if (!projectId.value || !videoId.value) return

        try {
            await resetFrame(projectId.value, videoId.value, currentFrameIdx.value)
            maskCache.delete(currentFrameIdx.value)
            localPrompts.value.delete(currentFrameIdx.value)
            await loadFrameData(currentFrameIdx.value)
        } catch (e) {
            console.error('Failed to reset frame:', e)
        }
    }

    const handleResetVideo = async () => {
        if (!projectId.value || !videoId.value) return

        try {
            await resetVideo(projectId.value, videoId.value)
            localPrompts.value.clear()
            maskCache.clear()
            prefetchedUpTo = -1
            currentMask.value = null
        } catch (e) {
            console.error('Failed to reset video:', e)
        }
    }

    watch(currentFrameIdx, async (newFrameIdx) => {
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
            const cachedMask = maskCache.get(frameIdx)
            if (cachedMask !== undefined) {
                currentMask.value = cachedMask
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
        localPrompts.value.clear()
        maskCache.clear()
        prefetchedUpTo = -1
    })

    // Clear cache and reload when mask view mode changes
    watch(maskViewMode, () => {
        maskCache.clear()
        prefetchedUpTo = -1
        loadFrameData(currentFrameIdx.value)
    })

    watch([projectId, videoId, currentFrameIdx], async ([pid, vid, frameIdx]) => {
        if (pid && vid !== null && frameIdx !== undefined && !isPlaying.value) {
            await prefetchMasks(frameIdx)
            prefetchMasks(frameIdx + PREFETCH_BATCH_SIZE)
        }
    }, { immediate: true })

    onUnmounted(() => {
        if (debounceTimeout) {
            clearTimeout(debounceTimeout)
        }
        if (animationFrameId !== null) {
            cancelAnimationFrame(animationFrameId)
        }
        maskCache.clear()
    })

    const clearMaskCache = (startFrame?: number, endFrame?: number) => {
        if (startFrame !== undefined && endFrame !== undefined) {
            for (let i = startFrame; i <= endFrame; i++) {
                maskCache.delete(i)
            }
        } else {
            maskCache.clear()
        }
        prefetchedUpTo = -1
    }

    return {
        activeTool,
        currentMask,
        currentPrompts,
        isSegmenting,
        loadFrameData,
        seekToFrame,
        togglePositivePointTool,
        toggleNegativePointTool,
        handlePointComplete,
        handleResetFrame,
        handleResetVideo,
        clearMaskCache,
    }
}
