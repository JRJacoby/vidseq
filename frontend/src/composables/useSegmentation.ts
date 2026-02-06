import { ref, watch, onUnmounted, computed, type Ref } from 'vue'
import {
    getTrackerMask,
    getTrackerMasks,
    getDetectorMasks,
    submitPrompt,
    deleteSegmentation,
    deleteVideoSegmentation,
    getDetectorMask,
    getDetectorBbox,
    getFinalMask,
    getFinalMasks,
    type DetectorBbox,
} from '@/services/api'
import { LruCache } from '@/utils/LruCache'

// ============================================================================
// Types & Constants
// ============================================================================

export type ToolType = 'none' | 'positive_point' | 'negative_point'

export interface UseSegmentationReturn {
    activeTool: Ref<ToolType>
    currentMask: Ref<ImageBitmap | null>
    detectorBbox: Ref<DetectorBbox | null>
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
    maskViewMode: Ref<'tracker' | 'detector' | 'final'> = ref('tracker'),
): UseSegmentationReturn {

    // ========================================================================
    // State
    // ========================================================================

    const activeTool = ref<ToolType>('none')
    const currentMask = ref<ImageBitmap | null>(null)
    const detectorBbox = ref<DetectorBbox | null>(null)
    const isSegmenting = ref(false)
    const intendedFrameIdx = ref(0)

    // Local prompts tracking (not persisted to server)
    const localPrompts = ref<Map<number, Array<{ x: number; y: number; type: 'positive_point' | 'negative_point' }>>>(new Map())

    const currentPrompts = computed(() => {
        return localPrompts.value.get(currentFrameIdx.value) || []
    })

    // Cache and prefetch state
    const maskCache = new LruCache<number, ImageBitmap>(
        MASK_CACHE_MAX_SIZE,
        (bitmap) => bitmap.close()
    )
    let isPrefetching = false
    let prefetchedUpTo = -1

    // Playback sync state
    let animationFrameId: number | null = null
    let lastDisplayedFrame = -1
    let debounceTimeout: number | null = null

    // ========================================================================
    // Cache Management
    // ========================================================================

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
            const batchFn = maskViewMode.value === 'detector'
                ? getDetectorMasks
                : maskViewMode.value === 'final'
                    ? getFinalMasks
                    : getTrackerMasks

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

    /**
     * Clear masks from the cache. This is the ONLY way to clear the cache safely.
     * Always use this instead of calling maskCache.delete() or maskCache.clear() directly.
     *
     * The cache's onEvict callback closes ImageBitmaps, so we must null currentMask
     * first if it references a bitmap being cleared - otherwise Vue will try to
     * render a detached bitmap.
     *
     * @param startFrame - Start of range to clear (inclusive). Omit for full clear.
     * @param endFrame - End of range to clear (inclusive). Omit for full clear.
     */
    const clearMaskCache = (startFrame?: number, endFrame?: number) => {
        if (startFrame !== undefined && endFrame !== undefined) {
            if (currentFrameIdx.value >= startFrame && currentFrameIdx.value <= endFrame) {
                currentMask.value = null
            }
            for (let i = startFrame; i <= endFrame; i++) {
                maskCache.delete(i)
            }
        } else {
            currentMask.value = null
            maskCache.clear()
        }
        prefetchedUpTo = -1
    }

    // ========================================================================
    // Mask Loading
    // ========================================================================

    const fetchMaskForFrame = async (frameIdx: number): Promise<Blob | null> => {
        if (!projectId.value || !videoId.value) return null

        try {
            if (maskViewMode.value === 'detector') {
                // Fetch bbox instead of mask
                const result = await getDetectorBbox(projectId.value, videoId.value, frameIdx)
                detectorBbox.value = result.bbox
                return null  // No mask to render
            } else if (maskViewMode.value === 'final') {
                return await getFinalMask(projectId.value, videoId.value, frameIdx)
            } else {
                return await getTrackerMask(projectId.value, videoId.value, frameIdx)
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

            if (frameIdx !== intendedFrameIdx.value) return

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

    const seekToFrame = (frameIdx: number) => {
        intendedFrameIdx.value = frameIdx
        loadFrameData(frameIdx)
    }

    // ========================================================================
    // Tool State
    // ========================================================================

    const togglePositivePointTool = () => {
        activeTool.value = activeTool.value === 'positive_point' ? 'none' : 'positive_point'
    }

    const toggleNegativePointTool = () => {
        activeTool.value = activeTool.value === 'negative_point' ? 'none' : 'negative_point'
    }

    // ========================================================================
    // Segmentation Actions
    // ========================================================================

    const handlePointComplete = async (point: { x: number; y: number; type: 'positive_point' | 'negative_point' }) => {
        if (!projectId.value || !videoId.value) return

        isSegmenting.value = true

        // Add to local prompts first
        const framePrompts = localPrompts.value.get(currentFrameIdx.value) || []
        framePrompts.push({ x: point.x, y: point.y, type: point.type })
        localPrompts.value.set(currentFrameIdx.value, framePrompts)

        try {
            // Always send all accumulated prompts - backend determines workflow
            const maskBlob = await submitPrompt(
                projectId.value,
                videoId.value,
                currentFrameIdx.value,
                framePrompts
            )

            const bitmap = await createImageBitmap(maskBlob)
            currentMask.value = bitmap
            maskCache.set(currentFrameIdx.value, bitmap)
        } catch (e) {
            console.error('Failed to submit prompt:', e)
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
            await deleteSegmentation(projectId.value, videoId.value, currentFrameIdx.value)
            clearMaskCache(currentFrameIdx.value, currentFrameIdx.value)
            localPrompts.value.delete(currentFrameIdx.value)
            await loadFrameData(currentFrameIdx.value)
        } catch (e) {
            console.error('Failed to reset frame:', e)
        }
    }

    const handleResetVideo = async () => {
        if (!projectId.value || !videoId.value) return

        try {
            await deleteVideoSegmentation(projectId.value, videoId.value)
            localPrompts.value.clear()
            clearMaskCache()
        } catch (e) {
            console.error('Failed to reset video:', e)
        }
    }

    // ========================================================================
    // Playback Synchronization
    // ========================================================================

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

    // ========================================================================
    // Watchers
    // ========================================================================

    // Debounced frame loading when scrubbing (not during playback)
    watch(currentFrameIdx, async (newFrameIdx) => {
        if (newFrameIdx === intendedFrameIdx.value) return

        intendedFrameIdx.value = newFrameIdx

        if (isPlaying.value) return

        if (debounceTimeout) {
            clearTimeout(debounceTimeout)
        }
        debounceTimeout = window.setTimeout(() => {
            loadFrameData(newFrameIdx)
        }, 100)
    })

    // Start/stop playback sync loop
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

    // Clear state when video changes
    watch(videoId, () => {
        localPrompts.value.clear()
        clearMaskCache()
    })

    // Clear cache and reload when mask view mode changes
    watch(maskViewMode, () => {
        clearMaskCache()
        detectorBbox.value = null
        loadFrameData(currentFrameIdx.value)
    })

    // Initial prefetch when video loads
    watch([projectId, videoId, currentFrameIdx], async ([pid, vid, frameIdx]) => {
        if (pid && vid !== null && frameIdx !== undefined && !isPlaying.value) {
            await prefetchMasks(frameIdx)
            prefetchMasks(frameIdx + PREFETCH_BATCH_SIZE)
        }
    }, { immediate: true })

    // ========================================================================
    // Cleanup & Return
    // ========================================================================

    onUnmounted(() => {
        if (debounceTimeout) {
            clearTimeout(debounceTimeout)
        }
        if (animationFrameId !== null) {
            cancelAnimationFrame(animationFrameId)
        }
        clearMaskCache()
    })

    return {
        activeTool,
        currentMask,
        detectorBbox,
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
