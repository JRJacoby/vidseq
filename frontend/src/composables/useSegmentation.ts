import { ref, watch, onUnmounted, computed, type Ref } from 'vue'
import {
    getMask,
    getMasksBatch,
    getBbox,
    getBboxesBatch,
    runSegmentation,
    resetFrame,
    resetVideo,
    generateTrainingMasks,
    getPromptsForFrame,
    type StoredPrompt,
    type Bbox,
} from '@/services/api'

export type ToolType = 'none' | 'positive_point' | 'negative_point'

export interface UseSegmentationReturn {
    activeTool: Ref<ToolType>
    currentMask: Ref<ImageBitmap | null>
    currentBbox: Ref<Bbox | null>
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
    handleGenerateTrainingMasks: () => Promise<void>
    clearMaskCache: (startFrame?: number, endFrame?: number) => void
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
    const currentBbox = ref<Bbox | null>(null)
    const isSegmenting = ref(false)
    const isPropagating = ref(false)
    const intendedFrameIdx = ref(0)

    const prompts = ref<Map<number, StoredPrompt[]>>(new Map())
    
    const currentPrompts = computed(() => {
        return prompts.value.get(currentFrameIdx.value) || []
    })
    
    const fetchPromptsForFrame = async (frameIdx: number) => {
        if (!projectId.value || !videoId.value) return
        try {
            const fetchedPrompts = await getPromptsForFrame(
                projectId.value,
                videoId.value,
                frameIdx
            )
            prompts.value.set(frameIdx, fetchedPrompts)
        } catch (e) {
            console.error('Failed to fetch prompts:', e)
            prompts.value.set(frameIdx, [])
        }
    }

    let debounceTimeout: number | null = null
    
    const maskCache = new Map<number, ImageBitmap>()
    const bboxCache = new Map<number, Bbox | null>()
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
            const [maskResponse, bboxResponse] = await Promise.all([
                getMasksBatch(
                    projectId.value,
                    videoId.value,
                    startFrame,
                    PREFETCH_BATCH_SIZE
                ),
                getBboxesBatch(
                    projectId.value,
                    videoId.value,
                    startFrame,
                    PREFETCH_BATCH_SIZE
                )
            ])
            
            for (const item of maskResponse.masks) {
                if (!maskCache.has(item.frame_idx)) {
                    const blob = base64ToBlob(item.png_base64)
                    const bitmap = await createImageBitmap(blob)
                    maskCache.set(item.frame_idx, bitmap)
                }
            }
            
            for (const item of bboxResponse.bboxes) {
                if (!bboxCache.has(item.frame_idx)) {
                    const bboxValue = item.bbox ? {
                        x1: item.bbox[0],
                        y1: item.bbox[1],
                        x2: item.bbox[2],
                        y2: item.bbox[3],
                    } : null
                    bboxCache.set(item.frame_idx, bboxValue)
                    if (item.frame_idx < 10) {
                        console.log(`[Prefetch] Cached bbox for frame ${item.frame_idx}:`, bboxValue)
                    }
                }
            }
            
            if (maskResponse.masks.length > 0) {
                prefetchedUpTo = maskResponse.masks[maskResponse.masks.length - 1].frame_idx
            }
        } catch (e) {
            console.error('Failed to prefetch masks:', e)
        } finally {
            isPrefetching = false
        }
    }

    const loadFrameData = async (frameIdx: number) => {
        if (!projectId.value || !videoId.value) return

        const cachedMask = maskCache.get(frameIdx)
        const cachedBbox = bboxCache.get(frameIdx)
        
        if (frameIdx < 10) {
            console.log(`[loadFrameData] Frame ${frameIdx}: cachedMask=${cachedMask !== undefined}, cachedBbox=${cachedBbox !== undefined}, intendedFrameIdx=${intendedFrameIdx.value}`)
        }
        
        if (cachedMask !== undefined && cachedBbox !== undefined) {
            if (frameIdx === intendedFrameIdx.value) {
                currentMask.value = cachedMask
                currentBbox.value = cachedBbox
                if (frameIdx < 10) {
                    console.log(`[loadFrameData] Set from cache - Frame ${frameIdx}: bbox=`, cachedBbox)
                }
            }
            return
        }

        try {
            const [maskBlob, bbox] = await Promise.all([
                getMask(projectId.value, videoId.value, frameIdx),
                getBbox(projectId.value, videoId.value, frameIdx)
            ])

            if (frameIdx !== intendedFrameIdx.value) {
                if (frameIdx < 10) {
                    console.log(`[loadFrameData] Frame ${frameIdx} changed, skipping (intended=${intendedFrameIdx.value})`)
                }
                return
            }

            currentMask.value = await createImageBitmap(maskBlob)
            currentBbox.value = bbox
            if (frameIdx < 10) {
                console.log(`[loadFrameData] Loaded from API - Frame ${frameIdx}: bbox=`, bbox)
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

        try {
            const maskBlob = await runSegmentation(
                projectId.value,
                videoId.value,
                currentFrameIdx.value,
                point.type,
                { x: point.x, y: point.y }
            )

            currentMask.value = await createImageBitmap(maskBlob)
            
            // Refresh prompts for current frame after adding a point
            await fetchPromptsForFrame(currentFrameIdx.value)
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
            maskCache.delete(currentFrameIdx.value)
            bboxCache.delete(currentFrameIdx.value)
            await loadFrameData(currentFrameIdx.value)
            // Refresh prompts after reset (should be empty)
            await fetchPromptsForFrame(currentFrameIdx.value)
        } catch (e) {
            console.error('Failed to reset frame:', e)
        }
    }

    const handleResetVideo = async () => {
        if (!projectId.value || !videoId.value) return

        try {
            await resetVideo(projectId.value, videoId.value)
            prompts.value.clear()
            maskCache.clear()
            bboxCache.clear()
            prefetchedUpTo = -1
            currentMask.value = null
            currentBbox.value = null
        } catch (e) {
            console.error('Failed to reset video:', e)
        }
    }

    const handleGenerateTrainingMasks = async () => {
        if (!projectId.value || !videoId.value) return

        isPropagating.value = true

        try {
            await generateTrainingMasks(
                projectId.value,
                videoId.value,
                currentFrameIdx.value,
                100
            )
            maskCache.clear()
            bboxCache.clear()
            prefetchedUpTo = -1
            await loadFrameData(currentFrameIdx.value)
        } catch (e) {
            console.error('Failed to generate training masks:', e)
        } finally {
            isPropagating.value = false
        }
    }

    watch(currentFrameIdx, async (newFrameIdx, oldFrameIdx) => {
        if (newFrameIdx === intendedFrameIdx.value) {
            return
        }
        
        intendedFrameIdx.value = newFrameIdx
        
        if (isPlaying.value) {
            return
        }
        
        // Fetch prompts for the new frame
        if (!prompts.value.has(newFrameIdx)) {
            await fetchPromptsForFrame(newFrameIdx)
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
            const cachedBbox = bboxCache.get(frameIdx)
            if (cachedMask !== undefined && cachedBbox !== undefined) {
                currentMask.value = cachedMask
                currentBbox.value = cachedBbox
                lastDisplayedFrame = frameIdx
                if (frameIdx < 10) {
                    console.log(`[syncMaskToVideo] Frame ${frameIdx}: bbox=`, cachedBbox)
                }
            } else if (frameIdx < 10) {
                console.log(`[syncMaskToVideo] Frame ${frameIdx}: missing cache (mask=${cachedMask !== undefined}, bbox=${cachedBbox !== undefined})`)
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
        prompts.value.clear()
        maskCache.clear()
        bboxCache.clear()
        prefetchedUpTo = -1
    })

    watch([projectId, videoId, currentFrameIdx], async ([pid, vid, frameIdx]) => {
        if (pid && vid !== null && frameIdx !== undefined && !isPlaying.value) {
            await prefetchMasks(frameIdx)
            prefetchMasks(frameIdx + PREFETCH_BATCH_SIZE)
            // Fetch prompts for the current frame
            if (!prompts.value.has(frameIdx)) {
                await fetchPromptsForFrame(frameIdx)
            }
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
                bboxCache.delete(i)
            }
        } else {
            maskCache.clear()
            bboxCache.clear()
        }
        prefetchedUpTo = -1
    }

    return {
        activeTool,
        currentMask,
        currentBbox,
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
        handleGenerateTrainingMasks,
        clearMaskCache,
    }
}
