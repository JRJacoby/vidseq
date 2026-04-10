import { ref, computed, type Ref } from 'vue'
import { createThresholdMasks } from '@/services/api'

export const CHUNK_SIZE = 150

export function useGraphCut(
    projectId: Ref<number>,
    videoId: Ref<number>,
) {
    const graphcutRegion = ref<{ start: number; end: number } | null>(null)
    const threshold = ref(30)
    const clickPoint = ref<{ x: number; y: number; frame: number } | null>(null)
    const isRunning = ref(false)

    const isGraphCutMode = computed(() => graphcutRegion.value !== null)

    function placeRegion(frameIdx: number, numFrames: number) {
        const end = Math.min(frameIdx + CHUNK_SIZE - 1, numFrames - 1)
        graphcutRegion.value = { start: frameIdx, end }
        clickPoint.value = null
    }

    function setClickPoint(x: number, y: number, frame: number) {
        clickPoint.value = { x, y, frame }
    }

    async function runThresholdSegment(clearMaskCache: (start: number, end: number) => void) {
        if (!graphcutRegion.value) return
        if (!clickPoint.value) return

        isRunning.value = true
        try {
            await createThresholdMasks(
                projectId.value,
                videoId.value,
                graphcutRegion.value.start,
                graphcutRegion.value.end,
                threshold.value,
                clickPoint.value.x,
                clickPoint.value.y,
                clickPoint.value.frame,
            )
            clearMaskCache(graphcutRegion.value.start, graphcutRegion.value.end)
        } finally {
            isRunning.value = false
        }
    }

    function exitGraphCutMode() {
        graphcutRegion.value = null
        clickPoint.value = null
    }

    return {
        graphcutRegion,
        threshold,
        clickPoint,
        isRunning,
        isGraphCutMode,
        placeRegion,
        setClickPoint,
        runThresholdSegment,
        exitGraphCutMode,
    }
}
