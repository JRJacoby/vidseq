import { ref, computed, type Ref } from 'vue'
import { createGraphcutMasks, type GraphCutSeed } from '@/services/api'

export const GRAPHCUT_CHUNK_SIZE = 150

export function useGraphCut(
    projectId: Ref<number>,
    videoId: Ref<number>,
) {
    const graphcutRegion = ref<{ start: number; end: number } | null>(null)
    const seeds = ref<Map<number, GraphCutSeed[]>>(new Map())
    const brushSize = ref(5)
    const seedsVisible = ref(true)
    const isRunning = ref(false)

    const isGraphCutMode = computed(() => graphcutRegion.value !== null)

    function placeRegion(frameIdx: number, numFrames: number) {
        const end = Math.min(frameIdx + GRAPHCUT_CHUNK_SIZE - 1, numFrames - 1)
        graphcutRegion.value = { start: frameIdx, end }
        seeds.value = new Map()
    }

    function paintSeed(frameIdx: number, x: number, y: number, label: number) {
        if (!graphcutRegion.value) return
        if (frameIdx < graphcutRegion.value.start || frameIdx > graphcutRegion.value.end) return

        const existing = seeds.value.get(frameIdx) ?? []
        existing.push({ x, y, label })
        seeds.value.set(frameIdx, existing)
    }

    function getSeedsForFrame(frameIdx: number): GraphCutSeed[] {
        return seeds.value.get(frameIdx) ?? []
    }

    async function runGraphCut(clearMaskCache: (start: number, end: number) => void) {
        if (!graphcutRegion.value) return
        if (seeds.value.size === 0) return

        const seedsObj: Record<string, GraphCutSeed[]> = {}
        for (const [frameIdx, points] of seeds.value) {
            seedsObj[String(frameIdx)] = points
        }

        isRunning.value = true
        try {
            await createGraphcutMasks(
                projectId.value,
                videoId.value,
                graphcutRegion.value.start,
                graphcutRegion.value.end,
                seedsObj,
            )
            clearMaskCache(graphcutRegion.value.start, graphcutRegion.value.end)
        } finally {
            isRunning.value = false
        }
    }

    function clearSeeds() {
        seeds.value = new Map()
    }

    function exitGraphCutMode() {
        graphcutRegion.value = null
        seeds.value = new Map()
    }

    return {
        graphcutRegion,
        seeds,
        brushSize,
        seedsVisible,
        isRunning,
        isGraphCutMode,
        placeRegion,
        paintSeed,
        getSeedsForFrame,
        runGraphCut,
        clearSeeds,
        exitGraphCutMode,
    }
}
