import { ref, type Ref } from 'vue'
import {
    getPoseLabels,
    savePoseLabel,
    deletePoseLabel,
    type PoseLabel,
} from '@/services/api'

export type LabelingState = 'idle' | 'awaiting_front' | 'awaiting_rear'

export interface UsePoseLabelsReturn {
    currentLabel: Ref<PoseLabel | null>
    labeledFrameCount: Ref<number>
    labelingState: Ref<LabelingState>
    pendingFront: Ref<{ x: number; y: number } | null>
    loadLabel: (frameIdx: number) => Promise<void>
    saveLabel: (frameIdx: number, frontX: number, frontY: number, rearX: number, rearY: number) => Promise<void>
    deleteLabel: (frameIdx: number) => Promise<void>
    refresh: () => Promise<void>
    startLabeling: () => void
    stopLabeling: () => void
    handleClick: (x: number, y: number, frameIdx: number, advanceFrame: () => void) => Promise<void>
}

export function usePoseLabels(
    projectId: Ref<number>,
    videoId: Ref<number>,
): UsePoseLabelsReturn {
    const currentLabel = ref<PoseLabel | null>(null)
    const labeledFrameCount = ref(0)
    const labelingState = ref<LabelingState>('idle')
    const pendingFront = ref<{ x: number; y: number } | null>(null)

    // Cache: frame_idx -> PoseLabel
    const labelCache = new Map<number, PoseLabel | null>()

    const loadLabel = async (frameIdx: number) => {
        if (labelCache.has(frameIdx)) {
            currentLabel.value = labelCache.get(frameIdx) ?? null
            return
        }
        // Frame not in cache — no label exists for it
        currentLabel.value = null
    }

    const refresh = async () => {
        if (!projectId.value || !videoId.value) return
        try {
            const labels = await getPoseLabels(projectId.value, videoId.value)
            labelCache.clear()
            for (const label of labels) {
                labelCache.set(label.frame_idx, label)
            }
            labeledFrameCount.value = labels.length
        } catch (e) {
            console.error('Failed to refresh pose labels:', e)
        }
    }

    const saveLabel = async (
        frameIdx: number,
        frontX: number,
        frontY: number,
        rearX: number,
        rearY: number,
    ) => {
        await savePoseLabel(projectId.value, videoId.value, frameIdx, frontX, frontY, rearX, rearY)
        const label: PoseLabel = { frame_idx: frameIdx, front_x: frontX, front_y: frontY, rear_x: rearX, rear_y: rearY }
        labelCache.set(frameIdx, label)
        currentLabel.value = label
        labeledFrameCount.value = labelCache.size
    }

    const deleteLabel = async (frameIdx: number) => {
        await deletePoseLabel(projectId.value, videoId.value, frameIdx)
        labelCache.delete(frameIdx)
        currentLabel.value = null
        labeledFrameCount.value = labelCache.size
    }

    const startLabeling = () => {
        labelingState.value = 'awaiting_front'
        pendingFront.value = null
    }

    const stopLabeling = () => {
        labelingState.value = 'idle'
        pendingFront.value = null
    }

    const handleClick = async (x: number, y: number, frameIdx: number, advanceFrame: () => void) => {
        if (labelingState.value === 'awaiting_front') {
            pendingFront.value = { x, y }
            labelingState.value = 'awaiting_rear'
        } else if (labelingState.value === 'awaiting_rear' && pendingFront.value) {
            const front = pendingFront.value
            await saveLabel(frameIdx, front.x, front.y, x, y)
            pendingFront.value = null
            labelingState.value = 'awaiting_front'
            advanceFrame()
        }
    }

    return {
        currentLabel,
        labeledFrameCount,
        labelingState,
        pendingFront,
        loadLabel,
        saveLabel,
        deleteLabel,
        refresh,
        startLabeling,
        stopLabeling,
        handleClick,
    }
}
