import { ref, watch, type Ref } from 'vue'
import {
    getFrameRanges,
    getTrainingRangeValidation,
    createTrainingRange,
    deleteTrainingRange,
} from '@/services/api'

export interface UseFrameRangesReturn {
    maskedRanges: Ref<[number, number][]>
    trainingRanges: Ref<[number, number][]>
    isLoading: Ref<boolean>
    refresh: () => Promise<void>
    markTraining: (startFrame: number, endFrame: number) => Promise<{ success: boolean; error?: string; missingFrames?: number[] }>
    unmarkTraining: (startFrame: number, endFrame: number) => Promise<void>
    validateRange: (startFrame: number, endFrame: number) => Promise<{ valid: boolean; missingFrames?: number[] }>
}

export function useFrameRanges(
    projectId: Ref<number | null>,
    videoId: Ref<number | null>
): UseFrameRangesReturn {
    const maskedRanges = ref<[number, number][]>([])
    const trainingRanges = ref<[number, number][]>([])
    const isLoading = ref(false)

    const refresh = async () => {
        if (!projectId.value || !videoId.value) return
        
        isLoading.value = true
        try {
            const response = await getFrameRanges(projectId.value, videoId.value)
            maskedRanges.value = response.masked_ranges
            trainingRanges.value = response.training_ranges
        } catch (e) {
            console.error('Failed to fetch frame ranges:', e)
            maskedRanges.value = []
            trainingRanges.value = []
        } finally {
            isLoading.value = false
        }
    }

    const validateRange = async (startFrame: number, endFrame: number): Promise<{ valid: boolean; missingFrames?: number[] }> => {
        if (!projectId.value || !videoId.value) {
            return { valid: false, missingFrames: [] }
        }
        
        try {
            const response = await getTrainingRangeValidation(
                projectId.value,
                videoId.value,
                startFrame,
                endFrame
            )
            return {
                valid: response.valid,
                missingFrames: response.missing_frames,
            }
        } catch (e) {
            console.error('Failed to validate training range:', e)
            return { valid: false }
        }
    }

    const markTraining = async (startFrame: number, endFrame: number): Promise<{ success: boolean; error?: string; missingFrames?: number[] }> => {
        if (!projectId.value || !videoId.value) {
            return { success: false, error: 'No project or video selected' }
        }
        
        try {
            await createTrainingRange(
                projectId.value,
                videoId.value,
                startFrame,
                endFrame
            )
            await refresh()
            return { success: true }
        } catch (e) {
            const error = e instanceof Error ? e.message : 'Unknown error'
            return { success: false, error }
        }
    }

    const unmarkTraining = async (startFrame: number, endFrame: number): Promise<void> => {
        if (!projectId.value || !videoId.value) return
        
        try {
            await deleteTrainingRange(
                projectId.value,
                videoId.value,
                startFrame,
                endFrame
            )
            await refresh()
        } catch (e) {
            console.error('Failed to unmark training range:', e)
        }
    }

    watch([projectId, videoId], () => {
        maskedRanges.value = []
        trainingRanges.value = []
        refresh()
    }, { immediate: true })

    return {
        maskedRanges,
        trainingRanges,
        isLoading,
        refresh,
        markTraining,
        unmarkTraining,
        validateRange,
    }
}
