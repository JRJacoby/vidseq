import { ref, watch, onUnmounted, type Ref } from 'vue'
import {
    getDetectionStatus,
    createDetectionTraining,
    deleteDetectionTraining,
    updateDetectionConfig,
    type DetectorStatus,
} from '@/services/api'

export interface UseDetectorReturn {
    isTraining: Ref<boolean>
    modelExists: Ref<boolean>
    detectorType: Ref<string>
    startTraining: (maxEpochs?: number, videoIds?: number[], fromCheckpoint?: boolean) => Promise<void>
    stopTraining: () => Promise<void>
    checkStatus: () => Promise<void>
    setDetectorType: (type: string) => Promise<void>
}

export function useDetector(projectId: Ref<number | null>): UseDetectorReturn {
    const isTraining = ref(false)
    const modelExists = ref(false)
    const detectorType = ref('rtdetr')

    const checkStatus = async () => {
        if (!projectId.value) return
        try {
            const status = await getDetectionStatus(projectId.value)
            modelExists.value = status.model_exists
            isTraining.value = status.is_training
            detectorType.value = status.detector_type
        } catch (e) {
            console.error('Failed to check detector status:', e)
        }
    }

    const setDetectorType = async (type: string) => {
        if (!projectId.value) return
        try {
            await updateDetectionConfig(projectId.value, type)
            detectorType.value = type
        } catch (e) {
            console.error('Failed to set detector type:', e)
        }
    }

    const startTraining = async (maxEpochs: number = 1000, videoIds: number[] = [], fromCheckpoint: boolean = false) => {
        if (!projectId.value || isTraining.value) return
        isTraining.value = true
        try {
            await createDetectionTraining(projectId.value, maxEpochs, videoIds, 10, 20, fromCheckpoint)
        } catch (e) {
            console.error('Failed to start detector training:', e)
            isTraining.value = false
            throw e
        }
    }

    const stopTraining = async () => {
        if (!projectId.value) return
        try {
            await deleteDetectionTraining(projectId.value)
        } catch (e) {
            console.error('Failed to stop training:', e)
            throw e
        }
    }

    // Check status on mount and when projectId changes
    watch(projectId, async (newId) => {
        if (newId !== null) {
            await checkStatus()
        }
    }, { immediate: true })

    // Poll while training
    let pollInterval: number | null = null

    watch(isTraining, (training) => {
        if (training) {
            if (pollInterval === null && projectId.value !== null) {
                pollInterval = window.setInterval(async () => {
                    await checkStatus()
                    if (!isTraining.value && pollInterval !== null) {
                        clearInterval(pollInterval)
                        pollInterval = null
                    }
                }, 2000)
            }
        } else {
            if (pollInterval !== null) {
                clearInterval(pollInterval)
                pollInterval = null
            }
        }
    })

    // Cleanup on unmount
    onUnmounted(() => {
        if (pollInterval !== null) {
            clearInterval(pollInterval)
            pollInterval = null
        }
    })

    return {
        isTraining,
        modelExists,
        detectorType,
        startTraining,
        stopTraining,
        checkStatus,
        setDetectorType,
    }
}
