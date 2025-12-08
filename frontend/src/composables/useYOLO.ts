import { ref, watch, type Ref } from 'vue'
import {
    trainInitialDetectionModel,
    runInitialDetection,
    getYOLOModelStatus,
    type YOLOModelStatus,
} from '@/services/api'

export interface UseYOLOReturn {
    isTraining: Ref<boolean>
    isApplying: Ref<boolean>
    modelExists: Ref<boolean>
    trainModel: (projectId: number) => Promise<void>
    runInitialDetection: (projectId: number) => Promise<void>
    checkModelStatus: (projectId: number) => Promise<void>
}

export function useYOLO(
    projectId: Ref<number | null>,
): UseYOLOReturn {
    const isTraining = ref(false)
    const isApplying = ref(false)
    const modelExists = ref(false)

    const checkModelStatus = async (projectIdValue: number) => {
        try {
            const status = await getYOLOModelStatus(projectIdValue)
            modelExists.value = status.exists
            isTraining.value = status.is_training
            isApplying.value = status.is_applying
        } catch (e) {
            console.error('Failed to check model status:', e)
        }
    }

    const trainModel = async (projectIdValue: number) => {
        if (isTraining.value) return

        isTraining.value = true
        try {
            await trainInitialDetectionModel(projectIdValue)
            await checkModelStatus(projectIdValue)
        } catch (e) {
            console.error('Failed to train model:', e)
            throw e
        } finally {
            isTraining.value = false
        }
    }

    const runInitialDetectionFn = async (projectIdValue: number) => {
        if (isApplying.value || !modelExists.value) return

        isApplying.value = true
        try {
            await runInitialDetection(projectIdValue)
            await checkModelStatus(projectIdValue)
        } catch (e) {
            console.error('Failed to run initial detection:', e)
            throw e
        } finally {
            isApplying.value = false
        }
    }

    watch(projectId, async (newProjectId) => {
        if (newProjectId !== null) {
            await checkModelStatus(newProjectId)
        }
    }, { immediate: true })

    let statusPollInterval: number | null = null

    watch([isTraining, isApplying], ([training, applying]) => {
        if (training || applying) {
            if (statusPollInterval === null && projectId.value !== null) {
                statusPollInterval = window.setInterval(async () => {
                    if (projectId.value !== null) {
                        await checkModelStatus(projectId.value)
                        if (!isTraining.value && !isApplying.value) {
                            if (statusPollInterval !== null) {
                                clearInterval(statusPollInterval)
                                statusPollInterval = null
                            }
                        }
                    }
                }, 1000)
            }
        } else {
            if (statusPollInterval !== null) {
                clearInterval(statusPollInterval)
                statusPollInterval = null
            }
        }
    })

    return {
        isTraining,
        isApplying,
        modelExists,
        trainModel,
        runInitialDetection: runInitialDetectionFn,
        checkModelStatus,
    }
}

