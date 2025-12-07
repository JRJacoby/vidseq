import { ref, watch, type Ref } from 'vue'
import {
    trainUNetModel,
    applyUNetModel,
    testApplyUNetModel,
    getUNetModelStatus,
    type UNetModelStatus,
} from '@/services/api'

// Note: This composable is named useUNet for backward compatibility,
// but it now uses YOLO (YOLOv8-nano) for bounding box detection instead of UNet.

export interface UseUNetReturn {
    isTraining: Ref<boolean>
    isApplying: Ref<boolean>
    modelExists: Ref<boolean>
    trainModel: (projectId: number) => Promise<void>
    applyModel: (projectId: number, videoId: number) => Promise<void>
    testApplyModel: (projectId: number, videoId: number, startFrame: number) => Promise<void>
    checkModelStatus: (projectId: number) => Promise<void>
}

export function useUNet(
    projectId: Ref<number | null>,
): UseUNetReturn {
    const isTraining = ref(false)
    const isApplying = ref(false)
    const modelExists = ref(false)

    const checkModelStatus = async (projectIdValue: number) => {
        try {
            const status = await getUNetModelStatus(projectIdValue)
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
            await trainUNetModel(projectIdValue)
            await checkModelStatus(projectIdValue)
        } catch (e) {
            console.error('Failed to train model:', e)
            throw e
        } finally {
            isTraining.value = false
        }
    }

    const applyModel = async (projectIdValue: number, videoId: number) => {
        if (isApplying.value || !modelExists.value) return

        isApplying.value = true
        try {
            await applyUNetModel(projectIdValue, videoId)
            await checkModelStatus(projectIdValue)
        } catch (e) {
            console.error('Failed to apply model:', e)
            throw e
        } finally {
            isApplying.value = false
        }
    }

    const testApplyModel = async (projectIdValue: number, videoId: number, startFrame: number) => {
        if (isApplying.value || !modelExists.value) return

        isApplying.value = true
        try {
            await testApplyUNetModel(projectIdValue, videoId, startFrame)
            await checkModelStatus(projectIdValue)
        } catch (e) {
            console.error('Failed to test apply model:', e)
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
        applyModel,
        testApplyModel,
        checkModelStatus,
    }
}

