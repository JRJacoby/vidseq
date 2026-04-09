import { ref, watch, onUnmounted, type Ref } from 'vue'
import {
    getPoseStatus,
    createPoseTraining,
    deletePoseTraining,
} from '@/services/api'

export interface UsePoseDetectorReturn {
    isTraining: Ref<boolean>
    modelExists: Ref<boolean>
    startTraining: (maxEpochs?: number, videoIds?: number[]) => Promise<void>
    stopTraining: () => Promise<void>
    checkStatus: () => Promise<void>
}

export function usePoseDetector(projectId: Ref<number | null>): UsePoseDetectorReturn {
    const isTraining = ref(false)
    const modelExists = ref(false)

    const checkStatus = async () => {
        if (!projectId.value) return
        try {
            const status = await getPoseStatus(projectId.value)
            modelExists.value = status.model_exists
            isTraining.value = status.is_training
        } catch (e) {
            console.error('Failed to check pose status:', e)
        }
    }

    const startTraining = async (maxEpochs: number = 300, videoIds: number[] = []) => {
        if (!projectId.value || isTraining.value) return
        isTraining.value = true
        try {
            await createPoseTraining(projectId.value, maxEpochs, videoIds)
        } catch (e) {
            console.error('Failed to start pose training:', e)
            isTraining.value = false
            throw e
        }
    }

    const stopTraining = async () => {
        if (!projectId.value) return
        try {
            await deletePoseTraining(projectId.value)
        } catch (e) {
            console.error('Failed to stop pose training:', e)
            throw e
        }
    }

    watch(projectId, async (newId) => {
        if (newId !== null) {
            await checkStatus()
        }
    }, { immediate: true })

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

    onUnmounted(() => {
        if (pollInterval !== null) {
            clearInterval(pollInterval)
            pollInterval = null
        }
    })

    return {
        isTraining,
        modelExists,
        startTraining,
        stopTraining,
        checkStatus,
    }
}
