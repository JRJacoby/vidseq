import { ref, watch, onUnmounted, type Ref } from 'vue'
import {
    getSegDetectionStatus,
    createSegTraining,
    deleteSegTraining,
} from '@/services/api'

export function useSegDetector(projectId: Ref<number | null>) {
    const isTraining = ref(false)
    const modelExists = ref(false)

    const checkStatus = async () => {
        if (!projectId.value) return
        try {
            const status = await getSegDetectionStatus(projectId.value)
            modelExists.value = status.model_exists
            isTraining.value = status.is_training
        } catch (e) {
            console.error('Failed to check seg detector status:', e)
        }
    }

    const startTraining = async (maxEpochs: number = 1000, videoIds: number[] = []) => {
        if (!projectId.value || isTraining.value) return
        isTraining.value = true
        try {
            await createSegTraining(projectId.value, maxEpochs, videoIds)
        } catch (e) {
            console.error('Failed to start seg training:', e)
            isTraining.value = false
            throw e
        }
    }

    const stopTraining = async () => {
        if (!projectId.value) return
        try {
            await deleteSegTraining(projectId.value)
        } catch (e) {
            console.error('Failed to stop seg training:', e)
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
