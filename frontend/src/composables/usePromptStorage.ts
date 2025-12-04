import { ref, computed, type Ref, type ComputedRef } from 'vue'

export interface StoredPrompt {
    type: 'bbox' | 'positive_point' | 'negative_point'
    details: Record<string, number>
    createdAt: string
}

export interface UsePromptStorageReturn {
    prompts: Ref<Map<number, StoredPrompt[]>>
    addPrompt: (frameIdx: number, prompt: Omit<StoredPrompt, 'createdAt'>) => void
    removePromptsForFrame: (frameIdx: number) => void
    getPromptsForFrame: (frameIdx: number) => StoredPrompt[]
    clearAll: () => void
    hasPromptsForFrame: (frameIdx: number) => boolean
    totalPromptCount: ComputedRef<number>
}

export function usePromptStorage(
    projectId: Ref<number | null>,
    videoId: Ref<number | null>
): UsePromptStorageReturn {
    const prompts = ref<Map<number, StoredPrompt[]>>(new Map())

    const storageKey = computed(() => {
        if (projectId.value === null || videoId.value === null) return null
        return `vidseq-prompts-${projectId.value}-${videoId.value}`
    })

    function loadFromStorage(): void {
        if (!storageKey.value) return
        try {
            const stored = localStorage.getItem(storageKey.value)
            if (stored) {
                const parsed = JSON.parse(stored) as [number, StoredPrompt[]][]
                prompts.value = new Map(parsed)
            }
        } catch (e) {
            console.error('Failed to load prompts from storage:', e)
            prompts.value = new Map()
        }
    }

    function saveToStorage(): void {
        if (!storageKey.value) return
        try {
            const serializable = Array.from(prompts.value.entries())
            localStorage.setItem(storageKey.value, JSON.stringify(serializable))
        } catch (e) {
            console.error('Failed to save prompts to storage:', e)
        }
    }

    function addPrompt(frameIdx: number, prompt: Omit<StoredPrompt, 'createdAt'>): void {
        const fullPrompt: StoredPrompt = {
            ...prompt,
            createdAt: new Date().toISOString()
        }
        const framePrompts = prompts.value.get(frameIdx) || []
        framePrompts.push(fullPrompt)
        prompts.value.set(frameIdx, framePrompts)
        saveToStorage()
    }

    function removePromptsForFrame(frameIdx: number): void {
        prompts.value.delete(frameIdx)
        saveToStorage()
    }

    function getPromptsForFrame(frameIdx: number): StoredPrompt[] {
        return prompts.value.get(frameIdx) || []
    }

    function hasPromptsForFrame(frameIdx: number): boolean {
        const framePrompts = prompts.value.get(frameIdx)
        return !!framePrompts && framePrompts.length > 0
    }

    function clearAll(): void {
        prompts.value = new Map()
        if (storageKey.value) {
            localStorage.removeItem(storageKey.value)
        }
    }

    const totalPromptCount = computed(() => {
        let count = 0
        for (const framePrompts of prompts.value.values()) {
            count += framePrompts.length
        }
        return count
    })

    loadFromStorage()

    return {
        prompts,
        addPrompt,
        removePromptsForFrame,
        getPromptsForFrame,
        clearAll,
        hasPromptsForFrame,
        totalPromptCount
    }
}

