<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { getDirectoryListing, type DirectoryEntry } from '@/services/api'

const props = defineProps<{
  initialPath?: string
}>()

const emit = defineEmits<{
  'files-selected': [paths: string[]]
  'cancel': []
}>()

const currentPath = ref(props.initialPath || '/')
const entries = ref<DirectoryEntry[]>([])
const selectedPaths = ref<string[]>([])
const isLoading = ref(false)

const loadDirectory = async (path: string) => {
  isLoading.value = true
  try {
    entries.value = await getDirectoryListing(path)
    currentPath.value = path
  } catch (error) {
    console.error('Error loading directory:', error)
  } finally {
    isLoading.value = false
  }
}

onMounted(() => {
  loadDirectory(currentPath.value)
})

const handlePathClick = (path: string) => {
  loadDirectory(path)
}

const handleGoUp = () => {
  const parts = currentPath.value.split('/').filter(p => p)
  if (parts.length === 0) return
  const parentPath = '/' + parts.slice(0, -1).join('/')
  loadDirectory(parentPath)
}

const canGoUp = () => {
  return currentPath.value !== '/'
}

const handleEntrySelect = (path: string) => {
  const index = selectedPaths.value.indexOf(path)
  if (index === -1) {
    selectedPaths.value.push(path)
  } else {
    selectedPaths.value.splice(index, 1)
  }
}

const handleConfirm = () => {
  emit('files-selected', selectedPaths.value)
}

const handleCancel = () => {
  emit('cancel')
}
</script>

<template>
  <div class="modal-overlay" @click.self="handleCancel">
    <div class="modal-content">
      <div class="modal-header">
        <h2 class="modal-title">Select Files or Directories</h2>
        <button class="modal-close" @click="handleCancel">×</button>
      </div>
      
      <div class="modal-body">
        <div class="path-navigation">
          <button 
            class="up-button" 
            @click="handleGoUp"
            :disabled="!canGoUp()"
            title="Go to parent directory"
          >
            ↑
          </button>
          <span class="path-label">Current path:</span>
          <span class="path-value">{{ currentPath }}</span>
        </div>
        
        <div class="directory-listing">
          <div v-if="isLoading" class="loading-state">
            Loading directory...
          </div>
          <div v-else-if="entries.length === 0" class="empty-state">
            No files or directories found.
          </div>
          <div v-else class="entries-list">
            <div
              v-for="entry in entries"
              :key="entry.path"
              class="entry-item"
              :class="{ 'entry-selected': selectedPaths.includes(entry.path) }"
              @click="handleEntrySelect(entry.path)"
            >
              <input
                type="checkbox"
                :checked="selectedPaths.includes(entry.path)"
                @click.stop="handleEntrySelect(entry.path)"
              />
              <span class="entry-icon">{{ entry.isDirectory ? '📁' : '📄' }}</span>
              <span class="entry-name">{{ entry.name }}</span>
              <button
                v-if="entry.isDirectory"
                class="entry-enter"
                @click.stop="handlePathClick(entry.path)"
              >
                Enter
              </button>
            </div>
          </div>
        </div>
      </div>
      
      <div class="modal-footer">
        <div class="selected-count">
          {{ selectedPaths.length }} selected
        </div>
        <div class="modal-actions">
          <button class="button-secondary" @click="handleCancel">Cancel</button>
          <button
            class="button-primary"
            @click="handleConfirm"
            :disabled="selectedPaths.length === 0"
          >
            Select
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-color: rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
}

.modal-content {
  background-color: white;
  border-radius: 8px;
  width: 90%;
  max-width: 800px;
  max-height: 90vh;
  display: flex;
  flex-direction: column;
}

.modal-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1rem;
  border-bottom: 1px solid #e0e0e0;
  flex-shrink: 0;
}

.modal-title {
  margin: 0;
}

.modal-close {
  background: none;
  border: none;
  font-size: 1.5rem;
  cursor: pointer;
  padding: 0;
  width: 2rem;
  height: 2rem;
  display: flex;
  align-items: center;
  justify-content: center;
}

.modal-body {
  display: flex;
  flex-direction: column;
  flex: 1;
  min-height: 0;
  padding: 1rem;
  overflow: auto;
}

.path-navigation {
  display: flex;
  gap: 0.5rem;
  margin-bottom: 1rem;
  flex-shrink: 0;
  align-items: center;
}

.up-button {
  background-color: #f0f0f0;
  border: 1px solid #ccc;
  border-radius: 4px;
  padding: 0.25rem 0.5rem;
  cursor: pointer;
  font-size: 1.2rem;
  line-height: 1;
  transition: background-color 0.2s;
}

.up-button:hover:not(:disabled) {
  background-color: #e0e0e0;
}

.up-button:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

.path-label {
  font-weight: bold;
}

.path-value {
  font-family: monospace;
}

.directory-listing {
  display: flex;
  flex-direction: column;
  flex: 1;
  min-height: 0;
  overflow: auto;
}

.loading-state,
.empty-state {
  padding: 2rem;
  text-align: center;
}

.entries-list {
  display: flex;
  flex-direction: column;
}

.entry-item {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem;
  cursor: pointer;
  border: 1px solid transparent;
}

.entry-item:hover {
  background-color: #f5f5f5;
}

.entry-selected {
  background-color: #e3f2fd;
  border-color: #2196f3;
}

.entry-icon {
  font-size: 1.2rem;
}

.entry-name {
  flex: 1;
}

.entry-enter {
  padding: 0.25rem 0.5rem;
  margin-left: auto;
}

.modal-footer {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1rem;
  border-top: 1px solid #e0e0e0;
  flex-shrink: 0;
}

.selected-count {
  font-weight: bold;
}

.modal-actions {
  display: flex;
  gap: 0.5rem;
}

.button-primary,
.button-secondary {
  padding: 0.5rem 1rem;
  border: 1px solid #ccc;
  border-radius: 4px;
  cursor: pointer;
}

.button-primary:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
</style>

