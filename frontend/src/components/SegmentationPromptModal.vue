<script setup lang="ts">
import { ref } from 'vue'

const emit = defineEmits<{
  'confirm': [prompt: string]
  'cancel': []
}>()

const prompt = ref('mouse, excluding tail')

const handleConfirm = () => {
  emit('confirm', prompt.value)
}

const handleCancel = () => {
  emit('cancel')
}
</script>

<template>
  <div class="modal-overlay" @click.self="handleCancel">
    <div class="modal-content">
      <div class="modal-header">
        <h2 class="modal-title">Segmentation Prompt</h2>
        <button class="modal-close" @click="handleCancel">×</button>
      </div>
      
      <div class="modal-body">
        <label class="prompt-label">
          Enter segmentation prompt:
        </label>
        <textarea 
          v-model="prompt"
          class="prompt-input"
          rows="4"
          placeholder="Describe what to segment..."
        />
        <p class="prompt-help">
          Describe the object you want to segment in the video frames.
        </p>
      </div>
      
      <div class="modal-footer">
        <div class="modal-actions">
          <button class="button-secondary" @click="handleCancel">Cancel</button>
          <button class="button-primary" @click="handleConfirm">
            Run Segmentation
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
  max-width: 500px;
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
  padding: 1.5rem;
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.prompt-label {
  font-weight: bold;
  font-size: 0.95rem;
}

.prompt-input {
  width: 100%;
  padding: 0.5rem;
  border: 1px solid #ccc;
  border-radius: 4px;
  font-family: inherit;
  font-size: 0.95rem;
  resize: vertical;
}

.prompt-input:focus {
  outline: none;
  border-color: #4a90e2;
  box-shadow: 0 0 0 2px rgba(74, 144, 226, 0.1);
}

.prompt-help {
  margin: 0;
  font-size: 0.85rem;
  color: #666;
}

.modal-footer {
  padding: 1rem;
  border-top: 1px solid #e0e0e0;
  flex-shrink: 0;
}

.modal-actions {
  display: flex;
  gap: 0.5rem;
  justify-content: flex-end;
}

.button-secondary,
.button-primary {
  padding: 0.5rem 1rem;
  border-radius: 4px;
  border: 1px solid;
  cursor: pointer;
  font-size: 0.95rem;
}

.button-secondary {
  background-color: white;
  border-color: #ccc;
  color: #333;
}

.button-secondary:hover {
  background-color: #f5f5f5;
}

.button-primary {
  background-color: #4a90e2;
  border-color: #4a90e2;
  color: white;
}

.button-primary:hover {
  background-color: #357abd;
  border-color: #357abd;
}
</style>

