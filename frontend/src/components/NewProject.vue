<script setup lang="ts">
import { ref } from 'vue'

const projectName = ref('')
const directoryPath = ref('')
const errorMessage = ref('')

const handleCreateProject = () => {
  // TODO: Implement project creation logic
}
</script>

<template>
  <h3 class="screen-title">New Project</h3>
  <div class="screen-content">
    <form @submit.prevent="handleCreateProject" class="new-project-form">
      <div class="form-field">
        <label for="project-name">Project Name</label>
        <input
          id="project-name"
          v-model="projectName"
          type="text"
          placeholder="Enter project name"
          required
        />
      </div>

      <div class="form-field">
        <label for="directory-path">
          Choose which folder your new project folder will be created within. This folder will contain all results from modeling and intermediate pipeline steps.
        </label>
        <input
          id="directory-path"
          v-model="directoryPath"
          type="text"
          placeholder="Enter path to directory"
          required
        />
      </div>

      <button type="submit" class="create-project-button">
        Create Project
      </button>

      <p v-if="errorMessage" class="error-message">
        {{ errorMessage }}
      </p>
    </form>
  </div>
</template>

<style scoped>
.new-project-form {
  max-width: 600px;
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}

.form-field {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

label {
  font-weight: 500;
  color: #333;
}

input {
  padding: 0.75rem;
  border: 1px solid #ddd;
  border-radius: 0.25rem;
  font-size: 1rem;
}

input:focus {
  outline: none;
  border-color: #3498db;
}

.create-project-button {
  padding: 0.75rem 1.5rem;
  background-color: #3498db;
  color: white;
  border: none;
  border-radius: 0.25rem;
  font-size: 1rem;
  cursor: pointer;
  transition: background-color 0.2s;
  align-self: flex-start;
}

.create-project-button:hover {
  background-color: #2980b9;
}

.error-message {
  color: #e74c3c;
  margin: 0;
}
</style>

