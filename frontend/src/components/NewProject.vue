<script setup lang="ts">
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { createProject } from '@/services/api'

const router = useRouter()

const projectName = ref('')
const directoryPath = ref('')
const errorMessage = ref('')

const handleCreateProject = async () => {
  errorMessage.value = ''
  
  if (!projectName.value.trim()) {
    errorMessage.value = 'Project name is required'
    return
  }
  
  if (!directoryPath.value.trim()) {
    errorMessage.value = 'Directory path is required'
    return
  }
  
  try {
    const project = await createProject(projectName.value, directoryPath.value)
    router.push(`/project/${project.id}`)
  } catch (error) {
    if (error instanceof Error) {
      errorMessage.value = error.message
    } else {
      errorMessage.value = 'Failed to create project'
    }
  }
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