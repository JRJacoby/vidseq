import { defineStore } from 'pinia'
import { getProject } from '@/services/api'

export const useProjectStore = defineStore('project', {
  state: () => ({
    currentProjectId: (() => {
      const stored = localStorage.getItem('currentProjectId')
      return stored ? Number(stored) : null
    })() as number | null,
  }),
  
  actions: {
    setCurrentProject(id: number) {
      this.currentProjectId = id
      localStorage.setItem('currentProjectId', String(id))
    },
    
    clearCurrentProject() {
      this.currentProjectId = null
      localStorage.removeItem('currentProjectId')
    },
    
    async validateCurrentProject() {
      if (this.currentProjectId === null) {
        return
      }
      
      try {
        await getProject(this.currentProjectId)
      } catch (error) {
        console.warn('Stored project no longer exists, clearing')
        this.clearCurrentProject()
      }
    }
  }
})

