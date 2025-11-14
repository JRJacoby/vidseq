const API_BASE = '/api'

export interface Project {
    id: number
    name: string
    path: string
    created_at: string
    updated_at: string
}

export async function getProjects(): Promise<Project[]> {
    const response = await fetch(`${API_BASE}/projects`)
    if (!response.ok) {
        throw new Error(`Failed to fetch projects: ${response.statusText}`)
    }
    return response.json()
}

export async function createProject(name: string, path: string): Promise<Project> {
    const response = await fetch(`${API_BASE}/projects`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ name, path }),
    })
    
    if (!response.ok) {
        const error = await response.json().catch(() => ({}))
        throw new Error(error.detail || 'Failed to create project for unknown reason.')
    }
    
    return response.json()
}

export interface Video {
    id: number
    name: string
    path: string
}

export async function getVideos(projectId: number): Promise<Video[]> {
    const response = await fetch(`${API_BASE}/projects/${projectId}/videos`)
    if (!response.ok) {
        throw new Error(`Failed to fetch videos: ${response.statusText}`)
    }
    return response.json()
}