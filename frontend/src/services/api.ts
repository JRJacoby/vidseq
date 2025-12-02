export const API_BASE = '/api'

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

export async function getProject(projectId: number): Promise<Project> {
    const response = await fetch(`${API_BASE}/projects/${projectId}`)
    if (!response.ok) {
        throw new Error(`Failed to fetch project: ${response.statusText}`)
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

export async function deleteProject(projectId: number): Promise<void> {
    const response = await fetch(`${API_BASE}/projects/${projectId}`, {
        method: 'DELETE',
    })
    
    if (!response.ok) {
        const error = await response.json().catch(() => ({}))
        throw new Error(error.detail || 'Failed to delete project')
    }
}

export interface Video {
    id: number
    name: string
    path: string
    fps: number
}

export async function getVideos(projectId: number): Promise<Video[]> {
    const response = await fetch(`${API_BASE}/projects/${projectId}/videos`)
    if (!response.ok) {
        throw new Error(`Failed to fetch videos: ${response.statusText}`)
    }
    return response.json()
}

export async function addVideos(projectId: number, paths: string[]): Promise<void> {
    const response = await fetch(`${API_BASE}/projects/${projectId}/videos`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ paths }),
    })
    
    if (!response.ok) {
        const error = await response.json().catch(() => ({}))
        throw new Error(error.detail || 'Failed to add videos')
    }
}

export async function getVideo(projectId: number, videoId: number): Promise<Video> {
    const response = await fetch(`${API_BASE}/projects/${projectId}/videos/${videoId}`)
    if (!response.ok) {
        throw new Error(`Failed to fetch video: ${response.statusText}`)
    }
    return response.json()
}

export function getVideoStreamUrl(projectId: number, videoId: number): string {
    return `${API_BASE}/projects/${projectId}/videos/${videoId}/stream`
}

export interface DirectoryEntry {
    name: string
    path: string
    isDirectory: boolean
}

export async function getDirectoryListing(path: string): Promise<DirectoryEntry[]> {
    const response = await fetch(`${API_BASE}/filesystem/list?path=${encodeURIComponent(path)}`)
    if (!response.ok) {
        throw new Error(`Failed to fetch directory listing: ${response.statusText}`)
    }
    return response.json()
}

export interface Job {
    id: number
    type: string
    status: string
    project_id: number
    details: object
    log_path: string
    created_at: string
    updated_at: string
}

export async function getJobs(): Promise<Job[]> {
    const response = await fetch(`${API_BASE}/jobs`)
    if (!response.ok) {
        throw new Error(`Failed to fetch jobs: ${response.statusText}`)
    }
    return response.json()
}

export interface Prompt {
    id: number
    video_id: number
    frame_idx: number
    type: string
    details: Record<string, number>
    created_at: string
}

export async function addPrompt(
    projectId: number,
    videoId: number,
    frameIdx: number,
    type: string,
    details: Record<string, number>
): Promise<Prompt> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/${videoId}/prompts`,
        {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ frame_idx: frameIdx, type, details }),
        }
    )
    if (!response.ok) {
        const error = await response.json().catch(() => ({}))
        throw new Error(error.detail || 'Failed to add prompt')
    }
    return response.json()
}

export async function getPrompts(
    projectId: number,
    videoId: number,
    frameIdx: number
): Promise<Prompt[]> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/${videoId}/prompts?frame_idx=${frameIdx}`
    )
    if (!response.ok) {
        throw new Error(`Failed to fetch prompts: ${response.statusText}`)
    }
    return response.json()
}

export async function deletePrompt(
    projectId: number,
    videoId: number,
    promptId: number
): Promise<Blob> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/${videoId}/prompts/${promptId}`,
        { method: 'DELETE' }
    )
    if (!response.ok) {
        const error = await response.json().catch(() => ({}))
        throw new Error(error.detail || 'Failed to delete prompt')
    }
    return response.blob()
}

export async function getMask(
    projectId: number,
    videoId: number,
    frameNumber: number
): Promise<Blob> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/${videoId}/mask/${frameNumber}`
    )
    if (!response.ok) {
        throw new Error(`Failed to fetch mask: ${response.statusText}`)
    }
    return response.blob()
}

export interface SAM3Status {
    status: 'not_loaded' | 'loading_model' | 'ready' | 'error'
    error: string | null
}

export async function getSAM3Status(): Promise<SAM3Status> {
    const response = await fetch(`${API_BASE}/sam3/status`)
    if (!response.ok) {
        throw new Error(`Failed to fetch SAM3 status: ${response.statusText}`)
    }
    return response.json()
}

export async function preloadSAM3(): Promise<void> {
    const response = await fetch(`${API_BASE}/sam3/preload`, { method: 'POST' })
    if (!response.ok) {
        throw new Error(`Failed to start SAM3 preload: ${response.statusText}`)
    }
}

export interface VideoSessionInfo {
    video_id: number
    num_frames: number
    height: number
    width: number
}

export async function initVideoSession(
    projectId: number,
    videoId: number
): Promise<VideoSessionInfo> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/${videoId}/session`,
        { method: 'POST' }
    )
    if (!response.ok) {
        const error = await response.json().catch(() => ({}))
        throw new Error(error.detail || 'Failed to init video session')
    }
    return response.json()
}

export async function closeVideoSession(
    projectId: number,
    videoId: number
): Promise<void> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/${videoId}/session`,
        { method: 'DELETE' }
    )
    if (!response.ok) {
        const error = await response.json().catch(() => ({}))
        throw new Error(error.detail || 'Failed to close video session')
    }
}

export async function resetFrame(
    projectId: number,
    videoId: number,
    frameNumber: number
): Promise<void> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/${videoId}/frame/${frameNumber}`,
        { method: 'DELETE' }
    )
    if (!response.ok) {
        const error = await response.json().catch(() => ({}))
        throw new Error(error.detail || 'Failed to reset frame')
    }
}