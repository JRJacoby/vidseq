export const API_BASE = '/api'

async function getErrorMessage(response: Response, fallback: string): Promise<string> {
    const body = await response.json().catch(() => ({}))
    return body.detail || fallback
}

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
        throw new Error(await getErrorMessage(response, 'Failed to fetch projects'))
    }
    return response.json()
}

export async function getProject(projectId: number): Promise<Project> {
    const response = await fetch(`${API_BASE}/projects/${projectId}`)
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to fetch project'))
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
        throw new Error(await getErrorMessage(response, 'Failed to create project'))
    }
    
    return response.json()
}

export async function deleteProject(projectId: number): Promise<void> {
    const response = await fetch(`${API_BASE}/projects/${projectId}`, {
        method: 'DELETE',
    })
    
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to delete project'))
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
        throw new Error(await getErrorMessage(response, 'Failed to fetch videos'))
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
        throw new Error(await getErrorMessage(response, 'Failed to add videos'))
    }
}

export async function getVideo(projectId: number, videoId: number): Promise<Video> {
    const response = await fetch(`${API_BASE}/projects/${projectId}/videos/${videoId}`)
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to fetch video'))
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
        throw new Error(await getErrorMessage(response, 'Failed to fetch directory listing'))
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
        throw new Error(await getErrorMessage(response, 'Failed to fetch jobs'))
    }
    return response.json()
}

export async function runSegmentation(
    projectId: number,
    videoId: number,
    frameIdx: number,
    type: 'positive_point' | 'negative_point',
    details: { x: number; y: number }
): Promise<Blob> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/${videoId}/segment`,
        {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ frame_idx: frameIdx, type, details }),
        }
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to run segmentation'))
    }
    return response.blob()
}

export async function getMask(
    projectId: number,
    videoId: number,
    frameIdx: number
): Promise<Blob> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/${videoId}/mask/${frameIdx}`
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to fetch mask'))
    }
    return response.blob()
}

export interface MaskBatchItem {
    frame_idx: number
    png_base64: string
}

export interface MaskBatchResponse {
    masks: MaskBatchItem[]
}

export async function getMasksBatch(
    projectId: number,
    videoId: number,
    startFrame: number,
    count: number = 100
): Promise<MaskBatchResponse> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/${videoId}/masks-batch?start_frame=${startFrame}&count=${count}`
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to fetch mask batch'))
    }
    return response.json()
}

export async function getConditioningFrames(
    projectId: number,
    videoId: number
): Promise<number[]> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/${videoId}/conditioning-frames`
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to fetch conditioning frames'))
    }
    const data = await response.json()
    return data.conditioning_frames
}

export interface SegmentationStatus {
    status: 'not_loaded' | 'loading_model' | 'ready' | 'error'
    error: string | null
}

export async function getSegmentationStatus(): Promise<SegmentationStatus> {
    const response = await fetch(`${API_BASE}/segmentation/status`)
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to fetch segmentation status'))
    }
    return response.json()
}

export async function preloadSegmentation(): Promise<void> {
    const response = await fetch(`${API_BASE}/segmentation/preload`, { method: 'POST' })
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to start segmentation preload'))
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
        throw new Error(await getErrorMessage(response, 'Failed to init video session'))
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
        throw new Error(await getErrorMessage(response, 'Failed to close video session'))
    }
}

export async function resetFrame(
    projectId: number,
    videoId: number,
    frameIdx: number
): Promise<void> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/${videoId}/frame/${frameIdx}`,
        { method: 'DELETE' }
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to reset frame'))
    }
}

export async function resetVideo(
    projectId: number,
    videoId: number
): Promise<void> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/${videoId}/all-frames`,
        { method: 'DELETE' }
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to reset video'))
    }
}

export interface PropagateResponse {
    frames_processed: number
}

export async function propagateForward(
    projectId: number,
    videoId: number,
    startFrameIdx: number,
    maxFrames: number = 1000
): Promise<PropagateResponse> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/${videoId}/propagate-forward`,
        {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ start_frame_idx: startFrameIdx, max_frames: maxFrames }),
        }
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to propagate forward'))
    }
    return response.json()
}

export async function propagateBackward(
    projectId: number,
    videoId: number,
    startFrameIdx: number,
    maxFrames: number = 1000
): Promise<PropagateResponse> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/${videoId}/propagate-backward`,
        {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ start_frame_idx: startFrameIdx, max_frames: maxFrames }),
        }
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to propagate backward'))
    }
    return response.json()
}
