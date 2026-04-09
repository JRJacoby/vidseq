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
    width: number
    height: number
    num_frames: number
    segmentation_status: 'in_progress' | 'segmented' | null
    min_confidence?: number
    p50_confidence?: number
    p95_confidence?: number
    // Associated video fields
    is_associated: boolean
    associated_with_id: number | null
    associated_video_id: number | null
    training_frame_count: number | null
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

export async function deleteVideos(projectId: number, videoIds: number[]): Promise<void> {
    const response = await fetch(`${API_BASE}/projects/${projectId}/videos`, {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ video_ids: videoIds })
    })
    if (!response.ok) throw new Error(await getErrorMessage(response, 'Failed to delete videos'))
}

export async function deleteVideosSegmentation(projectId: number, videoIds: number[]): Promise<void> {
    const response = await fetch(`${API_BASE}/projects/${projectId}/videos/segmentation`, {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ video_ids: videoIds })
    })
    if (!response.ok) throw new Error(await getErrorMessage(response, 'Failed to delete segmentations'))
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

export async function getFrameImage(
    projectId: number,
    videoId: number,
    frameIdx: number
): Promise<Blob> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/${videoId}/frame/${frameIdx}`
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to fetch frame image'))
    }
    return response.blob()
}

export async function getAssociatedVideo(
    projectId: number,
    videoId: number,
): Promise<Video> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/${videoId}/associated`,
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to get associated video'))
    }
    return response.json()
}

export async function addAssociatedVideos(
    projectId: number,
    jsonPath: string,
): Promise<Video[]> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/associated`,
        {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ json_path: jsonPath }),
        },
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to add associated videos'))
    }
    return response.json()
}

export async function coSegmentVideos(
    projectId: number,
    videoIds: number[],
    confidenceThreshold: number = 0.9,
): Promise<void> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/associated/segmentation`,
        {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                video_ids: videoIds,
                confidence_threshold: confidenceThreshold,
            }),
        },
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to co-segment videos'))
    }
}

export async function resetAssociatedSegmentation(
    projectId: number,
    videoId: number,
): Promise<void> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/${videoId}/associated-segmentation`,
        { method: 'DELETE' },
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to reset associated segmentation'))
    }
}

export interface DirectoryEntry {
    name: string
    path: string
    isDirectory: boolean
}

export async function getDirectoryListing(
    path: string,
    accept?: string[],
): Promise<DirectoryEntry[]> {
    let url = `${API_BASE}/filesystem/list?path=${encodeURIComponent(path)}`
    if (accept && accept.length > 0) {
        url += `&accept=${encodeURIComponent(accept.join(','))}`
    }
    const response = await fetch(url)
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to fetch directory listing'))
    }
    return response.json()
}

export interface PointPrompt {
    x: number
    y: number
    type: 'positive_point' | 'negative_point'
}

export async function submitPrompt(
    projectId: number,
    videoId: number,
    frameIdx: number,
    points: PointPrompt[]
): Promise<Blob> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/${videoId}/prompt/${frameIdx}`,
        {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ points }),
        }
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to submit prompt'))
    }
    return response.blob()
}

export async function submitBoxPrompt(
    projectId: number,
    videoId: number,
    frameIdx: number,
    box: { x1: number; y1: number; x2: number; y2: number }
): Promise<Blob> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/${videoId}/box-prompt/${frameIdx}`,
        {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(box),
        }
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to submit box prompt'))
    }
    return response.blob()
}

export async function getTrackerMask(
    projectId: number,
    videoId: number,
    frameIdx: number
): Promise<Blob> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/${videoId}/segmentation/tracker-masks/${frameIdx}`
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to fetch tracker mask'))
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

export interface MaskScore {
    frame_idx: number
    score: number // IoU score, -1.0 if not available
}

export async function getTrackerMasks(
    projectId: number,
    videoId: number,
    startFrame: number,
    count: number = 100
): Promise<MaskBatchResponse> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/${videoId}/segmentation/tracker-masks?start_frame=${startFrame}&count=${count}`
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to fetch tracker masks'))
    }
    return response.json()
}

export interface BboxResult {
    x1: number
    y1: number
    x2: number
    y2: number
}

export async function getTrackerMaskBbox(
    projectId: number,
    videoId: number,
    frameIdx: number,
): Promise<BboxResult | null> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/${videoId}/segmentation/tracker-mask-bboxes/${frameIdx}`,
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to fetch tracker mask bbox'))
    }
    const data = await response.json()
    return data.bbox
}

export interface BboxBatchItem {
    frame_idx: number
    x1: number
    y1: number
    x2: number
    y2: number
}

export async function getTrackerMaskBboxes(
    projectId: number,
    videoId: number,
    startFrame: number,
    count: number = 100,
): Promise<BboxBatchItem[]> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/${videoId}/segmentation/tracker-mask-bboxes?start_frame=${startFrame}&count=${count}`,
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to fetch tracker mask bboxes'))
    }
    const data = await response.json()
    return data.bboxes
}

export async function getScores(
    projectId: number,
    videoId: number,
    startFrame: number,
    count: number = 1000
): Promise<MaskScore[]> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/${videoId}/segmentation/scores?start_frame=${startFrame}&count=${count}`
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to fetch scores'))
    }
    const data = await response.json()
    return data.scores
}

export interface ScoresDownsampledResponse {
    scores: MaskScore[]
    total_count: number
}

export async function getScoresDownsampled(
    projectId: number,
    videoId: number,
    maxSamples: number = 800,
    startFrame?: number,
    endFrame?: number
): Promise<ScoresDownsampledResponse> {
    const params = new URLSearchParams({ max_samples: maxSamples.toString() })
    if (startFrame !== undefined) {
        params.set('start_frame', startFrame.toString())
    }
    if (endFrame !== undefined) {
        params.set('end_frame', endFrame.toString())
    }

    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/${videoId}/segmentation/scores-downsampled?${params}`
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to fetch downsampled scores'))
    }
    return response.json()
}

export async function getDetectorScoresDownsampled(
    projectId: number,
    videoId: number,
    maxSamples: number = 800,
    startFrame?: number,
    endFrame?: number
): Promise<ScoresDownsampledResponse> {
    const params = new URLSearchParams({ max_samples: maxSamples.toString() })
    if (startFrame !== undefined) {
        params.set('start_frame', startFrame.toString())
    }
    if (endFrame !== undefined) {
        params.set('end_frame', endFrame.toString())
    }
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/${videoId}/segmentation/detector-scores-downsampled?${params}`
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to fetch detector scores'))
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

export async function createSegmentationLoadedModel(): Promise<void> {
    const response = await fetch(`${API_BASE}/segmentation/loaded-model`, { method: 'POST' })
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to load segmentation model'))
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

export async function deleteSegmentation(
    projectId: number,
    videoId: number,
    frameIdx: number
): Promise<void> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/${videoId}/segmentation/${frameIdx}`,
        { method: 'DELETE' }
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to delete segmentation'))
    }
}

export async function deleteSegmentationRange(
    projectId: number,
    videoId: number,
    startFrame: number,
    endFrame: number,
): Promise<void> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/${videoId}/segmentation/range?start_frame=${startFrame}&end_frame=${endFrame}`,
        { method: 'DELETE' }
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to delete segmentation range'))
    }
}

export async function deleteVideoSegmentation(
    projectId: number,
    videoId: number
): Promise<void> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/${videoId}/segmentation`,
        { method: 'DELETE' }
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to delete video segmentation'))
    }
}

export async function createVideosSegmentation(projectId: number, videoIds: number[]): Promise<{ job_ids: number[] }> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/segmentation`,
        {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ video_ids: videoIds }),
        }
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to create videos segmentation'))
    }
    return response.json()
}

export async function createSimpleSegmentation(projectId: number, videoIds: number[]): Promise<{ job_ids: number[] }> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/segmentation/simple`,
        {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ video_ids: videoIds }),
        }
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to create simple segmentation'))
    }
    return response.json()
}

export interface PropagateResponse {
    frames_processed: number
}

export async function createPropagation(
    projectId: number,
    videoId: number,
    startFrameIdx: number,
    maxFrames: number = 1000
): Promise<PropagateResponse> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/${videoId}/propagation`,
        {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ start_frame_idx: startFrameIdx, max_frames: maxFrames }),
        }
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to create propagation'))
    }
    return response.json()
}

export async function createPropagationWithoutMemory(
    projectId: number,
    videoId: number,
    startFrameIdx: number,
    maxFrames: number = 1000
): Promise<PropagateResponse> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/${videoId}/propagation-without-memory`,
        {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ start_frame_idx: startFrameIdx, max_frames: maxFrames }),
        }
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to propagate without memory'))
    }
    return response.json()
}

export interface FrameRangesResponse {
    tracker_masked_ranges: [number, number][]
    training_ranges: [number, number][]
}

export async function getFrameRanges(
    projectId: number,
    videoId: number
): Promise<FrameRangesResponse> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/${videoId}/frame-ranges`
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to fetch frame ranges'))
    }
    return response.json()
}

export interface ValidateTrainingRangeResponse {
    valid: boolean
    missing_frames?: number[]
}

export async function getTrainingRangeValidation(
    projectId: number,
    videoId: number,
    startFrame: number,
    endFrame: number
): Promise<ValidateTrainingRangeResponse> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/${videoId}/training-range/validation?start_frame=${startFrame}&end_frame=${endFrame}`
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to validate training range'))
    }
    return response.json()
}

export async function createTrainingRange(
    projectId: number,
    videoId: number,
    startFrame: number,
    endFrame: number
): Promise<void> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/${videoId}/training-range?start_frame=${startFrame}&end_frame=${endFrame}`,
        { method: 'POST' }
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to create training range'))
    }
}

export async function deleteTrainingRange(
    projectId: number,
    videoId: number,
    startFrame: number,
    endFrame: number
): Promise<void> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/${videoId}/training-range?start_frame=${startFrame}&end_frame=${endFrame}`,
        { method: 'DELETE' }
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to delete training range'))
    }
}

// Cropped Video API

export async function createVideosExtraction(projectId: number, videoIds: number[]): Promise<{ status: string; video_count: number; message?: string }> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/extraction`,
        {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ video_ids: videoIds }),
        }
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to start video extraction'))
    }
    return response.json()
}

export interface CroppedVideoExistsResponse {
    exists: boolean
    path?: string
}

export async function getCroppedVideoExists(
    projectId: number,
    videoId: number
): Promise<CroppedVideoExistsResponse> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/${videoId}/cropped-video/exists`
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to check cropped video existence'))
    }
    return response.json()
}

export function getCroppedVideoStreamUrl(projectId: number, videoId: number): string {
    return `${API_BASE}/projects/${projectId}/videos/${videoId}/cropped-video/stream`
}

// --- Aligned Video ---

export interface AlignedVideoExistsResponse {
    exists: boolean
    path?: string
}

export async function getAlignedVideoExists(
    projectId: number,
    videoId: number
): Promise<AlignedVideoExistsResponse> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/${videoId}/aligned-video/exists`
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to check aligned video existence'))
    }
    return response.json()
}

export function getAlignedVideoStreamUrl(projectId: number, videoId: number): string {
    return `${API_BASE}/projects/${projectId}/videos/${videoId}/aligned-video/stream`
}

export async function getCroppedVideoFrame(
    projectId: number,
    videoId: number,
    frameIdx: number
): Promise<Blob> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/${videoId}/cropped-video/frame/${frameIdx}`
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to fetch cropped video frame'))
    }
    return response.blob()
}

// --- PCA API ---

export interface PCAStatus {
    has_pca: boolean
    n_components: number | null
    explained_variance_ratio: number[] | null
    total_frames: number | null
}

export interface PCARunResult {
    n_components: number
    explained_variance_ratio: number[]
    total_frames: number
}

export async function createPCA(
    projectId: number,
    nComponents: number,
    videoIds: number[],
): Promise<PCARunResult> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/pca`,
        {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ video_ids: videoIds, n_components: nComponents }),
        }
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to run PCA'))
    }
    return response.json()
}

export async function getPCAStatus(projectId: number): Promise<PCAStatus> {
    const response = await fetch(`${API_BASE}/projects/${projectId}/pca/status`)
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to get PCA status'))
    }
    return response.json()
}

export function getPCAScreePlotUrl(projectId: number): string {
    return `${API_BASE}/projects/${projectId}/pca/scree-plot`
}

export function getPCAComponentsPlotUrl(projectId: number): string {
    return `${API_BASE}/projects/${projectId}/pca/components-plot`
}

export interface PCAScorePoint {
    frame_idx: number
    score: number
}

export interface PCAScoresResponse {
    n_components: number
    scores: Record<string, PCAScorePoint[]>
}

export async function getPCAScoresDownsampled(
    projectId: number,
    videoId: number,
    pcIndices: number[],
    maxSamples: number = 800,
    startFrame?: number,
    endFrame?: number,
): Promise<PCAScoresResponse> {
    const params = new URLSearchParams({
        pc_indices: pcIndices.join(','),
        max_samples: maxSamples.toString(),
    })
    if (startFrame !== undefined) {
        params.set('start_frame', startFrame.toString())
    }
    if (endFrame !== undefined) {
        params.set('end_frame', endFrame.toString())
    }

    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/${videoId}/pca-scores?${params}`
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to fetch PCA scores'))
    }
    return response.json()
}

export async function checkPCAScoresExist(
    projectId: number,
    videoId: number,
): Promise<boolean> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/${videoId}/pca-scores/exists`
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to check PCA scores'))
    }
    const data = await response.json()
    return data.exists
}

// --- ARHMM ---

export interface ARHMMSearchEntry {
    step: number
    kappa: number
    log_kappa: number
    median_duration_ms: number
    result: 'too_low' | 'too_high' | 'found'
}

export interface ARHMMProgress {
    is_running: boolean
    status: 'idle' | 'searching' | 'fitting' | 'completed' | 'failed'
    phase: 'search' | 'final_fit'
    search_history: ARHMMSearchEntry[]
    log_low: number
    log_high: number
    current_kappa: number
    current_iteration: number
    total_iterations: number
    chosen_kappa: number
    final_median_duration_ms: number
    fps: number
    num_videos: number
    total_frames: number
    started_at: number | null
    error: string | null
    elapsed_seconds: number
    iterations_per_second: number
    eta_seconds: number
}

export interface ARHMMResults {
    kappa: number
    median_duration_ms: number
    num_videos: number
    total_frames: number
    fps: number
    search_history: ARHMMSearchEntry[]
    n_components: number
}

export async function createARHMMTraining(projectId: number): Promise<{ status: string }> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/arhmm/training`,
        { method: 'POST' }
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to start ARHMM training'))
    }
    return response.json()
}

export async function deleteARHMMTraining(projectId: number): Promise<void> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/arhmm/training`,
        { method: 'DELETE' }
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to stop ARHMM training'))
    }
}

export async function getARHMMStatus(projectId: number): Promise<ARHMMProgress> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/arhmm/status`
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to get ARHMM status'))
    }
    return response.json()
}

export async function getARHMMResults(projectId: number): Promise<ARHMMResults> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/arhmm/results`
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to get ARHMM results'))
    }
    return response.json()
}

export function getARHMMStreamUrl(projectId: number): string {
    return `${API_BASE}/projects/${projectId}/arhmm/stream`
}

export function connectARHMMStream(projectId: number): EventSource {
    return new EventSource(getARHMMStreamUrl(projectId))
}

export interface ARHMMAnalysis {
    duration_histogram: {
        bin_edges: number[]
        bin_centers: number[]
        counts: number[]
    }
    syllable_frequencies: {
        syllables: number[]
        counts: number[]
    }
    total_runs: number
    median_duration_ms: number
}

export async function getARHMMAnalysis(projectId: number): Promise<ARHMMAnalysis> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/arhmm/analysis`
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to get ARHMM analysis'))
    }
    return response.json()
}

// --- Crowd Movies ---

export interface CrowdMovieProgress {
    is_running: boolean
    status: 'idle' | 'generating' | 'completed' | 'failed'
    total_syllables: number
    completed_syllables: number
    skipped_syllables: number
    current_syllable: number
    error: string | null
}

export interface CrowdMovieEntry {
    syllable: number
    instance_count: number
    sampled: number
}

export async function createCrowdMoviesGeneration(projectId: number): Promise<{ status: string }> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/arhmm/crowd-movies/generation`,
        { method: 'POST' }
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to start crowd movie generation'))
    }
    return response.json()
}

export async function deleteCrowdMoviesGeneration(projectId: number): Promise<void> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/arhmm/crowd-movies/generation`,
        { method: 'DELETE' }
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to stop crowd movie generation'))
    }
}

export async function getCrowdMovieStatus(projectId: number): Promise<CrowdMovieProgress> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/arhmm/crowd-movies/status`
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to get crowd movie status'))
    }
    return response.json()
}

export async function getCrowdMovieList(projectId: number): Promise<CrowdMovieEntry[]> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/arhmm/crowd-movies/list`
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to get crowd movie list'))
    }
    return response.json()
}

export function getCrowdMovieStreamUrl(projectId: number): string {
    return `${API_BASE}/projects/${projectId}/arhmm/crowd-movies/stream`
}

export function connectCrowdMovieStream(projectId: number): EventSource {
    return new EventSource(getCrowdMovieStreamUrl(projectId))
}

export function getCrowdMovieVideoUrl(projectId: number, syllable: number): string {
    return `${API_BASE}/projects/${projectId}/arhmm/crowd-movies/${syllable}/video`
}

// --- Detector API ---

export interface DetectorStatus {
    model_exists: boolean
    is_training: boolean
    detector_type: string
}

export interface DetectorTrainingProgress {
    is_training: boolean
    status: 'idle' | 'training' | 'applying' | 'completed' | 'stopped' | 'failed'
    current_epoch: number
    max_epochs: number
    current_train_loss: number
    current_val_loss: number
    train_loss_history: number[]
    val_loss_history: number[]
    best_val_loss: number | null
    best_epoch: number
    current_lr: number
    epochs_without_improvement: number
    lr_patience: number
    early_stop_patience: number
    num_train_frames: number
    num_val_frames: number
    current_batch: number
    total_batches: number
    batch_loss: number
    apply_current: number
    apply_total: number
    error_message: string | null
}

export async function getDetectionStatus(projectId: number): Promise<DetectorStatus> {
    const response = await fetch(`${API_BASE}/projects/${projectId}/detection/status`)
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to get detection status'))
    }
    return response.json()
}

export async function updateDetectionConfig(projectId: number, detectorType: string): Promise<void> {
    const response = await fetch(`${API_BASE}/projects/${projectId}/detection/config`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ detector_type: detectorType })
    })
    if (!response.ok) throw new Error(await getErrorMessage(response, 'Failed to update detector config'))
}

export async function createDetectionTraining(
    projectId: number,
    maxEpochs: number,
    videoIds: number[],
    lrPatience: number = 10,
    earlyStopPatience: number = 20,
    fromCheckpoint: boolean = false,
): Promise<void> {
    const response = await fetch(`${API_BASE}/projects/${projectId}/detection/training`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            video_ids: videoIds,
            max_epochs: maxEpochs,
            lr_patience: lrPatience,
            early_stop_patience: earlyStopPatience,
            from_checkpoint: fromCheckpoint,
        }),
    })
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to start detection training'))
    }
}

export async function applyDetector(projectId: number, videoIds: number[]): Promise<{ videos_processed: number }> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/detection`,
        {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ video_ids: videoIds }),
        }
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to apply detector'))
    }
    return response.json()
}

export async function deleteDetectionTraining(projectId: number): Promise<void> {
    const response = await fetch(`${API_BASE}/projects/${projectId}/detection/training`, {
        method: 'DELETE',
    })
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to stop detection training'))
    }
}

// ----- Seg Detector -----

export interface SegDetectorStatus {
    model_exists: boolean
    is_training: boolean
}

export async function getSegDetectionStatus(projectId: number): Promise<SegDetectorStatus> {
    const response = await fetch(`${API_BASE}/projects/${projectId}/detection/seg/status`)
    if (!response.ok) throw new Error(await getErrorMessage(response, 'Failed to get seg status'))
    return response.json()
}

export async function createSegTraining(
    projectId: number,
    maxEpochs: number,
    videoIds: number[],
    earlyStopPatience: number = 20,
): Promise<void> {
    const response = await fetch(`${API_BASE}/projects/${projectId}/detection/seg/training`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            video_ids: videoIds,
            max_epochs: maxEpochs,
            early_stop_patience: earlyStopPatience,
        }),
    })
    if (!response.ok) throw new Error(await getErrorMessage(response, 'Failed to start seg training'))
}

export async function deleteSegTraining(projectId: number): Promise<void> {
    const response = await fetch(`${API_BASE}/projects/${projectId}/detection/seg/training`, {
        method: 'DELETE',
    })
    if (!response.ok) throw new Error(await getErrorMessage(response, 'Failed to stop seg training'))
}

export async function applySegDetector(projectId: number, videoIds: number[]): Promise<{ videos_processed: number }> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/seg-detection`,
        {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ video_ids: videoIds }),
        }
    )
    if (!response.ok) throw new Error(await getErrorMessage(response, 'Failed to apply seg detector'))
    return response.json()
}

export async function exportDetectorBboxes(
    projectId: number,
    videoIds: number[],
): Promise<{ path: string; row_count: number }> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/exports/detector-bboxes`,
        {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ video_ids: videoIds }),
        },
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to export detector bboxes'))
    }
    return response.json()
}

export async function exportDetectorMasks(
    projectId: number,
    videoIds: number[],
): Promise<{ path: string; row_count: number }> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/exports/detector-masks`,
        {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ video_ids: videoIds }),
        },
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to export seg detector masks'))
    }
    return response.json()
}

export async function segDetectorMasksExist(projectId: number, videoId: number): Promise<{ exists: boolean }> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/${videoId}/seg-detector-masks/exists`
    )
    if (!response.ok) throw new Error(await getErrorMessage(response, 'Failed to check seg masks'))
    return response.json()
}

// ----- OBB Detector -----

export interface ObbBbox {
    corners: [number, number][]
}

export interface ObbDetectorStatus {
    model_exists: boolean
    is_training: boolean
}

export async function getObbDetectionStatus(projectId: number): Promise<ObbDetectorStatus> {
    const response = await fetch(`${API_BASE}/projects/${projectId}/detection/obb/status`)
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to get OBB status'))
    }
    return response.json()
}

export async function createObbTraining(
    projectId: number,
    maxEpochs: number,
    videoIds: number[],
    lrPatience: number = 10,
    earlyStopPatience: number = 20,
): Promise<void> {
    const response = await fetch(`${API_BASE}/projects/${projectId}/detection/obb/training`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            video_ids: videoIds,
            max_epochs: maxEpochs,
            lr_patience: lrPatience,
            early_stop_patience: earlyStopPatience,
        }),
    })
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to start OBB training'))
    }
}

export async function deleteObbTraining(projectId: number): Promise<void> {
    const response = await fetch(`${API_BASE}/projects/${projectId}/detection/obb/training`, {
        method: 'DELETE',
    })
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to stop OBB training'))
    }
}

export async function applyObbDetector(projectId: number, videoIds: number[]): Promise<{ videos_processed: number }> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/obb-detection`,
        {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ video_ids: videoIds }),
        }
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to apply OBB detector'))
    }
    return response.json()
}

export async function getObbBbox(projectId: number, videoId: number, frameIdx: number): Promise<{ bbox: ObbBbox | null }> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/${videoId}/obb-bboxes/${frameIdx}`
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to get OBB bbox'))
    }
    return response.json()
}

export async function obbBboxesExist(projectId: number, videoId: number): Promise<{ exists: boolean }> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/${videoId}/obb-bboxes/exists`
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to check OBB bboxes'))
    }
    return response.json()
}

export async function getObbScoresDownsampled(
    projectId: number,
    videoId: number,
    maxSamples: number = 800,
    startFrame: number = 0,
    endFrame?: number,
): Promise<{ scores: { frame_idx: number; score: number }[]; total_count: number }> {
    let url = `${API_BASE}/projects/${projectId}/videos/${videoId}/segmentation/obb-scores-downsampled?max_samples=${maxSamples}&start_frame=${startFrame}`
    if (endFrame !== undefined) url += `&end_frame=${endFrame}`
    const response = await fetch(url)
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to get OBB scores'))
    }
    return response.json()
}

export async function getDetectionTraining(projectId: number): Promise<DetectorTrainingProgress> {
    const response = await fetch(`${API_BASE}/projects/${projectId}/detection/training`)
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to get detection training status'))
    }
    return response.json()
}

export function getDetectionTrainingStreamUrl(projectId: number): string {
    return `${API_BASE}/projects/${projectId}/detection/training/stream`
}

export function connectDetectorTrainingStream(projectId: number): EventSource {
    return new EventSource(getDetectionTrainingStreamUrl(projectId))
}

export async function getDetectorMask(
    projectId: number,
    videoId: number,
    frameIdx: number,
): Promise<Blob> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/${videoId}/detector-masks/${frameIdx}`
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to fetch detector mask'))
    }
    return response.blob()
}

export async function getDetectorMasks(
    projectId: number,
    videoId: number,
    startFrame: number,
    count: number = 100
): Promise<MaskBatchResponse> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/${videoId}/detector-masks?start_frame=${startFrame}&count=${count}`
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to fetch detector masks'))
    }
    return response.json()
}

export interface DetectorBbox {
  x1: number
  y1: number
  x2: number
  y2: number
}

export interface DetectorBboxResponse {
  bbox: DetectorBbox | null
}

export interface DetectorBboxesBatchResponse {
  bboxes: Array<{
    frame_idx: number
    x1: number
    y1: number
    x2: number
    y2: number
  }>
}

export async function getDetectorBbox(
  projectId: number,
  videoId: number,
  frameIdx: number,
): Promise<DetectorBboxResponse> {
  const response = await fetch(
    `${API_BASE}/projects/${projectId}/videos/${videoId}/detector-bboxes/${frameIdx}`,
  )
  if (!response.ok) throw new Error(await getErrorMessage(response, 'Failed to get detector bbox'))
  return response.json()
}

export async function getDetectorBboxes(
  projectId: number,
  videoId: number,
  startFrame: number,
  count: number = 100,
): Promise<DetectorBboxesBatchResponse> {
  const response = await fetch(
    `${API_BASE}/projects/${projectId}/videos/${videoId}/detector-bboxes?start_frame=${startFrame}&count=${count}`,
  )
  if (!response.ok) throw new Error(await getErrorMessage(response, 'Failed to get detector bboxes'))
  return response.json()
}

export async function detectorMasksExist(
    projectId: number,
    videoId: number,
): Promise<{ exists: boolean }> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/${videoId}/detector-masks/exists`
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to check detector masks'))
    }
    return response.json()
}

// ----- Final masks (tracker-detector fusion) -----

export async function getFinalMask(
    projectId: number,
    videoId: number,
    frameIdx: number,
): Promise<Blob> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/${videoId}/segmentation/final-mask/${frameIdx}`
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to fetch final mask'))
    }
    return response.blob()
}

export async function getFinalMasks(
    projectId: number,
    videoId: number,
    startFrame: number,
    count: number = 100
): Promise<MaskBatchResponse> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/${videoId}/segmentation/final-masks?start_frame=${startFrame}&count=${count}`
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to fetch final masks'))
    }
    return response.json()
}

export async function finalMasksExist(
    projectId: number,
    videoId: number,
): Promise<{ exists: boolean }> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/${videoId}/segmentation/final-masks/exists`
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to check final masks'))
    }
    return response.json()
}

// --- Pose API ---

export interface PoseLabel {
    frame_idx: number
    front_x: number
    front_y: number
    rear_x: number
    rear_y: number
}

export interface PosePrediction {
    front_x: number
    front_y: number
    rear_x: number
    rear_y: number
    score: number
}

export interface PoseStatus {
    model_exists: boolean
    is_training: boolean
}

export async function getPoseLabels(projectId: number, videoId: number): Promise<PoseLabel[]> {
    const response = await fetch(`${API_BASE}/projects/${projectId}/videos/${videoId}/pose/labels`)
    if (!response.ok) throw new Error(await getErrorMessage(response, 'Failed to fetch pose labels'))
    return response.json()
}

export async function savePoseLabel(
    projectId: number,
    videoId: number,
    frameIdx: number,
    frontX: number,
    frontY: number,
    rearX: number,
    rearY: number,
): Promise<void> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/${videoId}/pose/labels/${frameIdx}`,
        {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ front_x: frontX, front_y: frontY, rear_x: rearX, rear_y: rearY }),
        },
    )
    if (!response.ok) throw new Error(await getErrorMessage(response, 'Failed to save pose label'))
}

export async function deletePoseLabel(projectId: number, videoId: number, frameIdx: number): Promise<void> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/${videoId}/pose/labels/${frameIdx}`,
        { method: 'DELETE' },
    )
    if (!response.ok) throw new Error(await getErrorMessage(response, 'Failed to delete pose label'))
}

export async function getPoseLabelCount(projectId: number, videoId: number): Promise<number> {
    const response = await fetch(`${API_BASE}/projects/${projectId}/videos/${videoId}/pose/label-count`)
    if (!response.ok) throw new Error(await getErrorMessage(response, 'Failed to fetch pose label count'))
    const data = await response.json()
    return data.count
}

export async function getPosePrediction(
    projectId: number,
    videoId: number,
    frameIdx: number,
): Promise<PosePrediction | null> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/${videoId}/pose/prediction/${frameIdx}`,
    )
    if (!response.ok) return null
    const data = await response.json()
    return data.prediction
}

export async function getPoseStatus(projectId: number): Promise<PoseStatus> {
    const response = await fetch(`${API_BASE}/projects/${projectId}/pose/status`)
    if (!response.ok) throw new Error(await getErrorMessage(response, 'Failed to fetch pose status'))
    return response.json()
}

export async function createPoseTraining(
    projectId: number,
    maxEpochs: number,
    videoIds: number[],
): Promise<void> {
    const response = await fetch(`${API_BASE}/projects/${projectId}/pose/training`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ video_ids: videoIds, max_epochs: maxEpochs }),
    })
    if (!response.ok) throw new Error(await getErrorMessage(response, 'Failed to start pose training'))
}

export async function deletePoseTraining(projectId: number): Promise<void> {
    const response = await fetch(`${API_BASE}/projects/${projectId}/pose/training`, { method: 'DELETE' })
    if (!response.ok) throw new Error(await getErrorMessage(response, 'Failed to stop pose training'))
}

export function connectPoseTrainingStream(projectId: number): EventSource {
    return new EventSource(`${API_BASE}/projects/${projectId}/pose/training/stream`)
}

export async function applyPose(projectId: number, videoIds: number[]): Promise<void> {
    const response = await fetch(`${API_BASE}/projects/${projectId}/videos/pose`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ video_ids: videoIds }),
    })
    if (!response.ok) throw new Error(await getErrorMessage(response, 'Failed to apply pose model'))
}

export async function getPoseScoresDownsampled(
    projectId: number,
    videoId: number,
    maxSamples: number = 800,
    startFrame?: number,
    endFrame?: number,
): Promise<{ scores: MaskScore[] }> {
    const params = new URLSearchParams({ max_samples: maxSamples.toString() })
    if (startFrame !== undefined) params.set('start_frame', startFrame.toString())
    if (endFrame !== undefined) params.set('end_frame', endFrame.toString())
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/${videoId}/pose-scores-downsampled?${params}`,
    )
    if (!response.ok) throw new Error(await getErrorMessage(response, 'Failed to fetch pose scores'))
    return response.json()
}
