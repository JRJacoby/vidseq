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
    segmentation_status: 'in_progress' | 'segmented' | null
    min_confidence?: number
    p50_confidence?: number
    p95_confidence?: number
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

export interface PointPrompt {
    x: number
    y: number
    type: 'positive_point' | 'negative_point'
}

export async function refineMaskMultiPoint(
    projectId: number,
    videoId: number,
    frameIdx: number,
    points: PointPrompt[]
): Promise<Blob> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/${videoId}/refine-mask`,
        {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ frame_idx: frameIdx, points }),
        }
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to refine mask'))
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

export interface MaskScore {
    frame_idx: number
    score: number // IoU score, -1.0 if not available
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

export async function getScoresBatch(
    projectId: number,
    videoId: number,
    startFrame: number,
    count: number = 1000
): Promise<MaskScore[]> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/${videoId}/scores-batch?start_frame=${startFrame}&count=${count}`
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to fetch scores batch'))
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
        `${API_BASE}/projects/${projectId}/videos/${videoId}/scores-downsampled?${params}`
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to fetch downsampled scores'))
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
        `${API_BASE}/projects/${projectId}/videos/${videoId}/frame-data/${frameIdx}`,
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
        `${API_BASE}/projects/${projectId}/videos/${videoId}/frame-data`,
        { method: 'DELETE' }
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to reset video'))
    }
}

export interface GenerateTrainingMasksResponse {
    frames_processed: number
}

export async function generateTrainingMasks(
    projectId: number,
    videoId: number,
    startFrameIdx: number,
    maxFrames: number = 1000
): Promise<GenerateTrainingMasksResponse> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/${videoId}/generate-training-masks`,
        {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ start_frame_idx: startFrameIdx, max_frames: maxFrames }),
        }
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to generate training masks'))
    }
    return response.json()
}

export async function segmentAllVideos(projectId: number): Promise<{ job_ids: number[] }> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/segmentation`,
        { method: 'POST' }
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to segment all videos'))
    }
    return response.json()
}

export interface PropagateResponse {
    frames_processed: number
}

export async function propagateMask(
    projectId: number,
    videoId: number,
    startFrameIdx: number,
    maxFrames: number = 1000
): Promise<PropagateResponse> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/${videoId}/propagate-mask`,
        {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ start_frame_idx: startFrameIdx, max_frames: maxFrames }),
        }
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to propagate mask'))
    }
    return response.json()
}

export interface FrameRangesResponse {
    masked_ranges: [number, number][]
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

export async function validateTrainingRange(
    projectId: number,
    videoId: number,
    startFrame: number,
    endFrame: number
): Promise<ValidateTrainingRangeResponse> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/${videoId}/validate-training-range?start_frame=${startFrame}&end_frame=${endFrame}`,
        { method: 'POST' }
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to validate training range'))
    }
    return response.json()
}

export async function markTrainingRange(
    projectId: number,
    videoId: number,
    startFrame: number,
    endFrame: number
): Promise<void> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/${videoId}/mark-training?start_frame=${startFrame}&end_frame=${endFrame}`,
        { method: 'POST' }
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to mark training range'))
    }
}

export async function unmarkTrainingRange(
    projectId: number,
    videoId: number,
    startFrame: number,
    endFrame: number
): Promise<void> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/${videoId}/mark-training?start_frame=${startFrame}&end_frame=${endFrame}`,
        { method: 'DELETE' }
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to unmark training range'))
    }
}

// Cropped Video API

export async function extractCroppedVideos(projectId: number): Promise<{ job_ids: number[] }> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/extract-cropped-videos`,
        { method: 'POST' }
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to extract cropped videos'))
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

// Alignment API

export interface AlignmentLabel {
    id: number
    video_id: number
    frame_idx: number
    front_x: number
    front_y: number
    rear_x: number
    rear_y: number
}

export interface AlignmentStatus {
    label_count: number
    model_trained: boolean
    is_training: boolean
    is_applying: boolean
    all_videos_cropped: boolean
}

export interface RandomFrame {
    video_id: number
    frame_idx: number
}

export async function getAlignmentStatus(projectId: number): Promise<AlignmentStatus> {
    const response = await fetch(`${API_BASE}/projects/${projectId}/alignment/status`)
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to fetch alignment status'))
    }
    return response.json()
}

export async function getRandomAlignmentFrame(projectId: number): Promise<RandomFrame> {
    const response = await fetch(`${API_BASE}/projects/${projectId}/alignment/random-frame`)
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to fetch random frame'))
    }
    return response.json()
}

export async function saveAlignmentLabel(
    projectId: number,
    videoId: number,
    frameIdx: number,
    frontX: number,
    frontY: number,
    rearX: number,
    rearY: number
): Promise<AlignmentLabel> {
    const response = await fetch(`${API_BASE}/projects/${projectId}/alignment/labels`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            video_id: videoId,
            frame_idx: frameIdx,
            front_x: frontX,
            front_y: frontY,
            rear_x: rearX,
            rear_y: rearY,
        }),
    })
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to save alignment label'))
    }
    return response.json()
}

export async function getAllAlignmentLabels(projectId: number): Promise<AlignmentLabel[]> {
    const response = await fetch(`${API_BASE}/projects/${projectId}/alignment/labels`)
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to fetch alignment labels'))
    }
    return response.json()
}

export async function clearAllAlignmentLabels(projectId: number): Promise<{ deleted_count: number }> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/alignment/labels`,
        { method: 'DELETE' }
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to clear alignment labels'))
    }
    return response.json()
}

export async function clearAlignmentModel(projectId: number): Promise<{ deleted: boolean }> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/alignment/model`,
        { method: 'DELETE' }
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to clear alignment model'))
    }
    return response.json()
}

export async function trainAlignmentModel(
    projectId: number,
    epochs: number = 100,
    augment: boolean = true,
    earlyStopPatience: number = 5,
    lrPatience: number = 3,
): Promise<void> {
    const params = new URLSearchParams({
        epochs: epochs.toString(),
        augment: augment.toString(),
        early_stop_patience: earlyStopPatience.toString(),
        lr_patience: lrPatience.toString(),
    })
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/alignment/train?${params}`,
        { method: 'POST' }
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to train alignment model'))
    }
}

export function getAlignmentPredictionUrl(
    projectId: number,
    videoId: number,
    frameIdx: number
): string {
    return `${API_BASE}/projects/${projectId}/alignment/predict/${videoId}/${frameIdx}`
}

export async function applyAlignment(projectId: number): Promise<void> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/alignment/apply`,
        { method: 'POST' }
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to apply alignment'))
    }
}

// --- Training Progress (for real-time monitoring) ---

export interface TrainingProgress {
    is_training: boolean
    current_epoch: number
    max_epochs: number

    // Training loss
    current_train_loss: number
    train_loss_history: number[]

    // Validation loss
    current_val_loss: number
    val_loss_history: number[]

    // Best model tracking (based on validation loss)
    best_val_loss: number | null
    best_epoch: number

    // Learning rate
    current_lr: number

    // Patience counters
    epochs_without_improvement: number
    lr_patience: number
    early_stop_patience: number
    lr_reduced_this_plateau: boolean

    // Status
    status: 'idle' | 'training' | 'completed' | 'stopped' | 'failed'
    started_at: number | null

    // Dataset info
    num_train_labels: number
    num_val_labels: number

    // Backward-compatible aliases (from backend to_dict)
    current_loss: number       // = current_train_loss
    best_loss: number | null   // = best_val_loss
    loss_history: number[]     // = train_loss_history
}

export async function getTrainingStatus(projectId: number): Promise<TrainingProgress> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/alignment/training/status`
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to get training status'))
    }
    return response.json()
}

export function getAlignmentTrainingStreamUrl(projectId: number): string {
    return `${API_BASE}/projects/${projectId}/alignment/training/stream`
}

// --- Alignment Apply Progress (for real-time monitoring) ---

export interface AlignmentApplyProgress {
    is_aligning: boolean
    status: string  // idle, aligning, completed, failed

    // Video-level progress
    current_video_index: number
    total_videos: number
    current_video_name: string

    // Frame-level progress
    current_frame: number
    total_frames: number

    // Performance metrics
    fps: number
    eta_seconds: number

    // Timestamps
    started_at: number | null
}

export async function getAlignmentApplyStatus(projectId: number): Promise<AlignmentApplyProgress> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/alignment/apply/status`
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to get alignment apply status'))
    }
    return response.json()
}

export function getAlignmentApplyStreamUrl(projectId: number): string {
    return `${API_BASE}/projects/${projectId}/alignment/apply/stream`
}

// --- Stored Alignment Predictions (for debugging) ---

export interface AlignmentPredictionsExistsResponse {
    exists: boolean
}

export async function hasAlignmentPredictions(
    projectId: number,
    videoId: number
): Promise<boolean> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/${videoId}/alignment-predictions/exists`
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to check alignment predictions'))
    }
    const data: AlignmentPredictionsExistsResponse = await response.json()
    return data.exists
}

export function getStoredAlignmentPredictionUrl(
    projectId: number,
    videoId: number,
    frameIdx: number
): string {
    return `${API_BASE}/projects/${projectId}/videos/${videoId}/alignment-prediction/${frameIdx}`
}

// --- Video-specific Alignment Labels ---

export interface VideoAlignmentLabelsResponse {
    frame_indices: number[]
}

export async function getVideoAlignmentLabels(
    projectId: number,
    videoId: number
): Promise<VideoAlignmentLabelsResponse> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/${videoId}/alignment-labels`
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to fetch video alignment labels'))
    }
    return response.json()
}

export async function deleteAlignmentLabel(
    projectId: number,
    videoId: number,
    frameIdx: number
): Promise<{ deleted: boolean }> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/${videoId}/alignment-labels/${frameIdx}`,
        { method: 'DELETE' }
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to delete alignment label'))
    }
    return response.json()
}

export async function deleteVideoAlignmentLabels(
    projectId: number,
    videoId: number
): Promise<{ deleted_count: number }> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/${videoId}/alignment-labels`,
        { method: 'DELETE' }
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to delete video alignment labels'))
    }
    return response.json()
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

export async function runPCA(
    projectId: number,
    nComponents: number = 20
): Promise<PCARunResult> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/pca/run?n_components=${nComponents}`,
        { method: 'POST' }
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
        `${API_BASE}/projects/${projectId}/videos/${videoId}/pca-scores-exists`
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

export async function startARHMM(projectId: number): Promise<{ status: string }> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/arhmm/run`,
        { method: 'POST' }
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to start ARHMM'))
    }
    return response.json()
}

export async function stopARHMM(projectId: number): Promise<{ status: string }> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/arhmm/stop`,
        { method: 'POST' }
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to stop ARHMM'))
    }
    return response.json()
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

export async function generateCrowdMovies(projectId: number): Promise<{ status: string }> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/arhmm/crowd-movies/generate`,
        { method: 'POST' }
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to start crowd movie generation'))
    }
    return response.json()
}

export async function stopCrowdMovies(projectId: number): Promise<{ status: string }> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/arhmm/crowd-movies/stop`,
        { method: 'POST' }
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to stop crowd movie generation'))
    }
    return response.json()
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

export function getCrowdMovieVideoUrl(projectId: number, syllable: number): string {
    return `${API_BASE}/projects/${projectId}/arhmm/crowd-movies/${syllable}/video`
}

// --- Detector API ---

export interface DetectorStatus {
    model_exists: boolean
    is_training: boolean
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

export async function getDetectorStatus(projectId: number): Promise<DetectorStatus> {
    const response = await fetch(`${API_BASE}/projects/${projectId}/detector/status`)
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to get detector status'))
    }
    return response.json()
}

export async function trainDetector(
    projectId: number,
    maxEpochs: number = 1000,
    lrPatience: number = 10,
    earlyStopPatience: number = 20,
): Promise<void> {
    const response = await fetch(`${API_BASE}/projects/${projectId}/detector/train`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            max_epochs: maxEpochs,
            lr_patience: lrPatience,
            early_stop_patience: earlyStopPatience,
        }),
    })
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to start detector training'))
    }
}

export async function stopDetectorTraining(projectId: number): Promise<void> {
    const response = await fetch(`${API_BASE}/projects/${projectId}/detector/stop`, {
        method: 'POST',
    })
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to stop detector training'))
    }
}

export async function getDetectorTrainingStatus(projectId: number): Promise<DetectorTrainingProgress> {
    const response = await fetch(`${API_BASE}/projects/${projectId}/detector/training/status`)
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to get detector training status'))
    }
    return response.json()
}

export function getDetectorTrainingStreamUrl(projectId: number): string {
    return `${API_BASE}/projects/${projectId}/detector/training/stream`
}

export async function getDetectorMask(
    projectId: number,
    videoId: number,
    frameIdx: number,
): Promise<Blob> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/${videoId}/detector-mask/${frameIdx}`
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to fetch detector mask'))
    }
    return response.blob()
}

export async function getDetectorMasksBatch(
    projectId: number,
    videoId: number,
    startFrame: number,
    count: number = 100
): Promise<MaskBatchResponse> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/${videoId}/detector-masks-batch?start_frame=${startFrame}&count=${count}`
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to fetch detector mask batch'))
    }
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
        `${API_BASE}/projects/${projectId}/videos/${videoId}/final-mask/${frameIdx}`
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to fetch final mask'))
    }
    return response.blob()
}

export async function getFinalMasksBatch(
    projectId: number,
    videoId: number,
    startFrame: number,
    count: number = 100
): Promise<MaskBatchResponse> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/${videoId}/final-masks-batch?start_frame=${startFrame}&count=${count}`
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to fetch final mask batch'))
    }
    return response.json()
}

export async function finalMasksExist(
    projectId: number,
    videoId: number,
): Promise<{ exists: boolean }> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/${videoId}/final-masks/exists`
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to check final masks'))
    }
    return response.json()
}
