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

export interface Bbox {
    x1: number
    y1: number
    x2: number
    y2: number
}

export async function getBbox(
    projectId: number,
    videoId: number,
    frameIdx: number
): Promise<Bbox | null> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/${videoId}/bbox/${frameIdx}`
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to fetch bbox'))
    }
    const data = await response.json()
    return data === null ? null : data as Bbox
}

export interface BboxBatchItem {
    frame_idx: number
    bbox: [number, number, number, number] | null
}

export interface BboxBatchResponse {
    bboxes: BboxBatchItem[]
}

export async function getBboxesBatch(
    projectId: number,
    videoId: number,
    startFrame: number,
    count: number = 100
): Promise<BboxBatchResponse> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/${videoId}/bboxes-batch?start_frame=${startFrame}&count=${count}`
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to fetch bboxes batch'))
    }
    const data = await response.json()
    return data
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

export interface StoredPrompt {
    type: 'positive_point' | 'negative_point'
    details: { x: number; y: number }
    createdAt: string
}

export async function getPromptsForFrame(
    projectId: number,
    videoId: number,
    frameIdx: number
): Promise<StoredPrompt[]> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/${videoId}/prompts/${frameIdx}`
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to fetch prompts'))
    }
    const prompts = await response.json()
    // Map backend format to frontend StoredPrompt format
    return prompts.map((p: { type: string; x: number; y: number }) => ({
        type: p.type as 'positive_point' | 'negative_point',
        details: { x: p.x, y: p.y },
        createdAt: new Date().toISOString(), // Backend doesn't store timestamps
    }))
}

interface BackendPrompt {
    type: string
    x: number
    y: number
}

type PromptsDict = Record<string, BackendPrompt[]>

export async function getAllPrompts(
    projectId: number,
    videoId: number
): Promise<Map<number, StoredPrompt[]>> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/${videoId}/prompts`
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to fetch prompts'))
    }
    const promptsDict: PromptsDict = await response.json()
    // Convert dict to Map and map backend format to frontend StoredPrompt format
    const result = new Map<number, StoredPrompt[]>()
    for (const [frameIdxStr, prompts] of Object.entries(promptsDict)) {
        const frameIdx = parseInt(frameIdxStr, 10)
        result.set(
            frameIdx,
            prompts.map((p) => ({
                type: p.type as 'positive_point' | 'negative_point',
                details: { x: p.x, y: p.y },
                createdAt: new Date().toISOString(),
            }))
        )
    }
    return result
}

export interface YOLOModelStatus {
    exists: boolean
    model_path: string | null
    is_training: boolean
    is_applying: boolean
}

export async function trainInitialDetectionModel(projectId: number): Promise<void> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/train-model`,
        { method: 'POST' }
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to train initial detection model'))
    }
}

export async function runInitialDetection(
    projectId: number
): Promise<void> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/run-initial-detection`,
        { method: 'POST' }
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to run initial detection'))
    }
}

export async function segmentAllVideos(projectId: number): Promise<{ job_ids: number[] }> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/segment-all-videos`,
        { method: 'POST' }
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to segment all videos'))
    }
    return response.json()
}

export async function getYOLOModelStatus(
    projectId: number
): Promise<YOLOModelStatus> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/model-status`
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to get YOLO model status'))
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
