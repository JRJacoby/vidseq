<script setup lang="ts">
import { ref, onMounted, onUnmounted, computed, watch } from 'vue'
import { useProjectStore } from '@/stores/project'
import {
    getAlignmentTrainingStreamUrl,
    getAlignmentTraining,
    getVideosAlignmentStatus,
    getVideosAlignmentStreamUrl,
    type TrainingProgress,
    type AlignmentApplyProgress
} from '@/services/api'
import { Chart, registerables } from 'chart.js'

Chart.register(...registerables)

const projectStore = useProjectStore()
const projectId = computed(() => projectStore.currentProjectId)

// Training progress
const progress = ref<TrainingProgress | null>(null)
const chartCanvas = ref<HTMLCanvasElement | null>(null)
let chart: Chart | null = null
let eventSource: EventSource | null = null

// Alignment apply progress
const alignProgress = ref<AlignmentApplyProgress | null>(null)
let alignEventSource: EventSource | null = null

// Initialize chart with two datasets (train and val)
function initChart() {
    if (!chartCanvas.value) return

    chart = new Chart(chartCanvas.value, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Training Loss',
                    data: [],
                    borderColor: 'rgb(59, 130, 246)',  // Blue
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    fill: false,
                    tension: 0.1,
                },
                {
                    label: 'Validation Loss',
                    data: [],
                    borderColor: 'rgb(239, 68, 68)',  // Red
                    backgroundColor: 'rgba(239, 68, 68, 0.1)',
                    fill: false,
                    tension: 0.1,
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                }
            },
            scales: {
                x: { title: { display: true, text: 'Epoch' } },
                y: {
                    type: 'logarithmic',
                    title: { display: true, text: 'Loss (log scale)' },
                }
            },
            animation: { duration: 0 }  // Disable animation for real-time updates
        }
    })
}

// Update chart with train and val loss histories
function updateChart(trainHistory: number[], valHistory: number[]) {
    if (!chart || !chart.data.datasets[0] || !chart.data.datasets[1]) return

    const maxLen = Math.max(trainHistory.length, valHistory.length)
    chart.data.labels = Array.from({ length: maxLen }, (_, i) => i + 1)
    chart.data.datasets[0].data = trainHistory
    chart.data.datasets[1].data = valHistory
    chart.update('none')  // No animation
}

// Connect to SSE stream
function connectToStream() {
    if (!projectId.value) return

    // Close existing connection if any
    if (eventSource) {
        eventSource.close()
        eventSource = null
    }

    const url = getAlignmentTrainingStreamUrl(projectId.value)
    eventSource = new EventSource(url)

    eventSource.onmessage = (event) => {
        const data = JSON.parse(event.data) as TrainingProgress
        progress.value = data
        updateChart(data.train_loss_history, data.val_loss_history)

        // Close connection if training finished
        if (['completed', 'stopped', 'failed'].includes(data.status)) {
            eventSource?.close()
            eventSource = null
        }
    }

    eventSource.onerror = () => {
        eventSource?.close()
        eventSource = null
    }
}

// Load initial training status
async function loadTrainingStatus() {
    if (!projectId.value) return

    try {
        progress.value = await getAlignmentTraining(projectId.value)
        if (progress.value.train_loss_history.length > 0 || progress.value.val_loss_history.length > 0) {
            updateChart(progress.value.train_loss_history, progress.value.val_loss_history)
        }

        // If training in progress, connect to stream
        if (progress.value.is_training) {
            connectToStream()
        }
    } catch (e) {
        console.error('Failed to load training status:', e)
    }
}

// Connect to alignment apply SSE stream
function connectToAlignStream() {
    if (!projectId.value) return

    // Close existing connection if any
    if (alignEventSource) {
        alignEventSource.close()
        alignEventSource = null
    }

    const url = getVideosAlignmentStreamUrl(projectId.value)
    alignEventSource = new EventSource(url)

    alignEventSource.onmessage = (event) => {
        const data = JSON.parse(event.data) as AlignmentApplyProgress
        alignProgress.value = data

        // Close connection if alignment finished
        if (['completed', 'failed'].includes(data.status)) {
            alignEventSource?.close()
            alignEventSource = null
        }
    }

    alignEventSource.onerror = () => {
        alignEventSource?.close()
        alignEventSource = null
    }
}

// Load initial alignment apply status
async function loadAlignmentStatus() {
    if (!projectId.value) return

    try {
        alignProgress.value = await getVideosAlignmentStatus(projectId.value)

        // If alignment in progress, connect to stream
        if (alignProgress.value.is_aligning) {
            connectToAlignStream()
        }
    } catch (e) {
        console.error('Failed to load alignment status:', e)
    }
}

onMounted(() => {
    initChart()
    loadTrainingStatus()
    loadAlignmentStatus()
})

onUnmounted(() => {
    eventSource?.close()
    alignEventSource?.close()
    chart?.destroy()
})

// Reconnect if project changes
watch(() => projectId.value, () => {
    eventSource?.close()
    eventSource = null
    alignEventSource?.close()
    alignEventSource = null
    loadTrainingStatus()
    loadAlignmentStatus()
})

// Reconnect if training starts
watch(() => progress.value?.is_training, (isTraining) => {
    if (isTraining && !eventSource) {
        connectToStream()
    }
})

// Reconnect if alignment starts
watch(() => alignProgress.value?.is_aligning, (isAligning) => {
    if (isAligning && !alignEventSource) {
        connectToAlignStream()
    }
})

// Format helpers
const formatLR = (lr: number) => lr.toExponential(2)
const formatLoss = (loss: number | null | undefined) => {
    if (loss === Infinity || loss === null || loss === undefined) return '--'
    return loss.toFixed(6)
}
const formatDuration = (startedAt: number | null) => {
    if (!startedAt) return '--'
    const seconds = Math.floor((Date.now() / 1000) - startedAt)
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins}:${secs.toString().padStart(2, '0')}`
}
const formatEta = (seconds: number) => {
    if (seconds <= 0) return '--'
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    if (mins > 0) {
        return `~${mins}m ${secs}s`
    }
    return `~${secs}s`
}
const formatFps = (fps: number) => {
    if (fps <= 0) return '--'
    return fps.toFixed(1)
}

const statusColor = computed(() => {
    switch (progress.value?.status) {
        case 'training': return '#3b82f6'  // Blue
        case 'completed': return '#22c55e'  // Green
        case 'stopped': return '#f59e0b'    // Amber
        case 'failed': return '#ef4444'     // Red
        default: return '#6b7280'           // Gray
    }
})

// Calculate patience bar widths
const earlyStopProgress = computed(() => {
    if (!progress.value) return 0
    return (progress.value.epochs_without_improvement / progress.value.early_stop_patience) * 100
})

const lrReductionProgress = computed(() => {
    if (!progress.value) return 0
    const effectiveEpochs = Math.min(
        progress.value.epochs_without_improvement,
        progress.value.lr_patience
    )
    return (effectiveEpochs / progress.value.lr_patience) * 100
})

// Alignment apply status color
const alignStatusColor = computed(() => {
    switch (alignProgress.value?.status) {
        case 'aligning': return '#3b82f6'  // Blue
        case 'completed': return '#22c55e'  // Green
        case 'failed': return '#ef4444'     // Red
        default: return '#6b7280'           // Gray
    }
})

// Frame progress percentage for alignment
const frameProgressPercent = computed(() => {
    if (!alignProgress.value || alignProgress.value.total_frames === 0) return 0
    return (alignProgress.value.current_frame / alignProgress.value.total_frames) * 100
})
</script>

<template>
    <div class="alignment-page">
        <h1>Alignment</h1>

        <!-- Training Section -->
        <section class="section">
            <h2 class="section-title">Training</h2>
            <div class="training-grid">
                <!-- Status Panel -->
                <div class="status-panel">
                    <h3>Status</h3>
                <div class="status-indicator" :style="{ backgroundColor: statusColor }">
                    {{ progress?.status ?? 'idle' }}
                </div>

                <div class="stats-grid" v-if="progress">
                    <div class="stat">
                        <span class="stat-label">Epoch</span>
                        <span class="stat-value">{{ progress.current_epoch }} / {{ progress.max_epochs }}</span>
                    </div>
                    <div class="stat">
                        <span class="stat-label">Dataset Split</span>
                        <span class="stat-value">{{ progress.num_train_labels }} train / {{ progress.num_val_labels }} val</span>
                    </div>
                    <div class="stat">
                        <span class="stat-label">Train Loss</span>
                        <span class="stat-value loss-train">{{ formatLoss(progress.current_train_loss) }}</span>
                    </div>
                    <div class="stat">
                        <span class="stat-label">Val Loss</span>
                        <span class="stat-value loss-val">{{ formatLoss(progress.current_val_loss) }}</span>
                    </div>
                    <div class="stat">
                        <span class="stat-label">Best Val Loss</span>
                        <span class="stat-value">{{ formatLoss(progress.best_val_loss) }} (epoch {{ progress.best_epoch
                            }})</span>
                    </div>
                    <div class="stat">
                        <span class="stat-label">Learning Rate</span>
                        <span class="stat-value">{{ formatLR(progress.current_lr) }}</span>
                    </div>
                </div>

                <!-- Patience Counters -->
                <div class="patience-section" v-if="progress">
                    <h3>Early Stopping</h3>
                    <div class="patience-bar">
                        <div class="patience-fill patience-early-stop" :style="{ width: `${earlyStopProgress}%` }" />
                        <span class="patience-text">
                            {{ progress.epochs_without_improvement }} / {{ progress.early_stop_patience }} epochs
                            without improvement
                        </span>
                    </div>

                    <h3>LR Reduction</h3>
                    <div class="patience-bar">
                        <div class="patience-fill patience-lr"
                            :class="{ 'patience-triggered': progress.lr_reduced_this_plateau }"
                            :style="{ width: `${lrReductionProgress}%` }" />
                        <span class="patience-text">
                            {{ progress.lr_reduced_this_plateau ? 'LR reduced this plateau' :
                                `${Math.min(progress.epochs_without_improvement, progress.lr_patience)} / ${progress.lr_patience} to reduction`
                            }}
                        </span>
                    </div>
                </div>

                <!-- No data message -->
                <div v-if="!progress || progress.status === 'idle'" class="no-data-message">
                    <p>No training data available.</p>
                    <p class="hint">Start training from the Video Pipeline page to see progress here.</p>
                </div>
            </div>

            <!-- Chart Panel -->
            <div class="chart-panel">
                <h3>Loss Over Time</h3>
                <div class="chart-container">
                    <canvas ref="chartCanvas"></canvas>
                </div>
            </div>
            </div>
        </section>

        <!-- Apply Alignment Section -->
        <section class="section">
            <h2 class="section-title">Apply Alignment</h2>
            <div class="apply-panel">
                <div class="status-row">
                    <span class="label">Status:</span>
                    <div class="status-indicator" :style="{ backgroundColor: alignStatusColor }">
                        {{ alignProgress?.status ?? 'idle' }}
                    </div>
                </div>

                <div class="apply-stats" v-if="alignProgress && alignProgress.status !== 'idle'">
                    <div class="stat-row">
                        <span class="label">Video:</span>
                        <span class="value">{{ alignProgress.current_video_index }} / {{ alignProgress.total_videos }}</span>
                    </div>
                    <div class="stat-row" v-if="alignProgress.current_video_name">
                        <span class="label">Current:</span>
                        <span class="value filename">{{ alignProgress.current_video_name }}</span>
                    </div>

                    <!-- Progress bar -->
                    <div class="progress-container">
                        <div class="progress-bar">
                            <div class="progress-fill" :style="{ width: `${frameProgressPercent}%` }"></div>
                        </div>
                        <span class="progress-text">{{ Math.round(frameProgressPercent) }}%</span>
                    </div>

                    <div class="stat-row">
                        <span class="label">Frames:</span>
                        <span class="value">{{ alignProgress.current_frame.toLocaleString() }} / {{ alignProgress.total_frames.toLocaleString() }}</span>
                    </div>
                    <div class="stat-row">
                        <span class="label">Speed:</span>
                        <span class="value">{{ formatFps(alignProgress.fps) }} fps</span>
                    </div>
                    <div class="stat-row">
                        <span class="label">ETA:</span>
                        <span class="value">{{ formatEta(alignProgress.eta_seconds) }}</span>
                    </div>
                </div>

                <div v-if="!alignProgress || alignProgress.status === 'idle'" class="no-data-message">
                    <p>No alignment in progress.</p>
                    <p class="hint">Start alignment from the Video Pipeline page.</p>
                </div>
            </div>
        </section>
    </div>
</template>

<style scoped>
.alignment-page {
    padding: 20px;
    max-width: 1200px;
}

h1 {
    margin-bottom: 20px;
}

.section {
    margin-bottom: 24px;
}

.section-title {
    margin-bottom: 12px;
    font-size: 1.2em;
    border-bottom: 1px solid #e0e0e0;
    padding-bottom: 8px;
}

h3 {
    margin: 12px 0 6px;
    font-size: 0.95em;
    color: #333;
}

.training-grid {
    display: grid;
    grid-template-columns: 320px 1fr;
    gap: 20px;
}

.status-panel,
.chart-panel {
    background: #f8f8f8;
    border-radius: 8px;
    padding: 16px;
}

.status-indicator {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 4px;
    color: white;
    font-weight: 500;
    text-transform: uppercase;
    font-size: 0.85em;
    margin-bottom: 16px;
}

.stats-grid {
    display: grid;
    gap: 12px;
}

.stat {
    display: flex;
    flex-direction: column;
}

.stat-label {
    font-size: 0.8em;
    color: #666;
}

.stat-value {
    font-size: 1.1em;
    font-weight: 500;
    font-family: monospace;
}

.stat-value.loss-train {
    color: rgb(59, 130, 246);  /* Blue - matches chart */
}

.stat-value.loss-val {
    color: rgb(239, 68, 68);  /* Red - matches chart */
}

.patience-section {
    margin-top: 20px;
    padding-top: 16px;
    border-top: 1px solid #ddd;
}

.patience-bar {
    position: relative;
    height: 24px;
    background: #e5e7eb;
    border-radius: 4px;
    overflow: hidden;
}

.patience-fill {
    height: 100%;
    transition: width 0.3s ease;
}

.patience-early-stop {
    background: #f59e0b;
}

.patience-lr {
    background: #3b82f6;
}

.patience-lr.patience-triggered {
    background: #22c55e;
}

.patience-text {
    position: absolute;
    left: 8px;
    top: 50%;
    transform: translateY(-50%);
    font-size: 0.75em;
    color: #333;
}

.chart-panel {
    min-height: 400px;
}

.chart-container {
    height: 350px;
}

.no-data-message {
    margin-top: 20px;
    padding: 16px;
    background: #fff;
    border-radius: 4px;
    text-align: center;
    color: #666;
}

.no-data-message .hint {
    font-size: 0.85em;
    color: #999;
    margin-top: 8px;
}

@media (max-width: 900px) {
    .training-grid {
        grid-template-columns: 1fr;
    }
}

/* Apply Alignment Section */
.apply-panel {
    background: #f8f8f8;
    border-radius: 8px;
    padding: 16px;
    max-width: 500px;
}

.status-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 16px;
}

.apply-stats {
    display: flex;
    flex-direction: column;
    gap: 10px;
}

.stat-row {
    display: flex;
    align-items: baseline;
    gap: 8px;
}

.stat-row .label {
    font-size: 0.85em;
    color: #666;
    min-width: 60px;
}

.stat-row .value {
    font-family: monospace;
    font-weight: 500;
}

.stat-row .value.filename {
    font-size: 0.9em;
    color: #555;
    word-break: break-all;
}

.progress-container {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 8px 0;
}

.progress-bar {
    flex: 1;
    height: 20px;
    background: #e5e7eb;
    border-radius: 4px;
    overflow: hidden;
}

.progress-fill {
    height: 100%;
    background: #3b82f6;
    transition: width 0.3s ease;
}

.progress-text {
    font-family: monospace;
    font-weight: 500;
    min-width: 40px;
    text-align: right;
}
</style>
