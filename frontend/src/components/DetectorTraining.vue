<script setup lang="ts">
import { ref, onMounted, onUnmounted, computed, watch } from 'vue'
import { useProjectStore } from '@/stores/project'
import {
    getDetectorTrainingStreamUrl,
    getDetectorTrainingStatus,
    stopDetectorTraining,
    type DetectorTrainingProgress,
} from '@/services/api'
import { Chart, registerables } from 'chart.js'

Chart.register(...registerables)

const projectStore = useProjectStore()
const projectId = computed(() => projectStore.currentProjectId)

// Training progress
const progress = ref<DetectorTrainingProgress | null>(null)
const chartCanvas = ref<HTMLCanvasElement | null>(null)
let chart: Chart | null = null
let eventSource: EventSource | null = null

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

    const url = getDetectorTrainingStreamUrl(projectId.value)
    eventSource = new EventSource(url)

    eventSource.onmessage = (event) => {
        const data = JSON.parse(event.data) as DetectorTrainingProgress
        progress.value = data
        updateChart(data.train_loss_history, data.val_loss_history)

        // Close connection if training finished
        if (['completed', 'stopped', 'failed', 'idle'].includes(data.status)) {
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
        progress.value = await getDetectorTrainingStatus(projectId.value)
        if (progress.value.train_loss_history.length > 0 || progress.value.val_loss_history.length > 0) {
            updateChart(progress.value.train_loss_history, progress.value.val_loss_history)
        }

        // If training in progress, connect to stream
        if (progress.value.is_training || progress.value.status === 'applying') {
            connectToStream()
        }
    } catch (e) {
        console.error('Failed to load training status:', e)
    }
}

// Stop training handler
async function handleStopTraining() {
    if (!projectId.value) return
    try {
        await stopDetectorTraining(projectId.value)
    } catch (e) {
        console.error('Failed to stop training:', e)
    }
}

onMounted(() => {
    initChart()
    loadTrainingStatus()
})

onUnmounted(() => {
    eventSource?.close()
    chart?.destroy()
})

// Reconnect if project changes
watch(() => projectId.value, () => {
    eventSource?.close()
    eventSource = null
    loadTrainingStatus()
})

// Reconnect if training starts
watch(() => progress.value?.is_training, (isTraining) => {
    if (isTraining && !eventSource) {
        connectToStream()
    }
})

// Format helpers
const formatLR = (lr: number) => lr.toExponential(2)
const formatLoss = (loss: number | null | undefined) => {
    if (loss === Infinity || loss === null || loss === undefined) return '--'
    return loss.toFixed(6)
}

const statusColor = computed(() => {
    switch (progress.value?.status) {
        case 'training': return '#3b82f6'  // Blue
        case 'applying': return '#8b5cf6'  // Purple
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

// Apply progress
const applyProgressPercent = computed(() => {
    if (!progress.value || progress.value.apply_total === 0) return 0
    return (progress.value.apply_current / progress.value.apply_total) * 100
})
</script>

<template>
    <div class="detector-training-page">
        <h1>Detector Training</h1>

        <!-- Training Section -->
        <section class="section">
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
                            <span class="stat-value">{{ progress.num_train_frames }} train / {{ progress.num_val_frames }} val</span>
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
                            <span class="stat-value">{{ formatLoss(progress.best_val_loss) }} (epoch {{ progress.best_epoch }})</span>
                        </div>
                        <div class="stat">
                            <span class="stat-label">Learning Rate</span>
                            <span class="stat-value">{{ formatLR(progress.current_lr) }}</span>
                        </div>
                    </div>

                    <!-- Patience Counters -->
                    <div class="patience-section" v-if="progress && progress.is_training">
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
                                :style="{ width: `${lrReductionProgress}%` }" />
                            <span class="patience-text">
                                {{ Math.min(progress.epochs_without_improvement, progress.lr_patience) }} / {{ progress.lr_patience }} to reduction
                            </span>
                        </div>
                    </div>

                    <!-- Apply Progress -->
                    <div class="apply-section" v-if="progress && progress.status === 'applying'">
                        <h3>Applying to Training Frames</h3>
                        <div class="progress-container">
                            <div class="progress-bar">
                                <div class="progress-fill" :style="{ width: `${applyProgressPercent}%` }"></div>
                            </div>
                            <span class="progress-text">{{ Math.round(applyProgressPercent) }}%</span>
                        </div>
                        <div class="stat">
                            <span class="stat-label">Frames</span>
                            <span class="stat-value">{{ progress.apply_current }} / {{ progress.apply_total }}</span>
                        </div>
                    </div>

                    <!-- Stop Button -->
                    <div class="actions" v-if="progress?.is_training">
                        <button class="btn btn-stop" @click="handleStopTraining">
                            Stop Training
                        </button>
                    </div>

                    <!-- Error Message -->
                    <div v-if="progress?.error_message" class="error-message">
                        {{ progress.error_message }}
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
    </div>
</template>

<style scoped>
.detector-training-page {
    padding: 20px;
    max-width: 1200px;
}

h1 {
    margin-bottom: 20px;
}

.section {
    margin-bottom: 24px;
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

.patience-section,
.apply-section {
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

.patience-text {
    position: absolute;
    left: 8px;
    top: 50%;
    transform: translateY(-50%);
    font-size: 0.75em;
    color: #333;
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
    background: #8b5cf6;
    transition: width 0.3s ease;
}

.progress-text {
    font-family: monospace;
    font-weight: 500;
    min-width: 40px;
    text-align: right;
}

.actions {
    margin-top: 16px;
}

.btn {
    padding: 8px 16px;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-weight: 500;
}

.btn-stop {
    background: #ef4444;
    color: white;
}

.btn-stop:hover {
    background: #dc2626;
}

.error-message {
    margin-top: 16px;
    padding: 12px;
    background: #fef2f2;
    border: 1px solid #fecaca;
    border-radius: 4px;
    color: #dc2626;
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
</style>
