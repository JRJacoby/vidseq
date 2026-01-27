<script setup lang="ts">
import { ref, onMounted, onUnmounted, computed, watch, nextTick } from 'vue'
import { useProjectStore } from '@/stores/project'
import {
    getARHMMStatus,
    startARHMM,
    stopARHMM,
    getARHMMStreamUrl,
    getARHMMAnalysis,
    type ARHMMProgress,
    type ARHMMAnalysis,
} from '@/services/api'
import { Chart, registerables } from 'chart.js'
import zoomPlugin from 'chartjs-plugin-zoom'

Chart.register(...registerables, zoomPlugin)

const projectStore = useProjectStore()
const projectId = computed(() => projectStore.currentProjectId)

const progress = ref<ARHMMProgress | null>(null)
const isLoading = ref(true)
const actionError = ref<string | null>(null)
let eventSource: EventSource | null = null

// Analysis charts
const analysisData = ref<ARHMMAnalysis | null>(null)
const durationChartCanvas = ref<HTMLCanvasElement | null>(null)
const frequencyChartCanvas = ref<HTMLCanvasElement | null>(null)
let durationChart: Chart | null = null
let frequencyChart: Chart | null = null

// Load initial status
async function loadStatus() {
    if (!projectId.value) return
    isLoading.value = true
    try {
        progress.value = await getARHMMStatus(projectId.value)
        if (progress.value.is_running) {
            connectToStream()
        }
        if (progress.value.status === 'completed') {
            loadAnalysis()
        }
    } catch (e) {
        console.error('Failed to load ARHMM status:', e)
    } finally {
        isLoading.value = false
    }
}

// SSE streaming
function connectToStream() {
    if (!projectId.value) return

    if (eventSource) {
        eventSource.close()
        eventSource = null
    }

    const url = getARHMMStreamUrl(projectId.value)
    eventSource = new EventSource(url)

    eventSource.onmessage = (event) => {
        const data = JSON.parse(event.data) as ARHMMProgress
        progress.value = data

        if (['completed', 'failed'].includes(data.status)) {
            eventSource?.close()
            eventSource = null
            if (data.status === 'completed') {
                loadAnalysis()
            }
        }
    }

    eventSource.onerror = () => {
        eventSource?.close()
        eventSource = null
    }
}

async function handleStart() {
    if (!projectId.value) return
    actionError.value = null
    try {
        await startARHMM(projectId.value)
        // Give the backend a moment to initialize progress
        await new Promise(r => setTimeout(r, 300))
        await loadStatus()
        connectToStream()
    } catch (e: any) {
        actionError.value = e.message || 'Failed to start ARHMM'
    }
}

async function handleStop() {
    if (!projectId.value) return
    actionError.value = null
    try {
        await stopARHMM(projectId.value)
    } catch (e: any) {
        actionError.value = e.message || 'Failed to stop ARHMM'
    }
}

// --- Analysis charts ---

async function loadAnalysis() {
    if (!projectId.value) return
    try {
        analysisData.value = await getARHMMAnalysis(projectId.value)
        await nextTick()
        initCharts()
    } catch (e) {
        console.error('Failed to load ARHMM analysis:', e)
    }
}

function initCharts() {
    durationChart?.destroy()
    durationChart = null
    frequencyChart?.destroy()
    frequencyChart = null
    initDurationChart()
    initFrequencyChart()
}

function initDurationChart() {
    if (!durationChartCanvas.value || !analysisData.value) return
    const { bin_centers, counts } = analysisData.value.duration_histogram
    const labels = bin_centers.map(c => `${c.toFixed(0)}`)

    durationChart = new Chart(durationChartCanvas.value, {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                label: 'Runs',
                data: counts,
                backgroundColor: 'rgba(59, 130, 246, 0.7)',
                borderColor: 'rgb(59, 130, 246)',
                borderWidth: 1,
                barPercentage: 1.0,
                categoryPercentage: 1.0,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                zoom: {
                    zoom: {
                        drag: {
                            enabled: true,
                            backgroundColor: 'rgba(59, 130, 246, 0.1)',
                            borderColor: 'rgba(59, 130, 246, 0.5)',
                            borderWidth: 1,
                        },
                        pinch: { enabled: true },
                        mode: 'xy',
                    },
                    pan: {
                        enabled: true,
                        mode: 'xy',
                        modifierKey: 'ctrl',
                    },
                },
            },
            scales: {
                x: { title: { display: true, text: 'Duration (ms)' } },
                y: { title: { display: true, text: 'Count' }, beginAtZero: true },
            },
        }
    })
}

function initFrequencyChart() {
    if (!frequencyChartCanvas.value || !analysisData.value) return
    const { syllables, counts } = analysisData.value.syllable_frequencies
    const labels = syllables.map(s => `${s}`)

    frequencyChart = new Chart(frequencyChartCanvas.value, {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                label: 'Runs',
                data: counts,
                backgroundColor: 'rgba(139, 92, 246, 0.7)',
                borderColor: 'rgb(139, 92, 246)',
                borderWidth: 1,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                zoom: {
                    zoom: {
                        drag: {
                            enabled: true,
                            backgroundColor: 'rgba(139, 92, 246, 0.1)',
                            borderColor: 'rgba(139, 92, 246, 0.5)',
                            borderWidth: 1,
                        },
                        pinch: { enabled: true },
                        mode: 'xy',
                    },
                    pan: {
                        enabled: true,
                        mode: 'xy',
                        modifierKey: 'ctrl',
                    },
                },
            },
            scales: {
                x: { title: { display: true, text: 'Syllable' } },
                y: { title: { display: true, text: 'Number of Runs' }, beginAtZero: true },
            },
        }
    })
}

function resetDurationZoom() { durationChart?.resetZoom() }
function resetFrequencyZoom() { frequencyChart?.resetZoom() }

onMounted(() => {
    loadStatus()
})

onUnmounted(() => {
    eventSource?.close()
    durationChart?.destroy()
    frequencyChart?.destroy()
})

// Reconnect on project change
watch(() => projectId.value, () => {
    eventSource?.close()
    eventSource = null
    loadStatus()
})

// Reconnect if fitting starts
watch(() => progress.value?.is_running, (isRunning) => {
    if (isRunning && !eventSource) {
        connectToStream()
    }
})

// Computed helpers
const statusColor = computed(() => {
    switch (progress.value?.status) {
        case 'searching': return '#3b82f6'  // Blue
        case 'fitting': return '#8b5cf6'     // Purple
        case 'completed': return '#22c55e'   // Green
        case 'failed': return '#ef4444'      // Red
        default: return '#6b7280'            // Gray
    }
})

const fitProgressPercent = computed(() => {
    if (!progress.value || progress.value.total_iterations === 0) return 0
    return (progress.value.current_iteration / progress.value.total_iterations) * 100
})

const isIdle = computed(() => {
    return !progress.value || progress.value.status === 'idle'
})

const isActive = computed(() => {
    return progress.value?.status === 'searching' || progress.value?.status === 'fitting'
})

const formatKappa = (kappa: number) => {
    if (kappa === 0) return '--'
    return kappa.toExponential(2)
}

const formatTime = (seconds: number): string => {
    if (seconds <= 0) return '--'
    const h = Math.floor(seconds / 3600)
    const m = Math.floor((seconds % 3600) / 60)
    const s = Math.floor(seconds % 60)
    if (h > 0) return `${h}h ${m}m ${s}s`
    if (m > 0) return `${m}m ${s}s`
    return `${s}s`
}

const resultIcon = (result: string) => {
    switch (result) {
        case 'too_low': return '↑'
        case 'too_high': return '↓'
        case 'found': return '✓'
        default: return '?'
    }
}
</script>

<template>
    <div class="arhmm-page">
        <h1>ARHMM Modeling</h1>

        <div v-if="isLoading" class="loading-state">
            Loading ARHMM status...
        </div>

        <template v-else>
            <!-- Status Panel -->
            <section class="section">
                <h2 class="section-title">Status</h2>
                <div class="status-panel">
                    <div class="status-row">
                        <span class="label">Status:</span>
                        <div class="status-indicator" :style="{ backgroundColor: statusColor }">
                            {{ progress?.status ?? 'idle' }}
                        </div>
                    </div>

                    <div class="meta-row" v-if="progress && (progress.num_videos > 0 || progress.fps > 0)">
                        <span v-if="progress.fps > 0">FPS: {{ progress.fps }}</span>
                        <span v-if="progress.num_videos > 0">Videos: {{ progress.num_videos }}</span>
                        <span v-if="progress.total_frames > 0">Frames: {{ progress.total_frames.toLocaleString() }}</span>
                    </div>

                    <div class="action-row">
                        <button
                            class="action-btn action-btn-primary"
                            @click="handleStart"
                            :disabled="isActive"
                        >
                            Run ARHMM
                        </button>
                        <button
                            class="action-btn action-btn-danger"
                            @click="handleStop"
                            :disabled="!isActive"
                        >
                            Stop
                        </button>
                    </div>

                    <div v-if="actionError" class="error-message">{{ actionError }}</div>
                    <div v-if="progress?.error" class="error-message">{{ progress.error }}</div>
                </div>
            </section>

            <!-- Kappa Search Table -->
            <section class="section" v-if="progress && progress.search_history.length > 0">
                <h2 class="section-title">Kappa Search</h2>
                <div class="search-panel">
                    <table class="search-table">
                        <thead>
                            <tr>
                                <th>Step</th>
                                <th>log₁₀(κ)</th>
                                <th>κ</th>
                                <th>Duration</th>
                                <th></th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr
                                v-for="entry in progress.search_history"
                                :key="entry.step"
                                :class="{
                                    'row-found': entry.result === 'found',
                                    'row-too-low': entry.result === 'too_low',
                                    'row-too-high': entry.result === 'too_high',
                                }"
                            >
                                <td>{{ entry.step }}</td>
                                <td class="mono">{{ entry.log_kappa.toFixed(2) }}</td>
                                <td class="mono">{{ formatKappa(entry.kappa) }}</td>
                                <td class="mono">{{ entry.median_duration_ms.toFixed(0) }}ms</td>
                                <td class="result-icon">{{ resultIcon(entry.result) }}</td>
                            </tr>
                        </tbody>
                    </table>
                    <div class="bounds-info" v-if="isActive && progress.phase === 'search'">
                        Bounds: [{{ progress.log_low.toFixed(2) }}, {{ progress.log_high.toFixed(2) }}]
                    </div>
                </div>
            </section>

            <!-- Current Fit Progress -->
            <section class="section" v-if="progress && isActive">
                <h2 class="section-title">
                    {{ progress.phase === 'search' ? 'Search Iteration' : 'Final Fit' }}
                </h2>
                <div class="fit-panel">
                    <div class="stat-row">
                        <span class="label">κ:</span>
                        <span class="value mono">{{ formatKappa(progress.current_kappa) }}</span>
                    </div>
                    <div class="stat-row">
                        <span class="label">Iteration:</span>
                        <span class="value mono">
                            {{ progress.current_iteration }} / {{ progress.total_iterations }}
                        </span>
                    </div>
                    <div class="progress-container">
                        <div class="progress-bar">
                            <div
                                class="progress-fill"
                                :style="{ width: `${fitProgressPercent}%` }"
                            ></div>
                        </div>
                        <span class="progress-text">{{ Math.round(fitProgressPercent) }}%</span>
                    </div>
                    <div class="timing-row" v-if="progress.elapsed_seconds > 0">
                        <span>Elapsed: {{ formatTime(progress.elapsed_seconds) }}</span>
                        <span v-if="progress.iterations_per_second > 0">
                            Speed: {{ progress.iterations_per_second.toFixed(2) }} it/s
                        </span>
                        <span v-if="progress.eta_seconds > 0">
                            ETA: {{ formatTime(progress.eta_seconds) }}
                        </span>
                    </div>
                </div>
            </section>

            <!-- Results -->
            <section class="section" v-if="progress && progress.status === 'completed' && progress.chosen_kappa > 0">
                <h2 class="section-title">Results</h2>
                <div class="results-panel">
                    <div class="stat-row">
                        <span class="label">Chosen κ:</span>
                        <span class="value mono">{{ formatKappa(progress.chosen_kappa) }}</span>
                    </div>
                    <div class="stat-row">
                        <span class="label">Median syllable duration:</span>
                        <span class="value mono">{{ progress.final_median_duration_ms }}ms</span>
                    </div>
                    <div class="stat-row">
                        <span class="label">Model saved to:</span>
                        <span class="value filename">arhmm/model.joblib</span>
                    </div>
                </div>
            </section>

            <!-- Analysis Charts -->
            <section class="section" v-if="progress && progress.status === 'completed' && analysisData">
                <h2 class="section-title">Analysis</h2>
                <div class="charts-container">
                    <div class="chart-section">
                        <div class="chart-header">
                            <h3>Syllable Duration Distribution</h3>
                            <button class="reset-zoom-btn" @click="resetDurationZoom">Reset Zoom</button>
                        </div>
                        <div class="chart-wrapper">
                            <canvas ref="durationChartCanvas"></canvas>
                        </div>
                    </div>
                    <div class="chart-section">
                        <div class="chart-header">
                            <h3>Syllable Frequency</h3>
                            <button class="reset-zoom-btn" @click="resetFrequencyZoom">Reset Zoom</button>
                        </div>
                        <div class="chart-wrapper">
                            <canvas ref="frequencyChartCanvas"></canvas>
                        </div>
                    </div>
                </div>
            </section>

            <!-- Idle message -->
            <div v-if="isIdle && (!progress || progress.search_history.length === 0)" class="no-data-message">
                <p>No ARHMM model has been fitted yet.</p>
                <p class="hint">Click "Run ARHMM" to start fitting. PCA must be computed first.</p>
            </div>
        </template>
    </div>
</template>

<style scoped>
.arhmm-page {
    padding: 20px;
    max-width: 900px;
}

h1 {
    margin-bottom: 20px;
}

.loading-state {
    padding: 3rem;
    text-align: center;
    color: #888;
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

/* Status Panel */
.status-panel {
    background: #f8f8f8;
    border-radius: 8px;
    padding: 16px;
}

.status-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 12px;
}

.status-indicator {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 4px;
    color: white;
    font-weight: 500;
    text-transform: uppercase;
    font-size: 0.85em;
}

.meta-row {
    display: flex;
    gap: 16px;
    margin-bottom: 12px;
    font-size: 0.9em;
    color: #555;
}

.action-row {
    display: flex;
    gap: 8px;
}

.action-btn {
    padding: 8px 20px;
    border: none;
    border-radius: 4px;
    font-size: 0.9em;
    font-weight: 500;
    cursor: pointer;
    transition: opacity 0.2s;
}

.action-btn:disabled {
    opacity: 0.4;
    cursor: not-allowed;
}

.action-btn-primary {
    background: #3b82f6;
    color: white;
}

.action-btn-primary:hover:not(:disabled) {
    background: #2563eb;
}

.action-btn-danger {
    background: #ef4444;
    color: white;
}

.action-btn-danger:hover:not(:disabled) {
    background: #dc2626;
}

.error-message {
    margin-top: 12px;
    padding: 8px 12px;
    background: #fef2f2;
    border: 1px solid #fecaca;
    border-radius: 4px;
    color: #dc2626;
    font-size: 0.85em;
}

/* Search Table */
.search-panel {
    background: #f8f8f8;
    border-radius: 8px;
    padding: 16px;
}

.search-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.9em;
}

.search-table th {
    text-align: left;
    padding: 8px 12px;
    border-bottom: 2px solid #ddd;
    font-weight: 600;
    color: #555;
}

.search-table td {
    padding: 6px 12px;
    border-bottom: 1px solid #eee;
}

.search-table .mono {
    font-family: monospace;
}

.search-table .result-icon {
    text-align: center;
    font-weight: 700;
    font-size: 1.1em;
}

.row-found {
    background: #f0fdf4;
}

.row-found .result-icon {
    color: #22c55e;
}

.row-too-low .result-icon {
    color: #3b82f6;
}

.row-too-high .result-icon {
    color: #f59e0b;
}

.bounds-info {
    margin-top: 8px;
    font-size: 0.85em;
    color: #666;
    font-family: monospace;
}

/* Fit Progress Panel */
.fit-panel,
.results-panel {
    background: #f8f8f8;
    border-radius: 8px;
    padding: 16px;
}

.stat-row {
    display: flex;
    align-items: baseline;
    gap: 8px;
    margin-bottom: 8px;
}

.stat-row .label {
    font-size: 0.85em;
    color: #666;
    min-width: 80px;
}

.stat-row .value {
    font-weight: 500;
}

.stat-row .value.mono {
    font-family: monospace;
}

.stat-row .value.filename {
    font-size: 0.9em;
    color: #555;
    font-family: monospace;
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

.timing-row {
    display: flex;
    gap: 16px;
    margin-top: 8px;
    font-size: 0.85em;
    color: #555;
    font-family: monospace;
}

/* Analysis Charts */
.charts-container {
    display: flex;
    flex-direction: column;
    gap: 24px;
}

.chart-section {
    background: #f8f8f8;
    border-radius: 8px;
    padding: 16px;
}

.chart-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
}

.chart-header h3 {
    margin: 0;
    font-size: 1em;
    color: #333;
}

.reset-zoom-btn {
    padding: 4px 10px;
    border: 1px solid #ccc;
    border-radius: 4px;
    background: white;
    font-size: 0.8em;
    cursor: pointer;
    color: #555;
}

.reset-zoom-btn:hover {
    background: #eee;
    border-color: #aaa;
}

.chart-wrapper {
    height: 300px;
    position: relative;
}

/* No data */
.no-data-message {
    margin-top: 20px;
    padding: 16px;
    background: #f8f8f8;
    border-radius: 8px;
    text-align: center;
    color: #666;
}

.no-data-message .hint {
    font-size: 0.85em;
    color: #999;
    margin-top: 8px;
}
</style>
