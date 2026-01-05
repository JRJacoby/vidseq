<script setup lang="ts">
import { ref, computed, onMounted, watch, nextTick } from 'vue'
import { useProjectStore } from '@/stores/project'
import {
  getAlignmentStatus,
  getRandomAlignmentFrame,
  saveAlignmentLabel,
  trainAlignmentModel,
  getAlignmentPredictionUrl,
  applyAlignment,
  clearAllAlignmentLabels,
  clearAlignmentModel,
  getCroppedVideoFrame,
  type AlignmentStatus,
} from '@/services/api'

// Logging prefix for all alignment logs
const LOG_PREFIX = '[Alignment]'

const projectStore = useProjectStore()
const projectId = computed(() => projectStore.currentProjectId)

// State
const isTrainingMode = ref(false)
const status = ref<AlignmentStatus | null>(null)
const currentVideoId = ref<number | null>(null)
const currentFrameIdx = ref<number | null>(null)
const frontPoint = ref<{ x: number; y: number } | null>(null)
const rearPoint = ref<{ x: number; y: number } | null>(null)
const isLoading = ref(false)
const isSaving = ref(false)
const isTraining = ref(false)
const isApplying = ref(false)
const isClearing = ref(false)
const isClearingModel = ref(false)
const epochs = ref(10)
const error = ref<string | null>(null)

// Canvas refs
const frameCanvasRef = ref<HTMLCanvasElement | null>(null)
const overlayCanvasRef = ref<HTMLCanvasElement | null>(null)
const frameImage = ref<HTMLImageElement | null>(null)
const heatmapImage = ref<HTMLImageElement | null>(null)

// Canvas dimensions
const canvasWidth = ref(0)
const canvasHeight = ref(0)

// Computed
const canStartTraining = computed(() => status.value?.all_videos_cropped ?? false)
const hasLabels = computed(() => (status.value?.label_count ?? 0) > 0)
const modelTrained = computed(() => status.value?.model_trained ?? false)

// Load status
async function loadStatus() {
  console.log(`${LOG_PREFIX} loadStatus: projectId=${projectId.value}`)
  if (!projectId.value) {
    console.log(`${LOG_PREFIX} loadStatus: skipping - no projectId`)
    return
  }
  try {
    status.value = await getAlignmentStatus(projectId.value)
    console.log(`${LOG_PREFIX} loadStatus: response=`, {
      label_count: status.value.label_count,
      model_trained: status.value.model_trained,
      is_training: status.value.is_training,
      is_applying: status.value.is_applying,
      all_videos_cropped: status.value.all_videos_cropped,
    })
  } catch (e) {
    console.error(`${LOG_PREFIX} loadStatus: FAILED`, e)
  }
}

// Load a random frame for labeling
async function loadRandomFrame() {
  console.log(`${LOG_PREFIX} loadRandomFrame: projectId=${projectId.value}`)
  if (!projectId.value) {
    console.log(`${LOG_PREFIX} loadRandomFrame: skipping - no projectId`)
    return
  }
  isLoading.value = true
  error.value = null
  frontPoint.value = null
  rearPoint.value = null

  try {
    console.log(`${LOG_PREFIX} loadRandomFrame: fetching random frame...`)
    const frame = await getRandomAlignmentFrame(projectId.value)
    currentVideoId.value = frame.video_id
    currentFrameIdx.value = frame.frame_idx
    console.log(`${LOG_PREFIX} loadRandomFrame: got frame video_id=${frame.video_id}, frame_idx=${frame.frame_idx}`)

    // Load the frame image
    console.log(`${LOG_PREFIX} loadRandomFrame: fetching cropped frame image...`)
    const blob = await getCroppedVideoFrame(projectId.value, frame.video_id, frame.frame_idx)
    console.log(`${LOG_PREFIX} loadRandomFrame: frame blob size=${blob.size} bytes`)
    const url = URL.createObjectURL(blob)

    // Wait for image to load
    await new Promise<void>((resolve, reject) => {
      const img = new Image()
      img.onload = async () => {
        frameImage.value = img
        canvasWidth.value = img.width
        canvasHeight.value = img.height
        console.log(`${LOG_PREFIX} loadRandomFrame: image loaded, dimensions=${img.width}x${img.height}`)

        // Set isLoading false BEFORE nextTick so the canvas will be rendered
        isLoading.value = false
        console.log(`${LOG_PREFIX} loadRandomFrame: isLoading set to false`)

        // Wait for Vue to update the DOM (canvas is conditionally rendered)
        await nextTick()
        console.log(`${LOG_PREFIX} loadRandomFrame: nextTick complete, rendering frame`)
        renderFrame()

        // Load heatmap prediction if model is trained
        if (modelTrained.value) {
          console.log(`${LOG_PREFIX} loadRandomFrame: model is trained, loading heatmap`)
          loadHeatmap()
        } else {
          console.log(`${LOG_PREFIX} loadRandomFrame: model not trained, skipping heatmap`)
        }
        resolve()
      }
      img.onerror = () => {
        console.error(`${LOG_PREFIX} loadRandomFrame: image failed to load`)
        isLoading.value = false
        reject(new Error('Failed to load frame image'))
      }
      img.src = url
    })
  } catch (e: any) {
    error.value = e.message || 'Failed to load frame'
    console.error(`${LOG_PREFIX} loadRandomFrame: FAILED`, e)
    isLoading.value = false
  }
}

// Load heatmap prediction
async function loadHeatmap() {
  console.log(`${LOG_PREFIX} loadHeatmap: projectId=${projectId.value}, videoId=${currentVideoId.value}, frameIdx=${currentFrameIdx.value}`)
  if (!projectId.value || currentVideoId.value === null || currentFrameIdx.value === null) {
    console.log(`${LOG_PREFIX} loadHeatmap: skipping - missing parameters`)
    return
  }
  if (!modelTrained.value) {
    console.log(`${LOG_PREFIX} loadHeatmap: skipping - model not trained`)
    return
  }

  const url = getAlignmentPredictionUrl(projectId.value, currentVideoId.value, currentFrameIdx.value)
  console.log(`${LOG_PREFIX} loadHeatmap: fetching from ${url}`)
  const img = new Image()
  img.crossOrigin = 'anonymous'
  img.onload = () => {
    console.log(`${LOG_PREFIX} loadHeatmap: heatmap loaded, dimensions=${img.width}x${img.height}`)
    heatmapImage.value = img
    renderOverlay()
  }
  img.onerror = () => {
    console.error(`${LOG_PREFIX} loadHeatmap: FAILED to load heatmap image`)
    heatmapImage.value = null
  }
  img.src = url
}

// Render frame to canvas
function renderFrame() {
  const canvas = frameCanvasRef.value
  if (!canvas || !frameImage.value) {
    console.log(`${LOG_PREFIX} renderFrame: skipping - canvas=${!!canvas}, frameImage=${!!frameImage.value}`)
    return
  }

  canvas.width = canvasWidth.value
  canvas.height = canvasHeight.value
  console.log(`${LOG_PREFIX} renderFrame: canvas dimensions set to ${canvas.width}x${canvas.height}`)

  const ctx = canvas.getContext('2d')
  if (!ctx) return

  ctx.clearRect(0, 0, canvas.width, canvas.height)
  ctx.drawImage(frameImage.value, 0, 0)
  console.log(`${LOG_PREFIX} renderFrame: frame drawn to canvas`)
}

// Render overlay (heatmap + points)
function renderOverlay() {
  const canvas = overlayCanvasRef.value
  if (!canvas) {
    console.log(`${LOG_PREFIX} renderOverlay: skipping - no canvas`)
    return
  }

  canvas.width = canvasWidth.value
  canvas.height = canvasHeight.value
  console.log(`${LOG_PREFIX} renderOverlay: overlay canvas dimensions set to ${canvas.width}x${canvas.height}`)

  const ctx = canvas.getContext('2d')
  if (!ctx) return

  ctx.clearRect(0, 0, canvas.width, canvas.height)

  // Draw heatmap if available
  if (heatmapImage.value) {
    console.log(`${LOG_PREFIX} renderOverlay: rendering heatmap, source dimensions=${heatmapImage.value.width}x${heatmapImage.value.height}`)
    // Create a temporary canvas to read heatmap pixels
    const tempCanvas = document.createElement('canvas')
    tempCanvas.width = heatmapImage.value.width
    tempCanvas.height = heatmapImage.value.height
    const tempCtx = tempCanvas.getContext('2d')
    if (tempCtx) {
      tempCtx.drawImage(heatmapImage.value, 0, 0)
      const imageData = tempCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height)
      const data = imageData.data

      // Log sample pixel values for debugging
      const centerX = Math.floor(tempCanvas.width / 2)
      const centerY = Math.floor(tempCanvas.height / 2)
      const centerIdx = (centerY * tempCanvas.width + centerX) * 4
      console.log(`${LOG_PREFIX} renderOverlay: heatmap center pixel (${centerX},${centerY}): R=${data[centerIdx]}, G=${data[centerIdx+1]}, B=${data[centerIdx+2]}, A=${data[centerIdx+3]}`)

      // Find max values in the heatmap
      let maxFront = 0, maxRear = 0
      for (let i = 0; i < data.length; i += 4) {
        maxFront = Math.max(maxFront, data[i] ?? 0)
        maxRear = Math.max(maxRear, data[i + 1] ?? 0)
      }
      console.log(`${LOG_PREFIX} renderOverlay: heatmap max values: front(R)=${maxFront}, rear(G)=${maxRear}`)

      // Create overlay image data
      const overlayData = ctx.createImageData(canvas.width, canvas.height)
      const scaleX = tempCanvas.width / canvas.width
      const scaleY = tempCanvas.height / canvas.height
      console.log(`${LOG_PREFIX} renderOverlay: scale factors scaleX=${scaleX.toFixed(4)}, scaleY=${scaleY.toFixed(4)}`)

      for (let y = 0; y < canvas.height; y++) {
        for (let x = 0; x < canvas.width; x++) {
          const srcX = Math.floor(x * scaleX)
          const srcY = Math.floor(y * scaleY)
          const srcIdx = (srcY * tempCanvas.width + srcX) * 4
          const dstIdx = (y * canvas.width + x) * 4

          const frontProb = (data[srcIdx] ?? 0) / 255     // R channel
          const rearProb = (data[srcIdx + 1] ?? 0) / 255  // G channel

          // Blend: green for front, red for rear
          overlayData.data[dstIdx] = Math.floor(rearProb * 255)      // R
          overlayData.data[dstIdx + 1] = Math.floor(frontProb * 255) // G
          overlayData.data[dstIdx + 2] = 0                            // B
          overlayData.data[dstIdx + 3] = Math.floor(Math.max(frontProb, rearProb) * 150) // A
        }
      }
      ctx.putImageData(overlayData, 0, 0)
      console.log(`${LOG_PREFIX} renderOverlay: heatmap overlay applied`)
    }
  } else {
    console.log(`${LOG_PREFIX} renderOverlay: no heatmap available`)
  }

  // Draw labeled points
  if (frontPoint.value) {
    const pixelX = frontPoint.value.x * canvas.width
    const pixelY = frontPoint.value.y * canvas.height
    console.log(`${LOG_PREFIX} renderOverlay: drawing front point at normalized=(${frontPoint.value.x.toFixed(4)}, ${frontPoint.value.y.toFixed(4)}), pixel=(${pixelX.toFixed(1)}, ${pixelY.toFixed(1)})`)
    drawPoint(ctx, pixelX, pixelY, '#22c55e', 'Front')
  }
  if (rearPoint.value) {
    const pixelX = rearPoint.value.x * canvas.width
    const pixelY = rearPoint.value.y * canvas.height
    console.log(`${LOG_PREFIX} renderOverlay: drawing rear point at normalized=(${rearPoint.value.x.toFixed(4)}, ${rearPoint.value.y.toFixed(4)}), pixel=(${pixelX.toFixed(1)}, ${pixelY.toFixed(1)})`)
    drawPoint(ctx, pixelX, pixelY, '#ef4444', 'Rear')
  }
}

function drawPoint(ctx: CanvasRenderingContext2D, x: number, y: number, color: string, label: string) {
  const radius = 10

  // Draw circle
  ctx.beginPath()
  ctx.arc(x, y, radius, 0, Math.PI * 2)
  ctx.fillStyle = color
  ctx.fill()
  ctx.strokeStyle = '#fff'
  ctx.lineWidth = 2
  ctx.stroke()

  // Draw crosshair
  ctx.strokeStyle = '#fff'
  ctx.lineWidth = 2
  ctx.beginPath()
  ctx.moveTo(x - 5, y)
  ctx.lineTo(x + 5, y)
  ctx.moveTo(x, y - 5)
  ctx.lineTo(x, y + 5)
  ctx.stroke()

  // Draw label
  ctx.font = '12px sans-serif'
  ctx.fillStyle = color
  ctx.fillText(label, x + 15, y + 4)
}

// Handle canvas click
function handleCanvasClick(event: MouseEvent) {
  console.log(`${LOG_PREFIX} handleCanvasClick: isTrainingMode=${isTrainingMode.value}`)
  if (!isTrainingMode.value) {
    console.log(`${LOG_PREFIX} handleCanvasClick: ignoring click - not in training mode`)
    return
  }
  const canvas = overlayCanvasRef.value
  if (!canvas) {
    console.log(`${LOG_PREFIX} handleCanvasClick: ignoring click - no canvas`)
    return
  }

  const rect = canvas.getBoundingClientRect()
  console.log(`${LOG_PREFIX} handleCanvasClick: canvas rect - left=${rect.left.toFixed(1)}, top=${rect.top.toFixed(1)}, width=${rect.width.toFixed(1)}, height=${rect.height.toFixed(1)}`)
  console.log(`${LOG_PREFIX} handleCanvasClick: click clientX=${event.clientX}, clientY=${event.clientY}`)
  const x = (event.clientX - rect.left) / rect.width
  const y = (event.clientY - rect.top) / rect.height
  console.log(`${LOG_PREFIX} handleCanvasClick: normalized coordinates x=${x.toFixed(4)}, y=${y.toFixed(4)}`)

  // Clamp to 0-1
  if (x < 0 || x > 1 || y < 0 || y > 1) {
    console.log(`${LOG_PREFIX} handleCanvasClick: ignoring click - out of bounds`)
    return
  }

  if (!frontPoint.value) {
    // First click: set front point
    console.log(`${LOG_PREFIX} handleCanvasClick: setting FRONT point at (${x.toFixed(4)}, ${y.toFixed(4)})`)
    frontPoint.value = { x, y }
    renderOverlay()
  } else if (!rearPoint.value) {
    // Second click: set rear point and save
    console.log(`${LOG_PREFIX} handleCanvasClick: setting REAR point at (${x.toFixed(4)}, ${y.toFixed(4)})`)
    rearPoint.value = { x, y }
    renderOverlay()
    saveCurrentLabel()
  } else {
    console.log(`${LOG_PREFIX} handleCanvasClick: ignoring click - both points already set`)
  }
}

// Save current label and load next frame
async function saveCurrentLabel() {
  console.log(`${LOG_PREFIX} saveCurrentLabel: projectId=${projectId.value}, videoId=${currentVideoId.value}, frameIdx=${currentFrameIdx.value}`)
  if (!projectId.value || currentVideoId.value === null || currentFrameIdx.value === null) {
    console.log(`${LOG_PREFIX} saveCurrentLabel: skipping - missing parameters`)
    return
  }
  if (!frontPoint.value || !rearPoint.value) {
    console.log(`${LOG_PREFIX} saveCurrentLabel: skipping - missing points (front=${!!frontPoint.value}, rear=${!!rearPoint.value})`)
    return
  }

  console.log(`${LOG_PREFIX} saveCurrentLabel: saving label with front=(${frontPoint.value.x.toFixed(4)}, ${frontPoint.value.y.toFixed(4)}), rear=(${rearPoint.value.x.toFixed(4)}, ${rearPoint.value.y.toFixed(4)})`)

  isSaving.value = true
  try {
    await saveAlignmentLabel(
      projectId.value,
      currentVideoId.value,
      currentFrameIdx.value,
      frontPoint.value.x,
      frontPoint.value.y,
      rearPoint.value.x,
      rearPoint.value.y
    )
    console.log(`${LOG_PREFIX} saveCurrentLabel: label saved successfully`)
    await loadStatus()
    // Load next frame after a brief delay
    console.log(`${LOG_PREFIX} saveCurrentLabel: loading next frame in 300ms...`)
    setTimeout(() => loadRandomFrame(), 300)
  } catch (e) {
    console.error(`${LOG_PREFIX} saveCurrentLabel: FAILED`, e)
    error.value = 'Failed to save label'
  } finally {
    isSaving.value = false
  }
}

// Start training mode
async function startTraining() {
  console.log(`${LOG_PREFIX} startTraining: entering training mode`)
  isTrainingMode.value = true
  heatmapImage.value = null
  await loadRandomFrame()
  console.log(`${LOG_PREFIX} startTraining: training mode active, first frame loaded`)
}

// Stop training mode
function stopTraining() {
  console.log(`${LOG_PREFIX} stopTraining: exiting training mode`)
  isTrainingMode.value = false
  frontPoint.value = null
  rearPoint.value = null
  currentVideoId.value = null
  currentFrameIdx.value = null
  frameImage.value = null
  heatmapImage.value = null
  console.log(`${LOG_PREFIX} stopTraining: all state cleared`)
}

// Train model
async function handleTrain() {
  console.log(`${LOG_PREFIX} handleTrain: projectId=${projectId.value}, hasLabels=${hasLabels.value}, epochs=${epochs.value}`)
  if (!projectId.value || !hasLabels.value) {
    console.log(`${LOG_PREFIX} handleTrain: skipping - missing requirements`)
    return
  }
  console.log(`${LOG_PREFIX} handleTrain: starting training with ${epochs.value} epochs...`)
  isTraining.value = true
  error.value = null
  try {
    await trainAlignmentModel(projectId.value, epochs.value)
    console.log(`${LOG_PREFIX} handleTrain: training completed successfully`)
    await loadStatus()
    // Reload heatmap for current frame
    if (currentVideoId.value !== null && currentFrameIdx.value !== null) {
      console.log(`${LOG_PREFIX} handleTrain: reloading heatmap for current frame`)
      loadHeatmap()
    }
  } catch (e: any) {
    error.value = e.message || 'Training failed'
    console.error(`${LOG_PREFIX} handleTrain: FAILED`, e)
  } finally {
    isTraining.value = false
  }
}

// Apply alignment to all videos
async function handleApplyAlignment() {
  console.log(`${LOG_PREFIX} handleApplyAlignment: projectId=${projectId.value}, modelTrained=${modelTrained.value}`)
  if (!projectId.value || !modelTrained.value) {
    console.log(`${LOG_PREFIX} handleApplyAlignment: skipping - missing requirements`)
    return
  }
  console.log(`${LOG_PREFIX} handleApplyAlignment: starting alignment of all videos...`)
  isApplying.value = true
  error.value = null
  try {
    await applyAlignment(projectId.value)
    console.log(`${LOG_PREFIX} handleApplyAlignment: alignment completed successfully`)
    alert('Alignment complete! Aligned videos saved to aligned_videos folder.')
  } catch (e: any) {
    error.value = e.message || 'Alignment failed'
    console.error(`${LOG_PREFIX} handleApplyAlignment: FAILED`, e)
  } finally {
    isApplying.value = false
  }
}

// Clear all alignment labels
async function handleClearLabels() {
  console.log(`${LOG_PREFIX} handleClearLabels: projectId=${projectId.value}, hasLabels=${hasLabels.value}`)
  if (!projectId.value || !hasLabels.value) {
    console.log(`${LOG_PREFIX} handleClearLabels: skipping - missing requirements`)
    return
  }

  // Confirm with user
  const labelCount = status.value?.label_count ?? 0
  const confirmed = confirm(`Are you sure you want to delete all ${labelCount} alignment labels? This cannot be undone.`)
  if (!confirmed) {
    console.log(`${LOG_PREFIX} handleClearLabels: cancelled by user`)
    return
  }

  console.log(`${LOG_PREFIX} handleClearLabels: clearing all labels...`)
  isClearing.value = true
  error.value = null
  try {
    const result = await clearAllAlignmentLabels(projectId.value)
    console.log(`${LOG_PREFIX} handleClearLabels: cleared ${result.deleted_count} labels`)
    await loadStatus()
  } catch (e: any) {
    error.value = e.message || 'Failed to clear labels'
    console.error(`${LOG_PREFIX} handleClearLabels: FAILED`, e)
  } finally {
    isClearing.value = false
  }
}

// Clear alignment model
async function handleClearModel() {
  console.log(`${LOG_PREFIX} handleClearModel: projectId=${projectId.value}, modelTrained=${modelTrained.value}`)
  if (!projectId.value || !modelTrained.value) {
    console.log(`${LOG_PREFIX} handleClearModel: skipping - missing requirements`)
    return
  }

  // Confirm with user
  const confirmed = confirm('Are you sure you want to delete the trained model? You will need to retrain before applying alignment.')
  if (!confirmed) {
    console.log(`${LOG_PREFIX} handleClearModel: cancelled by user`)
    return
  }

  console.log(`${LOG_PREFIX} handleClearModel: clearing model...`)
  isClearingModel.value = true
  error.value = null
  try {
    const result = await clearAlignmentModel(projectId.value)
    console.log(`${LOG_PREFIX} handleClearModel: deleted=${result.deleted}`)
    heatmapImage.value = null  // Clear heatmap display
    await loadStatus()
  } catch (e: any) {
    error.value = e.message || 'Failed to clear model'
    console.error(`${LOG_PREFIX} handleClearModel: FAILED`, e)
  } finally {
    isClearingModel.value = false
  }
}

// Watch for canvas dimension changes
watch([canvasWidth, canvasHeight], () => {
  console.log(`${LOG_PREFIX} watch: canvas dimensions changed to ${canvasWidth.value}x${canvasHeight.value}`)
  renderOverlay()
})

onMounted(() => {
  console.log(`${LOG_PREFIX} onMounted: component mounted, loading status...`)
  loadStatus()
})
</script>

<template>
  <div class="alignment-container">
    <div class="alignment-content">
      <h3 class="screen-title">Egocentric Alignment</h3>
      <div class="content-area">
        <div class="main-content">
          <!-- Prerequisites warning -->
          <div v-if="!canStartTraining" class="warning-banner">
            All videos must have completed cropping before alignment training can begin.
            Return to Video Pipeline and extract cropped videos first.
          </div>

          <!-- Frame viewer -->
          <div v-else-if="isTrainingMode" class="frame-viewer">
            <div v-if="isLoading" class="loading-state">Loading frame...</div>
            <div v-else-if="error" class="error-state">{{ error }}</div>
            <div v-else-if="!frameImage" class="empty-state">
              Click "Start Training" to begin labeling
            </div>
            <div v-else class="canvas-container">
              <canvas ref="frameCanvasRef" class="frame-canvas" />
              <canvas
                ref="overlayCanvasRef"
                class="overlay-canvas"
                @click="handleCanvasClick"
              />
            </div>
            <div v-if="currentVideoId !== null" class="frame-info">
              <span>Video {{ currentVideoId }}, Frame {{ currentFrameIdx }}</span>
              <span class="label-hint" v-if="!frontPoint">Click on the animal's front (nose)</span>
              <span class="label-hint" v-else-if="!rearPoint">Click on the animal's rear (tail)</span>
              <span class="label-hint saving" v-else-if="isSaving">Saving...</span>
            </div>
          </div>

          <!-- Idle state -->
          <div v-else class="idle-state">
            <p>Train a model to detect front/rear keypoints for egocentric alignment.</p>
            <p v-if="status">
              Labels: <strong>{{ status.label_count }}</strong> |
              Model trained: <strong>{{ status.model_trained ? 'Yes' : 'No' }}</strong>
            </p>
          </div>
        </div>

        <!-- Sidebar -->
        <aside class="sidebar">
          <h4 class="sidebar-section-title">Training Mode</h4>
          <button
            v-if="!isTrainingMode"
            class="sidebar-button start-button"
            @click="startTraining"
            :disabled="!canStartTraining || isLoading"
          >
            Start Training
          </button>
          <button
            v-else
            class="sidebar-button stop-button"
            @click="stopTraining"
          >
            Stop Training
          </button>

          <template v-if="isTrainingMode">
            <h4 class="sidebar-section-title">Instructions</h4>
            <div class="instructions">
              <p>1. Click on animal's front (nose)</p>
              <p>2. Click on animal's rear (tail)</p>
              <p>3. Frame auto-saves, next frame loads</p>
            </div>

            <h4 class="sidebar-section-title">Train Model</h4>
            <div class="epochs-input">
              <label for="epochs">Epochs:</label>
              <input
                id="epochs"
                type="number"
                v-model.number="epochs"
                min="1"
                max="1000"
                :disabled="isTraining"
              />
            </div>
            <button
              class="sidebar-button train-button"
              @click="handleTrain"
              :disabled="!hasLabels || isTraining"
            >
              {{ isTraining ? 'Training...' : 'Train' }}
            </button>
            <div class="label-count">
              Labels: {{ status?.label_count ?? 0 }}
            </div>
            <button
              class="sidebar-button clear-button"
              @click="handleClearLabels"
              :disabled="!hasLabels || isTraining || isApplying || isClearing"
            >
              {{ isClearing ? 'Clearing...' : 'Clear All Labels' }}
            </button>
          </template>

          <h4 class="sidebar-section-title">Apply Alignment</h4>
          <button
            class="sidebar-button apply-button"
            @click="handleApplyAlignment"
            :disabled="!modelTrained || isApplying"
          >
            {{ isApplying ? 'Applying...' : 'Align All Videos' }}
          </button>
          <button
            class="sidebar-button clear-button"
            @click="handleClearModel"
            :disabled="!modelTrained || isTraining || isApplying || isClearingModel"
          >
            {{ isClearingModel ? 'Clearing...' : 'Clear Model' }}
          </button>

          <div v-if="error" class="error-indicator">
            {{ error }}
          </div>
        </aside>
      </div>
    </div>
  </div>
</template>

<style scoped>
.alignment-container {
  display: flex;
  flex-direction: column;
  flex: 1;
  min-height: 0;
}

.alignment-content {
  display: flex;
  flex-direction: column;
  flex: 1;
  min-height: 0;
}

.screen-title {
  flex-shrink: 0;
  padding: 1rem;
  margin: 0;
  border-bottom: 1px solid #e0e0e0;
}

.content-area {
  display: flex;
  flex: 1;
  min-height: 0;
}

.main-content {
  display: flex;
  flex-direction: column;
  flex: 1;
  min-height: 0;
  padding: 1rem;
}

.sidebar {
  flex-shrink: 0;
  width: 250px;
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  padding: 1rem;
  border-left: 1px solid #e0e0e0;
  background-color: #fcfcfc;
}

.sidebar-button {
  padding: 0.75rem 1rem;
  border: 1px solid #ccc;
  border-radius: 4px;
  background-color: #f8f8f8;
  cursor: pointer;
  text-align: left;
  font-weight: 500;
  transition: all 0.2s;
}

.sidebar-button:hover:not(:disabled) {
  background-color: #e8e8e8;
  border-color: #999;
}

.sidebar-button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.sidebar-section-title {
  margin: 1.5rem 0 0.5rem 0;
  font-size: 0.85rem;
  font-weight: 700;
  color: #888;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.sidebar-section-title:first-child {
  margin-top: 0;
}

.warning-banner {
  padding: 1.5rem;
  background-color: #fff3cd;
  border: 1px solid #ffc107;
  border-radius: 6px;
  color: #856404;
  text-align: center;
}

.frame-viewer {
  display: flex;
  flex-direction: column;
  flex: 1;
  min-height: 0;
  align-items: center;
  justify-content: center;
}

.canvas-container {
  position: relative;
  display: inline-block;
  max-width: 100%;
  max-height: calc(100% - 50px);
}

.frame-canvas,
.overlay-canvas {
  display: block;
  max-width: 100%;
  max-height: 100%;
  object-fit: contain;
}

.overlay-canvas {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  cursor: crosshair;
}

.frame-info {
  margin-top: 1rem;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.9rem;
  color: #666;
}

.label-hint {
  color: #0366d6;
  font-weight: 500;
}

.label-hint.saving {
  color: #856404;
}

.loading-state,
.empty-state,
.error-state,
.idle-state {
  padding: 3rem;
  text-align: center;
  color: #888;
}

.error-state {
  color: #dc3545;
}

.idle-state p {
  margin: 0.5rem 0;
}

.instructions {
  font-size: 0.85rem;
  color: #666;
  padding: 0.5rem;
  background-color: #f8f8f8;
  border-radius: 4px;
}

.instructions p {
  margin: 0.25rem 0;
}

.epochs-input {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 0.5rem;
}

.epochs-input label {
  font-size: 0.9rem;
  color: #666;
}

.epochs-input input {
  width: 80px;
  padding: 0.4rem;
  border: 1px solid #ccc;
  border-radius: 4px;
  font-size: 0.9rem;
}

.label-count {
  font-size: 0.85rem;
  color: #666;
  margin-top: 0.5rem;
}

.error-indicator {
  margin-top: 1rem;
  padding: 0.75rem;
  font-size: 0.85rem;
  color: #dc3545;
  background-color: #f8d7da;
  border-radius: 4px;
}

.start-button {
  background-color: #e3f2fd;
  border-color: #2196f3;
}

.stop-button {
  background-color: #fce4ec;
  border-color: #e91e63;
}

.train-button {
  background-color: #e8f5e9;
  border-color: #4caf50;
}

.apply-button {
  background-color: #fff3e0;
  border-color: #ff9800;
}

.clear-button {
  background-color: #ffebee;
  border-color: #f44336;
  margin-top: 0.5rem;
}
</style>
