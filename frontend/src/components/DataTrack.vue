<script setup lang="ts">
import { computed, ref, watch, onMounted, onUnmounted } from 'vue'

const props = withDefaults(defineProps<{
  duration: number
  fps: number
  currentTime: number
  viewStart: number
  viewEnd: number
  trackerMaskedRanges: [number, number][]
  trainingRanges: [number, number][]
  showTrackerMaskedFrames: boolean
  showTrainingFrames: boolean
  showConfidencePlot: boolean
  isMarkingMode: boolean
  confidenceScores: { frame_idx: number; score: number }[]
  showDetectorConfidence?: boolean
  detectorScores?: { frame_idx: number; score: number }[]
  // Alignment label frames (individual frame indices)
  alignmentLabelFrames?: number[]
  showAlignmentLabels?: boolean
  // PCA scores (keyed by PC index string)
  pcaScores?: Record<string, { frame_idx: number; score: number }[]>
  visiblePCs?: number[]
  showPCAPlot?: boolean
}>(), {
  alignmentLabelFrames: () => [],
  showAlignmentLabels: false,
  pcaScores: () => ({}),
  visiblePCs: () => [],
  showPCAPlot: false,
  showDetectorConfidence: false,
  detectorScores: () => [],
})

const emit = defineEmits<{
  'mark-training': [startFrame: number, endFrame: number]
  'unmark-training': [startFrame: number, endFrame: number]
  'view-change': [viewStart: number, viewEnd: number]
}>()

const trackRef = ref<HTMLElement | null>(null)
const canvasRef = ref<HTMLCanvasElement | null>(null)
const isDragging = ref(false)
const dragStartFrame = ref<number | null>(null)
const dragEndFrame = ref<number | null>(null)
const selectedRange = ref<[number, number] | null>(null)
const isHovering = ref(false)
const plotMinScore = ref<number | null>(null)
const plotMaxScore = ref<number | null>(null)
const detectorMinScore = ref<number | null>(null)
const detectorMaxScore = ref<number | null>(null)

const visibleDuration = computed(() => props.viewEnd - props.viewStart)

const displayProgress = computed(() => {
  if (visibleDuration.value <= 0) return 0
  return ((props.currentTime - props.viewStart) / visibleDuration.value) * 100
})

const timeToFrame = (time: number): number => {
  return Math.floor(time * props.fps)
}

const frameToTime = (frame: number): number => {
  return frame / props.fps
}

const MIN_VISIBLE_DURATION = 1

const onWheel = (event: WheelEvent) => {
  event.preventDefault()
  if (!trackRef.value) return
  
  const rect = trackRef.value.getBoundingClientRect()
  const mouseX = event.clientX - rect.left
  const mousePercent = mouseX / rect.width
  
  const timeAtMouse = props.viewStart + mousePercent * visibleDuration.value
  
  const clampedDelta = Math.sign(event.deltaY) * Math.min(Math.abs(event.deltaY), 50)
  const zoomFactor = 1 + clampedDelta * 0.002
  let newDuration = visibleDuration.value * zoomFactor
  newDuration = Math.max(MIN_VISIBLE_DURATION, Math.min(props.duration, newDuration))
  
  let newStart = timeAtMouse - mousePercent * newDuration
  let newEnd = timeAtMouse + (1 - mousePercent) * newDuration
  
  if (newStart < 0) {
    newStart = 0
    newEnd = newDuration
  }
  if (newEnd > props.duration) {
    newEnd = props.duration
    newStart = props.duration - newDuration
  }
  
  emit('view-change', newStart, newEnd)
}

const percentToTime = (percent: number): number => {
  return props.viewStart + (percent / 100) * visibleDuration.value
}

const timeToPercent = (time: number): number => {
  return ((time - props.viewStart) / visibleDuration.value) * 100
}

const frameToPercent = (frame: number): number => {
  return timeToPercent(frameToTime(frame))
}

const rangeToStyle = (range: [number, number]): { left: string; width: string } | null => {
  const startPercent = frameToPercent(range[0])
  const endPercent = frameToPercent(range[1] + 1)
  
  if (endPercent < 0 || startPercent > 100) return null
  
  const clampedStart = Math.max(0, startPercent)
  const clampedEnd = Math.min(100, endPercent)
  
  return {
    left: `${clampedStart}%`,
    width: `${clampedEnd - clampedStart}%`,
  }
}

const trackerMaskedRangeStyles = computed(() => {
  if (!props.showTrackerMaskedFrames) return []
  return props.trackerMaskedRanges
    .map(r => rangeToStyle(r))
    .filter((s): s is { left: string; width: string } => s !== null)
})

const trainingRangeStyles = computed(() => {
  if (!props.showTrainingFrames) return []
  return props.trainingRanges
    .map(r => ({ range: r, style: rangeToStyle(r) }))
    .filter((item): item is { range: [number, number]; style: { left: string; width: string } } => item.style !== null)
})

const dragRangeStyle = computed(() => {
  if (dragStartFrame.value === null || dragEndFrame.value === null) return null
  const start = Math.min(dragStartFrame.value, dragEndFrame.value)
  const end = Math.max(dragStartFrame.value, dragEndFrame.value)
  return rangeToStyle([start, end])
})

const selectedRangeStyle = computed(() => {
  if (!selectedRange.value) return null
  return rangeToStyle(selectedRange.value)
})

const alignmentLabelStyles = computed(() => {
  if (!props.showAlignmentLabels) return []
  return props.alignmentLabelFrames
    .map(frameIdx => {
      const percent = frameToPercent(frameIdx)
      if (percent < 0 || percent > 100) return null
      return { frameIdx, left: `${percent}%` }
    })
    .filter((s): s is { frameIdx: number; left: string } => s !== null)
})

const getFrameFromEvent = (event: MouseEvent): number => {
  if (!trackRef.value) return 0
  const rect = trackRef.value.getBoundingClientRect()
  const clickX = event.clientX - rect.left
  const percent = Math.max(0, Math.min(100, (clickX / rect.width) * 100))
  const time = percentToTime(percent)
  return timeToFrame(time)
}

const findTrainingRangeAt = (frame: number): [number, number] | null => {
  for (const range of props.trainingRanges) {
    if (frame >= range[0] && frame <= range[1]) {
      return range
    }
  }
  return null
}

const onMouseDown = (event: MouseEvent) => {
  const frame = getFrameFromEvent(event)
  const clickedRange = findTrainingRangeAt(frame)
  
  if (clickedRange) {
    selectedRange.value = clickedRange
    return
  }
  
  if (!props.isMarkingMode) {
    selectedRange.value = null
    return
  }
  
  selectedRange.value = null
  isDragging.value = true
  dragStartFrame.value = frame
  dragEndFrame.value = frame
}

const onMouseMove = (event: MouseEvent) => {
  if (!isDragging.value) return
  dragEndFrame.value = getFrameFromEvent(event)
}

const onMouseUp = () => {
  if (!isDragging.value) return
  
  if (dragStartFrame.value !== null && dragEndFrame.value !== null) {
    const start = Math.min(dragStartFrame.value, dragEndFrame.value)
    const end = Math.max(dragStartFrame.value, dragEndFrame.value)
    emit('mark-training', start, end)
  }
  
  isDragging.value = false
  dragStartFrame.value = null
  dragEndFrame.value = null
}

const onKeyDown = (event: KeyboardEvent) => {
  if (!isHovering.value) return
  
  if ((event.key === 'Delete' || event.key === 'Backspace') && selectedRange.value) {
    event.preventDefault()
    emit('unmark-training', selectedRange.value[0], selectedRange.value[1])
    selectedRange.value = null
  }
}

const hoverX = ref<number | null>(null)
const hoverFrame = ref<number | null>(null)

const hoverScores = computed(() => {
  if (hoverFrame.value === null) return []
  const frame = hoverFrame.value
  const results: { label: string; value: number; color: string }[] = []

  // Find nearest tracker confidence score
  if (props.showConfidencePlot && props.confidenceScores.length > 0) {
    const nearest = findNearestScore(props.confidenceScores, frame)
    if (nearest !== null) {
      results.push({ label: 'Tracker', value: nearest.score, color: 'rgba(255, 235, 59, 0.8)' })
    }
  }

  // Find nearest detector confidence score
  if (props.showDetectorConfidence && props.detectorScores.length > 0) {
    const nearest = findNearestScore(props.detectorScores, frame)
    if (nearest !== null) {
      results.push({ label: 'Detector', value: nearest.score, color: 'rgba(255, 99, 71, 0.8)' })
    }
  }

  return results
})

const findNearestScore = (
  scores: { frame_idx: number; score: number }[],
  frame: number,
): { frame_idx: number; score: number } | null => {
  if (scores.length === 0) return null

  // Binary search for closest frame
  let lo = 0
  let hi = scores.length - 1
  while (lo < hi) {
    const mid = (lo + hi) >> 1
    if (scores[mid].frame_idx < frame) lo = mid + 1
    else hi = mid
  }

  // Check lo and lo-1, pick closest
  let best = scores[lo]
  if (lo > 0) {
    const prev = scores[lo - 1]
    if (Math.abs(prev.frame_idx - frame) < Math.abs(best.frame_idx - frame)) {
      best = prev
    }
  }

  // Only show if within reasonable range (half the visible frames)
  const visibleFrames = visibleDuration.value * props.fps
  if (Math.abs(best.frame_idx - frame) > visibleFrames / 2) return null
  if (best.score < 0) return null

  return best
}

const onTrackMouseMove = (event: MouseEvent) => {
  if (!trackRef.value) return
  const rect = trackRef.value.getBoundingClientRect()
  hoverX.value = event.clientX - rect.left
  hoverFrame.value = getFrameFromEvent(event)
}

const onMouseEnter = () => {
  isHovering.value = true
}

const onMouseLeave = () => {
  isHovering.value = false
  hoverX.value = null
  hoverFrame.value = null
}

// Color palette for PC score lines
const PC_COLORS = [
  'rgba(59, 130, 246, 0.8)',   // Blue - PC0
  'rgba(239, 68, 68, 0.8)',    // Red - PC1
  'rgba(34, 197, 94, 0.8)',    // Green - PC2
  'rgba(168, 85, 247, 0.8)',   // Purple - PC3
  'rgba(249, 115, 22, 0.8)',   // Orange - PC4
  'rgba(236, 72, 153, 0.8)',   // Pink - PC5
  'rgba(20, 184, 166, 0.8)',   // Teal - PC6
  'rgba(234, 179, 8, 0.8)',    // Yellow - PC7
]

const getPCColor = (pcIdx: number): string => {
  return PC_COLORS[pcIdx % PC_COLORS.length] ?? '#888888'
}

// Draw a single score line on the canvas. Returns {min, max} if drawn.
const drawScoreLine = (
  ctx: CanvasRenderingContext2D,
  scores: { frame_idx: number; score: number }[],
  color: string,
  width: number,
  height: number,
): { min: number; max: number } | null => {
  if (scores.length === 0) return null

  // Calculate min/max for auto-scaling
  let minScore = Infinity
  let maxScore = -Infinity

  for (const item of scores) {
    if (item.score < minScore) minScore = item.score
    if (item.score > maxScore) maxScore = item.score
  }

  // Handle edge cases
  if (!isFinite(minScore) || !isFinite(maxScore)) return null
  if (Math.abs(maxScore - minScore) < 0.0001) {
    // Flat line, expand range
    minScore = minScore - 1
    maxScore = maxScore + 1
  }

  const scoreRange = maxScore - minScore
  const duration = visibleDuration.value
  if (duration <= 0) return null

  ctx.beginPath()
  ctx.strokeStyle = color
  ctx.lineWidth = 2

  let hasStarted = false
  let lastFrameIdx = -2

  for (const item of scores) {
    const time = frameToTime(item.frame_idx)
    const x = ((time - props.viewStart) / duration) * width
    const normalizedScore = (item.score - minScore) / scoreRange
    const y = height * (1 - normalizedScore)

    // Check for gaps
    const isGap = (item.frame_idx - lastFrameIdx) > 1

    // Optimization: Skip drawing if way off screen
    if (x < -100 && !hasStarted) {
      lastFrameIdx = item.frame_idx
      continue
    }
    if (x > width + 100) break

    if (!hasStarted || isGap) {
      // Draw a dot at isolated points (moveTo alone is invisible)
      ctx.fillStyle = color
      ctx.fillRect(x - 1.5, y - 1.5, 3, 3)
      ctx.moveTo(x, y)
      hasStarted = true
    } else {
      ctx.lineTo(x, y)
    }

    lastFrameIdx = item.frame_idx
  }

  ctx.stroke()
  return { min: minScore, max: maxScore }
}

const drawPlot = () => {
  const canvas = canvasRef.value
  if (!canvas) return

  const ctx = canvas.getContext('2d')
  if (!ctx) return

  const width = canvas.width
  const height = canvas.height

  ctx.clearRect(0, 0, width, height)

  let activeRange: { min: number; max: number } | null = null

  // Draw confidence scores (yellow line)
  if (props.showConfidencePlot && props.confidenceScores.length > 0) {
    // Filter out invalid scores (score < 0)
    const validScores = props.confidenceScores.filter(s => s.score >= 0)
    if (validScores.length > 0) {
      activeRange = drawScoreLine(ctx, validScores, 'rgba(255, 235, 59, 0.8)', width, height)
    }
  }

  // Draw detector confidence scores (tomato line)
  let detectorRange: { min: number; max: number } | null = null
  if (props.showDetectorConfidence && props.detectorScores.length > 0) {
    const validScores = props.detectorScores.filter(s => s.score >= 0)
    if (validScores.length > 0) {
      detectorRange = drawScoreLine(ctx, validScores, 'rgba(255, 99, 71, 0.8)', width, height)
      if (detectorRange && !activeRange) activeRange = detectorRange
    }
  }

  // Draw PCA scores (multiple colored lines)
  if (props.showPCAPlot && props.pcaScores && props.visiblePCs.length > 0) {
    for (const pcIdx of props.visiblePCs) {
      const scores = props.pcaScores[pcIdx.toString()]
      if (scores && scores.length > 0) {
        const color = getPCColor(pcIdx)
        const range = drawScoreLine(ctx, scores, color, width, height)
        if (range && !activeRange) activeRange = range
      }
    }
  }

  // Update y-axis labels
  plotMinScore.value = activeRange?.min ?? null
  plotMaxScore.value = activeRange?.max ?? null
  detectorMinScore.value = detectorRange?.min ?? null
  detectorMaxScore.value = detectorRange?.max ?? null
}

const resizeCanvas = () => {
  const canvas = canvasRef.value
  const track = trackRef.value
  if (canvas && track) {
    canvas.width = track.clientWidth
    canvas.height = track.clientHeight
    drawPlot()
  }
}

watch(() => props.confidenceScores, drawPlot, { deep: true })
watch([() => props.viewStart, () => props.viewEnd], drawPlot)
watch(() => props.showConfidencePlot, drawPlot)
watch(() => props.detectorScores, drawPlot, { deep: true })
watch(() => props.showDetectorConfidence, drawPlot)
watch(() => props.pcaScores, drawPlot, { deep: true })
watch(() => props.visiblePCs, drawPlot, { deep: true })
watch(() => props.showPCAPlot, drawPlot)

onMounted(() => {
  window.addEventListener('mousemove', onMouseMove)
  window.addEventListener('mouseup', onMouseUp)
  window.addEventListener('keydown', onKeyDown)
  window.addEventListener('resize', resizeCanvas)
  // Initial draw
  setTimeout(resizeCanvas, 10)
})

onUnmounted(() => {
  window.removeEventListener('mousemove', onMouseMove)
  window.removeEventListener('mouseup', onMouseUp)
  window.removeEventListener('keydown', onKeyDown)
  window.removeEventListener('resize', resizeCanvas)
})
</script>

<template>
  <div class="timeline-row">
    <div class="timeline-col-left">
      <div class="data-track-label-col">
        <span v-if="plotMaxScore !== null" class="y-axis-label top">{{ plotMaxScore.toFixed(2) }}</span>
        <span class="data-track-label">Data</span>
        <span v-if="plotMinScore !== null" class="y-axis-label bottom">{{ plotMinScore.toFixed(2) }}</span>
      </div>
    </div>
    <div class="timeline-col-center">
      <div
        ref="trackRef"
        class="data-track"
        :class="{ 'marking-mode': isMarkingMode }"
        @mousedown="onMouseDown"
        @mouseenter="onMouseEnter"
        @mouseleave="onMouseLeave"
        @mousemove="onTrackMouseMove"
        @wheel="onWheel"
      >
        <div
          v-for="(style, idx) in trackerMaskedRangeStyles"
          :key="'tracker-masked-' + idx"
          class="range-overlay masked"
          :style="style"
        />
        
        <canvas
          ref="canvasRef"
          class="confidence-plot"
        />
        
        <div
          v-for="item in trainingRangeStyles"
          :key="'training-' + item.range[0]"
          class="range-overlay training"
          :class="{ selected: selectedRange && selectedRange[0] === item.range[0] && selectedRange[1] === item.range[1] }"
          :style="item.style"
        />

        <div
          v-for="item in alignmentLabelStyles"
          :key="'alignment-' + item.frameIdx"
          class="alignment-label-tick"
          :style="{ left: item.left }"
        />

        <div
          v-if="dragRangeStyle"
          class="range-overlay drag-selection"
          :style="dragRangeStyle"
        />
        
        <!-- Hover crosshair and tooltip -->
        <template v-if="hoverX !== null && hoverScores.length > 0">
          <div class="hover-crosshair" :style="{ left: hoverX + 'px' }" />
          <div class="hover-tooltip" :style="{ left: hoverX + 'px' }">
            <div class="hover-frame">F{{ hoverFrame }}</div>
            <div v-for="s in hoverScores" :key="s.label" class="hover-score-row">
              <span class="hover-dot" :style="{ backgroundColor: s.color }" />
              <span>{{ s.value.toFixed(2) }}</span>
            </div>
          </div>
        </template>

        <div
          v-if="displayProgress >= 0 && displayProgress <= 100"
          class="data-track-playhead"
          :style="{ left: displayProgress + '%' }"
        />
      </div>
    </div>
    <div class="timeline-col-right">
      <div v-if="detectorMaxScore !== null" class="data-track-label-col">
        <span class="y-axis-label top detector">{{ detectorMaxScore.toFixed(1) }}</span>
        <span class="y-axis-label bottom detector">{{ detectorMinScore?.toFixed(1) }}</span>
      </div>
    </div>
  </div>
</template>

<style scoped>
/* UI columns for TimelineSystem */


.timeline-col-left,
.timeline-col-right {
  height: 120px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.data-track-label-col {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: space-between;
  height: 100%;
  padding: 4px 0;
  box-sizing: border-box;
}

.data-track-label {
  font-size: 10px;
  color: #888;
  text-transform: uppercase;
}

.y-axis-label {
  font-size: 9px;
  color: #aaa;
  font-family: monospace;
  line-height: 1;
}

.y-axis-label.detector {
  color: rgba(255, 99, 71, 0.8);
}

.data-track {
  flex: 1;
  position: relative;
  height: 120px;
  background-color: #2a2a2a;
  border-radius: 4px;
  cursor: pointer;
  overflow: hidden;
}

.data-track.marking-mode {
  cursor: crosshair;
}

.range-overlay {
  position: absolute;
  top: 0;
  height: 100%;
  pointer-events: none;
}

.confidence-plot {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
  z-index: 5; /* Above masked regions, below drag/playhead */
}

.range-overlay.masked {
  background-color: rgba(59, 130, 246, 0.3);
}

.range-overlay.training {
  background-color: rgba(34, 197, 94, 0.5);
  pointer-events: auto;
  cursor: pointer;
}

.range-overlay.training.selected {
  background-color: rgba(34, 197, 94, 0.7);
  outline: 2px dashed rgba(34, 197, 94, 1);
  outline-offset: -2px;
  animation: pulse 1s ease-in-out infinite;
}

@keyframes pulse {
  0%, 100% {
    outline-color: rgba(34, 197, 94, 1);
  }
  50% {
    outline-color: rgba(34, 197, 94, 0.5);
  }
}

.range-overlay.drag-selection {
  background-color: rgba(34, 197, 94, 0.2);
  border: 2px dashed rgba(34, 197, 94, 0.8);
  box-sizing: border-box;
}

.data-track-playhead {
  position: absolute;
  top: 0;
  width: 2px;
  height: 100%;
  background-color: #e74c3c;
  opacity: 0.7;
  transform: translateX(-50%);
  pointer-events: none;
}

.hover-crosshair {
  position: absolute;
  top: 0;
  width: 1px;
  height: 100%;
  background-color: rgba(255, 255, 255, 0.3);
  pointer-events: none;
  z-index: 5;
}

.hover-tooltip {
  position: absolute;
  top: 4px;
  transform: translateX(-50%);
  background: rgba(30, 30, 30, 0.95);
  border: 1px solid #555;
  border-radius: 4px;
  padding: 4px 8px;
  pointer-events: none;
  z-index: 10;
  white-space: nowrap;
  font-size: 11px;
  font-family: monospace;
  color: #ddd;
}

.hover-frame {
  color: #999;
  font-size: 10px;
  margin-bottom: 2px;
}

.hover-score-row {
  display: flex;
  align-items: center;
  gap: 4px;
}

.hover-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  flex-shrink: 0;
}

.alignment-label-tick {
  position: absolute;
  top: 0;
  width: 3px;
  height: 100%;
  background-color: rgba(168, 85, 247, 0.8); /* Purple */
  transform: translateX(-50%);
  pointer-events: none;
  z-index: 4;
}
</style>
