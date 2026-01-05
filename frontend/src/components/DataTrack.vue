<script setup lang="ts">
import { computed, ref, watch, onMounted, onUnmounted } from 'vue'

const props = withDefaults(defineProps<{
  duration: number
  fps: number
  currentTime: number
  viewStart: number
  viewEnd: number
  maskedRanges: [number, number][]
  trainingRanges: [number, number][]
  showMaskedFrames: boolean
  showTrainingFrames: boolean
  showConfidencePlot: boolean
  isMarkingMode: boolean
  confidenceScores: { frame_idx: number; score: number }[]
  // Alignment label frames (individual frame indices)
  alignmentLabelFrames?: number[]
  showAlignmentLabels?: boolean
}>(), {
  alignmentLabelFrames: () => [],
  showAlignmentLabels: false,
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
  
  const zoomFactor = event.deltaY > 0 ? 1.25 : 0.8
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

const maskedRangeStyles = computed(() => {
  if (!props.showMaskedFrames) return []
  return props.maskedRanges
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

const onMouseEnter = () => {
  isHovering.value = true
}

const onMouseLeave = () => {
  isHovering.value = false
}

const drawPlot = () => {
  const canvas = canvasRef.value
  if (!canvas) return
  
  const ctx = canvas.getContext('2d')
  if (!ctx) return
  
  const width = canvas.width
  const height = canvas.height
  
  ctx.clearRect(0, 0, width, height)
  
  if (!props.showConfidencePlot || props.confidenceScores.length === 0) return
  
  // Calculate global min/max for auto-scaling
  let minScore = 2.0 // Initialize higher than max possible (1.0)
  let maxScore = -1.0 // Initialize lower than min possible (0.0)
  
  for (const item of props.confidenceScores) {
    if (item.score >= 0) {
      if (item.score < minScore) minScore = item.score
      if (item.score > maxScore) maxScore = item.score
    }
  }
  
  // Default range if no valid scores or flat line
  if (minScore > 1.0) { // No valid scores found
    minScore = 0.0
    maxScore = 1.0
  } else if (Math.abs(maxScore - minScore) < 0.0001) {
    // Avoid division by zero, center it
    minScore = Math.max(0, minScore - 0.1)
    maxScore = Math.min(1, maxScore + 0.1)
  }
  
  const scoreRange = maxScore - minScore

  ctx.beginPath()
  ctx.strokeStyle = 'rgba(255, 235, 59, 0.8)' // Yellowish
  ctx.lineWidth = 2
  
  let hasStarted = false
  
  const duration = visibleDuration.value
  if (duration <= 0) return

  let lastFrameIdx = -2
  
  for (const item of props.confidenceScores) {
    if (item.score < 0) continue // Skip invalid scores
    
    const time = frameToTime(item.frame_idx)
    
    // Calculate x
    const x = ((time - props.viewStart) / duration) * width
    
    // Normalize score
    const normalizedScore = (item.score - minScore) / scoreRange
    const y = height * (1 - normalizedScore)
    
    // Check for gaps
    // If the gap is more than 1 frame, we should not connect the line
    const isGap = (item.frame_idx - lastFrameIdx) > 1
    
    // Optimization: Skip drawing if way off screen
    if (x < -100 && !hasStarted) {
      // Just update lastFrameIdx
      lastFrameIdx = item.frame_idx
      continue
    }
    if (x > width + 100) break
    
    if (!hasStarted || isGap) {
      ctx.moveTo(x, y)
      hasStarted = true
    } else {
      ctx.lineTo(x, y)
    }
    
    lastFrameIdx = item.frame_idx
  }
  
  ctx.stroke()
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
// Also watch masked/training visibility if we want to change opacity or something? No.

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
      <div class="data-track-label">Data</div>
    </div>
    <div class="timeline-col-center">
      <div 
        ref="trackRef"
        class="data-track"
        :class="{ 'marking-mode': isMarkingMode }"
        @mousedown="onMouseDown"
        @mouseenter="onMouseEnter"
        @mouseleave="onMouseLeave"
        @wheel="onWheel"
      >
        <div 
          v-for="(style, idx) in maskedRangeStyles"
          :key="'masked-' + idx"
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
        
        <div 
          v-if="displayProgress >= 0 && displayProgress <= 100" 
          class="data-track-playhead" 
          :style="{ left: displayProgress + '%' }"
        />
      </div>
    </div>
    <div class="timeline-col-right">
      <!-- Spacer to align with time display -->
    </div>
  </div>
</template>

<style scoped>
/* UI columns for TimelineSystem */


.timeline-col-left {
  height: 120px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.data-track-label {
  font-size: 10px;
  color: #888;
  text-transform: uppercase;
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
