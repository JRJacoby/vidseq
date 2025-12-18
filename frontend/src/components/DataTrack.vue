<script setup lang="ts">
import { computed, ref, watch, onMounted, onUnmounted } from 'vue'

const props = defineProps<{
  duration: number
  fps: number
  currentTime: number
  viewStart: number
  viewEnd: number
  maskedRanges: [number, number][]
  trainingRanges: [number, number][]
  showMaskedFrames: boolean
  showTrainingFrames: boolean
  isMarkingMode: boolean
}>()

const emit = defineEmits<{
  'mark-training': [startFrame: number, endFrame: number]
  'unmark-training': [startFrame: number, endFrame: number]
  'view-change': [viewStart: number, viewEnd: number]
}>()

const trackRef = ref<HTMLElement | null>(null)
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

onMounted(() => {
  window.addEventListener('mousemove', onMouseMove)
  window.addEventListener('mouseup', onMouseUp)
  window.addEventListener('keydown', onKeyDown)
})

onUnmounted(() => {
  window.removeEventListener('mousemove', onMouseMove)
  window.removeEventListener('mouseup', onMouseUp)
  window.removeEventListener('keydown', onKeyDown)
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
        
        <div 
          v-for="item in trainingRangeStyles"
          :key="'training-' + item.range[0]"
          class="range-overlay training"
          :class="{ selected: selectedRange && selectedRange[0] === item.range[0] && selectedRange[1] === item.range[1] }"
          :style="item.style"
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


.data-track-label {
  font-size: 10px;
  color: #888;
  text-transform: uppercase;
  padding-top: 8px;
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
</style>
