<script setup lang="ts">
import { computed, ref, watch, onMounted, onUnmounted } from 'vue'

const props = defineProps<{
  currentTime: number
  duration: number
  isPlaying: boolean
  fps: number
}>()

const emit = defineEmits<{
  seek: [time: number]
  'toggle-play': []
}>()

const isDragging = ref(false)
const isHovering = ref(false)
const isPanning = ref(false)
const panStartX = ref(0)
const panStartViewStart = ref(0)
const timelineRef = ref<HTMLElement | null>(null)
const seekTarget = ref<number | null>(null)  // Timeline's own target for smooth display

const frameDuration = computed(() => 1 / props.fps)

// Clear seekTarget when video catches up
watch(() => props.currentTime, (videoTime) => {
  if (seekTarget.value !== null && Math.abs(videoTime - seekTarget.value) < 0.01) {
    seekTarget.value = null
  }
})

const viewStart = ref(0)
const viewEnd = ref(0)
const MIN_VISIBLE_DURATION = 1

watch(() => props.duration, (d) => {
  viewStart.value = 0
  viewEnd.value = d
}, { immediate: true })

const visibleDuration = computed(() => viewEnd.value - viewStart.value)

const displayProgress = computed(() => {
  // Use seekTarget for smooth display, fall back to video's actual time
  const time = seekTarget.value !== null ? seekTarget.value : props.currentTime
  if (visibleDuration.value <= 0) return 0
  return ((time - viewStart.value) / visibleDuration.value) * 100
})

const formatTime = (seconds: number) => {
  const h = Math.floor(seconds / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  const s = Math.floor(seconds % 60)
  if (h > 0) {
    return `${h}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`
  }
  return `${m}:${s.toString().padStart(2, '0')}`
}

const NICE_INTERVALS = [
  1, 2, 5, 10, 15, 30,
  60, 120, 300, 600, 900, 1800,
  3600, 7200, 14400, 21600,
  43200, 86400
]

const getNiceInterval = (duration: number, targetTicks: number = 8) => {
  const rawInterval = duration / targetTicks
  return NICE_INTERVALS.find(i => i >= rawInterval) ?? NICE_INTERVALS.at(-1)!
}

const ticks = computed(() => {
  if (visibleDuration.value === 0) return []
  
  const interval = getNiceInterval(visibleDuration.value)
  const firstTick = Math.ceil(viewStart.value / interval) * interval
  
  const result = []
  for (let t = firstTick; t <= viewEnd.value; t += interval) {
    result.push({
      time: t,
      percent: ((t - viewStart.value) / visibleDuration.value) * 100,
      label: formatTime(t)
    })
  }
  return result
})

const seekFromEvent = (event: MouseEvent) => {
  if (!timelineRef.value) return
  const rect = timelineRef.value.getBoundingClientRect()
  const clickX = event.clientX - rect.left
  const percent = Math.max(0, Math.min(1, clickX / rect.width))
  const time = viewStart.value + percent * visibleDuration.value
  seekTarget.value = time
  emit('seek', time)
}

const onMouseDown = (event: MouseEvent) => {
  isDragging.value = true
  seekFromEvent(event)
}

const onMouseMove = (event: MouseEvent) => {
  if (isDragging.value) {
    seekFromEvent(event)
  } else if (isPanning.value && timelineRef.value) {
    const rect = timelineRef.value.getBoundingClientRect()
    const deltaX = event.clientX - panStartX.value
    const deltaTime = (deltaX / rect.width) * visibleDuration.value
    
    let newStart = panStartViewStart.value - deltaTime
    let newEnd = newStart + visibleDuration.value
    
    if (newStart < 0) {
      newStart = 0
      newEnd = visibleDuration.value
    }
    if (newEnd > props.duration) {
      newEnd = props.duration
      newStart = props.duration - visibleDuration.value
    }
    
    viewStart.value = newStart
    viewEnd.value = newEnd
  }
}

const onMouseUp = () => {
  isDragging.value = false
  isPanning.value = false
  // Don't clear seekTarget here - it clears automatically when video catches up
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

const togglePlay = () => {
  emit('toggle-play')
}

const onMouseEnter = () => {
  isHovering.value = true
}

const onMouseLeave = () => {
  isHovering.value = false
}

const onTicksMouseDown = (event: MouseEvent) => {
  isPanning.value = true
  panStartX.value = event.clientX
  panStartViewStart.value = viewStart.value
}

const onKeyDown = (event: KeyboardEvent) => {
  if (!isHovering.value) return
  
  // Use seekTarget if we have one (for consecutive arrow presses), otherwise use video time
  const baseTime = seekTarget.value !== null ? seekTarget.value : props.currentTime
  
  if (event.key === 'ArrowLeft') {
    event.preventDefault()
    const newTime = Math.max(0, baseTime - frameDuration.value)
    seekTarget.value = newTime
    emit('seek', newTime)
  } else if (event.key === 'ArrowRight') {
    event.preventDefault()
    const newTime = Math.min(props.duration, baseTime + frameDuration.value)
    seekTarget.value = newTime
    emit('seek', newTime)
  }
}

const onWheel = (event: WheelEvent) => {
  event.preventDefault()
  if (!timelineRef.value) return
  
  const rect = timelineRef.value.getBoundingClientRect()
  const mouseX = event.clientX - rect.left
  const mousePercent = mouseX / rect.width
  
  const timeAtMouse = viewStart.value + mousePercent * visibleDuration.value
  
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
  
  viewStart.value = newStart
  viewEnd.value = newEnd
}
</script>

<template>
  <div class="timeline-container">
    <button class="play-button" @click="togglePlay">
      <span v-if="isPlaying" class="pause-icon">❚❚</span>
      <span v-else class="play-icon">▶</span>
    </button>
    
    <div class="timeline-wrapper">
      <div 
        ref="timelineRef" 
        class="timeline-track" 
        @mousedown="onMouseDown" 
        @wheel="onWheel"
        @mouseenter="onMouseEnter"
        @mouseleave="onMouseLeave"
      >
        <div 
          v-if="displayProgress >= 0 && displayProgress <= 100" 
          class="timeline-playhead" 
          :style="{ left: displayProgress + '%' }"
        ></div>
      </div>
      
      <div class="timeline-ticks" @mousedown="onTicksMouseDown" @wheel="onWheel">
        <span 
          v-for="tick in ticks" 
          :key="tick.time" 
          class="tick-label"
          :style="{ left: tick.percent + '%' }"
        >
          {{ tick.label }}
        </span>
      </div>
    </div>
    
    <div class="time-display">
      {{ formatTime(seekTarget !== null ? seekTarget : currentTime) }} / {{ formatTime(duration) }}
    </div>
  </div>
</template>

<style scoped>
.timeline-container {
  display: flex;
  align-items: flex-start;
  gap: 12px;
  padding: 12px 16px;
  background-color: #1a1a1a;
  border-radius: 6px;
  width: 100%;
  box-sizing: border-box;
}

.play-button {
  flex-shrink: 0;
  width: 40px;
  height: 40px;
  border: none;
  border-radius: 50%;
  background-color: #333;
  color: #fff;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 14px;
  transition: background-color 0.15s;
}

.play-button:hover {
  background-color: #444;
}

.play-icon {
  margin-left: 2px;
}

.pause-icon {
  font-size: 12px;
  letter-spacing: 2px;
}

.timeline-wrapper {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.timeline-track {
  position: relative;
  height: 48px;
  background-color: #333;
  border-radius: 4px;
  cursor: pointer;
  overflow: visible;
}

.timeline-playhead {
  position: absolute;
  top: -3px;
  width: 4px;
  height: 54px;
  background-color: #e74c3c;
  border-radius: 2px;
  transform: translateX(-50%);
  pointer-events: none;
  box-shadow: 0 0 4px rgba(231, 76, 60, 0.5);
}

.timeline-ticks {
  position: relative;
  height: 18px;
  font-size: 11px;
  color: #888;
  user-select: none;
  cursor: grab;
}

.timeline-ticks:active {
  cursor: grabbing;
}

.tick-label {
  position: absolute;
  transform: translateX(-50%);
  white-space: nowrap;
}

.tick-label:first-child {
  transform: translateX(0);
}

.tick-label:last-child {
  transform: translateX(-100%);
}

.time-display {
  flex-shrink: 0;
  font-size: 12px;
  color: #aaa;
  font-family: monospace;
  min-width: 90px;
  text-align: right;
  padding-top: 4px;
}
</style>

