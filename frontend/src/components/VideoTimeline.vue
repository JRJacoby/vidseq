<script setup lang="ts">
import { computed, ref, onMounted, onUnmounted } from 'vue'

const props = defineProps<{
  currentTime: number
  duration: number
  isPlaying: boolean
}>()

const emit = defineEmits<{
  seek: [time: number]
  'toggle-play': []
}>()

const isDragging = ref(false)
const timelineRef = ref<HTMLElement | null>(null)
const dragTime = ref<number | null>(null)

const displayProgress = computed(() => {
  const time = dragTime.value !== null ? dragTime.value : props.currentTime
  return props.duration > 0 ? (time / props.duration) * 100 : 0
})

const formatTime = (seconds: number) => {
  const m = Math.floor(seconds / 60)
  const s = Math.floor(seconds % 60)
  return `${m}:${s.toString().padStart(2, '0')}`
}

const ticks = computed(() => {
  if (props.duration === 0) return []
  
  let interval: number
  if (props.duration > 600) interval = 120
  else if (props.duration > 300) interval = 60
  else if (props.duration > 120) interval = 30
  else if (props.duration > 60) interval = 15
  else if (props.duration > 30) interval = 10
  else interval = 5
  
  const result = []
  for (let t = 0; t <= props.duration; t += interval) {
    result.push({
      time: t,
      percent: (t / props.duration) * 100,
      label: formatTime(t)
    })
  }
  if (result[result.length - 1]?.time !== Math.floor(props.duration)) {
    result.push({
      time: props.duration,
      percent: 100,
      label: formatTime(props.duration)
    })
  }
  return result
})

const seekFromEvent = (event: MouseEvent) => {
  if (!timelineRef.value) return
  const rect = timelineRef.value.getBoundingClientRect()
  const clickX = event.clientX - rect.left
  const percent = Math.max(0, Math.min(1, clickX / rect.width))
  const time = percent * props.duration
  dragTime.value = time
  emit('seek', time)
}

const onMouseDown = (event: MouseEvent) => {
  isDragging.value = true
  seekFromEvent(event)
}

const onMouseMove = (event: MouseEvent) => {
  if (isDragging.value) {
    seekFromEvent(event)
  }
}

const onMouseUp = () => {
  isDragging.value = false
  dragTime.value = null
}

onMounted(() => {
  window.addEventListener('mousemove', onMouseMove)
  window.addEventListener('mouseup', onMouseUp)
})

onUnmounted(() => {
  window.removeEventListener('mousemove', onMouseMove)
  window.removeEventListener('mouseup', onMouseUp)
})

const togglePlay = () => {
  emit('toggle-play')
}
</script>

<template>
  <div class="timeline-container">
    <button class="play-button" @click="togglePlay">
      <span v-if="isPlaying" class="pause-icon">❚❚</span>
      <span v-else class="play-icon">▶</span>
    </button>
    
    <div class="timeline-wrapper">
      <div ref="timelineRef" class="timeline-track" @mousedown="onMouseDown">
        <div class="timeline-progress" :style="{ width: displayProgress + '%' }"></div>
        <div class="timeline-playhead" :style="{ left: displayProgress + '%' }"></div>
      </div>
      
      <div class="timeline-ticks">
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
      {{ formatTime(dragTime !== null ? dragTime : currentTime) }} / {{ formatTime(duration) }}
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
  height: 24px;
  background-color: #333;
  border-radius: 4px;
  cursor: pointer;
  overflow: visible;
}

.timeline-progress {
  position: absolute;
  top: 0;
  left: 0;
  height: 100%;
  background-color: #c0392b;
  border-radius: 4px 0 0 4px;
  pointer-events: none;
}

.timeline-playhead {
  position: absolute;
  top: -3px;
  width: 4px;
  height: 30px;
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

