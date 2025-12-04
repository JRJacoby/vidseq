<script setup lang="ts">
import { ref, watch, onMounted, onUnmounted } from 'vue'
import type { StoredPrompt } from '@/composables/usePromptStorage'

export type ToolType = 'none' | 'bbox' | 'positive_point' | 'negative_point'

const props = defineProps<{
  videoWidth: number
  videoHeight: number
  activeTool: ToolType
  mask: ImageBitmap | null
  prompts: StoredPrompt[]
}>()

const emit = defineEmits<{
  (e: 'bbox-complete', bbox: { x1: number; y1: number; x2: number; y2: number }): void
  (e: 'point-complete', point: { x: number; y: number; type: 'positive_point' | 'negative_point' }): void
}>()

const canvasRef = ref<HTMLCanvasElement | null>(null)
const isDrawing = ref(false)
const startPoint = ref<{ x: number; y: number } | null>(null)
const currentPoint = ref<{ x: number; y: number } | null>(null)
const pendingPoint = ref<{ x: number; y: number; type: 'positive_point' | 'negative_point' } | null>(null)

function getNormalizedCoords(event: MouseEvent): { x: number; y: number } | null {
  const canvas = canvasRef.value
  if (!canvas) return null
  
  const rect = canvas.getBoundingClientRect()
  const x = (event.clientX - rect.left) / rect.width
  const y = (event.clientY - rect.top) / rect.height
  
  if (x < 0 || x > 1 || y < 0 || y > 1) {
    return null
  }
  
  return { x, y }
}

function onMouseDown(event: MouseEvent) {
  const coords = getNormalizedCoords(event)
  if (!coords) return
  
  // Handle point tools - single click
  if (props.activeTool === 'positive_point' || props.activeTool === 'negative_point') {
    pendingPoint.value = { x: coords.x, y: coords.y, type: props.activeTool }
    render()
    emit('point-complete', { x: coords.x, y: coords.y, type: props.activeTool })
    return
  }
  
  // Handle bbox tool - drag
  if (props.activeTool !== 'bbox') return
  
  isDrawing.value = true
  startPoint.value = coords
  currentPoint.value = coords
}

function onMouseMove(event: MouseEvent) {
  if (!isDrawing.value || props.activeTool !== 'bbox') return
  
  const coords = getNormalizedCoords(event)
  if (!coords) return
  
  currentPoint.value = coords
  render()
}

function onMouseUp(event: MouseEvent) {
  if (!isDrawing.value || props.activeTool !== 'bbox') return
  
  const coords = getNormalizedCoords(event)
  if (!coords || !startPoint.value) {
    isDrawing.value = false
    startPoint.value = null
    currentPoint.value = null
    render()
    return
  }
  
  const x1 = Math.min(startPoint.value.x, coords.x)
  const y1 = Math.min(startPoint.value.y, coords.y)
  const x2 = Math.max(startPoint.value.x, coords.x)
  const y2 = Math.max(startPoint.value.y, coords.y)
  
  const minSize = 0.01
  if (x2 - x1 > minSize && y2 - y1 > minSize) {
    emit('bbox-complete', { x1, y1, x2, y2 })
  }
  
  isDrawing.value = false
  startPoint.value = null
  currentPoint.value = null
  render()
}

function render() {
  const canvas = canvasRef.value
  if (!canvas) return
  
  const ctx = canvas.getContext('2d', { willReadFrequently: true })
  if (!ctx) return
  
  ctx.clearRect(0, 0, canvas.width, canvas.height)
  
  if (props.mask) {
    // Draw mask to get pixel data
    ctx.drawImage(props.mask, 0, 0, canvas.width, canvas.height)
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height)
    const data = imageData.data
    
    // Convert: where mask > 0, set to semi-transparent blue; else transparent
    for (let i = 0; i < data.length; i += 4) {
      const maskValue = data[i] // grayscale, R=G=B
      if (maskValue > 0) {
        data[i] = 102       // R (0x66)
        data[i + 1] = 179   // G (0xb3)  
        data[i + 2] = 255   // B (0xff)
        data[i + 3] = 102   // A (~0.4 opacity)
      } else {
        data[i + 3] = 0     // Fully transparent
      }
    }
    
    ctx.putImageData(imageData, 0, 0)
  }
  
  ctx.strokeStyle = '#ffcc00'
  ctx.lineWidth = 2
  ctx.setLineDash([6, 4])
  
  for (const prompt of props.prompts) {
    if (prompt.type === 'bbox') {
      const { x1, y1, x2, y2 } = prompt.details
      ctx.strokeRect(
        x1 * canvas.width, 
        y1 * canvas.height, 
        (x2 - x1) * canvas.width, 
        (y2 - y1) * canvas.height
      )
    } else if (prompt.type === 'positive_point' || prompt.type === 'negative_point') {
      const { x, y } = prompt.details
      const px = x * canvas.width
      const py = y * canvas.height
      const radius = 8
      
      // Draw point circle
      ctx.beginPath()
      ctx.arc(px, py, radius, 0, Math.PI * 2)
      ctx.fillStyle = prompt.type === 'positive_point' ? '#22c55e' : '#ef4444'
      ctx.fill()
      ctx.strokeStyle = '#fff'
      ctx.lineWidth = 2
      ctx.setLineDash([])
      ctx.stroke()
      
      // Draw + or - symbol inside
      ctx.strokeStyle = '#fff'
      ctx.lineWidth = 2
      ctx.beginPath()
      // Horizontal line (for both + and -)
      ctx.moveTo(px - 4, py)
      ctx.lineTo(px + 4, py)
      if (prompt.type === 'positive_point') {
        // Vertical line (for + only)
        ctx.moveTo(px, py - 4)
        ctx.lineTo(px, py + 4)
      }
      ctx.stroke()
      
      // Reset dash for next bbox
      ctx.setLineDash([6, 4])
    }
  }
  
  if (isDrawing.value && startPoint.value && currentPoint.value) {
    ctx.strokeStyle = '#00ff00'
    ctx.lineWidth = 2
    ctx.setLineDash([])
    
    const x = Math.min(startPoint.value.x, currentPoint.value.x) * canvas.width
    const y = Math.min(startPoint.value.y, currentPoint.value.y) * canvas.height
    const w = Math.abs(currentPoint.value.x - startPoint.value.x) * canvas.width
    const h = Math.abs(currentPoint.value.y - startPoint.value.y) * canvas.height
    
    ctx.strokeRect(x, y, w, h)
  }
  
  // Draw pending point (waiting for backend response)
  if (pendingPoint.value) {
    const px = pendingPoint.value.x * canvas.width
    const py = pendingPoint.value.y * canvas.height
    const radius = 8
    
    // Outer ring (pulsing indicator)
    ctx.beginPath()
    ctx.arc(px, py, radius + 4, 0, Math.PI * 2)
    ctx.strokeStyle = pendingPoint.value.type === 'positive_point' ? '#22c55e' : '#ef4444'
    ctx.lineWidth = 2
    ctx.setLineDash([4, 4])
    ctx.stroke()
    
    // Inner circle (semi-transparent)
    ctx.beginPath()
    ctx.arc(px, py, radius, 0, Math.PI * 2)
    ctx.fillStyle = pendingPoint.value.type === 'positive_point' ? 'rgba(34, 197, 94, 0.5)' : 'rgba(239, 68, 68, 0.5)'
    ctx.fill()
    ctx.strokeStyle = '#fff'
    ctx.lineWidth = 2
    ctx.setLineDash([])
    ctx.stroke()
    
    // + or - symbol
    ctx.strokeStyle = '#fff'
    ctx.lineWidth = 2
    ctx.beginPath()
    ctx.moveTo(px - 4, py)
    ctx.lineTo(px + 4, py)
    if (pendingPoint.value.type === 'positive_point') {
      ctx.moveTo(px, py - 4)
      ctx.lineTo(px, py + 4)
    }
    ctx.stroke()
  }
}

watch(() => [props.mask, props.prompts], () => {
  pendingPoint.value = null
  render()
}, { deep: true })

watch(() => [props.videoWidth, props.videoHeight], () => {
  if (canvasRef.value) {
    canvasRef.value.width = props.videoWidth
    canvasRef.value.height = props.videoHeight
    render()
  }
})

onMounted(() => {
  if (canvasRef.value) {
    canvasRef.value.width = props.videoWidth
    canvasRef.value.height = props.videoHeight
    render()
  }
  
  window.addEventListener('mouseup', onMouseUp)
})

onUnmounted(() => {
  window.removeEventListener('mouseup', onMouseUp)
})
</script>

<template>
  <canvas
    ref="canvasRef"
    class="video-overlay"
    :class="{ 'tool-active': activeTool !== 'none' }"
    @mousedown="onMouseDown"
    @mousemove="onMouseMove"
  />
</template>

<style scoped>
.video-overlay {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  object-fit: contain;
  pointer-events: none;
}

.video-overlay.tool-active {
  pointer-events: auto;
  cursor: crosshair;
}
</style>
