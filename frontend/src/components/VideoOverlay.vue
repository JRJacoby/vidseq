<script setup lang="ts">
import { ref, watch, onMounted } from 'vue'
import type { StoredPrompt, Bbox } from '@/services/api'

export type ToolType = 'none' | 'positive_point' | 'negative_point'

const props = defineProps<{
  videoWidth: number
  videoHeight: number
  activeTool: ToolType
  mask: ImageBitmap | null
  bbox: Bbox | null
  prompts: StoredPrompt[]
  showMask?: boolean
  showBbox?: boolean
  showPrompts?: boolean
}>()

const emit = defineEmits<{
  (e: 'point-complete', point: { x: number; y: number; type: 'positive_point' | 'negative_point' }): void
}>()

const canvasRef = ref<HTMLCanvasElement | null>(null)
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
  
  if (props.activeTool === 'positive_point' || props.activeTool === 'negative_point') {
    pendingPoint.value = { x: coords.x, y: coords.y, type: props.activeTool }
    render()
    emit('point-complete', { x: coords.x, y: coords.y, type: props.activeTool })
  }
}

function render() {
  const canvas = canvasRef.value
  if (!canvas) return
  
  const ctx = canvas.getContext('2d', { willReadFrequently: true })
  if (!ctx) return
  
  ctx.clearRect(0, 0, canvas.width, canvas.height)
  
  if (props.mask && props.showMask !== false) {
    ctx.drawImage(props.mask, 0, 0, canvas.width, canvas.height)
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height)
    const data = imageData.data
    
    for (let i = 0; i < data.length; i += 4) {
      const maskValue = data[i]
      if (maskValue > 0) {
        data[i] = 102       // R
        data[i + 1] = 179   // G
        data[i + 2] = 255   // B
        data[i + 3] = 102   // A
      } else {
        data[i + 3] = 0
      }
    }
    
    ctx.putImageData(imageData, 0, 0)
  }
  
  // Draw bounding box
  if (props.bbox && props.showBbox !== false) {
    const { x1, y1, x2, y2 } = props.bbox
    // Ensure coordinates are within canvas bounds
    const clampedX1 = Math.max(0, Math.min(x1, canvas.width))
    const clampedY1 = Math.max(0, Math.min(y1, canvas.height))
    const clampedX2 = Math.max(0, Math.min(x2, canvas.width))
    const clampedY2 = Math.max(0, Math.min(y2, canvas.height))
    
    ctx.strokeStyle = '#22c55e'
    ctx.lineWidth = 3
    ctx.setLineDash([])
    ctx.strokeRect(clampedX1, clampedY1, clampedX2 - clampedX1, clampedY2 - clampedY1)
  }
  
  // Draw prompts (points only)
  if (props.showPrompts === false) return
  for (const prompt of props.prompts) {
    if (prompt.type === 'positive_point' || prompt.type === 'negative_point') {
      const { x, y } = prompt.details
      const px = x * canvas.width
      const py = y * canvas.height
      const radius = 8
      
      ctx.beginPath()
      ctx.arc(px, py, radius, 0, Math.PI * 2)
      ctx.fillStyle = prompt.type === 'positive_point' ? '#22c55e' : '#ef4444'
      ctx.fill()
      ctx.strokeStyle = '#fff'
      ctx.lineWidth = 2
      ctx.setLineDash([])
      ctx.stroke()
      
      ctx.strokeStyle = '#fff'
      ctx.lineWidth = 2
      ctx.beginPath()
      ctx.moveTo(px - 4, py)
      ctx.lineTo(px + 4, py)
      if (prompt.type === 'positive_point') {
        ctx.moveTo(px, py - 4)
        ctx.lineTo(px, py + 4)
      }
      ctx.stroke()
    }
  }
  
  // Draw pending point
  if (pendingPoint.value) {
    const px = pendingPoint.value.x * canvas.width
    const py = pendingPoint.value.y * canvas.height
    const radius = 8
    
    ctx.beginPath()
    ctx.arc(px, py, radius + 4, 0, Math.PI * 2)
    ctx.strokeStyle = pendingPoint.value.type === 'positive_point' ? '#22c55e' : '#ef4444'
    ctx.lineWidth = 2
    ctx.setLineDash([4, 4])
    ctx.stroke()
    
    ctx.beginPath()
    ctx.arc(px, py, radius, 0, Math.PI * 2)
    ctx.fillStyle = pendingPoint.value.type === 'positive_point' ? 'rgba(34, 197, 94, 0.5)' : 'rgba(239, 68, 68, 0.5)'
    ctx.fill()
    ctx.strokeStyle = '#fff'
    ctx.lineWidth = 2
    ctx.setLineDash([])
    ctx.stroke()
    
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

watch(() => [props.mask, props.bbox, props.prompts, props.showMask, props.showBbox, props.showPrompts], () => {
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
})
</script>

<template>
  <canvas
    ref="canvasRef"
    class="video-overlay"
    :class="{ 'tool-active': activeTool !== 'none' }"
    @mousedown="onMouseDown"
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

