<script setup lang="ts">
import { ref, watch, onMounted } from 'vue'
import type { ToolType } from '@/composables/useSegmentation'

// Simple prompt type for local/ephemeral prompts
export interface LocalPrompt {
  x: number
  y: number
  type: 'positive_point' | 'negative_point'
}

const props = defineProps<{
  videoWidth: number
  videoHeight: number
  activeTool: ToolType
  mask: ImageBitmap | null
  prompts: LocalPrompt[]
  showMask?: boolean
  showPrompts?: boolean
  detectorBbox?: { x1: number; y1: number; x2: number; y2: number } | null
  obbBbox?: { corners: [number, number][] } | null
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
    // Mask PNG is already rendered as semi-transparent blue overlay by backend
    ctx.drawImage(props.mask, 0, 0, canvas.width, canvas.height)
  }

  // Draw detector bbox if provided
  if (props.detectorBbox) {
    ctx.strokeStyle = 'rgba(255, 99, 71, 0.8)'
    ctx.lineWidth = 3
    const bbox = props.detectorBbox
    // bbox coords are in original video coordinates, canvas size matches video size
    const scaleX = canvas.width / props.videoWidth
    const scaleY = canvas.height / props.videoHeight
    ctx.strokeRect(
      bbox.x1 * scaleX, bbox.y1 * scaleY,
      (bbox.x2 - bbox.x1) * scaleX, (bbox.y2 - bbox.y1) * scaleY,
    )
  }

  // Draw OBB if provided
  if (props.obbBbox && props.obbBbox.corners.length === 4) {
    ctx.strokeStyle = 'rgba(0, 188, 212, 0.8)'  // cyan
    ctx.lineWidth = 3
    const scaleX = canvas.width / props.videoWidth
    const scaleY = canvas.height / props.videoHeight
    const corners = props.obbBbox.corners

    ctx.beginPath()
    ctx.moveTo(corners[0][0] * scaleX, corners[0][1] * scaleY)
    ctx.lineTo(corners[1][0] * scaleX, corners[1][1] * scaleY)
    ctx.lineTo(corners[2][0] * scaleX, corners[2][1] * scaleY)
    ctx.lineTo(corners[3][0] * scaleX, corners[3][1] * scaleY)
    ctx.closePath()
    ctx.stroke()
  }

  // Draw prompts (points only)
  if (props.showPrompts === false) return
  for (const prompt of props.prompts) {
    const px = prompt.x * canvas.width
    const py = prompt.y * canvas.height
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

watch(() => [props.mask, props.prompts, props.detectorBbox, props.obbBbox, props.showMask, props.showPrompts], () => {
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

watch(() => props.activeTool, () => {
  render()
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
