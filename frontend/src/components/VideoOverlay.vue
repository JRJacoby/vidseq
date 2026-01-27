<script setup lang="ts">
import { ref, watch, onMounted } from 'vue'
import type { StoredPrompt, Bbox } from '@/services/api'

export type ToolType = 'none' | 'positive_point' | 'negative_point' | 'bounding_box'

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
  (e: 'bbox-complete', bbox: { x1: number; y1: number; x2: number; y2: number }): void
}>()

const canvasRef = ref<HTMLCanvasElement | null>(null)
const pendingPoint = ref<{ x: number; y: number; type: 'positive_point' | 'negative_point' } | null>(null)

// Bounding box drawing state
const drawingBox = ref<{ startX: number; startY: number; endX: number; endY: number } | null>(null)
const placedBox = ref<{ x1: number; y1: number; x2: number; y2: number } | null>(null)
const editMode = ref<'none' | 'draw' | 'move' | 'resize-tl' | 'resize-tr' | 'resize-bl' | 'resize-br'>('none')
const editStart = ref<{ x: number; y: number; box: { x1: number; y1: number; x2: number; y2: number } } | null>(null)

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

const HANDLE_SIZE = 10

function getHitZone(coords: { x: number; y: number }, box: { x1: number; y1: number; x2: number; y2: number }): 'tl' | 'tr' | 'bl' | 'br' | 'inside' | 'outside' {
  const canvas = canvasRef.value
  if (!canvas) return 'outside'

  const px = coords.x * canvas.width
  const py = coords.y * canvas.height
  const bx1 = box.x1 * canvas.width
  const by1 = box.y1 * canvas.height
  const bx2 = box.x2 * canvas.width
  const by2 = box.y2 * canvas.height

  const threshold = HANDLE_SIZE

  if (Math.abs(px - bx1) < threshold && Math.abs(py - by1) < threshold) return 'tl'
  if (Math.abs(px - bx2) < threshold && Math.abs(py - by1) < threshold) return 'tr'
  if (Math.abs(px - bx1) < threshold && Math.abs(py - by2) < threshold) return 'bl'
  if (Math.abs(px - bx2) < threshold && Math.abs(py - by2) < threshold) return 'br'

  if (px >= bx1 && px <= bx2 && py >= by1 && py <= by2) return 'inside'

  return 'outside'
}

function onMouseDown(event: MouseEvent) {
  const coords = getNormalizedCoords(event)
  if (!coords) return

  if (props.activeTool === 'positive_point' || props.activeTool === 'negative_point') {
    pendingPoint.value = { x: coords.x, y: coords.y, type: props.activeTool }
    render()
    emit('point-complete', { x: coords.x, y: coords.y, type: props.activeTool })
  } else if (props.activeTool === 'bounding_box') {
    if (placedBox.value) {
      const zone = getHitZone(coords, placedBox.value)
      if (zone === 'tl' || zone === 'tr' || zone === 'bl' || zone === 'br') {
        editMode.value = `resize-${zone}` as typeof editMode.value
        editStart.value = { x: coords.x, y: coords.y, box: { ...placedBox.value } }
        return
      } else if (zone === 'inside') {
        editMode.value = 'move'
        editStart.value = { x: coords.x, y: coords.y, box: { ...placedBox.value } }
        return
      }
    }
    // Start new box draw
    editMode.value = 'draw'
    drawingBox.value = { startX: coords.x, startY: coords.y, endX: coords.x, endY: coords.y }
    placedBox.value = null
  }
}

function onMouseMove(event: MouseEvent) {
  const coords = getNormalizedCoords(event)
  if (!coords) return

  if (props.activeTool === 'bounding_box') {
    // Update cursor based on hover zone
    if (editMode.value === 'none' && placedBox.value) {
      const canvas = canvasRef.value
      if (canvas) {
        const zone = getHitZone(coords, placedBox.value)
        if (zone === 'tl' || zone === 'br') canvas.style.cursor = 'nwse-resize'
        else if (zone === 'tr' || zone === 'bl') canvas.style.cursor = 'nesw-resize'
        else if (zone === 'inside') canvas.style.cursor = 'move'
        else canvas.style.cursor = 'crosshair'
      }
    }

    if (editMode.value === 'draw' && drawingBox.value) {
      drawingBox.value.endX = coords.x
      drawingBox.value.endY = coords.y
      render()
    } else if (editMode.value === 'move' && editStart.value) {
      const dx = coords.x - editStart.value.x
      const dy = coords.y - editStart.value.y
      const orig = editStart.value.box
      placedBox.value = {
        x1: Math.max(0, Math.min(1, orig.x1 + dx)),
        y1: Math.max(0, Math.min(1, orig.y1 + dy)),
        x2: Math.max(0, Math.min(1, orig.x2 + dx)),
        y2: Math.max(0, Math.min(1, orig.y2 + dy)),
      }
      render()
    } else if (editMode.value.startsWith('resize-') && editStart.value) {
      const orig = editStart.value.box
      const corner = editMode.value.replace('resize-', '')
      const newBox = { ...orig }

      if (corner === 'tl') { newBox.x1 = coords.x; newBox.y1 = coords.y }
      else if (corner === 'tr') { newBox.x2 = coords.x; newBox.y1 = coords.y }
      else if (corner === 'bl') { newBox.x1 = coords.x; newBox.y2 = coords.y }
      else if (corner === 'br') { newBox.x2 = coords.x; newBox.y2 = coords.y }

      placedBox.value = {
        x1: Math.max(0, Math.min(newBox.x1, newBox.x2)),
        y1: Math.max(0, Math.min(newBox.y1, newBox.y2)),
        x2: Math.min(1, Math.max(newBox.x1, newBox.x2)),
        y2: Math.min(1, Math.max(newBox.y1, newBox.y2)),
      }
      render()
    }
  }
}

function onMouseUp(_event: MouseEvent) {
  if (props.activeTool !== 'bounding_box') return

  if (editMode.value === 'draw' && drawingBox.value) {
    const x1 = Math.max(0, Math.min(drawingBox.value.startX, drawingBox.value.endX))
    const y1 = Math.max(0, Math.min(drawingBox.value.startY, drawingBox.value.endY))
    const x2 = Math.min(1, Math.max(drawingBox.value.startX, drawingBox.value.endX))
    const y2 = Math.min(1, Math.max(drawingBox.value.startY, drawingBox.value.endY))

    if (Math.abs(x2 - x1) > 0.01 && Math.abs(y2 - y1) > 0.01) {
      placedBox.value = { x1, y1, x2, y2 }
      submitBbox()
    }

    drawingBox.value = null
  } else if (editMode.value === 'move' || editMode.value.startsWith('resize-')) {
    if (placedBox.value) {
      submitBbox()
    }
  }

  editMode.value = 'none'
  editStart.value = null
}

function submitBbox() {
  if (!placedBox.value) return
  const canvas = canvasRef.value
  if (!canvas) return

  emit('bbox-complete', {
    x1: placedBox.value.x1 * canvas.width,
    y1: placedBox.value.y1 * canvas.height,
    x2: placedBox.value.x2 * canvas.width,
    y2: placedBox.value.y2 * canvas.height,
  })
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
      const maskValue = data[i]!
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

  // Draw stored bounding box prompts
  for (const prompt of props.prompts) {
    if (prompt.type === 'bounding_box') {
      const details = prompt.details as { x1: number; y1: number; x2: number; y2: number }
      const bx1 = details.x1 * canvas.width
      const by1 = details.y1 * canvas.height
      const bx2 = details.x2 * canvas.width
      const by2 = details.y2 * canvas.height

      ctx.fillStyle = 'rgba(59, 130, 246, 0.1)'
      ctx.fillRect(bx1, by1, bx2 - bx1, by2 - by1)

      ctx.strokeStyle = '#3b82f6'
      ctx.lineWidth = 2
      ctx.setLineDash([])
      ctx.strokeRect(bx1, by1, bx2 - bx1, by2 - by1)
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

  // Draw in-progress bounding box (while dragging)
  if (drawingBox.value && props.activeTool === 'bounding_box') {
    const x1 = Math.min(drawingBox.value.startX, drawingBox.value.endX) * canvas.width
    const y1 = Math.min(drawingBox.value.startY, drawingBox.value.endY) * canvas.height
    const x2 = Math.max(drawingBox.value.startX, drawingBox.value.endX) * canvas.width
    const y2 = Math.max(drawingBox.value.startY, drawingBox.value.endY) * canvas.height

    ctx.strokeStyle = '#3b82f6'
    ctx.lineWidth = 2
    ctx.setLineDash([6, 4])
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1)
    ctx.fillStyle = 'rgba(59, 130, 246, 0.1)'
    ctx.fillRect(x1, y1, x2 - x1, y2 - y1)
  }

  // Draw placed bounding box (editable)
  if (placedBox.value && props.activeTool === 'bounding_box') {
    const bx1 = placedBox.value.x1 * canvas.width
    const by1 = placedBox.value.y1 * canvas.height
    const bx2 = placedBox.value.x2 * canvas.width
    const by2 = placedBox.value.y2 * canvas.height

    ctx.fillStyle = 'rgba(59, 130, 246, 0.1)'
    ctx.fillRect(bx1, by1, bx2 - bx1, by2 - by1)

    ctx.strokeStyle = '#3b82f6'
    ctx.lineWidth = 2
    ctx.setLineDash([])
    ctx.strokeRect(bx1, by1, bx2 - bx1, by2 - by1)

    // Corner handles
    const handleSize = 6
    for (const [cx, cy] of [[bx1, by1], [bx2, by1], [bx1, by2], [bx2, by2]]) {
      ctx.fillStyle = '#fff'
      ctx.fillRect(cx - handleSize / 2, cy - handleSize / 2, handleSize, handleSize)
      ctx.strokeStyle = '#3b82f6'
      ctx.lineWidth = 1
      ctx.strokeRect(cx - handleSize / 2, cy - handleSize / 2, handleSize, handleSize)
    }
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

watch(() => props.activeTool, () => {
  placedBox.value = null
  drawingBox.value = null
  editMode.value = 'none'
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
    @mousemove="onMouseMove"
    @mouseup="onMouseUp"
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

