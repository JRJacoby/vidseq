<script setup lang="ts">
import { ref, watch, onMounted } from 'vue'
import type { ToolType, LocalPrompt } from '@/composables/useSegmentation'

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
  // NEW: Pose keypoints
  poseLabel?: { front_x: number; front_y: number; rear_x: number; rear_y: number } | null
  posePrediction?: { front_x: number; front_y: number; rear_x: number; rear_y: number } | null
  pendingFront?: { x: number; y: number } | null
  showPoseKeypoints?: boolean
  isLabelingKeypoints?: boolean
  brushSize?: number
  graphcutSeeds?: { x: number; y: number; label: number }[]
  showSeeds?: boolean
}>()

const emit = defineEmits<{
  (e: 'point-complete', point: { x: number; y: number; type: 'positive_point' | 'negative_point' }): void
  (e: 'box-complete', box: { x1: number; y1: number; x2: number; y2: number }): void
  (e: 'keypoint-click', point: { x: number; y: number }): void
  (e: 'brush-stroke', point: { x: number; y: number; label: number }): void
}>()

const canvasRef = ref<HTMLCanvasElement | null>(null)
const pendingPoint = ref<{ x: number; y: number; type: 'positive_point' | 'negative_point' } | null>(null)
const pendingBox = ref<{ x1: number; y1: number; x2: number; y2: number } | null>(null)
const isDragging = ref(false)
const dragStart = ref<{ x: number; y: number } | null>(null)

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

function getNativeCoords(event: MouseEvent): { x: number; y: number } | null {
  const canvas = canvasRef.value
  if (!canvas) return null
  const rect = canvas.getBoundingClientRect()
  const scaleX = props.videoWidth / rect.width
  const scaleY = props.videoHeight / rect.height
  const x = Math.floor((event.clientX - rect.left) * scaleX)
  const y = Math.floor((event.clientY - rect.top) * scaleY)
  if (x < 0 || x >= props.videoWidth || y < 0 || y >= props.videoHeight) return null
  return { x, y }
}

const isBrushPainting = ref(false)
const brushButton = ref(0)

function emitBrushStroke(cx: number, cy: number, label: number) {
  const canvas = canvasRef.value
  if (!canvas) return
  const rect = canvas.getBoundingClientRect()
  const scaleX = props.videoWidth / rect.width
  const r = Math.max(1, Math.round((props.brushSize ?? 5) * scaleX))
  for (let dy = -r; dy <= r; dy++) {
    for (let dx = -r; dx <= r; dx++) {
      if (dx * dx + dy * dy <= r * r) {
        const x = cx + dx
        const y = cy + dy
        if (x >= 0 && x < props.videoWidth && y >= 0 && y < props.videoHeight) {
          emit('brush-stroke', { x, y, label })
        }
      }
    }
  }
  render()
}

function onMouseDown(event: MouseEvent) {
  if (props.activeTool === 'graphcut_brush') {
    const coords = getNativeCoords(event)
    if (!coords) return
    isBrushPainting.value = true
    brushButton.value = event.button
    const label = event.button === 2 ? 2 : 1
    emitBrushStroke(coords.x, coords.y, label)
    return
  }

  const coords = getNormalizedCoords(event)
  if (!coords) return

  if (props.isLabelingKeypoints) {
    emit('keypoint-click', { x: coords.x, y: coords.y })
    return
  }

  if (props.activeTool === 'positive_point' || props.activeTool === 'negative_point') {
    pendingPoint.value = { x: coords.x, y: coords.y, type: props.activeTool }
    render()
    emit('point-complete', { x: coords.x, y: coords.y, type: props.activeTool })
  } else if (props.activeTool === 'bounding_box') {
    isDragging.value = true
    dragStart.value = coords
    pendingBox.value = { x1: coords.x, y1: coords.y, x2: coords.x, y2: coords.y }
    render()
  }
}

function onMouseMove(event: MouseEvent) {
  if (isBrushPainting.value) {
    const coords = getNativeCoords(event)
    if (!coords) return
    const label = brushButton.value === 2 ? 2 : 1
    emitBrushStroke(coords.x, coords.y, label)
    return
  }

  if (!isDragging.value || !dragStart.value) return

  const coords = getNormalizedCoords(event)
  if (!coords) return

  pendingBox.value = {
    x1: Math.min(dragStart.value.x, coords.x),
    y1: Math.min(dragStart.value.y, coords.y),
    x2: Math.max(dragStart.value.x, coords.x),
    y2: Math.max(dragStart.value.y, coords.y),
  }
  render()
}

function onMouseUp(_event: MouseEvent) {
  if (isBrushPainting.value) {
    isBrushPainting.value = false
    return
  }

  if (!isDragging.value || !pendingBox.value) return

  isDragging.value = false
  dragStart.value = null

  const box = pendingBox.value
  // Only emit if box has meaningful size (not a click)
  const minSize = 0.01
  if (Math.abs(box.x2 - box.x1) > minSize && Math.abs(box.y2 - box.y1) > minSize) {
    emit('box-complete', { x1: box.x1, y1: box.y1, x2: box.x2, y2: box.y2 })
  } else {
    pendingBox.value = null
    render()
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

    const [c0, c1, c2, c3] = corners as [[number, number], [number, number], [number, number], [number, number]]
    ctx.beginPath()
    ctx.moveTo(c0[0] * scaleX, c0[1] * scaleY)
    ctx.lineTo(c1[0] * scaleX, c1[1] * scaleY)
    ctx.lineTo(c2[0] * scaleX, c2[1] * scaleY)
    ctx.lineTo(c3[0] * scaleX, c3[1] * scaleY)
    ctx.closePath()
    ctx.stroke()
  }

  // Draw pose keypoints
  if (props.showPoseKeypoints !== false) {
    const drawKeypoint = (x: number, y: number, color: string, filled: boolean) => {
      const px = x * canvas.width
      const py = y * canvas.height
      const radius = 8
      ctx.beginPath()
      ctx.arc(px, py, radius, 0, Math.PI * 2)
      if (filled) {
        ctx.fillStyle = color
        ctx.fill()
      }
      ctx.strokeStyle = color
      ctx.lineWidth = 2
      ctx.setLineDash([])
      ctx.stroke()
    }

    // Hand-labeled keypoints (filled circles)
    if (props.poseLabel) {
      drawKeypoint(props.poseLabel.front_x, props.poseLabel.front_y, '#22c55e', true)
      drawKeypoint(props.poseLabel.rear_x, props.poseLabel.rear_y, '#ef4444', true)
    }

    // Predicted keypoints (outline circles, only if no label)
    if (props.posePrediction && !props.poseLabel) {
      drawKeypoint(props.posePrediction.front_x, props.posePrediction.front_y, '#22c55e', false)
      drawKeypoint(props.posePrediction.rear_x, props.posePrediction.rear_y, '#ef4444', false)
    }

    // Pending front point during labeling
    if (props.pendingFront) {
      drawKeypoint(props.pendingFront.x, props.pendingFront.y, '#22c55e', true)
    }
  }

  // Draw graph cut seeds
  if (props.showSeeds !== false && props.graphcutSeeds) {
    for (const seed of props.graphcutSeeds) {
      ctx.fillStyle = seed.label === 1
        ? 'rgba(34, 197, 94, 0.7)'   // green for foreground
        : 'rgba(239, 68, 68, 0.7)'   // red for background
      ctx.fillRect(seed.x, seed.y, 1, 1)
    }
  }

  // Draw prompts (points and boxes)
  if (props.showPrompts === false) return
  for (const prompt of props.prompts) {
    if (prompt.type === 'bounding_box') {
      // Draw completed box prompt
      const bx1 = prompt.x1 * canvas.width
      const by1 = prompt.y1 * canvas.height
      const bx2 = prompt.x2 * canvas.width
      const by2 = prompt.y2 * canvas.height
      ctx.strokeStyle = '#3b82f6'
      ctx.lineWidth = 3
      ctx.setLineDash([])
      ctx.strokeRect(bx1, by1, bx2 - bx1, by2 - by1)
    } else {
      // Draw point prompt (existing logic)
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

  // Draw pending box (during drag)
  if (pendingBox.value) {
    const bx1 = pendingBox.value.x1 * canvas.width
    const by1 = pendingBox.value.y1 * canvas.height
    const bx2 = pendingBox.value.x2 * canvas.width
    const by2 = pendingBox.value.y2 * canvas.height
    ctx.strokeStyle = '#3b82f6'
    ctx.lineWidth = 2
    ctx.setLineDash([4, 4])
    ctx.strokeRect(bx1, by1, bx2 - bx1, by2 - by1)
    ctx.setLineDash([])
  }
}

watch(() => [props.mask, props.prompts, props.detectorBbox, props.obbBbox, props.showMask, props.showPrompts, props.poseLabel, props.posePrediction, props.pendingFront, props.showPoseKeypoints, props.graphcutSeeds, props.showSeeds], () => {
  pendingPoint.value = null
  pendingBox.value = null
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
  pendingPoint.value = null
  pendingBox.value = null
  isDragging.value = false
  dragStart.value = null
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
    :class="{ 'tool-active': activeTool !== 'none' || isLabelingKeypoints }"
    @mousedown="onMouseDown"
    @mousemove="onMouseMove"
    @mouseup="onMouseUp"
    @contextmenu.prevent
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
