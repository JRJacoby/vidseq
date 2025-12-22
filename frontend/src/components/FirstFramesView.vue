<script setup lang="ts">
import { ref, onMounted, onUnmounted, computed, watch, nextTick } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { getVideos, getFrameImage, getBbox, type Video, type Bbox } from '@/services/api'

const route = useRoute()
const router = useRouter()

const projectId = computed(() => Number(route.params.id))

// Video data
const videos = ref<Video[]>([])
const isLoading = ref(false)
const error = ref<string | null>(null)

// Pagination state
const currentPage = ref(1)
const videosPerPage = 18

// Sorted videos (alphabetical by name)
const sortedVideos = computed(() =>
  [...videos.value].sort((a, b) => a.name.localeCompare(b.name))
)

const totalPages = computed(() => Math.ceil(sortedVideos.value.length / videosPerPage))

// Current page's videos
const pageVideos = computed(() => {
  const start = (currentPage.value - 1) * videosPerPage
  return sortedVideos.value.slice(start, start + videosPerPage)
})

// Per-cell state (Map keyed by video ID)
const frameImages = ref<Map<number, ImageBitmap>>(new Map())
const bboxes = ref<Map<number, Bbox | null>>(new Map())
const cellCanvases = ref<Map<number, HTMLCanvasElement>>(new Map())

// Page numbers for pagination UI (with ellipsis for large counts)
const visiblePageNumbers = computed(() => {
  const total = totalPages.value
  const current = currentPage.value

  if (total <= 7) {
    return Array.from({ length: total }, (_, i) => i + 1)
  }

  const pages: (number | string)[] = []

  // Always show first page
  pages.push(1)

  if (current > 3) {
    pages.push('...')
  }

  // Show pages around current
  const start = Math.max(2, current - 1)
  const end = Math.min(total - 1, current + 1)

  for (let i = start; i <= end; i++) {
    if (!pages.includes(i)) {
      pages.push(i)
    }
  }

  if (current < total - 2) {
    pages.push('...')
  }

  // Always show last page
  if (!pages.includes(total)) {
    pages.push(total)
  }

  return pages
})

const loadVideos = async () => {
  if (!projectId.value) return

  isLoading.value = true
  error.value = null

  try {
    videos.value = await getVideos(projectId.value)
    if (videos.value.length > 0) {
      currentPage.value = 1
      await loadPageData()
    }
  } catch (e) {
    error.value = e instanceof Error ? e.message : 'Failed to load videos'
    console.error('Failed to load videos:', e)
  } finally {
    isLoading.value = false
  }
}

const loadPageData = async () => {
  if (!projectId.value || pageVideos.value.length === 0) return

  isLoading.value = true
  error.value = null

  // Cleanup old bitmaps
  frameImages.value.forEach(img => img.close())
  frameImages.value.clear()
  bboxes.value.clear()

  try {
    // Load all frame images and bboxes for current page in parallel
    const loadPromises = pageVideos.value.map(async (video) => {
      try {
        const [frameBlob, bbox] = await Promise.all([
          getFrameImage(projectId.value, video.id, 0),
          getBbox(projectId.value, video.id, 0)
        ])

        const imageBitmap = await createImageBitmap(frameBlob)
        frameImages.value.set(video.id, imageBitmap)
        bboxes.value.set(video.id, bbox)

        // Draw canvas for this video
        nextTick(() => {
          drawCell(video.id)
        })
      } catch (e) {
        console.error(`Failed to load frame for video ${video.id}:`, e)
        // Continue loading other videos even if one fails
      }
    })

    await Promise.all(loadPromises)
  } catch (e) {
    error.value = e instanceof Error ? e.message : 'Failed to load page data'
    console.error('Failed to load page data:', e)
  } finally {
    isLoading.value = false
  }
}

const setCellCanvas = (videoId: number, el: HTMLCanvasElement | null) => {
  if (el) {
    cellCanvases.value.set(videoId, el)
    // Draw if image is already loaded
    if (frameImages.value.has(videoId)) {
      nextTick(() => drawCell(videoId))
    }
  } else {
    cellCanvases.value.delete(videoId)
  }
}

const drawCell = (videoId: number) => {
  const canvas = cellCanvases.value.get(videoId)
  const image = frameImages.value.get(videoId)

  if (!canvas || !image) return

  const ctx = canvas.getContext('2d')
  if (!ctx) return

  // Set canvas size to match image dimensions
  canvas.width = image.width
  canvas.height = image.height

  ctx.clearRect(0, 0, canvas.width, canvas.height)

  // Draw the frame image
  ctx.drawImage(image, 0, 0)

  // Draw bounding box if available
  const bbox = bboxes.value.get(videoId)
  if (bbox) {
    const { x1, y1, x2, y2 } = bbox
    const clampedX1 = Math.max(0, Math.min(x1, canvas.width))
    const clampedY1 = Math.max(0, Math.min(y1, canvas.height))
    const clampedX2 = Math.max(0, Math.min(x2, canvas.width))
    const clampedY2 = Math.max(0, Math.min(y2, canvas.height))

    ctx.strokeStyle = '#22c55e'
    ctx.lineWidth = 3
    ctx.setLineDash([])
    ctx.strokeRect(clampedX1, clampedY1, clampedX2 - clampedX1, clampedY2 - clampedY1)
  }
}

// Page navigation
const goToPage = (page: number | string) => {
  if (typeof page !== 'number') return
  if (page < 1 || page > totalPages.value) return
  currentPage.value = page
}

const prevPage = () => {
  if (currentPage.value > 1) {
    currentPage.value--
  }
}

const nextPage = () => {
  if (currentPage.value < totalPages.value) {
    currentPage.value++
  }
}

// Navigate to VideoDetail
const goToVideoDetail = (videoId: number) => {
  router.push({ name: 'video', params: { id: projectId.value, videoId } })
}

// Watch for page changes to reload data
watch(currentPage, async () => {
  await loadPageData()
})

// Keyboard navigation
const handleKeyPress = (event: KeyboardEvent) => {
  if (event.key === 'ArrowLeft') {
    event.preventDefault()
    prevPage()
  } else if (event.key === 'ArrowRight') {
    event.preventDefault()
    nextPage()
  }
}

onMounted(() => {
  loadVideos()
  window.addEventListener('keydown', handleKeyPress)
})

onUnmounted(() => {
  window.removeEventListener('keydown', handleKeyPress)
  // Cleanup all ImageBitmaps
  frameImages.value.forEach(img => img.close())
  frameImages.value.clear()
})
</script>

<template>
  <div class="first-frames-container">
    <!-- Initial loading state -->
    <div v-if="isLoading && videos.length === 0" class="loading-state">
      Loading videos...
    </div>

    <!-- Error state -->
    <div v-else-if="error && videos.length === 0" class="error-state">
      {{ error }}
    </div>

    <!-- Empty state -->
    <div v-else-if="videos.length === 0" class="empty-state">
      No videos found in this project.
    </div>

    <!-- Main grid view -->
    <div v-else class="grid-viewer">
      <!-- Header with pagination -->
      <div class="viewer-header">
        <h2>First Frames</h2>
        <div class="header-info">
          {{ sortedVideos.length }} videos total
        </div>
      </div>

      <!-- Pagination controls (top) -->
      <div v-if="totalPages > 1" class="pagination">
        <button
          class="page-btn nav-btn"
          @click="prevPage"
          :disabled="currentPage === 1 || isLoading"
        >
          ← Prev
        </button>

        <div class="page-numbers">
          <button
            v-for="page in visiblePageNumbers"
            :key="page"
            class="page-btn"
            :class="{
              active: page === currentPage,
              ellipsis: page === '...'
            }"
            :disabled="page === '...' || isLoading"
            @click="goToPage(page)"
          >
            {{ page }}
          </button>
        </div>

        <button
          class="page-btn nav-btn"
          @click="nextPage"
          :disabled="currentPage === totalPages || isLoading"
        >
          Next →
        </button>
      </div>

      <!-- Loading overlay for page transitions -->
      <div v-if="isLoading" class="page-loading">
        Loading page {{ currentPage }}...
      </div>

      <!-- Grid of first frames -->
      <div class="frames-grid">
        <div
          v-for="video in pageVideos"
          :key="video.id"
          class="frame-cell"
        >
          <div class="cell-canvas-wrapper">
            <canvas
              :ref="(el) => setCellCanvas(video.id, el as HTMLCanvasElement)"
              class="cell-canvas"
            />
            <div
              v-if="!frameImages.has(video.id)"
              class="cell-loading"
            >
              Loading...
            </div>
          </div>

          <div class="cell-footer">
            <span class="video-name" :title="video.name">{{ video.name }}</span>
            <button
              class="edit-btn"
              @click="goToVideoDetail(video.id)"
            >
              Edit
            </button>
          </div>
        </div>
      </div>

      <!-- Pagination controls (bottom) -->
      <div v-if="totalPages > 1" class="pagination pagination-bottom">
        <button
          class="page-btn nav-btn"
          @click="prevPage"
          :disabled="currentPage === 1 || isLoading"
        >
          ← Prev
        </button>

        <div class="page-numbers">
          <button
            v-for="page in visiblePageNumbers"
            :key="page"
            class="page-btn"
            :class="{
              active: page === currentPage,
              ellipsis: page === '...'
            }"
            :disabled="page === '...' || isLoading"
            @click="goToPage(page)"
          >
            {{ page }}
          </button>
        </div>

        <button
          class="page-btn nav-btn"
          @click="nextPage"
          :disabled="currentPage === totalPages || isLoading"
        >
          Next →
        </button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.first-frames-container {
  display: flex;
  flex-direction: column;
  flex: 1;
  min-height: 0;
  padding: 1rem;
  overflow: auto;
}

.loading-state,
.error-state,
.empty-state {
  display: flex;
  align-items: center;
  justify-content: center;
  flex: 1;
  font-size: 1.1rem;
  color: #666;
}

.error-state {
  color: #d32f2f;
}

.grid-viewer {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.viewer-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding-bottom: 0.5rem;
  border-bottom: 1px solid #e0e0e0;
}

.viewer-header h2 {
  margin: 0;
  font-size: 1.5rem;
}

.header-info {
  color: #666;
  font-size: 0.9rem;
}

/* Pagination */
.pagination {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
}

.pagination-bottom {
  margin-top: 1rem;
  padding-top: 1rem;
  border-top: 1px solid #e0e0e0;
}

.page-numbers {
  display: flex;
  gap: 0.25rem;
}

.page-btn {
  padding: 0.4rem 0.75rem;
  font-size: 0.9rem;
  border: 1px solid #ccc;
  border-radius: 4px;
  background-color: #f8f8f8;
  cursor: pointer;
  transition: background-color 0.2s, border-color 0.2s;
}

.page-btn:hover:not(:disabled):not(.ellipsis) {
  background-color: #e8e8e8;
  border-color: #999;
}

.page-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.page-btn.active {
  background-color: #1976d2;
  border-color: #1976d2;
  color: white;
}

.page-btn.ellipsis {
  border: none;
  background: transparent;
  cursor: default;
}

.nav-btn {
  min-width: 80px;
}

.page-loading {
  text-align: center;
  padding: 0.5rem;
  color: #666;
  font-style: italic;
}

/* Grid layout */
.frames-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 1rem;
}

.frame-cell {
  display: flex;
  flex-direction: column;
  border: 1px solid #e0e0e0;
  border-radius: 8px;
  overflow: hidden;
  background: #fafafa;
}

.cell-canvas-wrapper {
  position: relative;
  aspect-ratio: 16 / 9;
  background: #000;
  display: flex;
  align-items: center;
  justify-content: center;
}

.cell-canvas {
  max-width: 100%;
  max-height: 100%;
  display: block;
}

.cell-loading {
  position: absolute;
  color: #999;
  font-size: 0.9rem;
}

.cell-footer {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.5rem 0.75rem;
  background: white;
  border-top: 1px solid #e0e0e0;
}

.video-name {
  font-size: 0.85rem;
  color: #333;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  flex: 1;
  margin-right: 0.5rem;
}

.edit-btn {
  padding: 0.35rem 0.75rem;
  font-size: 0.8rem;
  border: 1px solid #1976d2;
  border-radius: 4px;
  background-color: #1976d2;
  color: white;
  cursor: pointer;
  transition: background-color 0.2s;
  flex-shrink: 0;
}

.edit-btn:hover {
  background-color: #1565c0;
}
</style>
