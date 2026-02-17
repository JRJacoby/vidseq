import { ref, watch, type Ref } from 'vue'
import {
  submitKeypointPrompt,
  getFrameKeypoints,
  getKeypointsRange,
  deleteKeypointFrame,
  deleteKeypointAllFrames,
  type KeypointResult,
} from '@/services/api'

export function useKeypointTracking(
  projectId: Ref<number>,
  videoId: Ref<number>,
  currentFrameIdx: Ref<number>,
) {
  const keypointCache = new Map<number, KeypointResult>()
  const activeTool = ref<'none' | 'front' | 'rear'>('none')
  const currentKeypoints = ref<KeypointResult | null>(null)
  const isPrompting = ref(false)

  async function loadFrameKeypoints(frameIdx: number) {
    if (keypointCache.has(frameIdx)) {
      currentKeypoints.value = keypointCache.get(frameIdx)!
      return
    }
    try {
      const result = await getFrameKeypoints(projectId.value, videoId.value, frameIdx)
      keypointCache.set(frameIdx, result)
      currentKeypoints.value = result
    } catch (e) {
      console.error('Failed to load keypoints:', e)
      currentKeypoints.value = null
    }
  }

  async function handlePointComplete(point: { x: number; y: number }) {
    if (activeTool.value === 'none') return
    isPrompting.value = true
    try {
      const result = await submitKeypointPrompt(
        projectId.value, videoId.value, currentFrameIdx.value,
        point.x, point.y, activeTool.value,
      )
      keypointCache.set(currentFrameIdx.value, result)
      currentKeypoints.value = result
      // Auto-cycle: front → rear, rear → front
      activeTool.value = activeTool.value === 'front' ? 'rear' : 'front'
    } catch (e) {
      console.error('Failed to submit keypoint prompt:', e)
    } finally {
      isPrompting.value = false
    }
  }

  watch(currentFrameIdx, (idx) => loadFrameKeypoints(idx))

  async function prefetchKeypoints(startFrame: number) {
    try {
      const { keypoints } = await getKeypointsRange(projectId.value, videoId.value, startFrame, 100)
      for (const kp of keypoints) {
        keypointCache.set(kp.frame_idx, kp)
      }
    } catch (e) {
      console.error('Failed to prefetch keypoints:', e)
    }
  }

  async function handleResetFrame() {
    try {
      await deleteKeypointFrame(projectId.value, videoId.value, currentFrameIdx.value)
      keypointCache.delete(currentFrameIdx.value)
      currentKeypoints.value = null
      activeTool.value = 'front'
    } catch (e) {
      console.error('Failed to reset frame:', e)
    }
  }

  async function handleResetVideo() {
    try {
      await deleteKeypointAllFrames(projectId.value, videoId.value)
      keypointCache.clear()
      currentKeypoints.value = null
    } catch (e) {
      console.error('Failed to reset video:', e)
    }
  }

  function clearCache() { keypointCache.clear() }

  return {
    activeTool, currentKeypoints, isPrompting,
    handlePointComplete, handleResetFrame, handleResetVideo,
    loadFrameKeypoints, prefetchKeypoints, clearCache,
    toggleFrontTool: () => { activeTool.value = activeTool.value === 'front' ? 'none' : 'front' },
    toggleRearTool: () => { activeTool.value = activeTool.value === 'rear' ? 'none' : 'rear' },
  }
}
