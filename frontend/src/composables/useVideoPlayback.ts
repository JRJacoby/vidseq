import { ref, type Ref } from 'vue'

export interface UseVideoPlaybackReturn {
  videoRef: Ref<HTMLVideoElement | null>
  currentTime: Ref<number>
  duration: Ref<number>
  isPlaying: Ref<boolean>
  videoWidth: Ref<number>
  videoHeight: Ref<number>
  onTimeUpdate: () => void
  onLoadedMetadata: () => void
  onPlay: () => void
  onPause: () => void
  seek: (time: number) => void
  togglePlay: () => void
  setMetadataCallback: (callback: () => void) => void
}

export function useVideoPlayback(): UseVideoPlaybackReturn {
  const videoRef = ref<HTMLVideoElement | null>(null)
  const currentTime = ref(0)
  const duration = ref(0)
  const isPlaying = ref(false)
  const videoWidth = ref(0)
  const videoHeight = ref(0)

  let metadataCallback: (() => void) | null = null

  const onTimeUpdate = () => {
    if (videoRef.value) {
      currentTime.value = videoRef.value.currentTime
    }
  }

  const onLoadedMetadata = () => {
    if (videoRef.value) {
      duration.value = videoRef.value.duration
      videoWidth.value = videoRef.value.videoWidth
      videoHeight.value = videoRef.value.videoHeight
      metadataCallback?.()
    }
  }

  const setMetadataCallback = (callback: () => void) => {
    metadataCallback = callback
  }

  const onPlay = () => {
    isPlaying.value = true
  }

  const onPause = () => {
    isPlaying.value = false
  }

  const seek = (time: number) => {
    currentTime.value = time
    if (videoRef.value) {
      videoRef.value.currentTime = time
    }
  }

  const togglePlay = () => {
    if (videoRef.value) {
      if (isPlaying.value) {
        videoRef.value.pause()
      } else {
        videoRef.value.play()
      }
    }
  }

  return {
    videoRef,
    currentTime,
    duration,
    isPlaying,
    videoWidth,
    videoHeight,
    onTimeUpdate,
    onLoadedMetadata,
    onPlay,
    onPause,
    seek,
    togglePlay,
    setMetadataCallback,
  }
}
