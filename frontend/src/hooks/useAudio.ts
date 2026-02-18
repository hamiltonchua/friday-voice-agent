// frontend/src/hooks/useAudio.ts
import { useRef, useCallback } from 'react'

export function useAudio() {
  const audioQueueRef = useRef<string[]>([])
  const currentAudioRef = useRef<HTMLAudioElement | null>(null)
  const isSpeakingRef = useRef(false)

  const playNext = useCallback(() => {
    if (currentAudioRef.current || audioQueueRef.current.length === 0) return

    isSpeakingRef.current = true
    const b64 = audioQueueRef.current.shift()!
    const bytes = Uint8Array.from(atob(b64), c => c.charCodeAt(0))
    const blob = new Blob([bytes], { type: 'audio/wav' })
    const url = URL.createObjectURL(blob)
    const audio = new Audio(url)
    currentAudioRef.current = audio

    const cleanup = () => {
      URL.revokeObjectURL(url)
      currentAudioRef.current = null
      if (audioQueueRef.current.length === 0) {
        isSpeakingRef.current = false
      }
      playNext()
    }

    audio.onended = cleanup
    audio.onerror = cleanup
    audio.play().catch(cleanup)
  }, [])

  const enqueueAudio = useCallback((b64: string) => {
    audioQueueRef.current.push(b64)
    playNext()
  }, [playNext])

  const stopPlayback = useCallback(() => {
    if (currentAudioRef.current) {
      currentAudioRef.current.pause()
      currentAudioRef.current.currentTime = 0
      currentAudioRef.current = null
    }
    audioQueueRef.current = []
    isSpeakingRef.current = false
  }, [])

  // Legacy single audio playback
  const playAudio = useCallback((b64: string) => {
    const bytes = Uint8Array.from(atob(b64), c => c.charCodeAt(0))
    const blob = new Blob([bytes], { type: 'audio/wav' })
    const url = URL.createObjectURL(blob)
    const audio = new Audio(url)
    audio.play()
    audio.onended = () => URL.revokeObjectURL(url)
  }, [])

  return { enqueueAudio, stopPlayback, playAudio, isSpeakingRef, currentAudioRef }
}
