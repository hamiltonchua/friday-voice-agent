// frontend/src/hooks/useManualRecording.ts
import { useRef, useCallback } from 'react'
import { SAMPLE_RATE } from '../constants'

function arrayBufferToBase64(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer)
  let binary = ''
  for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i])
  return btoa(binary)
}

interface UseManualRecordingOptions {
  onAudioLevel: (rms: number) => void
  onRecordingStart: () => void
  onRecordingEnd: (float32: Float32Array) => void
  onError: (msg: string) => void
}

export function useManualRecording({
  onAudioLevel,
  onRecordingStart,
  onRecordingEnd,
  onError,
}: UseManualRecordingOptions) {
  const streamRef = useRef<MediaStream | null>(null)
  const ctxRef = useRef<AudioContext | null>(null)
  const recorderRef = useRef<{ proc: ScriptProcessorNode; source: MediaStreamAudioSourceNode } | null>(null)
  const chunksRef = useRef<Float32Array[]>([])

  const start = useCallback(async () => {
    if (!streamRef.current) {
      try {
        streamRef.current = await navigator.mediaDevices.getUserMedia({
          audio: { sampleRate: SAMPLE_RATE, channelCount: 1, echoCancellation: true, noiseSuppression: true },
        })
      } catch (err: unknown) {
        const error = err as { name?: string; message?: string }
        if (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError') {
          onError('Mic access denied — enable microphone in browser settings')
        } else {
          onError('Could not access microphone: ' + (error.message ?? ''))
        }
        return false
      }
    }

    const ctx = new AudioContext({ sampleRate: SAMPLE_RATE })
    ctxRef.current = ctx
    const source = ctx.createMediaStreamSource(streamRef.current)
    const proc = ctx.createScriptProcessor(4096, 1, 1)

    chunksRef.current = []
    proc.onaudioprocess = (e) => {
      const float32 = e.inputBuffer.getChannelData(0)
      chunksRef.current.push(new Float32Array(float32))
      let sum = 0
      for (let i = 0; i < float32.length; i++) sum += float32[i] * float32[i]
      onAudioLevel(Math.sqrt(sum / float32.length))
    }

    source.connect(proc)
    proc.connect(ctx.destination)
    recorderRef.current = { proc, source }

    onRecordingStart()
    return true
  }, [onAudioLevel, onRecordingStart, onError])

  const stop = useCallback(() => {
    if (!recorderRef.current) return

    const { proc, source } = recorderRef.current
    proc.disconnect()
    source.disconnect()
    ctxRef.current?.close()
    ctxRef.current = null
    recorderRef.current = null

    onAudioLevel(0)

    const chunks = chunksRef.current
    chunksRef.current = []

    const total = chunks.reduce((n, c) => n + c.length, 0)
    const merged = new Float32Array(total)
    let offset = 0
    for (const chunk of chunks) {
      merged.set(chunk, offset)
      offset += chunk.length
    }

    onRecordingEnd(merged)
  }, [onAudioLevel, onRecordingEnd])

  const isRecording = useCallback(() => !!recorderRef.current, [])

  return { start, stop, isRecording }
}

export function float32ToBase64(float32: Float32Array): string {
  const int16 = new Int16Array(float32.length)
  for (let i = 0; i < float32.length; i++) {
    int16[i] = Math.max(-32768, Math.min(32767, Math.round(float32[i] * 32767)))
  }
  const bytes = new Uint8Array(int16.buffer)
  let binary = ''
  for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i])
  return btoa(binary)
}
