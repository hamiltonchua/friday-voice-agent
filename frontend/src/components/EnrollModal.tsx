// frontend/src/components/EnrollModal.tsx
import { useState, useRef, useCallback, useEffect } from 'react'
import { ENROLL_PROMPTS, ENROLL_MIN, ENROLL_MAX, SAMPLE_RATE } from '../constants'
import { float32ToBase64 } from '../hooks/useManualRecording'

interface EnrollModalProps {
  show: boolean
  step: number
  onCancel: () => void
  send: (msg: Record<string, unknown>) => void
}

export function EnrollModal({ show, step, onCancel, send }: EnrollModalProps) {
  const [recording, setRecording] = useState(false)
  const [btnText, setBtnText] = useState('🎤 Hold to Record')
  const streamRef = useRef<MediaStream | null>(null)
  const ctxRef = useRef<AudioContext | null>(null)
  const procRef = useRef<ScriptProcessorNode | null>(null)
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null)
  const chunksRef = useRef<Float32Array[]>([])

  const startRecording = useCallback(async () => {
    if (!streamRef.current) {
      streamRef.current = await navigator.mediaDevices.getUserMedia({
        audio: { sampleRate: SAMPLE_RATE, channelCount: 1, echoCancellation: true, noiseSuppression: true },
      })
    }
    const ctx = new AudioContext({ sampleRate: SAMPLE_RATE })
    ctxRef.current = ctx
    const source = ctx.createMediaStreamSource(streamRef.current)
    sourceRef.current = source
    const proc = ctx.createScriptProcessor(4096, 1, 1)
    procRef.current = proc
    chunksRef.current = []
    proc.onaudioprocess = (e) => {
      chunksRef.current.push(new Float32Array(e.inputBuffer.getChannelData(0)))
    }
    source.connect(proc)
    proc.connect(ctx.destination)
    setRecording(true)
    setBtnText('⏹ Recording...')
  }, [])

  const stopRecording = useCallback(() => {
    if (!recording) return
    setRecording(false)
    procRef.current?.disconnect()
    sourceRef.current?.disconnect()
    ctxRef.current?.close()

    const chunks = chunksRef.current
    chunksRef.current = []
    const total = chunks.reduce((n, c) => n + c.length, 0)
    if (total < SAMPLE_RATE * 0.5) {
      setBtnText('🎤 Too short, try again')
      return
    }
    const merged = new Float32Array(total)
    let off = 0
    for (const c of chunks) { merged.set(c, off); off += c.length }
    send({ type: 'enroll_sample', data: float32ToBase64(merged) })
    setBtnText('🎤 Hold to Record')
  }, [recording, send])

  // Auto-complete when max samples reached
  useEffect(() => {
    if (show && step >= ENROLL_MAX) {
      send({ type: 'enroll_complete' })
    }
  }, [show, step, send])

  if (!show) return null

  const isDone = step >= ENROLL_MAX
  const canFinish = step >= ENROLL_MIN && !isDone
  const currentPrompt = isDone ? 'Processing enrollment...' : ENROLL_PROMPTS[step % ENROLL_PROMPTS.length]
  const infoText = isDone ? '' : step < ENROLL_MIN
    ? `Sample ${step + 1} of ${ENROLL_MIN} required (up to ${ENROLL_MAX})`
    : `Sample ${step + 1} of ${ENROLL_MAX} — ${step} recorded, you can finish or keep going`

  const handleFinish = useCallback(() => {
    send({ type: 'enroll_complete' })
  }, [send])

  return (
    <div className="modal-overlay show">
      <div className="modal">
        <h2 id="enrollTitle">{isDone ? '✅ Enrolled!' : 'Voice Enrollment'}</h2>
        <p>Read each sentence aloud to enroll your voice. This lets Kismet verify it&apos;s you speaking.</p>
        <p style={{ fontSize: '0.8rem', opacity: 0.7 }}>Minimum {ENROLL_MIN} samples, up to {ENROLL_MAX} for better accuracy.</p>
        <div className="prompt-text">&quot;{currentPrompt}&quot;</div>
        <div className="progress">
          {Array.from({ length: Math.min(step + 1, ENROLL_MAX) }, (_, i) => (
            <div
              key={i}
              className={`dot ${i < step ? 'done' : i === step ? 'active' : ''}`}
            />
          ))}
        </div>
        <p style={{ fontSize: '0.8rem' }}>{infoText}</p>
        <div>
          {!isDone && (
            <button
              className="modal-btn"
              onMouseDown={(e) => { e.preventDefault(); startRecording() }}
              onMouseUp={(e) => { e.preventDefault(); stopRecording() }}
              onMouseLeave={() => { if (recording) stopRecording() }}
              onTouchStart={(e) => { e.preventDefault(); startRecording() }}
              onTouchEnd={(e) => { e.preventDefault(); stopRecording() }}
            >
              {btnText}
            </button>
          )}
          {canFinish && (
            <button className="modal-btn" onClick={handleFinish}>
              ✅ Done ({step} samples)
            </button>
          )}
          <button className="modal-btn secondary" onClick={onCancel}>
            {isDone ? 'Close' : 'Cancel'}
          </button>
        </div>
      </div>
    </div>
  )
}
