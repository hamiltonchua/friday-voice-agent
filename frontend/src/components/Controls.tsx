// frontend/src/components/Controls.tsx
import { useRef, useCallback } from 'react'
import { Trash2, Ear, EarOff, Radio } from 'lucide-react'
import { float32ToBase64 } from '../hooks/useManualRecording'
import { SAMPLE_RATE } from '../constants'
import type { MicVAD } from '@ricky0123/vad-web'

interface ControlsProps {
  micState: 'idle' | 'recording' | 'vad-active' | 'sleeping'
  wakeWordActive: boolean
  wakeWordEnabled: boolean
  audioLevelRingRef: React.RefObject<HTMLDivElement | null>
  meetingMode: boolean
  voiceEnabled: boolean
  onMicDown: (e: React.MouseEvent | React.TouchEvent) => void
  onMicUp: (e: React.MouseEvent | React.TouchEvent) => void
  onMicLeave: () => void
  onClear: () => void
  onWakeToggle: () => void
  onMeetingToggle: () => void
  send: (msg: Record<string, unknown>) => void
  vadRef: React.RefObject<MicVAD | null>
  meetingModeRef: React.RefObject<boolean>
  wakeWordEnabledRef: React.RefObject<boolean>
  startWakeStream: () => Promise<void>
  setStatus: (text: string, cls?: 'active' | 'listening' | 'sleeping' | '') => void
  wakeWordName: string | null
}

export function Controls({
  micState, wakeWordActive, wakeWordEnabled, audioLevelRingRef,
  meetingMode, voiceEnabled, onMicDown, onMicUp, onMicLeave,
  onClear, onWakeToggle, onMeetingToggle,
  send, vadRef, meetingModeRef,
  wakeWordEnabledRef, startWakeStream, setStatus, wakeWordName,
}: ControlsProps) {
  void wakeWordEnabledRef
  void startWakeStream

  const commandRecordingRef = useRef(false)
  const commandChunksRef = useRef<Float32Array[]>([])
  const commandStreamRef = useRef<MediaStream | null>(null)
  const commandCtxRef = useRef<AudioContext | null>(null)
  const commandProcessorRef = useRef<ScriptProcessorNode | null>(null)

  const micBtnClass = [
    'mic-btn',
    !voiceEnabled ? 'voice-disabled' : '',
    micState === 'recording' ? 'recording' : '',
    micState === 'vad-active' ? 'vad-active' : '',
    micState === 'sleeping' ? 'sleeping' : '',
  ].filter(Boolean).join(' ')

  const startCommandRecording = useCallback(async () => {
    if (commandRecordingRef.current) return
    commandRecordingRef.current = true
    commandChunksRef.current = []
    if (vadRef.current) vadRef.current.pause()

    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { sampleRate: 16000, channelCount: 1, echoCancellation: true },
    })
    commandStreamRef.current = stream
    const ctx = new AudioContext({ sampleRate: 16000 })
    commandCtxRef.current = ctx
    const source = ctx.createMediaStreamSource(stream)
    const processor = ctx.createScriptProcessor(4096, 1, 1)
    processor.onaudioprocess = (e) => {
      commandChunksRef.current.push(new Float32Array(e.inputBuffer.getChannelData(0)))
    }
    source.connect(processor)
    processor.connect(ctx.destination)
    commandProcessorRef.current = processor
    setStatus('Recording command...', 'listening')
  }, [vadRef, setStatus])

  const stopCommandRecording = useCallback(() => {
    if (!commandRecordingRef.current) return
    commandRecordingRef.current = false

    commandProcessorRef.current?.disconnect()
    commandCtxRef.current?.close()
    commandStreamRef.current?.getTracks().forEach(t => t.stop())
    commandStreamRef.current = null

    const chunks = commandChunksRef.current
    commandChunksRef.current = []
    const total = chunks.reduce((n, c) => n + c.length, 0)
    const merged = new Float32Array(total)
    let offset = 0
    for (const chunk of chunks) { merged.set(chunk, offset); offset += chunk.length }

    if (vadRef.current && meetingModeRef.current) vadRef.current.start()

    if (merged.length < SAMPLE_RATE * 0.3) {
      setStatus('Too short — hold longer', 'active')
      return
    }
    send({ type: 'meeting_command', data: float32ToBase64(merged) })
    setStatus('Processing command...', 'active')
  }, [vadRef, meetingModeRef, send, setStatus])

  const WakeIcon = wakeWordActive ? Ear : EarOff

  return (
    <>
      {/* Floating meeting command button — visible when in meeting mode */}
      {meetingMode && voiceEnabled && (
        <button
          style={{
            position: 'fixed',
            bottom: 80,
            right: 24,
            width: 52,
            height: 52,
            borderRadius: '50%',
            background: 'var(--accent)',
            border: 'none',
            color: 'white',
            fontSize: '1.3rem',
            cursor: 'pointer',
            zIndex: 50,
            boxShadow: '0 2px 12px rgba(0,0,0,0.4)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
          title="Hold to give Kismet a command"
          onMouseDown={(e) => { e.preventDefault(); startCommandRecording() }}
          onMouseUp={(e) => { e.preventDefault(); stopCommandRecording() }}
          onMouseLeave={() => { if (commandRecordingRef.current) stopCommandRecording() }}
          onTouchStart={(e) => { e.preventDefault(); startCommandRecording() }}
          onTouchEnd={(e) => { e.preventDefault(); stopCommandRecording() }}
        >
          💬
        </button>
      )}

      {/* Mic button */}
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 16 }}>
        <button
          className={micBtnClass}
          title="Hold to talk, or say the wake word"
          onMouseDown={onMicDown}
          onMouseUp={onMicUp}
          onMouseLeave={onMicLeave}
          onTouchStart={onMicDown}
          onTouchEnd={onMicUp}
        >
          <div className="audio-ring" ref={audioLevelRingRef} />
          <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/>
            <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
            <line x1="12" y1="19" x2="12" y2="23"/>
            <line x1="8" y1="23" x2="16" y2="23"/>
          </svg>
        </button>

        {/* Secondary buttons row */}
        <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
          <button
            className="ctrl-btn"
            title="Clear conversation"
            onClick={onClear}
          >
            <Trash2 size={18} />
          </button>

          <button
            className={`ctrl-btn ${wakeWordActive ? 'wake-active' : ''} ${!voiceEnabled ? 'voice-disabled' : ''}`}
            title={!voiceEnabled ? 'Enable voice input first' : wakeWordEnabled ? 'Toggle wake word mode' : 'Toggle hands-free listening'}
            onClick={onWakeToggle}
            disabled={!voiceEnabled}
          >
            <WakeIcon size={18} />
          </button>

          <button
            className={`ctrl-btn ${meetingMode ? 'meeting-active' : ''} ${!voiceEnabled ? 'voice-disabled' : ''}`}
            title={!voiceEnabled ? 'Enable voice input first' : meetingMode ? 'End meeting mode' : 'Start meeting mode'}
            onClick={onMeetingToggle}
            disabled={!voiceEnabled}
          >
            <Radio size={18} />
          </button>
        </div>

        {/* Wake word hint text */}
        {voiceEnabled && wakeWordActive && micState === 'sleeping' && wakeWordName && (
          <p style={{ fontSize: '0.75rem', color: 'var(--text2)', textAlign: 'center', marginTop: 4 }}>
            Say &ldquo;{wakeWordName}&rdquo; to wake me
          </p>
        )}
      </div>
    </>
  )
}
