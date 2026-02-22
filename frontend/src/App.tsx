// frontend/src/App.tsx
import { useState, useRef, useCallback, useEffect } from 'react'
import { Toaster, toast } from 'sonner'
import { useWebSocket } from './hooks/useWebSocket'
import { useAudio } from './hooks/useAudio'
import { useWakeWordStream } from './hooks/useWakeWordStream'
import { useVAD } from './hooks/useVAD'
import { useManualRecording, float32ToBase64 } from './hooks/useManualRecording'
import { useAudioLevel } from './hooks/useAudioLevel'
import { SAMPLE_RATE } from './constants'
import type { ChatMessage, MeetingEntry, ClientState, TimingInfo } from './types'
import { Header } from './components/Header'
import { ChatDisplay } from './components/ChatDisplay'
import { Controls } from './components/Controls'
import { EnrollModal } from './components/EnrollModal'
import { ReconnectBanner } from './components/ReconnectBanner'
import { MeetingBanner } from './components/MeetingBanner'
import { SettingsDrawer } from './components/SettingsDrawer'

export default function App() {
  // ---- Connection state ----
  const [_connectionStatus, setConnectionStatus] = useState<'connecting' | 'reconnecting' | 'disconnected' | 'connected'>('connecting')
  const [showReconnectBanner, setShowReconnectBanner] = useState(false)

  // ---- Server capabilities ----
  const [wakeWordEnabled, setWakeWordEnabled] = useState(false)
  const [wakeWordName, setWakeWordName] = useState<string | null>(null)
  const [_idleTimeout, setIdleTimeout] = useState(30)

  // ---- Client mode state ----
  const [wakeWordActive, setWakeWordActive] = useState(false)
  const [clientState, setClientState] = useState<ClientState>('awake')
  const clientStateRef = useRef<ClientState>('awake')

  // ---- Processing state ----
  const [_isProcessing, setIsProcessing] = useState(false)
  const isProcessingRef = useRef(false)

  // ---- Status display ----
  const [statusText, setStatusText] = useState('Connecting...')
  const [statusClass, setStatusClass] = useState<'active' | 'listening' | 'sleeping' | ''>('')

  // ---- Chat messages ----
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [_streamingId, setStreamingId] = useState<string | null>(null)
  const streamingTextRef = useRef('')

  // ---- Speaker verification ----
  const [enrolled, setEnrolled] = useState(false)
  const [verifyEnabled, setVerifyEnabled] = useState(false)
  const [lastVerifyScore, setLastVerifyScore] = useState<number | null>(null)
  const lastVerifyScoreRef = useRef<number | null>(null)
  const [showEnrollModal, setShowEnrollModal] = useState(false)
  const [enrollStep, setEnrollStep] = useState(0)

  // ---- Meeting mode ----
  const [meetingMode, setMeetingMode] = useState(false)
  const meetingModeRef = useRef(false)
  const [meetingEntries, setMeetingEntries] = useState<MeetingEntry[]>([])
  const meetingCommandCaptureRef = useRef(false)

  // ---- Mic state for UI ----
  const [micState, setMicState] = useState<'idle' | 'recording' | 'vad-active' | 'sleeping'>('idle')

  // ---- Audio level ring ----
  const audioLevelRingRef = useRef<HTMLDivElement | null>(null)
  const { updateAudioLevel } = useAudioLevel(audioLevelRingRef)

  // ---- Reconnect state ----
  const savedWakeWordActiveRef = useRef(false)

  // ---- Refs for callbacks that need current state ----
  const wakeWordActiveRef = useRef(false)
  const wakeWordEnabledRef = useRef(false)
  const wakeWordNameRef = useRef<string | null>(null)

  // Keep refs in sync
  useEffect(() => { clientStateRef.current = clientState }, [clientState])
  useEffect(() => { isProcessingRef.current = _isProcessing }, [_isProcessing])
  useEffect(() => { wakeWordActiveRef.current = wakeWordActive }, [wakeWordActive])
  useEffect(() => { wakeWordEnabledRef.current = wakeWordEnabled }, [wakeWordEnabled])
  useEffect(() => { wakeWordNameRef.current = wakeWordName }, [wakeWordName])
  useEffect(() => { meetingModeRef.current = meetingMode }, [meetingMode])

  const setStatus = useCallback((text: string, cls: 'active' | 'listening' | 'sleeping' | '' = '') => {
    setStatusText(text)
    setStatusClass(cls)
  }, [])

  // ---- Audio ----
  const { enqueueAudio, stopPlayback, playAudio, isSpeakingRef } = useAudio()

  // forward-declare send so we can reference it in callbacks
  const sendRef = useRef<(data: Record<string, unknown>) => void>(() => {})

  // ---- Interrupt ----
  const interrupt = useCallback(() => {
    console.log('[Interrupt] User interrupted')
    stopPlayback()
    setMessages(prev => prev.map(m =>
      m.streaming ? { ...m, streaming: false, interrupted: true } : m
    ))
    setStreamingId(null)
    streamingTextRef.current = ''
    sendRef.current({ type: 'cancel' })
    setIsProcessing(false)
    isProcessingRef.current = false
  }, [stopPlayback])

  // ---- Wake word streaming ----
  const shouldStreamWakeWord = useCallback(() => {
    return clientStateRef.current === 'sleeping' ||
      (meetingModeRef.current && !meetingCommandCaptureRef.current)
  }, [])

  const { start: startWakeStream, stop: stopWakeStream } = useWakeWordStream({
    onAudioLevel: updateAudioLevel,
    shouldStream: shouldStreamWakeWord,
    onSend: (b64) => sendRef.current({ type: 'audio_stream', data: b64 }),
  })

  // ---- sendAudio — ref-based to avoid circular dependency ----
  const sendAudioRef = useRef<(float32: Float32Array) => void>(() => {})

  // ---- VAD ----
  const handleVADSpeechStart = useCallback(() => {
    if (!meetingModeRef.current && (isSpeakingRef.current || isProcessingRef.current)) {
      interrupt()
    }
    setMicState('recording')
    setStatus(meetingModeRef.current ? 'Meeting mode — speaking detected...' : 'Listening...', 'listening')
  }, [interrupt, isSpeakingRef, setStatus])

  const handleVADSpeechEnd = useCallback((audio: Float32Array) => {
    setMicState(wakeWordActiveRef.current ? 'vad-active' : 'idle')
    if (audio.length > SAMPLE_RATE * 0.3) {
      sendAudioRef.current(audio)
    }
  }, [])

  const { start: startVAD, pause: pauseVAD, isListening, vadRef } = useVAD({
    onSpeechStart: handleVADSpeechStart,
    onSpeechEnd: handleVADSpeechEnd,
    onFrameProcessed: (probs) => {
      updateAudioLevel(probs.isSpeech * 0.5)
    },
  })

  // ---- Manual recording ----
  const handleManualRecordingEnd = useCallback((float32: Float32Array) => {
    setMicState(wakeWordActiveRef.current && clientStateRef.current === 'awake' ? 'vad-active' : 'idle')
    updateAudioLevel(0)
    if (clientStateRef.current === 'awake' && wakeWordActiveRef.current && vadRef.current) {
      vadRef.current.start()
    }
    if (float32.length < SAMPLE_RATE * 0.3) {
      setStatus('Too short — hold longer', 'active')
      return
    }
    sendAudioRef.current(float32)
  }, [updateAudioLevel, setStatus, vadRef])

  const { start: startManual, stop: stopManual, isRecording } = useManualRecording({
    onAudioLevel: updateAudioLevel,
    onRecordingStart: () => setMicState('recording'),
    onRecordingEnd: handleManualRecordingEnd,
    onError: (msg) => toast.error(msg),
  })

  const updateStatusForMode = useCallback(() => {
    if (wakeWordActiveRef.current) {
      if (clientStateRef.current === 'sleeping') {
        setStatus(`Say "${wakeWordNameRef.current}" to wake me`, 'sleeping')
      } else {
        setStatus('Listening — talk anytime', 'active')
      }
    } else if (isListening()) {
      setStatus('Listening...', 'listening')
    } else {
      setStatus('Ready — hold mic to talk', 'active')
    }
  }, [setStatus, isListening])

  // ---- Send audio helper ----
  const sendAudio = useCallback((float32: Float32Array) => {
    const b64 = float32ToBase64(float32)
    if (meetingModeRef.current && meetingCommandCaptureRef.current) {
      meetingCommandCaptureRef.current = false
      sendRef.current({ type: 'meeting_command', data: b64 })
      setStatus('Processing command...', 'active')
      if (wakeWordEnabledRef.current) startWakeStream()
    } else if (meetingModeRef.current) {
      sendRef.current({ type: 'meeting_audio', data: b64 })
      setStatus('Meeting mode — listening...', 'listening')
    } else {
      sendRef.current({ type: 'audio', data: b64 })
      setIsProcessing(true)
      isProcessingRef.current = true
      setStatus('Processing...', 'active')
    }
  }, [setStatus, startWakeStream])

  useEffect(() => { sendAudioRef.current = sendAudio }, [sendAudio])

  // ---- Wake word mode ----
  const enableWakeWordMode = useCallback(async () => {
    setWakeWordActive(true)
    wakeWordActiveRef.current = true
    setClientState('sleeping')
    clientStateRef.current = 'sleeping'
    setMicState('sleeping')
    if (vadRef.current) pauseVAD()
    sendRef.current({ type: 'set_state', state: 'sleeping' })
    try {
      await startWakeStream()
    } catch (e: unknown) {
      const err = e as Error
      if (err.message === 'MIC_DENIED') {
        toast.error('Mic access denied — enable microphone in browser settings', { duration: 6000 })
      }
    }
    setStatus(`Say "${wakeWordNameRef.current}" to wake me`, 'sleeping')
  }, [pauseVAD, startWakeStream, setStatus, vadRef])

  const disableWakeWordMode = useCallback(() => {
    setWakeWordActive(false)
    wakeWordActiveRef.current = false
    setClientState('awake')
    clientStateRef.current = 'awake'
    setMicState('idle')
    stopWakeStream()
    if (vadRef.current) pauseVAD()
    updateStatusForMode()
  }, [stopWakeStream, pauseVAD, updateStatusForMode, vadRef])

  const handleWake = useCallback(async () => {
    setClientState('awake')
    clientStateRef.current = 'awake'
    setMicState('vad-active')
    stopWakeStream()
    await new Promise(r => setTimeout(r, 200))
    setStatus('Listening...', 'listening')
    await startVAD()
  }, [stopWakeStream, setStatus, startVAD])

  const handleSleep = useCallback(() => {
    if (!wakeWordActiveRef.current) return
    setClientState('sleeping')
    clientStateRef.current = 'sleeping'
    setMicState('sleeping')
    if (vadRef.current) pauseVAD()
    startWakeStream()
    setStatus(`Say "${wakeWordNameRef.current}" to wake me`, 'sleeping')
  }, [pauseVAD, startWakeStream, setStatus, vadRef])

  // ---- Meeting mode ----
  const enterMeetingMode = useCallback(async () => {
    setMeetingMode(true)
    meetingModeRef.current = true
    setMessages([{ id: 'meeting-start', role: 'system', text: 'Meeting mode started. Transcribing all speakers... Say "Hey Friday" to ask a question.' }])
    setMeetingEntries([])
    sendRef.current({ type: 'meeting_start' })
    await startVAD()
    setMicState('vad-active')
    if (wakeWordEnabledRef.current) startWakeStream()
    setStatus('Meeting mode — listening...', 'listening')
  }, [setStatus, startVAD, startWakeStream])

  const exitMeetingMode = useCallback(() => {
    setMeetingMode(false)
    meetingModeRef.current = false
    sendRef.current({ type: 'meeting_stop' })
    pauseVAD()
    stopWakeStream()
    setMicState('idle')
    updateStatusForMode()
  }, [pauseVAD, stopWakeStream, updateStatusForMode])

  const handleMeetingWake = useCallback(async () => {
    meetingCommandCaptureRef.current = true
    stopWakeStream()
    try {
      const actx = new AudioContext()
      const osc = actx.createOscillator()
      const gain = actx.createGain()
      osc.frequency.value = 880
      gain.gain.value = 0.15
      osc.connect(gain).connect(actx.destination)
      osc.start()
      osc.stop(actx.currentTime + 0.15)
      osc.onended = () => actx.close()
    } catch { /* ignore */ }
    setStatus('🎤 Listening for your command...', 'listening')
    if (vadRef.current) pauseVAD()
    await new Promise(r => setTimeout(r, 300))
    if (vadRef.current) vadRef.current.start()
  }, [stopWakeStream, setStatus, pauseVAD, vadRef])

  // ---- WebSocket message handler ----
  const handleWSMessage = useCallback((msg: Record<string, unknown>) => {
    const type = msg.type as string

    if (type === 'ready') {
      const wwe = msg.wake_word_enabled as boolean
      let wwn = msg.wake_word as string | null
      // Extract friendly name from .ppn file path (e.g. "/path/to/hey-friday_en_mac.ppn" → "Hey Friday")
      if (wwn && wwn.endsWith('.ppn')) {
        const base = wwn.split('/').pop()!.replace(/_en.*\.ppn$/, '').replace(/[-_]/g, ' ')
        wwn = base.split(' ').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ')
      }
      const it = (msg.idle_timeout as number) || 30
      setWakeWordEnabled(wwe)
      wakeWordEnabledRef.current = wwe
      setWakeWordName(wwn)
      wakeWordNameRef.current = wwn
      setIdleTimeout(it)
      setConnectionStatus('connected')
      setShowReconnectBanner(false)

      if (wwe) {
        if (savedWakeWordActiveRef.current || wakeWordActiveRef.current) {
          enableWakeWordMode()
        } else {
          updateStatusForMode()
        }
      } else {
        updateStatusForMode()
      }
      sendRef.current({ type: 'check_enrollment' })

    } else if (type === 'error') {
      toast.error(msg.text as string)
      setIsProcessing(false)
      isProcessingRef.current = false
      updateStatusForMode()

    } else if (type === 'wake') {
      if (msg.meeting && meetingModeRef.current) {
        handleMeetingWake()
      } else {
        handleWake()
      }

    } else if (type === 'sleep') {
      handleSleep()

    } else if (type === 'status') {
      setStatus(msg.text as string, 'active')

    } else if (type === 'transcript') {
      const metaParts: string[] = []
      if (msg.time) metaParts.push(`${msg.time}s`)
      if (msg.role === 'user' && lastVerifyScoreRef.current !== null) {
        const score = lastVerifyScoreRef.current
        const badge = score >= 0.50 ? '✓' : '✗'
        metaParts.push(`speaker ${badge} ${score.toFixed(2)}`)
        lastVerifyScoreRef.current = null
      }
      setMessages(prev => [...prev, {
        id: crypto.randomUUID(),
        role: msg.role as 'user' | 'assistant',
        text: msg.text as string,
        meta: metaParts.join(' · ') || undefined,
      }])

    } else if (type === 'working') {
      setStatus('Using tools...', 'active')

    } else if (type === 'stream_start') {
      streamingTextRef.current = ''
      const id = crypto.randomUUID()
      setStreamingId(id)
      setMessages(prev => [...prev, {
        id,
        role: 'assistant',
        text: '',
        streaming: true,
      }])

    } else if (type === 'token') {
      streamingTextRef.current += (msg.text as string)
      const text = streamingTextRef.current
      setMessages(prev => prev.map(m =>
        m.streaming ? { ...m, text } : m
      ))

    } else if (type === 'audio_chunk') {
      enqueueAudio(msg.data as string)

    } else if (type === 'stream_end') {
      const t = msg.times as TimingInfo
      const parts = [`STT ${t.stt}s`, `LLM ${t.llm}s`, `TTS ${t.tts}s`]
      if (t.first_sentence) parts.push(`first audio ${t.first_sentence}s`)
      setMessages(prev => prev.map(m =>
        m.streaming ? { ...m, streaming: false, text: msg.text as string, meta: parts.join(' · ') } : m
      ))
      setStreamingId(null)
      streamingTextRef.current = ''
      setIsProcessing(false)
      isProcessingRef.current = false
      updateStatusForMode()

    } else if (type === 'meeting_transcript') {
      const entry: MeetingEntry = {
        id: crypto.randomUUID(),
        time: msg.time as string,
        speaker: msg.speaker as string,
        text: msg.text as string,
        isOwner: msg.is_owner as boolean,
        speakerType: (msg.is_owner as boolean) ? 'ham' : msg.speaker === 'Kismet' ? 'kismet' : 'other',
      }
      setMeetingEntries(prev => [...prev, entry])

    } else if (type === 'meeting_stopped') {
      const speakers = (msg.speakers as string[]).join(', ')
      setMessages(prev => [...prev, {
        id: crypto.randomUUID(),
        role: 'system',
        text: `Meeting ended. ${msg.entries} entries, ${(msg.speakers as string[]).length} speakers: ${speakers}`,
      }])

    } else if (type === 'cancelled') {
      setIsProcessing(false)
      isProcessingRef.current = false
      updateStatusForMode()

    } else if (type === 'audio') {
      playAudio(msg.data as string)
      const t = msg.times as TimingInfo
      setStatus(`STT ${t.stt}s · LLM ${t.llm}s · TTS ${t.tts}s`, 'active')
      setIsProcessing(false)
      isProcessingRef.current = false

    } else if (type === 'verified') {
      setLastVerifyScore(msg.score as number)
      lastVerifyScoreRef.current = msg.score as number

    } else if (type === 'rejected') {
      setLastVerifyScore(msg.score as number)
      lastVerifyScoreRef.current = msg.score as number
      setMessages(prev => [...prev, {
        id: crypto.randomUUID(),
        role: 'user',
        text: '[Voice not recognized]',
        meta: `Score: ${msg.score} · ${msg.time}s`,
        rejected: true,
      }])
      // Subtle double-beep so user knows (even when tab is in background)
      try {
        const actx = new AudioContext()
        const playTone = (freq: number, startTime: number, duration: number) => {
          const osc = actx.createOscillator()
          const gain = actx.createGain()
          osc.type = 'sine'
          osc.frequency.value = freq
          gain.gain.setValueAtTime(0.08, startTime)
          gain.gain.exponentialRampToValueAtTime(0.001, startTime + duration)
          osc.connect(gain).connect(actx.destination)
          osc.start(startTime)
          osc.stop(startTime + duration)
        }
        playTone(440, actx.currentTime, 0.12)
        playTone(330, actx.currentTime + 0.15, 0.12)
        setTimeout(() => actx.close(), 500)
      } catch (_) { /* ignore audio context errors */ }
      setIsProcessing(false)
      isProcessingRef.current = false
      updateStatusForMode()

    } else if (type === 'enroll_status') {
      if (msg.status === 'sample_received') {
        setEnrollStep(msg.samples as number)
      } else if (msg.status === 'complete') {
        setEnrolled(true)
        setVerifyEnabled(true)
        setShowEnrollModal(false)
      } else if (msg.status === 'error') {
        toast.error((msg.message as string) || 'Enrollment error')
      }

    } else if (type === 'enrollment_info') {
      setEnrolled(msg.enrolled as boolean)
      setVerifyEnabled(msg.verify_enabled as boolean)

    } else if (type === 'verify_toggled') {
      setVerifyEnabled(msg.enabled as boolean)
    }
  }, [
    enableWakeWordMode, updateStatusForMode, handleMeetingWake, handleWake, handleSleep,
    setStatus, enqueueAudio, playAudio, lastVerifyScore,
  ])

  // ---- WebSocket hook ----
  const { send, reconnect, reconnectAttemptsRef } = useWebSocket({
    onMessage: handleWSMessage,
    onOpen: () => {
      setConnectionStatus('connected')
      setShowReconnectBanner(false)
    },
    onClose: () => {
      stopWakeStream()
      stopPlayback()
      savedWakeWordActiveRef.current = wakeWordActiveRef.current
      if (reconnectAttemptsRef.current >= 10) {
        setShowReconnectBanner(true)
        setStatus('Connection lost', '')
      } else {
        setStatus(`Reconnecting (${reconnectAttemptsRef.current}/10)...`, '')
        setConnectionStatus('reconnecting')
      }
    },
  })

  useEffect(() => { sendRef.current = send }, [send])

  // ---- Wake toggle ----
  const handleWakeToggle = useCallback(async () => {
    if (!wakeWordEnabledRef.current) {
      if (isListening()) {
        pauseVAD()
        setMicState('idle')
        setWakeWordActive(false)
        wakeWordActiveRef.current = false
        setStatus('Click mic to talk', '')
      } else {
        setClientState('awake')
        clientStateRef.current = 'awake'
        setMicState('vad-active')
        setWakeWordActive(true)
        wakeWordActiveRef.current = true
        setStatus('Listening...', 'listening')
        await startVAD()
      }
      return
    }
    if (wakeWordActiveRef.current) {
      disableWakeWordMode()
    } else {
      enableWakeWordMode()
    }
  }, [isListening, pauseVAD, setStatus, startVAD, disableWakeWordMode, enableWakeWordMode])

  // ---- Clear ----
  const handleClear = useCallback(() => {
    setMessages([])
    setMeetingEntries([])
    stopPlayback()
    send({ type: 'clear' })
  }, [stopPlayback, send])

  // ---- Manual mic button ----
  const handleMicDown = useCallback(async (e: React.MouseEvent | React.TouchEvent) => {
    e.preventDefault()
    if (wakeWordActiveRef.current && clientStateRef.current === 'sleeping') {
      handleWake()
      send({ type: 'set_state', state: 'awake' })
    }
    if (isSpeakingRef.current || isProcessingRef.current) interrupt()
    if (vadRef.current) pauseVAD()
    await startManual()
  }, [handleWake, interrupt, pauseVAD, startManual, isSpeakingRef, vadRef, send])

  const handleMicUp = useCallback((e: React.MouseEvent | React.TouchEvent) => {
    e.preventDefault()
    stopManual()
  }, [stopManual])

  // ---- Spacebar support ----
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.code === 'Space' && !e.repeat && !isRecording()) {
        e.preventDefault()
        handleMicDown(e as unknown as React.MouseEvent)
      }
    }
    const onKeyUp = (e: KeyboardEvent) => {
      if (e.code === 'Space' && isRecording()) {
        e.preventDefault()
        stopManual()
      }
    }
    document.addEventListener('keydown', onKeyDown)
    document.addEventListener('keyup', onKeyUp)
    return () => {
      document.removeEventListener('keydown', onKeyDown)
      document.removeEventListener('keyup', onKeyUp)
    }
  }, [handleMicDown, stopManual, isRecording])

  // ---- Speaker verification ----
  const handleEnrollStart = useCallback(() => {
    setEnrollStep(0)
    setShowEnrollModal(true)
    send({ type: 'enroll_start' })
  }, [send])

  const handleEnrollCancel = useCallback(() => {
    send({ type: 'enroll_cancel' })
    setShowEnrollModal(false)
  }, [send])

  const handleVerifyToggle = useCallback(() => {
    const next = !verifyEnabled
    setVerifyEnabled(next)
    send({ type: 'toggle_verify', enabled: next })
  }, [verifyEnabled, send])

  // Derive connection dot state from existing state
  const connectionDot: 'connected' | 'disconnected' | 'sleeping' | 'connecting' = showReconnectBanner
    ? 'disconnected'
    : micState === 'sleeping'
    ? 'sleeping'
    : _connectionStatus === 'connecting' || _connectionStatus === 'reconnecting'
    ? 'connecting'
    : 'connected'

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', background: 'var(--bg)', color: 'var(--text)', overflow: 'hidden' }}>
      <Toaster position="top-right" theme="dark" />
      <ReconnectBanner
        show={showReconnectBanner}
        onReconnect={() => {
          setShowReconnectBanner(false)
          setStatus('Reconnecting...', '')
          reconnect()
        }}
      />

      {/* Mobile top bar */}
      <Header
        statusText={statusText}
        statusClass={statusClass}
        connectionDot={connectionDot}
        meetingMode={meetingMode}
        enrolled={enrolled}
        verifyEnabled={verifyEnabled}
        onEnroll={handleEnrollStart}
        onVerifyToggle={handleVerifyToggle}
        onMeetingToggle={() => meetingMode ? exitMeetingMode() : enterMeetingMode()}
      />

      {/* Two-panel grid — desktop only */}
      <div
        className="hidden md:grid"
        style={{
          flex: 1,
          display: 'grid',
          gridTemplateColumns: '320px 1fr',
          overflow: 'hidden',
        }}
      >
        {/* Left panel — Control Center */}
        <div
          style={{
            background: 'var(--surface)',
            borderRight: meetingMode ? '3px solid var(--accent)' : '1px solid rgba(255,255,255,0.08)',
            display: 'flex',
            flexDirection: 'column',
            padding: '28px 20px 20px',
            gap: 20,
            overflow: 'hidden',
          }}
        >
          {/* Top: Title + status */}
          <div>
            <h1 style={{ fontSize: '1.4rem', fontWeight: 700, marginBottom: 10 }}>Kismet</h1>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span className={`conn-dot ${connectionDot}`} />
              <span
                className={`status-${statusClass || 'default'}`}
                style={{ fontSize: '0.82rem', transition: 'color 0.2s' }}
              >
                {meetingMode ? 'Meeting Mode — Recording' : statusText}
              </span>
            </div>
          </div>

          {/* Middle: Button cluster */}
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 0 }}>
            <Controls
              micState={micState}
              wakeWordActive={wakeWordActive}
              wakeWordEnabled={wakeWordEnabled}
              audioLevelRingRef={audioLevelRingRef}
              meetingMode={meetingMode}
              onMicDown={handleMicDown}
              onMicUp={handleMicUp}
              onMicLeave={() => { if (isRecording()) stopManual() }}
              onClear={handleClear}
              onWakeToggle={handleWakeToggle}
              onMeetingToggle={() => meetingMode ? exitMeetingMode() : enterMeetingMode()}
              send={send}
              vadRef={vadRef}
              meetingModeRef={meetingModeRef}
              wakeWordEnabledRef={wakeWordEnabledRef}
              startWakeStream={startWakeStream}
              setStatus={setStatus}
              wakeWordName={wakeWordName}
            />
          </div>

          {/* Bottom: Settings gear */}
          <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
            <SettingsDrawer
              enrolled={enrolled}
              verifyEnabled={verifyEnabled}
              onEnroll={handleEnrollStart}
              onVerifyToggle={handleVerifyToggle}
            />
          </div>
        </div>

        {/* Right panel — Conversation */}
        <div style={{ display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
          <MeetingBanner show={meetingMode} meetingCommandCapture={meetingCommandCaptureRef} />
          <ChatDisplay
            messages={messages}
            meetingEntries={meetingEntries}
            meetingMode={meetingMode}
          />
        </div>
      </div>

      {/* Mobile layout — single column */}
      <div
        className="flex flex-col md:hidden"
        style={{ flex: 1, overflow: 'hidden' }}
      >
        <MeetingBanner show={meetingMode} meetingCommandCapture={meetingCommandCaptureRef} />
        <ChatDisplay
          messages={messages}
          meetingEntries={meetingEntries}
          meetingMode={meetingMode}
        />
        {/* Mobile bottom dock */}
        <div className="mobile-dock">
          <Controls
            micState={micState}
            wakeWordActive={wakeWordActive}
            wakeWordEnabled={wakeWordEnabled}
            audioLevelRingRef={audioLevelRingRef}
            meetingMode={meetingMode}
            onMicDown={handleMicDown}
            onMicUp={handleMicUp}
            onMicLeave={() => { if (isRecording()) stopManual() }}
            onClear={handleClear}
            onWakeToggle={handleWakeToggle}
            onMeetingToggle={() => meetingMode ? exitMeetingMode() : enterMeetingMode()}
            send={send}
            vadRef={vadRef}
            meetingModeRef={meetingModeRef}
            wakeWordEnabledRef={wakeWordEnabledRef}
            startWakeStream={startWakeStream}
            setStatus={setStatus}
            wakeWordName={wakeWordName}
          />
        </div>
      </div>

      <EnrollModal
        show={showEnrollModal}
        step={enrollStep}
        onCancel={handleEnrollCancel}
        send={send}
      />
    </div>
  )
}
