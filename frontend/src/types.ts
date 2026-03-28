// frontend/src/types.ts

export type ConnectionStatus = 'connecting' | 'connected' | 'reconnecting' | 'disconnected'
export type ClientState = 'sleeping' | 'awake'
export type StatusClass = 'active' | 'listening' | 'sleeping' | ''

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant' | 'system'
  text: string
  meta?: string
  streaming?: boolean
  interrupted?: boolean
  rejected?: boolean
}

export interface MeetingEntry {
  id: string
  time: string
  speaker: string
  text: string
  isOwner: boolean
  speakerType: 'ham' | 'kismet' | 'other'
}

export interface ServerReadyMsg {
  type: 'ready'
  wake_word_enabled: boolean
  wake_word: string | null
  idle_timeout: number
}

export interface TimingInfo {
  stt: number
  llm: number
  tts: number
  first_sentence?: number
}

export interface BackgroundTask {
  id: string
  description: string
  startTime: number
  status: 'running' | 'complete' | 'error'
  error?: string
}
