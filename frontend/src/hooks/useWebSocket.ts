// frontend/src/hooks/useWebSocket.ts
import { useRef, useCallback, useEffect } from 'react'
import { MAX_RECONNECT_ATTEMPTS, BASE_RECONNECT_MS } from '../constants'

export type WSMessage = Record<string, unknown>

export interface UseWebSocketOptions {
  onMessage: (msg: WSMessage) => void
  onOpen?: () => void
  onClose?: () => void
}

export function useWebSocket({ onMessage, onOpen, onClose }: UseWebSocketOptions) {
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectAttemptsRef = useRef(0)
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const wasConnectedRef = useRef(false)

  const send = useCallback((data: WSMessage) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data))
    }
  }, [])

  const connect = useCallback(() => {
    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current)
      reconnectTimerRef.current = null
    }

    const proto = location.protocol === 'https:' ? 'wss' : 'ws'
    const ws = new WebSocket(`${proto}://${location.host}/ws`)
    wsRef.current = ws

    ws.onopen = () => {
      console.log('[WS] Connected')
      reconnectAttemptsRef.current = 0
      onOpen?.()
      wasConnectedRef.current = true
    }

    ws.onmessage = (e) => {
      try {
        const msg = JSON.parse(e.data) as WSMessage
        onMessage(msg)
      } catch {
        console.error('[WS] Failed to parse message', e.data)
      }
    }

    ws.onclose = () => {
      console.log('[WS] Disconnected')
      onClose?.()
      wsRef.current = null

      if (reconnectAttemptsRef.current < MAX_RECONNECT_ATTEMPTS) {
        const delay = Math.min(
          BASE_RECONNECT_MS * Math.pow(2, reconnectAttemptsRef.current),
          30000
        )
        reconnectAttemptsRef.current++
        console.log(`[WS] Reconnecting in ${delay}ms (attempt ${reconnectAttemptsRef.current})`)
        reconnectTimerRef.current = setTimeout(connect, delay)
      }
    }

    ws.onerror = (err) => {
      console.error('[WS] Error:', err)
    }
  }, [onMessage, onOpen, onClose])

  const reconnect = useCallback(() => {
    reconnectAttemptsRef.current = 0
    connect()
  }, [connect])

  useEffect(() => {
    connect()
    return () => {
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current)
      wsRef.current?.close()
    }
  }, []) // intentionally empty — connect is stable but we only want to run on mount

  return { send, reconnect, wsRef, reconnectAttemptsRef, wasConnectedRef }
}
