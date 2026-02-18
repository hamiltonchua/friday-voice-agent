// frontend/src/components/ChatDisplay.tsx
import { useEffect, useRef } from 'react'
import type { ChatMessage, MeetingEntry } from '../types'

interface ChatDisplayProps {
  messages: ChatMessage[]
  meetingEntries: MeetingEntry[]
  meetingMode: boolean
}

export function ChatDisplay({ messages, meetingEntries, meetingMode }: ChatDisplayProps) {
  const chatRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (chatRef.current) {
      chatRef.current.scrollTop = chatRef.current.scrollHeight
    }
  }, [messages, meetingEntries])

  return (
    <div
      ref={chatRef}
      style={{
        flex: 1,
        overflowY: 'auto',
        paddingBottom: 32,
        scrollBehavior: 'smooth',
      }}
    >
      {messages.map(msg => (
        <div
          key={msg.id}
          className={[
            'msg-row',
            msg.role,
            msg.streaming ? 'streaming' : '',
            msg.interrupted ? 'interrupted' : '',
            msg.rejected ? 'rejected' : '',
          ].filter(Boolean).join(' ')}
        >
          <span className="role-label">{msg.role}</span>
          <span className="msg-text">{msg.text}</span>
          {msg.meta && <div className="msg-meta">{msg.meta}</div>}
        </div>
      ))}

      {meetingMode && meetingEntries.map(entry => (
        <div key={entry.id} className="meeting-row">
          <span className="meeting-time">{entry.time}</span>
          <span className={`meeting-speaker ${entry.speakerType}`}>{entry.speaker}:</span>
          <span style={{ color: 'var(--text)', flex: 1 }}>{entry.text}</span>
        </div>
      ))}
    </div>
  )
}
