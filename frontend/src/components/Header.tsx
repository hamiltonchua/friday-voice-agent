// frontend/src/components/Header.tsx
import * as React from 'react'

interface HeaderProps {
  statusText: string
  statusClass: 'active' | 'listening' | 'sleeping' | ''
  connectionDot: 'connected' | 'disconnected' | 'sleeping' | 'connecting'
  meetingMode: boolean
  enrolled: boolean
  verifyEnabled: boolean
  canvasEnabled: boolean
  onEnroll: () => void
  onVerifyToggle: () => void
  onMeetingToggle: () => void
}

export function Header({ statusText, statusClass, connectionDot, meetingMode, enrolled, verifyEnabled, canvasEnabled, onEnroll, onVerifyToggle, onMeetingToggle }: HeaderProps) {
  // Mobile-only top bar — hidden on desktop (md+)
  return (
    <header
      className="flex md:hidden items-center justify-between sticky top-0 z-30"
      style={{
        padding: '12px 16px',
        background: 'var(--surface)',
        borderBottom: '1px solid rgba(255,255,255,0.08)',
      }}
    >
      <h1 style={{ fontSize: '1.1rem', fontWeight: 600 }}>Kismet Labs</h1>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <span className={`conn-dot ${connectionDot}`} />
        {canvasEnabled && (
          <span style={{ fontSize: '0.6rem', background: 'var(--purple)', color: 'white', padding: '1px 5px', borderRadius: 4, fontWeight: 600 }}>
            CANVAS
          </span>
        )}
        <span
          className={`status-${statusClass || 'default'}`}
          style={{ fontSize: '0.8rem' }}
        >
          {statusText}
        </span>
      </div>
      <div style={{ display: 'flex', gap: 8 }}>
        <button
          style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text2)', fontSize: '0.75rem', padding: '4px 8px', borderRadius: 6, backgroundColor: 'var(--surface2)' } as React.CSSProperties}
          onClick={onEnroll}
        >
          {enrolled ? 'Re-enroll' : 'Enroll'}
        </button>
        {enrolled && (
          <button
            style={{ background: verifyEnabled ? 'var(--green)' : 'var(--surface2)', border: 'none', cursor: 'pointer', color: verifyEnabled ? 'var(--bg)' : 'var(--text2)', fontSize: '0.75rem', padding: '4px 8px', borderRadius: 6 } as React.CSSProperties}
            onClick={onVerifyToggle}
          >
            {verifyEnabled ? 'Verify: On' : 'Verify: Off'}
          </button>
        )}
        <button
          style={{ background: meetingMode ? 'var(--accent)' : 'var(--surface2)', border: 'none', cursor: 'pointer', color: meetingMode ? 'white' : 'var(--text2)', fontSize: '0.75rem', padding: '4px 8px', borderRadius: 6 } as React.CSSProperties}
          onClick={onMeetingToggle}
        >
          {meetingMode ? 'End Meeting' : 'Meeting'}
        </button>
      </div>
    </header>
  )
}
