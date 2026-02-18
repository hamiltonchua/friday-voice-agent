// frontend/src/components/ReconnectBanner.tsx
interface ReconnectBannerProps {
  show: boolean
  onReconnect: () => void
}

export function ReconnectBanner({ show, onReconnect }: ReconnectBannerProps) {
  if (!show) return null
  return (
    <div
      onClick={onReconnect}
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        background: 'var(--accent)',
        color: 'white',
        textAlign: 'center',
        padding: '10px',
        fontSize: '0.85rem',
        zIndex: 300,
        cursor: 'pointer',
        fontWeight: 600,
      }}
    >
      Connection lost. Click to retry.
    </div>
  )
}
