// frontend/src/components/SettingsDrawer.tsx
import { Settings } from 'lucide-react'
import { Sheet, SheetContent, SheetHeader, SheetTitle, SheetTrigger } from './ui/sheet'
import { ENROLL_MIN, ENROLL_MAX } from '../constants'

interface SettingsDrawerProps {
  enrolled: boolean
  verifyEnabled: boolean
  onEnroll: () => void
  onVerifyToggle: () => void
}

export function SettingsDrawer({ enrolled, verifyEnabled, onEnroll, onVerifyToggle }: SettingsDrawerProps) {
  return (
    <Sheet>
      <SheetTrigger asChild>
        <button
          className="ctrl-btn"
          title="Settings"
        >
          <Settings size={20} />
        </button>
      </SheetTrigger>
      <SheetContent side="left">
        <div style={{ padding: '24px 20px' }}>
        <SheetHeader>
          <SheetTitle>Settings</SheetTitle>
        </SheetHeader>

        <div style={{ marginTop: 24, display: 'flex', flexDirection: 'column', gap: 20 }}>
          {/* Speaker Verification */}
          <div>
            <p style={{ fontSize: '0.75rem', color: 'var(--text2)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 12 }}>
              Speaker Verification
            </p>

            <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
              <button
                className="ctrl-btn"
                onClick={onEnroll}
                title="Enroll your voice for speaker verification"
                style={{ width: '100%', borderRadius: 8, height: 40, fontSize: '0.85rem', color: 'var(--text2)', background: 'var(--surface2)', border: 'none', cursor: 'pointer', padding: '0 12px', display: 'flex', alignItems: 'center', gap: 8 }}
              >
                <span>🎤</span>
                <span>{enrolled ? 'Re-enroll Voice' : 'Enroll Voice'}</span>
              </button>

              {enrolled && (
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '8px 12px', background: 'var(--surface2)', borderRadius: 8 }}>
                  <span style={{ fontSize: '0.85rem', color: 'var(--text2)' }}>Verify Speaker</span>
                  <button
                    onClick={onVerifyToggle}
                    style={{
                      width: 40,
                      height: 22,
                      borderRadius: 11,
                      border: 'none',
                      cursor: 'pointer',
                      background: verifyEnabled ? 'var(--green)' : 'var(--surface)',
                      position: 'relative',
                      transition: 'background 0.2s',
                    }}
                    title="Toggle speaker verification"
                  >
                    <span style={{
                      position: 'absolute',
                      top: 2,
                      left: verifyEnabled ? 20 : 2,
                      width: 18,
                      height: 18,
                      borderRadius: '50%',
                      background: 'white',
                      transition: 'left 0.2s',
                    }} />
                  </button>
                </div>
              )}
            </div>

            <p style={{ fontSize: '0.7rem', color: 'var(--text2)', marginTop: 8 }}>
              {enrolled
                ? `Enrolled. Verification is ${verifyEnabled ? 'on' : 'off'}.`
                : `Not enrolled. Record ${ENROLL_MIN}–${ENROLL_MAX} voice samples to enable speaker verification.`}
            </p>
          </div>
        </div>
        </div>
      </SheetContent>
    </Sheet>
  )
}
