// frontend/src/hooks/useAudioLevel.ts
import { useRef, useCallback } from 'react'

export function useAudioLevel(ringRef: React.RefObject<HTMLDivElement | null>) {
  const rmsRef = useRef(0)

  const updateAudioLevel = useCallback((rms: number) => {
    rmsRef.current = rms
    const el = ringRef.current
    if (!el) return

    if (rms > 0.01) {
      el.classList.add('active')
      const spread = Math.min(rms * 40, 12)
      const alpha = Math.min(rms * 3, 0.8)
      el.style.boxShadow = `0 0 ${spread}px rgba(0,217,126,${alpha})`
    } else {
      el.classList.remove('active')
      el.style.boxShadow = 'none'
    }
  }, [ringRef])

  return { updateAudioLevel }
}
