// frontend/src/components/ChatInput.tsx
import { useState, useRef, useCallback, useEffect } from 'react'
import { SendHorizontal } from 'lucide-react'

interface ChatInputProps {
  onSend: (text: string) => void
  disabled?: boolean
}

export function ChatInput({ onSend, disabled }: ChatInputProps) {
  const [text, setText] = useState('')
  const inputRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    if (!disabled) {
      inputRef.current?.focus()
    }
  }, [disabled])

  const handleSend = useCallback(() => {
    const trimmed = text.trim()
    if (!trimmed || disabled) return
    onSend(trimmed)
    setText('')
    // Re-focus after send
    setTimeout(() => inputRef.current?.focus(), 0)
  }, [text, disabled, onSend])

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }, [handleSend])

  return (
    <div className="chat-input-bar">
      <textarea
        ref={inputRef}
        className="chat-input-textarea"
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="Type a message..."
        rows={1}
        disabled={disabled}
      />
      <button
        className="chat-input-send"
        onClick={handleSend}
        disabled={disabled || !text.trim()}
        title="Send message"
      >
        <SendHorizontal size={18} />
      </button>
    </div>
  )
}
