// frontend/src/components/ChatDisplay.tsx
import { useEffect, useRef } from 'react'
import type { ChatMessage, MeetingEntry } from '../types'

interface ChatDisplayProps {
  messages: ChatMessage[]
  meetingEntries: MeetingEntry[]
  meetingMode: boolean
}

type RichBlock =
  | { type: 'paragraph'; text: string }
  | { type: 'ul'; items: string[] }
  | { type: 'ol'; items: string[] }
  | { type: 'table'; headers: string[]; rows: string[][] }

function cleanInlineMarkdown(text: string): string {
  let s = text
  s = s.replace(/!\[([^\]]*)\]\([^)]+\)/g, '$1')
  s = s.replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
  s = s.replace(/`([^`]+)`/g, '$1')
  s = s.replace(/\*\*\*(.+?)\*\*\*/g, '$1')
  s = s.replace(/\*\*(.+?)\*\*/g, '$1')
  s = s.replace(/\*(.+?)\*/g, '$1')
  s = s.replace(/___(.+?)___/g, '$1')
  s = s.replace(/__(.+?)__/g, '$1')
  s = s.replace(/(?<!\w)_(.+?)_(?!\w)/g, '$1')
  return s.trim()
}

function parseTableRow(line: string): string[] {
  const trimmed = line.trim()
  const inner = trimmed.startsWith('|') ? trimmed.slice(1) : trimmed
  const noTail = inner.endsWith('|') ? inner.slice(0, -1) : inner
  return noTail.split('|').map(c => cleanInlineMarkdown(c.trim()))
}

function isTableSeparator(line: string): boolean {
  return /^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$/.test(line)
}

function parseRichBlocks(text: string): RichBlock[] {
  const lines = text.split('\n')
  const blocks: RichBlock[] = []
  let i = 0

  while (i < lines.length) {
    const line = lines[i]
    const trimmed = line.trim()

    if (!trimmed) {
      i++
      continue
    }

    if (
      i + 1 < lines.length &&
      lines[i].includes('|') &&
      isTableSeparator(lines[i + 1]) &&
      parseTableRow(lines[i]).length >= 2
    ) {
      const headers = parseTableRow(lines[i])
      i += 2
      const rows: string[][] = []
      while (i < lines.length && lines[i].includes('|') && lines[i].trim()) {
        const row = parseTableRow(lines[i])
        if (row.length) rows.push(row)
        i++
      }
      blocks.push({ type: 'table', headers, rows })
      continue
    }

    if (/^\s*[-*+]\s+/.test(line)) {
      const items: string[] = []
      while (i < lines.length && /^\s*[-*+]\s+/.test(lines[i])) {
        items.push(cleanInlineMarkdown(lines[i].replace(/^\s*[-*+]\s+/, '')))
        i++
      }
      blocks.push({ type: 'ul', items })
      continue
    }

    if (/^\s*\d+\.\s+/.test(line)) {
      const items: string[] = []
      while (i < lines.length && /^\s*\d+\.\s+/.test(lines[i])) {
        items.push(cleanInlineMarkdown(lines[i].replace(/^\s*\d+\.\s+/, '')))
        i++
      }
      blocks.push({ type: 'ol', items })
      continue
    }

    const paragraph: string[] = []
    while (i < lines.length) {
      const current = lines[i].trim()
      if (!current) break
      if (/^\s*[-*+]\s+/.test(lines[i])) break
      if (/^\s*\d+\.\s+/.test(lines[i])) break
      if (
        i + 1 < lines.length &&
        lines[i].includes('|') &&
        isTableSeparator(lines[i + 1]) &&
        parseTableRow(lines[i]).length >= 2
      ) break
      paragraph.push(cleanInlineMarkdown(current))
      i++
    }
    if (paragraph.length) blocks.push({ type: 'paragraph', text: paragraph.join(' ') })
  }

  return blocks.length ? blocks : [{ type: 'paragraph', text: cleanInlineMarkdown(text) }]
}

function RichMessage({ text }: { text: string }) {
  const blocks = parseRichBlocks(text)

  return (
    <div className="msg-rich">
      {blocks.map((block, idx) => {
        if (block.type === 'paragraph') {
          return <p key={idx} className="msg-paragraph">{block.text}</p>
        }
        if (block.type === 'ul') {
          return (
            <ul key={idx} className="msg-list msg-list-ul">
              {block.items.map((item, itemIdx) => <li key={itemIdx}>{item}</li>)}
            </ul>
          )
        }
        if (block.type === 'ol') {
          return (
            <ol key={idx} className="msg-list msg-list-ol">
              {block.items.map((item, itemIdx) => <li key={itemIdx}>{item}</li>)}
            </ol>
          )
        }
        return (
          <div key={idx} className="msg-table-wrap">
            <table className="msg-table">
              <thead>
                <tr>
                  {block.headers.map((h, hIdx) => <th key={hIdx}>{h}</th>)}
                </tr>
              </thead>
              <tbody>
                {block.rows.map((row, rowIdx) => (
                  <tr key={rowIdx}>
                    {block.headers.map((_, colIdx) => <td key={colIdx}>{row[colIdx] ?? ''}</td>)}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )
      })}
    </div>
  )
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
          <div className="msg-text"><RichMessage text={msg.text} /></div>
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
