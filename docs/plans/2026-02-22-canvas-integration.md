# Phase 10: Canvas Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add canvas output support so the LLM can push visual content (charts, tables, code) to the macOS a2ui canvas panel.

**Architecture:** Backend detects `<canvas>` blocks in LLM responses, strips them from voice output, and pushes content to the macOS canvas via `openclaw` CLI. Frontend has a toggle in settings to enable/disable canvas mode via WebSocket.

**Tech Stack:** Python (FastAPI/WebSocket), React/TypeScript, openclaw CLI

---

### Task 1: Backend — Canvas toggle and system prompt injection

**Files:**
- Modify: `server.py` (add canvas state, handle toggle message, modify system prompt)

**Step 1: Add canvas state + system prompt addition**

In `server.py`, add the canvas system prompt text after the existing SYSTEM_PROMPT/MEETING_SYSTEM_PROMPT definitions (~line 90):

```python
CANVAS_INSTRUCTION = (
    "\n\nYou can emit visual content for a canvas display panel. "
    "Wrap visual content in <canvas> tags. Two types:\n"
    "- <canvas type=\"html\" title=\"Title\">...full HTML...</canvas> for rich content (charts, tables, styled output). "
    "Include Chart.js from https://cdn.jsdelivr.net/npm/chart.js if needed.\n"
    "- <canvas type=\"text\" title=\"Title\">...plain text...</canvas> for simple text summaries.\n"
    "Canvas content is displayed visually, NOT spoken. Continue speaking normally outside canvas blocks. "
    "Use canvas for data that benefits from visual presentation."
)
```

**Step 2: Handle canvas_toggle message in WebSocket handler**

In the WebSocket endpoint, add a per-connection `canvas_enabled` variable (initially False) and handle the `canvas_toggle` message type:

```python
# After line 644 (enrollment state), add:
canvas_enabled = False

# In the message handling loop, add handler:
elif msg["type"] == "canvas_toggle":
    canvas_enabled = msg.get("enabled", False)
    print(f"[Canvas] {'Enabled' if canvas_enabled else 'Disabled'}")
    await ws.send_json({"type": "canvas_toggled", "enabled": canvas_enabled})
```

**Step 3: Inject canvas instruction into system prompt when enabled**

In `process_audio()`, modify the system prompt used for LLM calls. The `chat_stream()` function uses the global `SYSTEM_PROMPT`. We need to make `chat_stream` accept a system prompt parameter:

Change `chat_stream` signature to accept optional system_prompt override:
```python
async def chat_stream(user_text: str, cancel_event: asyncio.Event, system_prompt: str = None) -> AsyncIterator[tuple[str, str]]:
```
And use it: `messages = [{"role": "system", "content": system_prompt or SYSTEM_PROMPT}] + conversation_history`

Then in `process_audio`, pass the canvas-enhanced prompt:
```python
effective_prompt = SYSTEM_PROMPT + CANVAS_INSTRUCTION if canvas_enabled else SYSTEM_PROMPT
async for event_type, data in chat_stream(user_text, cancel_event, effective_prompt):
```

**Step 4: Commit**
```bash
git add server.py
git commit -m "feat(canvas): add canvas toggle and system prompt injection"
```

---

### Task 2: Backend — Canvas extraction, stripping, and push

**Files:**
- Modify: `server.py` (add canvas extraction + CLI push after LLM response)

**Step 1: Add canvas extraction function**

After the `SENTENCE_END_RE` pattern (~line 275), add:

```python
CANVAS_BLOCK_RE = re.compile(r'<canvas\s+type="(\w+)"(?:\s+title="([^"]*)")?\s*>(.*?)</canvas>', re.DOTALL)

def extract_canvas_blocks(text: str) -> tuple[str, list[dict]]:
    """Extract <canvas> blocks from text. Returns (cleaned_text, blocks)."""
    blocks = []
    for m in CANVAS_BLOCK_RE.finditer(text):
        blocks.append({
            "type": m.group(1),
            "title": m.group(2) or "",
            "content": m.group(3).strip(),
        })
    cleaned = CANVAS_BLOCK_RE.sub("", text).strip()
    # Collapse multiple newlines left by removal
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned, blocks
```

**Step 2: Add canvas push function**

```python
import subprocess

async def push_canvas(blocks: list[dict], loop):
    """Push canvas blocks to macOS a2ui canvas via openclaw CLI."""
    for block in blocks:
        try:
            if block["type"] == "text":
                content = block["content"]
                if block["title"]:
                    content = f"{block['title']}\n\n{content}"
                await loop.run_in_executor(None, lambda c=content: subprocess.run(
                    ["openclaw", "nodes", "canvas", "a2ui", "push", "--node", "prodigy", "--text", c],
                    timeout=10,
                ))
            elif block["type"] == "html":
                title = block["title"] or "Canvas"
                html_content = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>{title}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: #1a1a2e; color: #e0e0e0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif; padding: 24px; line-height: 1.6; }}
  h1, h2, h3 {{ color: #ffffff; margin-bottom: 12px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
  th, td {{ border: 1px solid #333; padding: 10px 14px; text-align: left; }}
  th {{ background: #16213e; color: #00d97e; }}
  tr:nth-child(even) {{ background: rgba(255,255,255,0.03); }}
  code, pre {{ background: #0f0f0f; padding: 2px 6px; border-radius: 4px; font-size: 0.9em; }}
  pre {{ padding: 16px; overflow-x: auto; margin: 12px 0; }}
  canvas {{ max-width: 100%; }}
</style>
</head><body>
<h2>{title}</h2>
{block['content']}
</body></html>"""
                # Write to temp file and navigate
                with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, dir='/tmp', prefix='canvas_') as f:
                    f.write(html_content)
                    tmp_path = f.name
                await loop.run_in_executor(None, lambda p=tmp_path: subprocess.run(
                    ["openclaw", "nodes", "canvas", "navigate", "--node", "prodigy", "--url", f"file://{p}"],
                    timeout=10,
                ))
                print(f"[Canvas] Pushed HTML to {tmp_path}")
        except Exception as e:
            print(f"[Canvas] Push error: {e}")
```

**Step 3: Integrate extraction into process_audio**

In `process_audio`, after the LLM stream completes (the `"done"` event), extract canvas blocks, strip from text, and push. We need to:

1. Collect the full text from the stream
2. After stream_end, extract canvas blocks and push them
3. The tricky part: sentences are emitted during streaming, but we need to strip canvas from them

Better approach: After the full response is collected, extract canvas blocks, strip the text, and the voice response (sentences already spoken) will have already included the canvas tags as text. Since TTS will just speak them (badly), we need to handle this differently.

**Revised approach:** Process canvas blocks post-completion. The sentences will unfortunately include canvas markup in the spoken text. To prevent this, we should buffer the full response when canvas is enabled, then emit sentences only from the cleaned text.

Actually simpler: Strip canvas blocks from each sentence/token as they stream. Since canvas blocks span multiple tokens, we should handle this at the full-text level. The cleanest approach:

- After `"done"` event with full_text, extract canvas blocks and push
- For TTS: the sentences are already emitted during streaming. Canvas tags will be in the spoken text.
- **Solution:** When canvas is enabled, skip TTS for sentences that contain `<canvas` or `</canvas>`. The canvas content between tags won't form clean sentences anyway.
- After stream completes, rebuild the clean text (without canvas), and if any sentences were skipped, synthesize only the clean text.

**Simplest v1 approach:**
- Let streaming happen normally (tokens + sentences flow to frontend for display)
- For TTS: skip any sentence containing `<canvas` fragments
- After "done", extract canvas blocks from full_text, push to canvas
- Send a `canvas_update` message to frontend with the cleaned text (so it can update the displayed message)

Let me revise — in `process_audio`, around the LLM streaming section:

```python
# Before the streaming loop, add:
canvas_blocks_to_push = []

# In the "sentence" handler, skip TTS for canvas content:
elif event_type == "sentence":
    if first_sentence_time is None:
        first_sentence_time = time.time() - llm_start
    if cancel_event.is_set():
        continue
    # Skip TTS for sentences containing canvas markup
    if canvas_enabled and ('<canvas' in data or '</canvas>' in data):
        continue
    await ws.send_json({"type": "status", "text": "Speaking..."})
    # ... existing TTS code ...

# In the "done" handler, add canvas extraction:
elif event_type == "done":
    # Extract and push canvas blocks if enabled
    if canvas_enabled and '<canvas' in data:
        cleaned_text, canvas_blocks_to_push = extract_canvas_blocks(data)
        # Update conversation history with cleaned text
        if conversation_history and conversation_history[-1]["role"] == "assistant":
            conversation_history[-1]["content"] = cleaned_text
        # Push canvas content
        if canvas_blocks_to_push:
            asyncio.create_task(push_canvas(canvas_blocks_to_push, loop))
            await ws.send_json({"type": "canvas_pushed", "count": len(canvas_blocks_to_push)})
        data = cleaned_text  # Use cleaned text for stream_end
    # ... existing stream_end code ...
```

**Step 4: Commit**
```bash
git add server.py
git commit -m "feat(canvas): extract canvas blocks, strip from voice, push to a2ui"
```

---

### Task 3: Frontend — Canvas toggle in SettingsDrawer

**Files:**
- Modify: `frontend/src/components/SettingsDrawer.tsx`
- Modify: `frontend/src/App.tsx`

**Step 1: Add canvas props and toggle to SettingsDrawer**

Update SettingsDrawerProps:
```typescript
interface SettingsDrawerProps {
  enrolled: boolean
  verifyEnabled: boolean
  canvasEnabled: boolean
  onEnroll: () => void
  onVerifyToggle: () => void
  onCanvasToggle: () => void
}
```

Add a "Canvas Output" section after the Speaker Verification section, same toggle style.

**Step 2: Add canvas state to App.tsx**

```typescript
// State
const [canvasEnabled, setCanvasEnabled] = useState(() => localStorage.getItem('canvas_enabled') === 'true')

// Toggle handler
const handleCanvasToggle = useCallback(() => {
  const next = !canvasEnabled
  setCanvasEnabled(next)
  localStorage.setItem('canvas_enabled', String(next))
  send({ type: 'canvas_toggle', enabled: next })
}, [canvasEnabled, send])

// On WS connect (in ready handler), send current canvas state
// After check_enrollment send:
if (localStorage.getItem('canvas_enabled') === 'true') {
  sendRef.current({ type: 'canvas_toggle', enabled: true })
}

// Handle canvas_toggled message
else if (type === 'canvas_toggled') {
  setCanvasEnabled(msg.enabled as boolean)
}

// Handle canvas_pushed message
else if (type === 'canvas_pushed') {
  // Optional: show toast or indicator
}
```

Pass to SettingsDrawer:
```tsx
<SettingsDrawer
  enrolled={enrolled}
  verifyEnabled={verifyEnabled}
  canvasEnabled={canvasEnabled}
  onEnroll={handleEnrollStart}
  onVerifyToggle={handleVerifyToggle}
  onCanvasToggle={handleCanvasToggle}
/>
```

**Step 3: Commit**
```bash
git add frontend/src/components/SettingsDrawer.tsx frontend/src/App.tsx
git commit -m "feat(canvas): add canvas toggle in settings drawer"
```

---

### Task 4: Frontend — Canvas indicator in header/status area

**Files:**
- Modify: `frontend/src/components/Header.tsx`
- Modify: `frontend/src/App.tsx`

**Step 1: Add canvas indicator to Header**

Add `canvasEnabled` prop to Header and show a small indicator:

```typescript
interface HeaderProps {
  // ... existing props
  canvasEnabled: boolean
}
```

Add a small canvas icon/badge next to status when canvas is active.

**Step 2: Add canvas indicator to desktop left panel**

In App.tsx, add a small "Canvas" badge near the status text on the desktop panel.

**Step 3: Commit**
```bash
git add frontend/src/components/Header.tsx frontend/src/App.tsx
git commit -m "feat(canvas): show canvas indicator in header"
```

---

### Task 5: Build and test

**Step 1: Build frontend**
```bash
cd frontend && npm run build
```

**Step 2: Verify build succeeds**

**Step 3: Final commit**
```bash
git add -A
git commit -m "Phase 10: canvas integration for a2ui visual output"
```

**Step 4: Notify**
```bash
openclaw system event --text "Done: Phase 10 canvas integration complete" --mode now
```
