# Voice Chat Roadmap

Goal: Transform push-to-talk voice chat into a natural, always-on voice assistant with wake word activation, streaming responses, and interruption support.

## Current State (v0.1 — Done)
- [x] Push-to-talk via mic button or spacebar
- [x] faster-whisper large-v3 STT on GPU
- [x] Kokoro ONNX TTS (local)
- [x] Routes through OpenClaw (Friday)
- [x] Web UI with HTTPS
- [x] Git repo initialized

---

## Phase 1: Streaming TTS (v0.2) ✅
*Biggest perceived latency improvement for least effort.*

- [x] Split LLM response into sentences as they arrive
- [x] Generate TTS per sentence and send each chunk to the client immediately
- [x] Client plays audio chunks sequentially (queue-based)
- [x] Switch OpenClaw API call to streaming (`stream: true` SSE)
- [x] Accumulate streamed text, detect sentence boundaries, fire TTS per sentence
- [x] Show transcript updating in real-time as tokens arrive

**Completed:** 2026-02-03

---

## Phase 2: Voice Activity Detection — No More Button (v0.3) ✅
*Remove push-to-talk. System listens continuously and detects speech automatically.*

- [x] Add VAD to the frontend (@ricky0123/vad-web — Silero VAD in browser)
- [x] Auto-start recording when speech detected
- [x] Auto-stop and send to STT when silence detected
- [x] Visual indicator: idle → listening → processing → speaking
- [x] Keep the mic button as a manual fallback
- [x] Toggle button to enable/disable VAD mode

**Completed:** 2026-02-03

---

## Phase 3: Interruption Support (v0.4) ✅
*If you talk while Friday is speaking, she stops and listens.*

- [x] Track playback state on the client (playing / idle)
- [x] Keep VAD active during TTS playback
- [x] On speech detected during playback:
  - Stop current audio immediately
  - Cancel any queued TTS chunks
  - Send cancel signal to server (abort in-flight LLM/TTS)
  - Capture new speech and send to STT
- [x] Server-side: handle cancel gracefully (abort streaming LLM call)
- [x] Interrupted messages shown with visual indicator

**Completed:** 2026-02-03

---

## Phase 4: Wake Word (v0.5)
*Only activate after hearing "Friday." Low power idle state.*

- [ ] Evaluate and pick a wake word engine:
  - **OpenWakeWord** (Python, CPU, custom words) — top choice
  - Porcupine (commercial)
  - Simple whisper-tiny on short chunks (wasteful but no extra deps)
- [ ] Decide where wake word runs:
  - **Option A: Server-side** — client streams low-quality audio continuously, server runs wake word detection on CPU. More control, slightly more bandwidth.
  - **Option B: Client-side** — run wake word in browser via WASM/JS. Zero bandwidth when idle. Harder to set up.
- [ ] Implement chosen approach
- [ ] State machine: `sleeping → wake_word_detected → listening → processing → speaking → sleeping`
- [ ] Visual UI states (pulsing dot when sleeping, active indicator when awake)
- [ ] Configurable timeout: return to sleep after N seconds of no interaction
- [ ] Custom wake word training (if using OpenWakeWord)

**Estimated effort:** 1-2 days
**Branch:** `feat/wake-word`
**Depends on:** Phase 3 (interruption)

---

## Phase 5: Polish & Hardening (v1.0)
*Production-quality touches.*

- [ ] Reconnection handling (WebSocket drops, server restarts)
- [ ] Graceful error messages in the UI
- [ ] Audio level visualizer (show mic input levels)
- [ ] Settings panel (voice selection, wake word toggle, VAD sensitivity)
- [ ] Mobile-friendly layout
- [ ] Conversation export (save transcript)
- [ ] Startup as a systemd service (optional)
- [ ] Performance profiling (GPU memory, latency benchmarks)

**Estimated effort:** 1-2 days
**Branch:** various `feat/*` and `fix/*`

---

## Architecture After All Phases

```
[Browser]
  │
  ├─ Wake word detection (idle, low power)
  ├─ VAD (speech start/end detection)
  ├─ Audio capture + streaming
  ├─ Audio playback (chunked, interruptible)
  │
  └─── WebSocket ───→ [Server on discovery:8765]
                         │
                         ├─ faster-whisper (GPU) — STT
                         ├─ OpenClaw API — LLM (Friday)
                         └─ Kokoro ONNX — TTS (sentence-level streaming)
```

## Notes
- Each phase is a separate git branch, merged to main when stable
- Phases are incremental — each one works standalone on top of the previous
- GPU VRAM budget: ~4GB used (whisper 3GB + kokoro 300MB), ~8GB headroom
- Wake word engine should run on CPU to avoid competing with STT/TTS for GPU
