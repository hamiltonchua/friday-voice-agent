# Voice Chat Roadmap

Goal: Transform push-to-talk voice chat into a natural, always-on voice assistant with wake word activation, streaming responses, and interruption support.

## Current State (v0.1 — Done)
- [x] Push-to-talk via mic button or spacebar
- [x] faster-whisper large-v3 STT on GPU
- [x] Kokoro ONNX TTS (local)
- [x] Routes through OpenClaw
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
*If you talk while the agent is speaking, it stops and listens.*

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

## Phase 4: Wake Word (v0.5) ✅
*Only activate after hearing "Hey Friday." Low power idle state.*

- [x] Server-side wake word detection — client streams audio chunks, server detects
- [x] State machine: `sleeping → wake_word_detected → listening → processing → speaking → sleeping`
- [x] Visual UI states (sleeping indicator, active when awake)
- [x] Configurable timeout: return to sleep after N seconds of no interaction
- [x] Initially OpenWakeWord (TFLite), later replaced with Picovoice Porcupine
- [x] Custom wake word model: `hey-friday_en_mac_v4_0_0.ppn`

**Completed:** 2026-02-08
**Depends on:** Phase 3 (interruption)

---

## Phase 5: Speaker Verification (v0.6) ✅
*Only respond to recognized voices. Reject strangers.*

- [x] SpeechBrain ECAPA-TDNN speaker verification module (`speaker_verify.py`)
  - Runs on CPU to keep GPU free for whisper + TTS
  - Embedding extraction, cosine similarity comparison
  - Enrollment: average multiple samples → save to `voices/`
  - Verification: compare incoming audio against enrolled embedding
- [x] Enrollment flow via WebSocket (enroll_start → enroll_sample × N → enroll_complete)
- [x] Runtime verification gate in `process_audio()` — reject unrecognized speakers before STT
- [x] UI: Enrollment modal with guided prompts
- [x] UI: Verification toggle, status badges, similarity scores
- [x] Configurable threshold via `SPEAKER_VERIFY_THRESHOLD` env var (default 0.65)
- [x] `SPEAKER_VERIFY` env var: "auto" (verify if enrolled), "true", "false"

**Completed:** 2026-02-08
**Depends on:** Phase 3 (interruption)

---

## Phase 6: Wake Word + Speaker Verification Combined (v0.7) ✅
*Wake word triggers listening, speaker verification gates processing.*

- [x] Integrate wake word (Phase 4) with speaker verification (Phase 5)
- [x] Flow: wake_word → record speech → verify speaker → if verified, transcribe + respond
- [x] On rejected speaker: subtle rejection beep + stay awake
- [x] Configurable: wake word only, verification only, or both (via env vars)

**Completed:** 2026-02-08
**Depends on:** Phase 4 + Phase 5

---

## Phase 7: Polish & Hardening (v1.0) ✅
*Production-quality touches.*

- [x] Reconnection handling (exponential backoff, state restoration)
- [x] Graceful error messages (toast notifications for all pipeline failures)
- [x] Audio level visualizer (green glow ring on mic button)
- [x] Multi-platform support (macOS Apple Silicon / Linux CUDA / CPU-only)
- [x] Auto-detect platform and select STT/TTS backends
- [x] Dual-environment enrollment (works across platforms)
- [x] `.env` support for secrets and configuration
- [x] Porcupine wake word (replaced OpenWakeWord for reliability)
- [x] VAD echo cancellation + sensitivity tuning
- [x] Preload models at startup with threading lock for MLX safety

**Completed:** 2026-02-14

---

## Phase 9: UI Overhaul ✅
*Complete frontend rewrite with modern React stack.*

- [x] React 19 + TypeScript + Vite scaffold
- [x] Tailwind CSS v4 + shadcn/ui (new-york dark theme)
- [x] Two-panel layout: left Control Center, right Conversation
- [x] Custom hooks: useWebSocket, useAudio, useVAD, useWakeWordStream, useManualRecording, useAudioLevel
- [x] Settings drawer (shadcn Sheet) for speaker verification and enrollment
- [x] Enrollment modal with guided 5-sample flow
- [x] Speaker score display on each user message
- [x] Meeting companion mode toggle + banner
- [x] Wake word support in meeting mode
- [x] Lucide React icons throughout

**Completed:** 2026-02-21
**Branch:** `phase9-ui-overhaul` (merged via PR #1)

---

## Phase 10: Canvas Display + Tool Activity ✅
*Visual output companion page and tool status indicator.*

- [x] Standalone `/canvas` page with WebSocket connection (`/ws/canvas`)
- [x] LLM emits `<canvas type="html">` or `<canvas type="text">` blocks
- [x] Backend extracts canvas blocks, suppresses from TTS and chat display
- [x] Canvas supports Chart.js, tables, styled content, inline scripts
- [x] Tool-working indicator ("Using tools...") in frontend
- [x] Stronger LLM instructions for canvas content generation

**Completed:** 2026-02-21
**Branch:** `phase10-canvas`

---

## Phase 12: Noise Suppression ✅
*Clean up audio input with real-time noise suppression.*

- [x] DeepFilterNet integration for noise suppression
- [x] UI toggle to enable/disable noise suppression
- [x] Runs on audio stream before STT processing

**Completed:** 2026-02-28
**Branch:** `phase12-noise-suppression`

---

## Phase 13: Pipeline Observability & Configuration ✅
*Timing, model selection, and system prompt customization.*

- [x] Pipeline timing metrics (STT duration, LLM TTFT, LLM total, TTS duration)
- [x] Configurable LLM model via `OPENCLAW_MODEL` env var
- [x] Custom system prompt (rename to Friday, canvas instructions)
- [x] Kokoro voice switched to `af_kore`
- [x] Restart script (`restart.sh`)

**Completed:** 2026-03-02

---

## Phase 14: WebAuthn Authentication ✅
*Secure access with passkeys / Touch ID. Multi-device support.*

- [x] WebAuthn (Touch ID / passkey) authentication for the web UI
- [x] Device registration page (requires existing auth)
- [x] QR code invite flow for multi-device passkey enrollment
- [x] Disabled extended thinking for voice requests (latency optimization)

**Completed:** 2026-03-04
**Branch:** `phase14-webauthn`

---

## Architecture

```
[Browser]
  │
  ├─ VAD (speech start/end detection)
  ├─ Audio capture + streaming
  ├─ Audio playback (chunked, interruptible)
  │
  └─── WebSocket ───→ [Server on :8765]
                         │
                         ├─ Wake Word: Porcupine (CPU, "Hey Friday")
                         ├─ Noise Suppression: DeepFilterNet
                         ├─ Speaker Verification: SpeechBrain ECAPA-TDNN (CPU)
                         ├─ STT: MLX Whisper (macOS) / faster-whisper (CUDA/CPU)
                         ├─ LLM: OpenClaw /v1/chat/completions → your agent
                         ├─ TTS: MLX Kokoro/Chatterbox (macOS) / Chatterbox (CUDA) / Kokoro ONNX (CPU)
                         ├─ Canvas: /ws/canvas WebSocket → visual display
                         └─ Auth: WebAuthn / passkey (Touch ID)
                       ← audio response + canvas content
```

---

## Text Chat Input ✅
*Type messages when voice isn't practical (noisy environments, public spaces).*

- [x] Text input field in the voice agent UI
- [x] Send pre-transcribed text through same LLM pipeline
- [x] Same WebSocket connection, no extra auth

**Completed:** 2026-03-13

---

## Future

### Phase 11: Token-Aware Context Management *(planned)*
*Replace naive sliding window with token counting and compaction.*

- [ ] Token counting for conversation history
- [ ] Smart compaction (summarize older messages)
- [ ] Context window budget management

### Phase 8: Meeting Companion *(parked)*
*Passive transcription with diarization. Only responds to Ham's voice on command.*

- [ ] Passive transcription mode (always listening, transcribing in background)
- [ ] Speaker diarization with pyannote (who said what)
- [ ] Command-on-demand: only responds when Ham speaks a command
- [ ] Meeting notes / transcript export
- [ ] Multi-speaker labeling

**Depends on:** Phase 5 (speaker verification)

### Phase 15: Quality of Life *(backlog)*
*UX and operational improvements.*

- [ ] Mobile-friendly layout
- [ ] Conversation export (save transcript)
- [ ] Performance profiling (GPU memory, latency benchmarks)
- [ ] Multi-speaker enrollment (recognize different users)
- [ ] Voice profile management UI

---

## Notes
- Each phase is a separate git branch, merged to main when stable
- Phases are incremental — each one works standalone on top of the previous
- Phase 8 was parked in favor of UI and infrastructure work (phases 9-14)
- Phase 11 and 13 were renumbered as priorities shifted
- Wake word + speaker verification run on CPU to avoid competing with STT/TTS for GPU
- Speaker verification adds ~50-100ms latency per request (CPU embedding extraction)
