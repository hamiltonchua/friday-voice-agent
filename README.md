# Kismet Voice Agent

Real-time voice interface for [OpenClaw](https://github.com/openclaw/openclaw) agents. Talk to your agent using your voice — speech-to-text, text-to-speech, wake word detection, and speaker verification all run locally.

Built by [Kismet Labs](https://kismetlabs.com), an AI consulting firm in the Philippines.

## How It Works

```
Browser (mic) → WebSocket → Server
                              ├─ Wake Word: Porcupine (CPU, "Hey Friday")
                              ├─ Speaker Verification: SpeechBrain ECAPA-TDNN (CPU)
                              ├─ STT: MLX Parakeet (macOS) / faster-whisper (CUDA)
                              ├─ Smart Turn: turn-taking prediction (avoid mid-thought cutoffs)
                              ├─ LLM: OpenClaw /v1/chat/completions → your agent
                              ├─ TTS: MLX Kokoro/Chatterbox (macOS) / Chatterbox (CUDA) / Kokoro ONNX (CPU)
                              └─ Canvas: /canvas WebSocket → visual display panel
                            ← audio response + canvas content
```

Everything except the LLM runs locally on your machine. No cloud STT/TTS APIs, no extra costs.

**Context management** is handled entirely by OpenClaw server-side. The voice agent sends only the system prompt and latest user message per request — OpenClaw's session transcript and [LCM (Lossless Context Management)](https://docs.openclaw.ai) handle conversation history, compaction, and memory automatically.

## Multi-Platform Support

The server auto-detects your hardware and selects the right backends:

| Platform | STT | TTS | Notes |
|---|---|---|---|
| **macOS Apple Silicon** | MLX Parakeet TDT 0.6B v3 | MLX Kokoro / Chatterbox | Metal acceleration, no CUDA needed |
| **Linux + NVIDIA GPU** | faster-whisper (large-v3) | Chatterbox Turbo | CUDA, 6GB+ VRAM recommended |
| **CPU-only** | faster-whisper (CPU mode) | Kokoro ONNX | Slower, but works anywhere |

Override auto-detection with `KISMET_PLATFORM=mlx|cuda|cpu` or set `STT_BACKEND` / `TTS_BACKEND` directly.

## Features

- **Wake Word** — say "Hey Friday" to activate (Picovoice Porcupine, runs on CPU)
- **Speaker Verification** — only responds to enrolled voices (SpeechBrain ECAPA-TDNN)
- **Streaming TTS** — responses spoken sentence-by-sentence as they arrive
- **Voice Activity Detection (VAD)** — hands-free, no button required
- **Interruption support** — talk over the agent and it stops to listen
- **Voice Cloning** — Chatterbox TTS supports cloning from a reference audio file
- **Noise Suppression** — MLX DeepFilterNet cleans up audio input in real time (Metal-accelerated, toggleable)
- **Smart Turn** — SmartTurn v3 endpoint detection predicts when the user has finished speaking, avoiding mid-thought cutoffs
- **WebAuthn / Touch ID** — passkey authentication with multi-device enrollment via QR invite
- **Auto-Reconnection** — WebSocket reconnects with exponential backoff
- **Audio Level Visualizer** — mic input levels shown on the mic button
- **Settings Drawer** — configure speaker verification and voice enrollment in-app
- **Speaker score display** — verification score shown on each user message
- **Canvas Display** — visual output (charts, tables, code) pushed to a companion canvas page
- **TTS Skip When Voice Disabled** — toggling voice off skips TTS synthesis server-side (saves CPU/GPU) and suppresses audio playback on the client
- **Tool Activity Indicator** — shows "Using tools..." when the LLM is running tools
- **Pipeline Timing** — STT, LLM TTFT, LLM total, and TTS latency metrics
- **Configurable LLM model** — set model via `LLM_MODEL` env var
- **Local processing** — STT, TTS, wake word, noise suppression, and speaker verify all run locally

## Requirements

### macOS (Apple Silicon)

- **Apple Silicon Mac** (M1/M2/M3/M4)
- **Python 3.11+**
- **conda** (recommended for environment management)
- **Picovoice Porcupine access key** (free tier at [picovoice.ai](https://picovoice.ai))
- **Node.js 18+** (for frontend development)
- **OpenClaw** with chat completions endpoint enabled

### Linux (NVIDIA GPU)

- **NVIDIA GPU** with 6GB+ VRAM (tested on RTX 3060 12GB)
- **Python 3.11** (required for Chatterbox TTS)
- **CUDA** toolkit installed
- **conda** (recommended)
- **Picovoice Porcupine access key**
- **Node.js 18+**
- **OpenClaw** with chat completions endpoint enabled

### Python Dependencies

```bash
pip install -r requirements.txt
```

## OpenClaw Setup

Enable the chat completions endpoint in your OpenClaw config:

```json
{
  "gateway": {
    "http": {
      "endpoints": {
        "chatCompletions": { "enabled": true }
      }
    }
  }
}
```

## SSL Certificate

Browsers require HTTPS to access the microphone over a network. Generate a self-signed cert:

```bash
openssl req -x509 -newkey rsa:2048 -keyout key.pem -out cert.pem -days 365 -nodes -subj "/CN=localhost"
```

## Environment Variables

Create a `.env` file and fill in your values:

### Core

| Variable | Default | Description |
|---|---|---|
| `KISMET_PLATFORM` | *(auto-detect)* | Force platform: `mlx`, `cuda`, or `cpu` |
| `SYSTEM_PROMPT` | *(built-in)* | System prompt for voice responses |
| `MEETING_SYSTEM_PROMPT` | *(built-in)* | System prompt used for meeting command responses |

### STT

| Variable | Default | Description |
|---|---|---|
| `STT_BACKEND` | *(auto)* | `mlx-audio` (macOS) or `faster-whisper` (CUDA/CPU) |
| `MLX_STT_MODEL` | `mlx-community/parakeet-tdt-0.6b-v3` | MLX STT model (Parakeet) |
| `WHISPER_MODEL` | `large-v3` | faster-whisper model size |
| `WHISPER_DEVICE` | `cuda` | Device for faster-whisper (`cuda`, `cpu`) |
| `WHISPER_COMPUTE` | `float16` | Compute type for faster-whisper |

### TTS

| Variable | Default | Description |
|---|---|---|
| `TTS_BACKEND` | *(auto)* | `mlx-audio`, `chatterbox-cuda`, or `kokoro-onnx` |
| `TTS_ENGINE` | `chatterbox` | Legacy: `chatterbox` or `kokoro` |
| `MLX_TTS_MODEL` | `mlx-community/chatterbox-fp16` | MLX TTS model |
| `MLX_TTS_MODEL_FALLBACK` | `mlx-community/Kokoro-82M-bf16` | Fallback MLX TTS model |
| `MLX_TTS_VOICE` | `af_sky` | Voice ID for MLX Kokoro |
| `CHATTERBOX_REF` | — | Reference audio for voice cloning |
| `KOKORO_VOICE` | `af_sky` | Kokoro ONNX voice ID |

### Wake Word & Speaker Verification

| Variable | Default | Description |
|---|---|---|
| `WAKE_WORD_ENABLED` | `true` | Enable wake word detection |
| `WAKE_WORD_KEYWORD` | `jarvis` | Built-in keyword name or path to custom `.ppn` |
| `PORCUPINE_ACCESS_KEY` | — | Picovoice access key (required) |
| `WAKE_WORD_SENSITIVITY` | `0.5` | Wake word detection sensitivity |
| `SPEAKER_VERIFY` | `auto` | Speaker verification (`auto`, `true`, `false`) |
| `IDLE_TIMEOUT_SEC` | `30` | Seconds before returning to sleep |

### Smart Turn (Endpoint Detection)

| Variable | Default | Description |
|---|---|---|
| `SMART_TURN_ENABLED` | `true` | Enable SmartTurn turn-taking prediction |
| `SMART_TURN_THRESHOLD` | `0.5` | Probability threshold for "turn complete" (0-1) |
| `SMART_TURN_MAX_WAIT_SEC` | `3.0` | Force-send after this many seconds of silence |

### LLM

| Variable | Default | Description |
|---|---|---|
| `LLM_URL` | `http://127.0.0.1:1234/v1/chat/completions` | OpenAI-compatible chat completions endpoint |
| `LLM_API_KEY` | — | Optional bearer token for the LLM endpoint |
| `LLM_MODEL` | `nvidia/nemotron-3-nano` | Model name sent to the endpoint |
| `MAX_HISTORY_MESSAGES` | `20` | Sliding window size for local conversation history |

### Memory (Forgetful RAG)

| Variable | Default | Description |
|---|---|---|
| `FORGETFUL_ENABLED` | `true` | Enable Forgetful semantic memory injection |
| `FORGETFUL_MAX_MEMORIES` | `3` | Top-K memories to inject per request |
| `FORGETFUL_MAX_CONTENT_CHARS` | `300` | Truncate each injected memory to this length |

### Push & Delegation

| Variable | Default | Description |
|---|---|---|
| `PUSH_SECRET` | — | Bearer token required by `/push` and `/webhook/*` |
| `PUSH_URL` | `https://prodigy.skunk-shark.ts.net:8765/push` | Public push endpoint URL used by delegated tools |
| `DELEGATE_ENABLED` | `true` | Enable external delegation tool |
| `DELEGATE_USE_ACP` | `true` | Prefer persistent `opencode acp` transport (falls back to `opencode run` on error) |
| `DELEGATE_CMD` | `opencode` | Delegate CLI command |
| `DELEGATE_MODEL` | `opencode/mimo-v2-pro-free` | Model passed to the delegate CLI |
| `DELEGATE_TIMEOUT` | `120` | Delegate timeout in seconds |

### Authentication

| Variable | Default | Description |
|---|---|---|
| `AUTH_SECRET` | *(random at startup)* | Secret used to sign auth session cookies |
| `AUTH_SESSION_HOURS` | `24` | Auth session lifetime in hours |
| `WEBAUTHN_RP_ID` | *(auto from request host)* | WebAuthn relying party ID override |
| `WEBAUTHN_RP_NAME` | `Kismet Voice Agent` | Display name for passkey prompts |
| `WEBAUTHN_ORIGIN` | *(auto from request host/scheme)* | Expected WebAuthn origin override |

## Usage

### Quick Start (macOS — Kokoro TTS)

```bash
./start-kokoro.sh
```

### Quick Start (Linux — Chatterbox TTS)

```bash
./start-chatterbox.sh
```

Open `https://<your-host>:8765` in your browser. Accept the self-signed cert warning.

**Controls:**
- **Wake word** — say "Hey Friday" to activate (when wake word mode is on)
- **Eye button** — toggle VAD (auto-listen mode)
- **Mic button** — hold to talk (manual mode), glows green to show audio levels
- **Spacebar** — hold to talk (when VAD is off)
- **Talk while Kismet speaks** — interrupts and listens to you
- **Trash icon** — clear conversation and reset context
- **Settings (⚙️)** — open the settings drawer to manage speaker verification and enrollment

### Speaker Enrollment

1. Click the **⚙️ settings icon** in the header to open the settings drawer
2. Click **Enroll Voice**
3. Record 5 guided sentences (hold to record each one)
4. Your voice embedding is saved to `~/.kismet/voices/`
5. Kismet will now only respond to your voice

You can toggle speaker verification on/off from the settings drawer at any time. Each user message shows the verification score (e.g., `speaker ✓ 0.78`).

## Canvas Display

The voice agent includes a companion canvas page at `/canvas` for visual output. When the Canvas Output toggle is enabled in settings, the LLM can emit rich visual content (charts, tables, code, structured data) that appears on the canvas while the voice response speaks a brief summary.

### Setup

1. Open `https://<your-host>:8765/canvas` in a browser (or point a WebView at it)
2. Toggle **Canvas Output** on in the voice agent settings drawer
3. Ask something visual — "compare Python and JavaScript", "show me a chart", etc.

The canvas page auto-reconnects via WebSocket. Canvas content is stripped from the spoken response — the LLM speaks a summary, the canvas shows the details.

### How It Works

- LLM emits `<canvas type="html">` or `<canvas type="text">` blocks in its response
- Backend extracts canvas blocks, suppresses them from TTS and chat display
- Canvas content is broadcast to connected `/ws/canvas` WebSocket clients
- HTML canvas blocks support Chart.js, tables, styled content, and inline scripts

## Frontend

The UI is a React 19 + TypeScript app built with Vite, Tailwind CSS, and [shadcn/ui](https://ui.shadcn.com/) components.

### Stack

| Layer | Library |
|---|---|
| Framework | React 19 + TypeScript |
| Build | Vite |
| Styling | Tailwind CSS v4 |
| UI Components | shadcn/ui (Radix UI primitives) |
| Icons | Lucide React |
| VAD | @ricky0123/vad-web v0.0.30 (Silero V5, ONNX) |
| Toasts | Sonner |

### Structure

```
frontend/
├── src/
│   ├── App.tsx              # Root component + state machine
│   ├── types.ts             # Shared TypeScript types
│   ├── constants.ts         # Enrollment constants
│   ├── hooks/
│   │   ├── useWebSocket.ts  # WebSocket connection + reconnect logic
│   │   ├── useAudio.ts      # Audio playback queue
│   │   ├── useAudioLevel.ts # Mic level visualizer
│   │   ├── useVAD.ts        # Silero VAD integration
│   │   ├── useManualRecording.ts  # Push-to-talk recording
│   │   └── useWakeWordStream.ts   # Audio streaming to server for wake word
│   └── components/
│       ├── Header.tsx       # App header with title and settings trigger
│       ├── StatusDisplay.tsx # Connection + agent state badges
│       ├── ChatDisplay.tsx  # Conversation transcript
│       ├── Controls.tsx     # Mic, VAD, and spacebar controls
│       ├── EnrollModal.tsx  # Guided voice enrollment flow
│       ├── SettingsDrawer.tsx  # Settings panel (shadcn Sheet)
│       ├── MeetingBanner.tsx   # Meeting companion mode banner
│       ├── ReconnectBanner.tsx # WebSocket reconnect status
│       └── ui/              # shadcn/ui primitives (Sheet, etc.)
```

### Development

```bash
cd frontend
npm install
npm run dev        # Dev server at http://localhost:5173
npm run build      # Production build → ../index.html + assets
npm run lint       # ESLint
```

The production build outputs directly to the root of the project so `server.py` can serve it.

## VRAM Usage (Linux/CUDA)

| Component | VRAM |
|---|---|
| faster-whisper large-v3 | ~3 GB |
| Chatterbox Turbo | ~2 GB |
| **Total** | **~5 GB** |

On macOS, MLX models use unified memory — no dedicated VRAM needed.

Wake word (Porcupine) and speaker verification (SpeechBrain) run on CPU on all platforms.

## Roadmap

- [x] **Phase 1:** Streaming TTS
- [x] **Phase 2:** Voice Activity Detection (VAD)
- [x] **Phase 3:** Interruption support
- [x] **Phase 4:** Wake word ("Hey Friday" via Porcupine)
- [x] **Phase 5:** Speaker verification (SpeechBrain ECAPA-TDNN)
- [x] **Phase 6:** Wake word + speaker verification combined
- [x] **Phase 7:** Polish & hardening — reconnection, multi-platform, audio visualizer
- [x] **Phase 9:** UI overhaul — React 19 + shadcn/ui rewrite, two-panel layout, settings drawer
- [x] **Phase 10:** Canvas display + tool activity — standalone canvas page, tool-working indicator
- [x] **Phase 12:** Noise suppression — DeepFilterNet with UI toggle
- [x] **Phase 13:** Pipeline observability — timing metrics, configurable model, restart script
- [x] **Phase 14:** WebAuthn authentication — Touch ID / passkey, multi-device QR invite
- [x] **Phase 16:** SmartTurn v3 endpoint detection — turn-taking prediction to avoid mid-thought cutoffs
- [x] **Text chat input** — type messages when voice isn't practical
- [x] **Context delegation** — removed client-side 40-message window; OpenClaw session + LCM manages context server-side
- ~~Phase 11 (token-aware context management)~~ — eliminated, superseded by context delegation
- [ ] **Phase 15:** Meeting companion — passive transcription with diarization (Sortformer via mlx-audio) *(parked)*
- [ ] **Phase 17:** Quality of life — mobile layout, conversation export, multi-speaker enrollment, performance profiling *(backlog)*

See [ROADMAP.md](ROADMAP.md) for full details on each phase.

## License

MIT
