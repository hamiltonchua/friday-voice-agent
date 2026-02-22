# Kismet Voice Agent

Real-time voice interface for [OpenClaw](https://github.com/openclaw/openclaw) agents. Talk to your agent using your voice — speech-to-text, text-to-speech, wake word detection, and speaker verification all run locally.

Built by [Kismet Labs](https://kismetlabs.com), an AI consulting firm in the Philippines.

## How It Works

```
Browser (mic) → WebSocket → Server
                              ├─ Wake Word: Porcupine (CPU, "Hey Friday")
                              ├─ Speaker Verification: SpeechBrain ECAPA-TDNN (CPU)
                              ├─ STT: MLX Whisper (macOS) / faster-whisper (CUDA)
                              ├─ LLM: OpenClaw /v1/chat/completions → your agent
                              ├─ TTS: MLX Kokoro/Chatterbox (macOS) / Chatterbox (CUDA) / Kokoro ONNX (CPU)
                              └─ Canvas: /canvas WebSocket → visual display panel
                            ← audio response + canvas content
```

Everything except the LLM runs locally on your machine. No cloud STT/TTS APIs, no extra costs.

## Multi-Platform Support

The server auto-detects your hardware and selects the right backends:

| Platform | STT | TTS | Notes |
|---|---|---|---|
| **macOS Apple Silicon** | MLX Whisper (large-v3-turbo) | MLX Kokoro / Chatterbox | Metal acceleration, no CUDA needed |
| **Linux + NVIDIA GPU** | faster-whisper (large-v3) | Chatterbox Turbo | CUDA, 6GB+ VRAM recommended |
| **CPU-only** | faster-whisper (CPU mode) | Kokoro ONNX | Slower, but works anywhere |

Override auto-detection with `KISMET_PLATFORM=mlx|cuda|cpu` or set `STT_BACKEND` / `TTS_BACKEND` directly.

## Features

- **Wake Word** — say "Hey Friday" to activate (Picovoice Porcupine, runs on CPU)
- **Speaker Verification** — only responds to enrolled voices (SpeechBrain ECAPA-TDNN)
- **Streaming TTS** — responses spoken sentence-by-sentence as they arrive
- **Voice Activity Detection (VAD)** — hands-free, no button required
- **Interruption support** — talk over Kismet and it stops to listen
- **Voice Cloning** — Chatterbox TTS supports cloning from a reference audio file
- **Auto-Reconnection** — WebSocket reconnects with exponential backoff
- **Audio Level Visualizer** — mic input levels shown on the mic button
- **Settings Drawer** — configure speaker verification and voice enrollment in-app
- **Speaker score display** — verification score shown on each user message
- **Canvas Display** — visual output (charts, tables, code) pushed to a companion canvas page
- **Tool Activity Indicator** — shows "Using tools..." when the LLM is running tools
- **Local processing** — STT, TTS, wake word, and speaker verify all run locally

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

Copy `.env.example` to `.env` and fill in your values:

### Core

| Variable | Default | Description |
|---|---|---|
| `KISMET_PLATFORM` | *(auto-detect)* | Force platform: `mlx`, `cuda`, or `cpu` |
| `OPENCLAW_URL` | `http://127.0.0.1:18789/v1/chat/completions` | OpenClaw endpoint |
| `OPENCLAW_TOKEN` | — | Gateway auth token |
| `OPENCLAW_AGENT` | `main` | Agent ID to route to |
| `SYSTEM_PROMPT` | *(built-in)* | System prompt for voice responses |

### STT

| Variable | Default | Description |
|---|---|---|
| `STT_BACKEND` | *(auto)* | `mlx-audio` (macOS) or `faster-whisper` (CUDA/CPU) |
| `MLX_STT_MODEL` | `mlx-community/whisper-large-v3-turbo-asr-fp16` | MLX Whisper model |
| `WHISPER_MODEL` | `large-v3` | faster-whisper model size |
| `WHISPER_DEVICE` | `cuda` | Device for faster-whisper (`cuda`, `cpu`) |

### TTS

| Variable | Default | Description |
|---|---|---|
| `TTS_BACKEND` | *(auto)* | `mlx-audio`, `chatterbox-cuda`, or `kokoro-onnx` |
| `TTS_ENGINE` | `chatterbox` | Legacy: `chatterbox` or `kokoro` |
| `MLX_TTS_MODEL` | `mlx-community/chatterbox-fp16` | MLX TTS model |
| `MLX_TTS_MODEL_FALLBACK` | `mlx-community/Kokoro-82M-bf16` | Fallback MLX TTS model |
| `MLX_TTS_VOICE` | `af_sky` | Voice ID for MLX Kokoro |
| `CHATTERBOX_REF` | — | Reference audio for voice cloning |
| `KOKORO_VOICE` | `af_heart` | Kokoro ONNX voice ID |

### Wake Word & Speaker Verification

| Variable | Default | Description |
|---|---|---|
| `WAKE_WORD_ENABLED` | `true` | Enable wake word detection |
| `WAKE_WORD` | `hey_friday` | Wake word name |
| `PORCUPINE_ACCESS_KEY` | — | Picovoice access key (required) |
| `PORCUPINE_MODEL_PATH` | — | Path to custom `.ppn` wake word file |
| `WAKE_WORD_THRESHOLD` | `0.5` | Wake word detection sensitivity |
| `SPEAKER_VERIFY` | `auto` | Speaker verification (`auto`, `true`, `false`) |
| `SPEAKER_VERIFY_THRESHOLD` | `0.65` | Cosine similarity threshold |
| `IDLE_TIMEOUT_SEC` | `30` | Seconds before returning to sleep |

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
| VAD | @ricky0123/vad-web (Silero VAD, ONNX) |
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
- [x] **Phase 7:** Polish & hardening (reconnection, toasts, audio visualizer)
- [x] **Phase 9:** UI overhaul — React 19 + shadcn/ui rewrite, settings drawer, speaker score display
- [x] **Phase 10:** Canvas display + tool activity — standalone canvas page via WebSocket, tool-working indicator, canvas token/TTS suppression
- [ ] **Phase 11:** Token-aware context management — replace naive sliding window with token counting and compaction
- [ ] **Phase 8:** Meeting companion — passive transcription with diarization, wake word commands *(parked)*

## License

MIT
