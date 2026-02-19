# Kismet Voice Agent

Real-time voice interface for [OpenClaw](https://github.com/openclaw/openclaw) agents. Talk to your agent using your voice — speech-to-text, text-to-speech, wake word detection, and speaker verification all run locally.

Built by [Kismet Labs](https://kismetlabs.com), an AI consulting firm in the Philippines.

## How It Works

```
Browser (mic) → WebSocket → Server
                              ├─ Wake Word: Porcupine (CPU, "Hey Friday")
                              ├─ Speaker Verification: SpeechBrain ECAPA-TDNN (CPU)
                              ├─ STT: faster-whisper large-v3 (GPU)
                              ├─ LLM: OpenClaw /v1/chat/completions → your agent
                              └─ TTS: Chatterbox Turbo (GPU)
                            ← audio response
```

Everything except the LLM runs locally on your machine. No cloud STT/TTS APIs, no extra costs.

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
- **Local processing** — STT, TTS, wake word, and speaker verify all run locally

## Requirements

- **GPU:** NVIDIA with 6GB+ VRAM (tested on RTX 3060 12GB)
- **Python 3.11** (required for Chatterbox TTS)
- **Node.js 18+** (for frontend development)
- **OpenClaw** with chat completions endpoint enabled
- **CUDA** toolkit installed
- **conda** (recommended for environment management)
- **Picovoice Porcupine access key** (free tier available at [picovoice.ai](https://picovoice.ai))

### Python Dependencies

```bash
pip install -r requirements.txt
```

### Optional: Kokoro TTS (lighter alternative)

If you prefer Kokoro over Chatterbox (less VRAM, no voice cloning):

```bash
pip install kokoro-onnx
curl -L -o kokoro-v1.0.onnx https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
curl -L -o voices-v1.0.bin https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
```

Set `TTS_ENGINE=kokoro` when running.

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

| Variable | Default | Description |
|---|---|---|
| `TTS_ENGINE` | `chatterbox` | TTS engine (`chatterbox` or `kokoro`) |
| `CHATTERBOX_REF` | — | Reference audio for voice cloning |
| `WHISPER_MODEL` | `large-v3` | Whisper model size |
| `WHISPER_DEVICE` | `cuda` | Device for STT (`cuda`, `cpu`) |
| `KOKORO_VOICE` | `af_heart` | Kokoro voice ID (when using Kokoro) |
| `WAKE_WORD_ENABLED` | `true` | Enable wake word detection |
| `WAKE_WORD` | `hey_friday` | Wake word (must match your `.ppn` model) |
| `PORCUPINE_ACCESS_KEY` | — | Picovoice Porcupine access key (required) |
| `PORCUPINE_MODEL_PATH` | — | Path to custom `.ppn` wake word file |
| `WAKE_WORD_THRESHOLD` | `0.5` | Wake word detection sensitivity |
| `SPEAKER_VERIFY` | `auto` | Speaker verification (`auto`, `true`, `false`) |
| `SPEAKER_VERIFY_THRESHOLD` | `0.65` | Cosine similarity threshold |
| `IDLE_TIMEOUT_SEC` | `30` | Seconds before returning to sleep |
| `OPENCLAW_URL` | `http://127.0.0.1:18789/v1/chat/completions` | OpenClaw endpoint |
| `OPENCLAW_TOKEN` | — | Gateway auth token |
| `OPENCLAW_AGENT` | `main` | Agent ID to route to |
| `SYSTEM_PROMPT` | *(built-in)* | System prompt for voice responses |

## Usage

### Quick Start (with Chatterbox TTS)

```bash
./start-chatterbox.sh
```

### Quick Start (with Kokoro TTS)

```bash
./start-kokoro.sh
```

Open `https://<your-host>:8765` in your browser. Accept the self-signed cert warning.

**Controls:**
- **Wake word** — say "Hey Friday" to activate (when wake word mode is on)
- **Eye button** — toggle VAD (auto-listen mode)
- **Mic button** — hold to talk (manual mode), glows green to show audio levels
- **Spacebar** — hold to talk (when VAD is off)
- **Talk while Kismet speaks** — interrupts and listens to you
- **Settings (⚙️)** — open the settings drawer to manage speaker verification and enrollment

### Speaker Enrollment

1. Click the **⚙️ settings icon** in the header to open the settings drawer
2. Click **Enroll Voice**
3. Record 3 guided sentences (hold to record each one)
4. Your voice embedding is saved to `~/.kismet/voices/`
5. Kismet will now only respond to your voice

You can toggle speaker verification on/off from the settings drawer at any time.

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
│   ├── constants.ts         # Enrollment constants (min/max samples)
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

## VRAM Usage

| Component | VRAM |
|---|---|
| faster-whisper large-v3 | ~3 GB |
| Chatterbox Turbo | ~2 GB |
| **Total** | **~5 GB** |

Wake word (Porcupine) and speaker verification (SpeechBrain) run on CPU.

Use `TTS_ENGINE=kokoro` (~300MB) or `WHISPER_MODEL=medium` (~1.5GB) to reduce VRAM.

## Roadmap

See [ROADMAP.md](ROADMAP.md) for the full development plan:

- [x] **Phase 1:** Streaming TTS
- [x] **Phase 2:** Voice Activity Detection (VAD)
- [x] **Phase 3:** Interruption support
- [x] **Phase 4:** Wake word ("Hey Friday" via Porcupine)
- [x] **Phase 5:** Speaker verification (SpeechBrain ECAPA-TDNN)
- [x] **Phase 6:** Wake word + speaker verification combined
- [x] **Phase 7:** Polish & hardening (reconnection, toasts, audio visualizer)
- [x] **Phase 8:** Meeting companion (passive transcription, diarization basics)
- [ ] **Phase 9:** UI overhaul (settings drawer, mobile layout, visual polish) ← *in progress*

## License

MIT
