# Friday Voice Agent

Real-time voice interface for [OpenClaw](https://github.com/openclaw/openclaw) agents. Talk to your agent using your voice — speech-to-text and text-to-speech run locally on your GPU, while the conversation routes through OpenClaw.

## How It Works

```
Browser (mic) → WebSocket → Server
                              ├─ STT: faster-whisper (GPU)
                              ├─ LLM: OpenClaw /v1/chat/completions → your agent
                              └─ TTS: Kokoro ONNX (local)
                            ← audio response
```

Everything except the LLM runs locally on your machine. No cloud STT/TTS APIs, no extra costs.

## Requirements

- **GPU:** NVIDIA with 4GB+ VRAM (tested on RTX 3060 12GB)
- **Python 3.10+**
- **OpenClaw** with chat completions endpoint enabled
- **CUDA** toolkit installed

### Python Dependencies

```bash
pip install faster-whisper fastapi uvicorn kokoro-onnx numpy httpx
```

### Model Files

Download Kokoro TTS models into the project directory:

```bash
curl -L -o kokoro-v1.0.onnx https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
curl -L -o voices-v1.0.bin https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
```

The faster-whisper model (`large-v3`, ~3GB) downloads automatically on first run.

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

Or via the gateway tool:

```
openclaw gateway config.patch '{"gateway":{"http":{"endpoints":{"chatCompletions":{"enabled":true}}}}}'
```

## SSL Certificate

Browsers require HTTPS to access the microphone over a network. Generate a self-signed cert:

```bash
openssl req -x509 -newkey rsa:2048 -keyout key.pem -out cert.pem -days 365 -nodes -subj "/CN=localhost"
```

If accessing from `localhost` only, this isn't needed — browsers allow mic access on localhost over HTTP.

## Usage

### Quick Start

```bash
./start.sh
```

### Manual Start

```bash
python3 server.py
```

Open `https://<your-host>:8765` in your browser. Accept the self-signed cert warning, then hold the mic button or press **spacebar** to talk.

### Configuration

Set environment variables to customize:

| Variable | Default | Description |
|---|---|---|
| `WHISPER_MODEL` | `large-v3` | Whisper model size (`tiny`, `base`, `small`, `medium`, `large-v3`) |
| `WHISPER_DEVICE` | `cuda` | Device for STT (`cuda`, `cpu`) |
| `KOKORO_VOICE` | `af_heart` | Kokoro voice ID |
| `OPENCLAW_URL` | `http://127.0.0.1:18789/v1/chat/completions` | OpenClaw endpoint |
| `OPENCLAW_TOKEN` | — | Gateway auth token |
| `OPENCLAW_AGENT` | `main` | Agent ID to route to |
| `SYSTEM_PROMPT` | *(built-in)* | System prompt for voice responses |

Example with a different voice and smaller model:

```bash
WHISPER_MODEL=medium KOKORO_VOICE=bf_emma python3 server.py
```

## VRAM Usage

| Component | VRAM |
|---|---|
| faster-whisper large-v3 | ~3 GB |
| Kokoro ONNX | ~300 MB |
| **Total** | **~3.3 GB** |

Fits comfortably on a 4GB+ GPU. Use `WHISPER_MODEL=medium` (~1.5GB) or `small` (~500MB) to reduce usage.

## Roadmap

See [ROADMAP.md](ROADMAP.md) for the development plan:

1. **Streaming TTS** — sentence-level audio streaming for lower perceived latency
2. **VAD** — voice activity detection, no more push-to-talk button
3. **Interruption** — talk over the agent and it stops speaking
4. **Wake word** — "Hey Friday" activation
5. **Polish** — reconnection, settings, mobile support

## License

MIT
