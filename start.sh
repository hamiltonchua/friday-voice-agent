#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

export WHISPER_MODEL="${WHISPER_MODEL:-large-v3}"
export WHISPER_DEVICE="${WHISPER_DEVICE:-cuda}"
export KOKORO_VOICE="${KOKORO_VOICE:-af_heart}"

echo "🔨 Starting Friday Voice Chat"
echo "   STT: faster-whisper ($WHISPER_MODEL on $WHISPER_DEVICE)"
echo "   LLM: OpenClaw → Friday (localhost:18789)"
echo "   TTS: Kokoro ($KOKORO_VOICE)"
echo "   URL: https://localhost:8765"
echo ""

exec python3 server.py
