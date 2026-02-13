#!/bin/bash
# Start Kismet Voice Agent with Kokoro TTS (CPU-only, GPU-friendly)

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate voice-agent

# Isolate from ~/.local site-packages (numpy conflicts)
export PYTHONNOUSERSITE=1

# Set TTS engine
export TTS_ENGINE=kokoro
export KOKORO_VOICE=af_sky

# Feature flags (set to "false" to disable)
export WAKE_WORD_ENABLED=${WAKE_WORD_ENABLED:-true}
export SPEAKER_VERIFY=${SPEAKER_VERIFY:-true}

# Unbuffered output for logging
export PYTHONUNBUFFERED=1

# Run the server
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
exec python3 server.py "$@"
