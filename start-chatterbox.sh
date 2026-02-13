#!/bin/bash
# Start Kismet Voice Agent with Chatterbox TTS

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate voice-agent

# Isolate from ~/.local site-packages (numpy conflicts)
export PYTHONNOUSERSITE=1

# Set TTS engine
export TTS_ENGINE=chatterbox

# Feature flags (set to "false" to disable)
export WAKE_WORD_ENABLED=${WAKE_WORD_ENABLED:-false}
export SPEAKER_VERIFY=${SPEAKER_VERIFY:-false}

# Voice cloning reference (Majel Barrett / TNG Ship Computer)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export CHATTERBOX_REF="$SCRIPT_DIR/voices/rosamund_pike.wav"

# Unbuffered output for logging
export PYTHONUNBUFFERED=1

# Run the server
cd "$SCRIPT_DIR"
exec python3 server.py "$@"
