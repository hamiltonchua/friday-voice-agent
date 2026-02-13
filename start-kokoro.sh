#!/bin/bash
# Start Kismet Voice Agent with Kokoro TTS (CPU-only, GPU-friendly)

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate chatterbox

# Isolate from ~/.local site-packages (numpy conflicts)
export PYTHONNOUSERSITE=1

# Set TTS engine
export TTS_ENGINE=kokoro

# Unbuffered output for logging
export PYTHONUNBUFFERED=1

# Run the server
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
exec python3 server.py "$@"
