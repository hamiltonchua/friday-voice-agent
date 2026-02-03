#!/bin/bash
# Start Kismet Voice Agent with Chatterbox TTS

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate chatterbox

# Set TTS engine
export TTS_ENGINE=chatterbox

# Voice cloning reference (Rosamund Pike)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export CHATTERBOX_REF="$SCRIPT_DIR/voices/rosamund_pike.wav"

# Unbuffered output for logging
export PYTHONUNBUFFERED=1

# Run the server
cd "$SCRIPT_DIR"
exec python3 server.py "$@"
