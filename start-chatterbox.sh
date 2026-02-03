#!/bin/bash
# Start Kismet Voice Agent with Chatterbox TTS

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate chatterbox

# Set TTS engine
export TTS_ENGINE=chatterbox

# Run the server
cd "$(dirname "$0")"
exec python3 server.py "$@"
