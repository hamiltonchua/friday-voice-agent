#!/bin/bash
# Start Kismet Voice Agent on macOS (Apple Silicon / MLX)

# Find and activate conda environment
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
elif [ -f /opt/homebrew/anaconda3/bin/conda ]; then
    eval "$(/opt/homebrew/anaconda3/bin/conda shell.bash hook)"
elif [ -f ~/miniforge3/bin/conda ]; then
    eval "$(~/miniforge3/bin/conda shell.bash hook)"
elif [ -f ~/miniconda3/bin/conda ]; then
    eval "$(~/miniconda3/bin/conda shell.bash hook)"
fi
conda activate voice-agent

# Isolate from ~/.local site-packages
export PYTHONNOUSERSITE=1

# Platform is auto-detected, but we can be explicit
export KISMET_PLATFORM=mlx
export STT_BACKEND=mlx-audio
export TTS_BACKEND=mlx-audio

# MLX models (defaults, override as needed)
export MLX_STT_MODEL=${MLX_STT_MODEL:-mlx-community/whisper-large-v3-turbo-asr-fp16}
export MLX_TTS_MODEL=${MLX_TTS_MODEL:-mlx-community/Kokoro-82M-bf16}
export MLX_TTS_VOICE=${MLX_TTS_VOICE:-af_sky}

# Voice cloning reference
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export CHATTERBOX_REF="${CHATTERBOX_REF:-$SCRIPT_DIR/voices/rosamund_pike.wav}"

# Feature flags (set to "false" to disable)
export WAKE_WORD_ENABLED=${WAKE_WORD_ENABLED:-false}
export PICOVOICE_ACCESS_KEY=${PICOVOICE_ACCESS_KEY:-""}
export WAKE_WORD_KEYWORD=${WAKE_WORD_KEYWORD:-"jarvis"}  # built-in or path to .ppn
export WAKE_WORD_SENSITIVITY=${WAKE_WORD_SENSITIVITY:-"0.5"}
export SPEAKER_VERIFY=${SPEAKER_VERIFY:-auto}

# Unbuffered output for logging
export PYTHONUNBUFFERED=1

# Run the server
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
exec python3 server.py "$@"
