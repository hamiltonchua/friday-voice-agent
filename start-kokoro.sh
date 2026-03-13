#!/bin/bash
# Start Kismet Voice Agent with Kokoro TTS (CPU-only, GPU-friendly)

# Load .env if present
SCRIPT_DIR_EARLY="$(cd "$(dirname "$0")" && pwd)"
if [ -f "$SCRIPT_DIR_EARLY/.env" ]; then
    set -a
    source "$SCRIPT_DIR_EARLY/.env"
    set +a
fi

# Activate conda environment (use full path so this works without conda in PATH)
CONDA_BASE=/opt/homebrew/anaconda3
eval "$($CONDA_BASE/bin/conda shell.bash hook)"
conda activate voice-agent

# Isolate from ~/.local site-packages (numpy conflicts)
export PYTHONNOUSERSITE=1

# Set TTS engine (Kokoro via MLX on Apple Silicon)
export TTS_ENGINE=kokoro
export KOKORO_VOICE=af_kore
export MLX_TTS_VOICE=af_kore
export MLX_TTS_MODEL=mlx-community/Kokoro-82M-bf16

# Feature flags (set to "false" to disable)
export WAKE_WORD_ENABLED=${WAKE_WORD_ENABLED:-true}
export SPEAKER_VERIFY=${SPEAKER_VERIFY:-true}

# Inherit gateway token from system env so it stays current after rotations
export OPENCLAW_TOKEN="${OPENCLAW_TOKEN:-$OPENCLAW_GATEWAY_TOKEN}"

# Unbuffered output for logging
export PYTHONUNBUFFERED=1

# Run the server
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
exec python3 server.py "$@"
