"""
Platform detection for Kismet Voice Agent.
Auto-detects Apple Silicon (MLX) vs CUDA vs CPU and sets backend defaults.
All values can be overridden via environment variables.
"""

import os
import platform
import sys


def _detect_platform() -> str:
    """Detect hardware platform: 'mlx', 'cuda', or 'cpu'."""
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return "mlx"
    # Check for CUDA
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


PLATFORM = os.getenv("KISMET_PLATFORM", _detect_platform())

# --- STT Backend ---
# "faster-whisper" (Linux/CUDA), "mlx-audio" (macOS/MLX)
_default_stt = "mlx-audio" if PLATFORM == "mlx" else "faster-whisper"
STT_BACKEND = os.getenv("STT_BACKEND", _default_stt)

# MLX STT model (only used when STT_BACKEND == "mlx-audio")
MLX_STT_MODEL = os.getenv("MLX_STT_MODEL", "mlx-community/parakeet-tdt-0.6b-v3")

# --- TTS Backend ---
# "mlx-audio" (macOS/MLX), "chatterbox-cuda" (Linux/CUDA), "kokoro-onnx" (CPU fallback)
_default_tts_map = {"mlx": "mlx-audio", "cuda": "chatterbox-cuda", "cpu": "kokoro-onnx"}
TTS_BACKEND = os.getenv("TTS_BACKEND", _default_tts_map.get(PLATFORM, "kokoro-onnx"))

# MLX TTS model (only used when TTS_BACKEND == "mlx-audio")
MLX_TTS_MODEL = os.getenv("MLX_TTS_MODEL", "mlx-community/chatterbox-fp16")
# Fallback MLX TTS model (kokoro, lighter weight)
MLX_TTS_MODEL_FALLBACK = os.getenv("MLX_TTS_MODEL_FALLBACK", "mlx-community/Kokoro-82M-bf16")


def print_config():
    """Print detected platform configuration."""
    print(f"[Platform] {PLATFORM} (system={platform.system()}, arch={platform.machine()})")
    print(f"[STT] Backend: {STT_BACKEND}" + (f" model: {MLX_STT_MODEL}" if STT_BACKEND == "mlx-audio" else ""))
    print(f"[TTS] Backend: {TTS_BACKEND}" + (f" model: {MLX_TTS_MODEL}" if TTS_BACKEND == "mlx-audio" else ""))
