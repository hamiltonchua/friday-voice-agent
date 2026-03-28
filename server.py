#!/usr/bin/env python3
"""
Kismet Voice Agent - Real-time voice chat with streaming TTS and interruption support.
Supports both Linux (CUDA) and macOS (Apple Silicon / MLX) backends.

v0.6: Multi-platform support (MLX on macOS, CUDA on Linux)
"""

import asyncio
import base64
import io
import json
import os
import re
import tempfile
import time
import uuid
import wave
from pathlib import Path
from typing import AsyncIterator

import httpx
import numpy as np
import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from platform_config import (
    PLATFORM, STT_BACKEND, TTS_BACKEND,
    MLX_STT_MODEL, MLX_TTS_MODEL, MLX_TTS_MODEL_FALLBACK,
    print_config,
)
from auth import mount_auth_routes, auth_middleware, is_ws_authenticated
from session_memory import (
    clear_session_memory,
    init_session_db,
    load_recent_session_messages,
    persist_session_message,
    remove_last_session_message,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# Legacy env vars (used by faster-whisper / CUDA backends)
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cuda")
WHISPER_COMPUTE = os.getenv("WHISPER_COMPUTE", "float16")
TTS_ENGINE = os.getenv("TTS_ENGINE", "chatterbox")  # "chatterbox" or "kokoro" (legacy, used by CUDA/ONNX backends)
KOKORO_VOICE = os.getenv("KOKORO_VOICE", "af_sky")
CHATTERBOX_REF = os.getenv("CHATTERBOX_REF", None)  # Optional reference audio for voice cloning
MLX_TTS_VOICE = os.getenv("MLX_TTS_VOICE", "af_sky")  # Voice for MLX Kokoro
STT_SAMPLE_RATE = 16000

# Wake word config (Porcupine)
WAKE_WORD_ENABLED = os.getenv("WAKE_WORD_ENABLED", "true").lower() == "true"
PICOVOICE_ACCESS_KEY = os.getenv("PICOVOICE_ACCESS_KEY", "")
# Built-in keywords: alexa, jarvis, computer, hey google, hey siri, picovoice, etc.
# For custom keywords, set to path of .ppn file (e.g. /path/to/hey-friday.ppn)
WAKE_WORD_KEYWORD = os.getenv("WAKE_WORD_KEYWORD", "jarvis")
WAKE_WORD_SENSITIVITY = float(os.getenv("WAKE_WORD_SENSITIVITY", "0.5"))  # 0.0-1.0
IDLE_TIMEOUT_SEC = float(os.getenv("IDLE_TIMEOUT_SEC", "30"))  # Return to sleep after this many seconds of silence

# Smart Turn endpoint detection (turn-taking prediction)
SMART_TURN_ENABLED = os.getenv("SMART_TURN_ENABLED", "true").lower() == "true"
SMART_TURN_THRESHOLD = float(os.getenv("SMART_TURN_THRESHOLD", "0.5"))  # Probability threshold for "turn complete"
SMART_TURN_MAX_WAIT_SEC = float(os.getenv("SMART_TURN_MAX_WAIT_SEC", "3.0"))  # Force-send after this silence

# LLM backend (OpenAI-compatible API — LM Studio, Ollama, vLLM, etc.)
LLM_URL = os.getenv("LLM_URL", "http://127.0.0.1:1234/v1/chat/completions")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")  # Optional — most local servers don't need auth
LLM_MODEL = os.getenv("LLM_MODEL", "nvidia/nemotron-3-nano")

# Conversation history — LM Studio is stateless, so we manage context locally
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "20"))  # Sliding window size (user+assistant pairs)

# Forgetful semantic memory (RAG — inject relevant memories into system prompt)
FORGETFUL_ENABLED = os.getenv("FORGETFUL_ENABLED", "true").lower() == "true"
FORGETFUL_MAX_MEMORIES = int(os.getenv("FORGETFUL_MAX_MEMORIES", "3"))  # Top-K memories to inject

SPEAKER_VERIFY = os.getenv("SPEAKER_VERIFY", "auto")  # "true", "false", or "auto" (verify if enrolled)

# Push endpoint — allows sub-agents to POST results back to the voice UI
PUSH_SECRET = os.getenv("PUSH_SECRET", "")
PUSH_URL = os.getenv("PUSH_URL", "https://prodigy.skunk-shark.ts.net:8765/push")

# Delegation — delegate tasks to an external AI agent via ACP or CLI fallback.
# Supports any ACP-compatible agent: opencode, hermes, claude, etc.
DELEGATE_CMD = os.getenv("DELEGATE_CMD", "opencode")
DELEGATE_MODEL = os.getenv("DELEGATE_MODEL", "opencode/mimo-v2-pro-free")
DELEGATE_TIMEOUT = int(os.getenv("DELEGATE_TIMEOUT", "120"))  # seconds
DELEGATE_ENABLED = os.getenv("DELEGATE_ENABLED", "true").lower() == "true"
DELEGATE_USE_ACP = os.getenv("DELEGATE_USE_ACP", "true").lower() == "true"


SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", (
    "You are Friday, a local voice assistant. "
    "Be accurate, concise, and action-oriented. "
    "Use memory only when it is relevant to the current request. "
    "If information is missing, ask one short clarifying question. "
    "Do not invent facts, prior conversations, or user preferences."
))

VOICE_OUTPUT_RULES = (
    "Voice output rules:\n"
    "- Respond in plain spoken English.\n"
    "- Keep it brief unless the user asks for detail.\n"
    "- Do not use markdown, bullet points, code fences, or emojis.\n"
    "- If listing multiple items, speak them naturally in sentence form."
)


MEETING_SYSTEM_PROMPT = os.getenv("MEETING_SYSTEM_PROMPT", (
    "You are Friday, a local voice assistant helping during a meeting. "
    "Answer using the meeting transcript when it is relevant. "
    "Be accurate, concise, and action-oriented. "
    "If information is missing, say so briefly."
))

CANVAS_INSTRUCTION = (
    "\n\n## CANVAS OUTPUT (IMPORTANT)\n"
    "You have a visual canvas display connected. When your response includes data that benefits from "
    "visual presentation (tables, charts, comparisons, code, lists, structured info), you MUST include "
    "a <canvas> block in your response. The canvas content is rendered visually on a separate screen — "
    "it is NOT spoken aloud. Always speak a brief summary AND include the canvas block.\n\n"
    "Two formats:\n"
    '1. Rich HTML: <canvas type="html" title="Title">...HTML content...</canvas>\n'
    '2. Plain text: <canvas type="text" title="Title">...text content...</canvas>\n\n'
    "For charts, include: <script src=\"https://cdn.jsdelivr.net/npm/chart.js\"></script>\n\n"
    "Example response:\n"
    "Here's the comparison you asked for.\n"
    '<canvas type="html" title="Python vs JavaScript">\n'
    "<table><tr><th>Feature</th><th>Python</th><th>JavaScript</th></tr>\n"
    "<tr><td>Typing</td><td>Dynamic</td><td>Dynamic</td></tr></table>\n"
    "</canvas>\n\n"
    "ALWAYS use canvas blocks for visual data. Do NOT describe tables/charts verbally — show them on canvas."
)

# Harmony format: channel instructions for gpt-oss models
# These are included in the system message content — LM Studio's chat template
# wraps them into proper Harmony <|start|>system<|message|>...<|end|> tokens.
HARMONY_CHANNEL_INSTRUCTIONS = (
    "\n\n# Valid channels: analysis, commentary, final. "
    "Channel must be included for every message.\n"
    "Calls to these tools must go to the commentary channel: 'functions'."
)

# Tool definitions in Harmony's TypeScript namespace syntax
# Included in a developer-role message (or system message if LM Studio doesn't support developer role)
HARMONY_TOOL_DEFS = (
    "\n\n# Tools\n\n"
    "## functions\n\n"
    "namespace functions {\n\n"
    "// Read the contents of a file on the local filesystem.\n"
    "// Use this for checking configs, logs, code, or any text file.\n"
    "// Returns the file contents (truncated to 8000 chars if very large).\n"
    "type read_file = (_: {\n"
    "// Absolute path to the file to read\n"
    "path: string,\n"
    "}) => any;\n\n"
    "// List files and directories at a given path.\n"
    "// Returns names with a trailing / for directories.\n"
    "type list_directory = (_: {\n"
    "// Absolute path to the directory to list\n"
    "path: string,\n"
    "}) => any;\n\n"
    "// Delegates a task to an external AI assistant for research, up-to-date information,\n"
    "// or complex analysis beyond your local knowledge.\n"
    "// Use delegate only when the task requires web access, multi-step reasoning,\n"
    "// or capabilities beyond simple file reading.\n"
    "// The result will be returned to you so you can summarize it for the user.\n"
    "type delegate = (_: {\n"
    "// The complete question or task to delegate, with full context\n"
    "task: string,\n"
    "}) => any;\n\n"
    "} // namespace functions"
)

# ---------------------------------------------------------------------------
# Global singletons
# ---------------------------------------------------------------------------
import threading

whisper_model = None
speaker_enrolled_embedding = None  # Cached enrollment embedding
_push_queue: asyncio.Queue = asyncio.Queue()  # Messages pushed via POST /push → delivered to active WS client
tts_model = None
tts_sample_rate = None
wake_word_model = None
deepfilter_model = None  # MLX DeepFilterNet model (mlx-audio)
smart_turn_model = None  # SmartTurn endpoint detector (mlx-audio)
conversation_history = []  # Sliding window of conversation context (sent to LLM each request)
_history_lock = asyncio.Lock()  # Protect conversation_history from concurrent mutations
_last_memory_context = ""  # Background-fetched memory context for next turn
_memory_fetch_task: asyncio.Task | None = None  # In-flight background memory query
_model_lock = threading.Lock()  # Prevent concurrent MLX/GPU model loads
_models_ready = False  # True once startup preload completes
_deepfilter_available = True  # Set to False if import fails
_delegate_acp_client = None  # Lazy-created OpencodeACPClient singleton

# Persistent HTTP client — reuses TCP connections to LLM server (avoids
# per-request connection setup overhead of ~50-200ms). Created at module level,
# closed on shutdown via lifespan handler.
_http_client = httpx.AsyncClient(
    limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
    timeout=httpx.Timeout(10.0, read=120.0),
)

# Forgetful semantic memory HTTP client (sidecar at localhost:8020)
# Start with: uvx forgetful-ai --transport http --port 8020
# Tight timeout — memory is optional enrichment, not critical path.
_forgetful_client = httpx.AsyncClient(
    base_url="http://localhost:8020",
    timeout=httpx.Timeout(30.0),
) if FORGETFUL_ENABLED else None

_CASUAL_MEMORY_SKIP_PHRASES = {
    'hey', 'hi', 'hello', 'yo', 'sup', 'what\'s up', 'whats up',
    'thanks', 'thank you', 'ok', 'okay', 'cool', 'nice', 'great',
    'bye', 'goodbye', 'see you',
}


def _should_query_memory(user_text: str) -> bool:
    normalized = ' '.join(user_text.strip().lower().split())
    if not normalized:
        return False
    if normalized in _CASUAL_MEMORY_SKIP_PHRASES:
        return False
    if len(normalized) <= 12 and len(normalized.split()) <= 3:
        return False
    return True


FORGETFUL_MAX_CONTENT_CHARS = int(os.getenv("FORGETFUL_MAX_CONTENT_CHARS", "300"))  # Truncate each memory's content


async def _fetch_memory_context(user_text: str) -> str:
    """Background task: query Forgetful and store result for next turn."""
    if not FORGETFUL_ENABLED or not _forgetful_client:
        return ""
    if not _should_query_memory(user_text):
        return ""
    t0 = time.time()
    try:
        resp = await _forgetful_client.post("/api/v1/memories/search", json={
            "query": user_text,
            "query_context": "Voice chat RAG — enriching response with relevant knowledge",
            "k": FORGETFUL_MAX_MEMORIES,
            "include_links": False,
        })
        search_ms = int((time.time() - t0) * 1000)
        if resp.status_code != 200:
            print(f"[Memory] Forgetful returned {resp.status_code} ({search_ms}ms)")
            return ""
        data = resp.json()
        memories = data.get("primary_memories", [])
        if not memories:
            print(f"[Memory] No relevant memories ({search_ms}ms)")
            return ""
        lines = []
        for mem in memories:
            title = mem.get("title", "")
            content = mem.get("content", "")
            if len(content) > FORGETFUL_MAX_CONTENT_CHARS:
                content = content[:FORGETFUL_MAX_CONTENT_CHARS] + "…"
            lines.append(f"- {title}: {content}")
        context = "\n".join(lines)
        print(f"[Memory] Fetched {len(memories)} memories ({len(context)} chars, {search_ms}ms)")
        return context
    except httpx.ConnectError:
        print(f"[Memory] Forgetful not reachable — skipping RAG ({int((time.time() - t0) * 1000)}ms)")
        return ""
    except Exception as e:
        print(f"[Memory] Error querying Forgetful: {type(e).__name__}: {e} ({int((time.time() - t0) * 1000)}ms)")
        return ""


async def _background_memory_fetch(user_text: str) -> None:
    """Fire-and-forget: fetch memories and stash for next turn."""
    global _last_memory_context
    try:
        result = await _fetch_memory_context(user_text)
        _last_memory_context = result
    except Exception as e:
        print(f"[Memory] Background fetch failed: {type(e).__name__}: {e}")


def schedule_memory_fetch(user_text: str) -> None:
    """Schedule a non-blocking memory fetch. Result available for the next turn."""
    global _memory_fetch_task
    if not FORGETFUL_ENABLED:
        return
    # Cancel any in-flight fetch
    if _memory_fetch_task and not _memory_fetch_task.done():
        _memory_fetch_task.cancel()
    _memory_fetch_task = asyncio.create_task(_background_memory_fetch(user_text))


def consume_memory_context() -> str:
    """Consume the pre-fetched memory context (returns it once, then clears)."""
    global _last_memory_context
    ctx = _last_memory_context
    _last_memory_context = ""
    return ctx


def _is_harmony_model() -> bool:
    """Check if the configured LLM model uses Harmony format."""
    return "gpt-oss" in LLM_MODEL.lower()


def _build_system_prompt(base_prompt: str, memory_context: str) -> str:
    """Build system prompt with optional memory context injection."""
    parts = [base_prompt.strip()]
    # Add Harmony channel instructions for gpt-oss models
    if _is_harmony_model():
        parts[0] += HARMONY_CHANNEL_INSTRUCTIONS
    if memory_context:
        parts.append(
            "Relevant memory:\n"
            f"{memory_context}\n\n"
            "Use this only if it helps answer the current request. "
            "If it seems unrelated, ignore it."
        )
    parts.append(VOICE_OUTPUT_RULES)
    # Add tool definitions for Harmony models with delegation enabled
    if _is_harmony_model() and DELEGATE_ENABLED:
        parts.append(HARMONY_TOOL_DEFS)
    return "\n\n".join(parts)


def get_whisper():
    global whisper_model
    with _model_lock:
        if whisper_model is None:
            if STT_BACKEND == "mlx-audio":
                from mlx_audio.stt.utils import load_model
                print(f"[STT] Loading MLX model: {MLX_STT_MODEL}...")
                whisper_model = load_model(MLX_STT_MODEL)
                print(f"[STT] MLX STT ready ({MLX_STT_MODEL}).")
            else:
                from faster_whisper import WhisperModel
                print(f"[STT] Loading {WHISPER_MODEL} on {WHISPER_DEVICE}...")
                whisper_model = WhisperModel(WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE)
                print("[STT] Ready.")
    return whisper_model


def get_tts():
    global tts_model, tts_sample_rate
    with _model_lock:
        if tts_model is None:
            if TTS_BACKEND == "mlx-audio":
                from mlx_audio.tts.utils import load_model
                print(f"[TTS] Loading MLX model: {MLX_TTS_MODEL}...")
                tts_model = load_model(MLX_TTS_MODEL)
                # MLX Chatterbox uses 24000Hz, Kokoro uses 24000Hz
                tts_sample_rate = 24000
                print(f"[TTS] MLX TTS ready. Model: {MLX_TTS_MODEL}")
            elif TTS_BACKEND == "chatterbox-cuda" or TTS_ENGINE == "chatterbox":
                from chatterbox.tts_turbo import ChatterboxTurboTTS
                print("[TTS] Loading Chatterbox Turbo...")
                tts_model = ChatterboxTurboTTS.from_pretrained(device="cuda")
                tts_sample_rate = tts_model.sr
                print(f"[TTS] Chatterbox ready. Sample rate: {tts_sample_rate}Hz")
                if CHATTERBOX_REF:
                    print(f"[TTS] Using reference voice: {CHATTERBOX_REF}")
            else:
                import kokoro_onnx
                print("[TTS] Loading Kokoro...")
                tts_model = kokoro_onnx.Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")
                tts_sample_rate = 22050  # Kokoro sample rate
                print(f"[TTS] Kokoro ready. Voice: {KOKORO_VOICE}")
    return tts_model, tts_sample_rate



def get_wake_word():
    global wake_word_model
    with _model_lock:
        if wake_word_model is None and WAKE_WORD_ENABLED:
            import pvporcupine
            if not PICOVOICE_ACCESS_KEY:
                print("[WakeWord] ERROR: PICOVOICE_ACCESS_KEY not set. Disabling wake word.")
                return None
            # Check if keyword is a file path (.ppn) or built-in name
            keyword = WAKE_WORD_KEYWORD
            if keyword.endswith(".ppn") or os.path.sep in keyword:
                print(f"[WakeWord] Loading custom keyword: {keyword}")
                wake_word_model = pvporcupine.create(
                    access_key=PICOVOICE_ACCESS_KEY,
                    keyword_paths=[keyword],
                    sensitivities=[WAKE_WORD_SENSITIVITY],
                )
            else:
                print(f"[WakeWord] Loading built-in keyword: {keyword}")
                wake_word_model = pvporcupine.create(
                    access_key=PICOVOICE_ACCESS_KEY,
                    keywords=[keyword],
                    sensitivities=[WAKE_WORD_SENSITIVITY],
                )
            print(f"[WakeWord] Porcupine ready. Keyword: {keyword}, Sensitivity: {WAKE_WORD_SENSITIVITY}")
    return wake_word_model


def preload_models():
    """Load all models at startup (called once). Prevents per-connection loading."""
    global _models_ready
    print("[Startup] Preloading models...")
    get_whisper()
    get_tts()
    if WAKE_WORD_ENABLED:
        get_wake_word()
    if SMART_TURN_ENABLED:
        get_smart_turn()
    _models_ready = True
    print("[Startup] All models ready.")


def _should_verify() -> bool:
    """Check if speaker verification is enabled."""
    if SPEAKER_VERIFY == "false":
        return False
    if SPEAKER_VERIFY == "true":
        return True
    # "auto" — verify only if enrollment exists
    import speaker_verify
    return speaker_verify.has_enrollment()


def _load_enrolled_embedding():
    """Load and cache enrolled embedding."""
    global speaker_enrolled_embedding
    import speaker_verify
    speaker_enrolled_embedding = speaker_verify.load_enrollment()
    return speaker_enrolled_embedding


def get_deepfilter():
    """Lazy-load MLX DeepFilterNet model (mlx-audio)."""
    global deepfilter_model, _deepfilter_available
    if not _deepfilter_available:
        return None
    with _model_lock:
        if deepfilter_model is None:
            try:
                from mlx_audio.sts.models.deepfilternet.model import DeepFilterNetModel
                print("[Denoise] Loading MLX DeepFilterNet model...")
                deepfilter_model = DeepFilterNetModel.from_pretrained()
                print("[Denoise] MLX DeepFilterNet ready.")
            except Exception as e:
                print(f"[Denoise] WARNING: Failed to load MLX DeepFilterNet: {e}")
                _deepfilter_available = False
                return None
    return deepfilter_model


def get_smart_turn():
    """Lazy-load SmartTurn v3 endpoint detector (mlx-audio)."""
    global smart_turn_model
    if not SMART_TURN_ENABLED:
        return None
    with _model_lock:
        if smart_turn_model is None:
            try:
                from mlx_audio.vad.utils import load_model
                print("[SmartTurn] Loading SmartTurn v3 model...")
                smart_turn_model = load_model("mlx-community/smart-turn-v3")
                print(f"[SmartTurn] Ready (threshold={SMART_TURN_THRESHOLD})")
            except Exception as e:
                print(f"[SmartTurn] WARNING: Failed to load: {e}")
                return None
    return smart_turn_model


def predict_turn_complete(audio_bytes: bytes) -> tuple[bool, float]:
    """Predict whether the user has finished their turn.
    Returns (is_complete, probability). Audio is raw PCM 16-bit 16kHz mono."""
    model = get_smart_turn()
    if model is None:
        return True, 1.0  # Fallback: always assume turn complete

    pcm = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    result = model.predict_endpoint(pcm, sample_rate=STT_SAMPLE_RATE, threshold=SMART_TURN_THRESHOLD)
    return bool(result.prediction), result.probability


def denoise(audio_bytes: bytes) -> bytes:
    """Apply MLX DeepFilterNet noise suppression to raw PCM bytes (16-bit 16kHz mono)."""
    df_model = get_deepfilter()
    if df_model is None:
        return audio_bytes

    t0 = time.time()

    from scipy.signal import resample_poly

    # PCM 16-bit → float32
    pcm = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    # Resample 16kHz → 48kHz (DeepFilterNet native rate)
    audio_48k = resample_poly(pcm, up=3, down=1).astype(np.float32)

    # MLX DeepFilterNet: enhance_array takes float32 numpy array, returns numpy array
    enhanced = df_model.enhance_array(audio_48k)

    # Resample 48kHz → 16kHz
    audio_16k = resample_poly(enhanced, up=1, down=3).astype(np.float32)

    # Float32 → PCM 16-bit
    audio_16k = np.clip(audio_16k, -1.0, 1.0)
    result = (audio_16k * 32767).astype(np.int16).tobytes()

    duration_ms = int((time.time() - t0) * 1000)
    print(f'[Denoise] Noise suppression applied ({duration_ms}ms)')
    return result


# ---------------------------------------------------------------------------
# Pipeline functions
# ---------------------------------------------------------------------------
def transcribe(audio_bytes: bytes, apply_denoise: bool = False) -> str:
    """Convert raw PCM 16-bit 16kHz mono audio to text."""
    if apply_denoise:
        audio_bytes = denoise(audio_bytes)
    model = get_whisper()

    # Write to temp WAV file (needed by both backends)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name
        with wave.open(f, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(STT_SAMPLE_RATE)
            wf.writeframes(audio_bytes)

    try:
        stt_start = time.time()
        if STT_BACKEND == "mlx-audio":
            result = model.generate(tmp_path)
            text = (result.text or "").strip()
            lang = getattr(result, "language", "?") or "?"
            stt_ms = int((time.time() - stt_start) * 1000)
            print(f"[STT] MLX ({lang}) → \"{text}\" ({stt_ms}ms)")
            return text
        else:
            segments, info = model.transcribe(tmp_path, language=None, vad_filter=True, beam_size=5)
            text = " ".join(seg.text.strip() for seg in segments).strip()
            stt_ms = int((time.time() - stt_start) * 1000)
            print(f"[STT] ({info.language}, {info.duration:.1f}s) → \"{text}\" ({stt_ms}ms)")
            return text
    finally:
        os.unlink(tmp_path)


def transcribe_array(pcm_int16: np.ndarray, apply_denoise: bool = False) -> str:
    """Transcribe a numpy int16 array directly (no temp file for MLX backend).
    Falls back to transcribe() for non-MLX backends."""
    if STT_BACKEND != "mlx-audio":
        return transcribe(pcm_int16.tobytes(), apply_denoise=apply_denoise)

    import mlx.core as mx
    model = get_whisper()

    audio_float = pcm_int16.astype(np.float32) / 32768.0
    if apply_denoise:
        # Denoise operates on bytes, convert round-trip
        denoised = denoise(pcm_int16.tobytes())
        audio_float = np.frombuffer(denoised, dtype=np.int16).astype(np.float32) / 32768.0

    audio_mx = mx.array(audio_float)
    result = model.generate(audio_mx)
    return (result.text or "").strip()


_porcupine_buffer = np.array([], dtype=np.int16)

def detect_wake_word(audio_chunk: np.ndarray) -> tuple[bool, float]:
    """
    Run wake word detection on an audio chunk using Porcupine.
    Returns (detected, confidence).
    Audio should be int16 @ 16kHz.
    """
    global _porcupine_buffer
    model = get_wake_word()
    if model is None:
        return False, 0.0

    # Porcupine requires exactly frame_length samples (512 for 16kHz)
    frame_len = model.frame_length
    _porcupine_buffer = np.concatenate([_porcupine_buffer, audio_chunk])

    detected = False
    while len(_porcupine_buffer) >= frame_len:
        frame = _porcupine_buffer[:frame_len]
        _porcupine_buffer = _porcupine_buffer[frame_len:]
        keyword_index = model.process(frame.tolist())
        if keyword_index >= 0:
            detected = True
            print(f"[WakeWord] Detected! (keyword index {keyword_index})")
            break

    return detected, 1.0 if detected else 0.0


# Sentence boundary pattern: ends with .!? followed by space or end
SENTENCE_END_RE = re.compile(r'[.!?](?:\s|$)')

# Canvas block extraction
CANVAS_BLOCK_RE = re.compile(r'<canvas\s+type="(\w+)"(?:\s+title="([^"]*)")?\s*>(.*?)</canvas>', re.DOTALL)



# Thinking block extraction — strip model reasoning from output
THINKING_BLOCK_RE = re.compile(r'<thinking>.*?</thinking>', re.DOTALL)




def extract_canvas_blocks(text: str) -> tuple[str, list[dict]]:
    """Extract <canvas> blocks from text. Returns (cleaned_text, blocks)."""
    blocks = []
    for m in CANVAS_BLOCK_RE.finditer(text):
        blocks.append({
            "type": m.group(1),
            "title": m.group(2) or "",
            "content": m.group(3).strip(),
        })
    cleaned = CANVAS_BLOCK_RE.sub("", text).strip()
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned, blocks


def strip_markdown_for_tts(text: str, spoken_structures: set[str] | None = None) -> str:
    """Strip markdown formatting from text for clean TTS output.
    Removes: bold/italic markers, headers, code blocks, bullet points, links, etc.
    Preserves the actual content text."""
    import re as _re
    print(f"[Strip] Input: {text[:60]}...")
    s = text
    lines = [line.rstrip() for line in s.splitlines()]

    # If a markdown table is detected, speak a concise cue once.
    has_table = False
    for i in range(len(lines) - 1):
        if "|" not in lines[i] or "|" not in lines[i + 1]:
            continue
        if _re.match(r'^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$', lines[i + 1]):
            has_table = True
            break
    if has_table:
        if spoken_structures is not None:
            if "table" in spoken_structures:
                return ""
            spoken_structures.add("table")
        return "I displayed that in a table below."

    # If a markdown list is detected, speak a concise cue once.
    has_list = any(
        _re.match(r'^\s*(?:[-*+]\s+|\d+\.\s+)', line)
        for line in lines
        if line.strip()
    )
    if has_list:
        if spoken_structures is not None:
            if "list" in spoken_structures:
                return ""
            spoken_structures.add("list")
        return "I displayed that as a list below."
    # Remove code blocks (``` ... ```)
    s = _re.sub(r'```[\s\S]*?```', '', s)
    # Remove inline code (`...`)
    s = _re.sub(r'`([^`]+)`', r'\1', s)
    # Remove bold/italic markers (**, *, __, _)
    s = _re.sub(r'\*\*\*(.+?)\*\*\*', r'\1', s)
    s = _re.sub(r'\*\*(.+?)\*\*', r'\1', s)
    s = _re.sub(r'\*(.+?)\*', r'\1', s)
    s = _re.sub(r'___(.+?)___', r'\1', s)
    s = _re.sub(r'__(.+?)__', r'\1', s)
    s = _re.sub(r'(?<!\w)_(.+?)_(?!\w)', r'\1', s)
    # Remove headers (# ## ### etc)
    s = _re.sub(r'^#{1,6}\s+', '', s, flags=_re.MULTILINE)
    # Remove link syntax [text](url) -> text
    s = _re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', s)
    # Remove image syntax ![alt](url)
    s = _re.sub(r'!\[([^\]]*)\]\([^)]+\)', r'\1', s)
    # Remove bullet points (- or * at start of line)
    s = _re.sub(r'^\s*[-*+]\s+', '', s, flags=_re.MULTILINE)
    # Remove numbered list markers (1. 2. etc)
    s = _re.sub(r'^\s*\d+\.\s+', '', s, flags=_re.MULTILINE)
    # Remove blockquotes
    s = _re.sub(r'^\s*>\s?', '', s, flags=_re.MULTILINE)
    # Remove horizontal rules
    s = _re.sub(r'^[-*_]{3,}\s*$', '', s, flags=_re.MULTILINE)
    # Remove table separators and loose pipe delimiters that sound noisy in TTS
    s = _re.sub(r'^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$', '', s, flags=_re.MULTILINE)
    s = s.replace('|', ' ')
    # Keep parenthetical content but strip the symbols themselves
    s = s.replace('(', ', ').replace(')', '')
    # Clean up extra whitespace
    s = _re.sub(r'[ \t]+', ' ', s)
    s = _re.sub(r'\n{3,}', '\n\n', s)
    s = _re.sub(r'\s+([,.;:!?])', r'\1', s)
    s = _re.sub(r',\s*,+', ', ', s)
    print(f"[Strip] Output: {s.strip()[:60]}...")
    return s.strip()


# Connected canvas display clients
_canvas_clients: set[WebSocket] = set()


async def push_canvas(blocks: list[dict], loop):
    """Push canvas blocks to connected canvas display clients via WebSocket."""
    if not _canvas_clients:
        print("[Canvas] No canvas clients connected, skipping push")
        return
    for block in blocks:
        msg = json.dumps({"type": "canvas_update", "block": block})
        dead = set()
        for client in _canvas_clients:
            try:
                await client.send_text(msg)
            except Exception:
                dead.add(client)
        _canvas_clients.difference_update(dead)
        print(f"[Canvas] Pushed {block['type']}: {block['title'] or '(untitled)'} to {len(_canvas_clients)} client(s)")


# ---------------------------------------------------------------------------
# Harmony response format filter (gpt-oss models)
# ---------------------------------------------------------------------------
# gpt-oss models emit multi-channel output using special tokens:
#   <|channel|>analysis  — chain-of-thought (suppress)
#   <|channel|>commentary — tool calls (suppress)
#   <|channel|>final     — user-facing text (keep)
# This filter extracts only 'final' channel content for the voice pipeline.
# Transparent passthrough for non-Harmony models (auto-detected on first token).
_HARMONY_CTRL_RE = re.compile(r'<\|(?:start|end|return|call|channel|constrain|message)\|>')


class HarmonyFilter:
    """Streaming filter that extracts only 'final' channel content from Harmony format."""

    def __init__(self):
        self._buf = ""
        self._passthrough = False  # True once we confirm non-Harmony output
        self._harmony = False      # True once Harmony tokens detected (never reverts)
        self._emitting = False     # True when inside final channel content
        self._tool_calls = []  # List of (tool_name, content_type, arguments_json)
        self._current_recipient = None  # e.g. "functions.delegate"
        self._current_constrain = None  # e.g. "json"

    def feed(self, token: str) -> str:
        """Feed a streaming token. Returns text to emit (empty if suppressed)."""
        if self._passthrough:
            return token

        self._buf += token

        # Already confirmed Harmony — go straight to processing
        if self._harmony:
            return self._process()

        # Fast path: first non-whitespace isn't <| → not Harmony, pass through
        stripped = self._buf.lstrip()
        if stripped and not stripped.startswith("<|"):
            self._passthrough = True
            out = self._buf
            self._buf = ""
            return out

        # Check for Harmony control tokens
        if "<|channel|>" in self._buf or "<|start|>" in self._buf:
            self._harmony = True
            return self._process()

        # Safety: 100+ chars with no Harmony tokens → passthrough
        if len(self._buf) > 100:
            self._passthrough = True
            out = self._buf
            self._buf = ""
            return out

        return ""

    def _process(self) -> str:
        output = ""
        while True:
            if self._emitting:
                # Emit content until end-of-block marker
                earliest_idx = len(self._buf)
                earliest_len = 0
                for marker in ("<|end|>", "<|return|>", "<|start|>", "<|call|>"):
                    idx = self._buf.find(marker)
                    if idx != -1 and idx < earliest_idx:
                        earliest_idx = idx
                        earliest_len = len(marker)

                if earliest_idx < len(self._buf):
                    output += self._buf[:earliest_idx]
                    self._buf = self._buf[earliest_idx + earliest_len:]
                    self._emitting = False
                    continue
                else:
                    # No end marker yet — emit safe portion, keep tail
                    safe = max(0, len(self._buf) - 12)
                    if safe > 0:
                        output += self._buf[:safe]
                        self._buf = self._buf[safe:]
                    break
            else:
                # Suppressing — look for <|channel|>final...<|message|>
                idx = self._buf.find("<|channel|>final")
                if idx != -1:
                    msg_idx = self._buf.find("<|message|>", idx)
                    if msg_idx != -1:
                        self._buf = self._buf[msg_idx + len("<|message|>"):]
                        self._emitting = True
                        continue
                    else:
                        # Keep from final marker onwards, wait for <|message|>
                        self._buf = self._buf[idx:]
                        break
                else:
                    # No final channel — check for tool calls
                    # Only attempt match when <|message|> is present AND has content after it
                    msg_pos = self._buf.find("<|message|>")
                    if msg_pos == -1:
                        # Pattern incomplete — keep buffering, wait for more tokens
                        break
                    msg_end = msg_pos + len("<|message|>")
                    if msg_end == len(self._buf):
                        # <|message|> at buffer end — content hasn't arrived yet
                        break
                    # Don't parse tool payload until we see a closing Harmony marker.
                    # Without this, we can capture partial JSON like '{"' mid-stream.
                    tail = self._buf[msg_end:]
                    end_idx = len(tail)
                    end_len = 0
                    for marker in ("<|call|>", "<|end|>", "<|return|>", "<|start|>"):
                        idx = tail.find(marker)
                        if idx != -1 and idx < end_idx:
                            end_idx = idx
                            end_len = len(marker)
                    if end_len == 0:
                        break
                    payload = tail[:end_idx]
                    header = self._buf[:msg_end]
                    parsed = self._parse_tool_call(header, payload)
                    if parsed is not None:
                        recipient, constrain, content = parsed
                        self._current_recipient = recipient
                        self._current_constrain = constrain
                        self._tool_calls.append((recipient, constrain, content))
                        self._buf = tail[end_idx + end_len:]
                        break
                    # Also check for legacy delegation pattern without explicit tool name
                    # e.g. <|channel|>commentary to=delegate
                    tool_match = re.search(
                        r'to=(delegate|read_file|list_directory)\b.*?<\|message\|>(.*)',
                        self._buf[:msg_end] + payload, re.DOTALL
                    )
                    if tool_match and tool_match.group(2):
                        matched_tool = tool_match.group(1)
                        content = _HARMONY_CTRL_RE.sub("", tool_match.group(2)).strip()
                        self._current_recipient = f"functions.{matched_tool}"
                        self._current_constrain = "json"
                        self._tool_calls.append((f"functions.{matched_tool}", "json", content))
                        self._buf = tail[end_idx + end_len:]
                        break
                    # <|message|> present but no tool pattern matched yet — keep buffering
                    break
        return output

    def _parse_tool_call(self, header: str, payload: str) -> tuple[str, str, str] | None:
        """Best-effort parse of Harmony commentary tool calls across header variants."""
        if "<|channel|>commentary" not in header:
            return None

        raw_content = _HARMONY_CTRL_RE.sub("", payload).strip()
        if not raw_content:
            return None

        to_match = re.search(r'\bto=([^\s<]+)', header)
        recipient = to_match.group(1).strip() if to_match else ""

        constrain_match = re.search(r'<\|constrain\|>([^<\s]+)', header)
        constrain = (constrain_match.group(1).strip() if constrain_match else "text")

        _KNOWN_TOOLS = {"delegate", "read_file", "list_directory"}

        # Recipient may appear in role position after <|start|>.
        if not recipient:
            start_role_match = re.search(r'<\|start\|>\s*([^\s<]+)', header)
            if start_role_match:
                candidate = start_role_match.group(1).strip()
                bare = candidate.removeprefix("functions.")
                if bare in _KNOWN_TOOLS or candidate.startswith("functions."):
                    recipient = candidate

        # Some outputs incorrectly place tool name in constrain slot.
        if not recipient:
            bare_constrain = constrain.removeprefix("functions.")
            if bare_constrain in _KNOWN_TOOLS or constrain.startswith("functions."):
                recipient = constrain if constrain.startswith("functions.") else f"functions.{constrain}"
                constrain = "json"

        # If recipient is still missing, try to infer from payload shape.
        if not recipient:
            looks_like_json = raw_content.startswith("{") or raw_content.startswith("[")
            if looks_like_json:
                recipient = "functions.delegate"
                if constrain == "text":
                    constrain = "json"
            else:
                return None

        # Normalize bare tool names to functions.* prefix
        if not recipient.startswith("functions."):
            recipient = f"functions.{recipient}"

        return recipient, constrain or "text", raw_content

    def flush(self) -> str:
        """Flush remaining buffer at end of stream."""
        # Catch tool calls if stream ended without <|call|> stop token.
        msg_pos = self._buf.find("<|message|>")
        if msg_pos != -1:
            msg_end = msg_pos + len("<|message|>")
            parsed = self._parse_tool_call(self._buf[:msg_end], self._buf[msg_end:])
            if parsed is not None:
                recipient, constrain, content = parsed
                self._current_recipient = recipient
                self._current_constrain = constrain
                self._tool_calls.append((recipient, constrain, content))
                self._buf = ""
                return ""

        tool_match = re.search(
            r'to=(delegate|read_file|list_directory)\b.*?<\|message\|>(.*)',
            self._buf, re.DOTALL
        )
        if tool_match and tool_match.group(2):
            matched_tool = tool_match.group(1)
            content = _HARMONY_CTRL_RE.sub("", tool_match.group(2)).strip()
            self._current_recipient = f"functions.{matched_tool}"
            self._current_constrain = "json"
            self._tool_calls.append((f"functions.{matched_tool}", "json", content))
            self._buf = ""
            return ""

        if self._passthrough or self._emitting:
            out = _HARMONY_CTRL_RE.sub("", self._buf)
            self._buf = ""
            return out.strip()
        self._buf = ""
        return ""

    @property
    def tool_calls(self) -> list:
        """Return detected tool calls: [(recipient, constrain_type, content), ...]"""
        return self._tool_calls


# ---------------------------------------------------------------------------
# Background Task Manager
# ---------------------------------------------------------------------------

class BackgroundTaskManager:
    """Manages per-connection background delegate tasks for async delegation."""

    def __init__(self, ws: WebSocket):
        self._ws = ws
        self._tasks: dict[str, asyncio.Task] = {}
        self._results: asyncio.Queue = asyncio.Queue()
        self._metadata: dict[str, dict] = {}

    def start_delegate(self, task_id: str, description: str, coro) -> None:
        """Start a background delegate task and send task_start message."""
        task = asyncio.create_task(coro)
        self._tasks[task_id] = task
        self._metadata[task_id] = {"description": description}

        # Send task_start message
        asyncio.get_event_loop().create_task(
            self._ws.send_json({"type": "task_start", "task_id": task_id, "description": description})
        )

        # Attach done callback
        task.add_done_callback(lambda t: self._on_task_done(task_id, t))

    def _on_task_done(self, task_id: str, task: asyncio.Task) -> None:
        """Handle task completion: put result in queue and send WS message."""
        description = self._metadata.get(task_id, {}).get("description", "")

        try:
            result = task.result()
            # Put result in queue for polling
            self._results.put_nowait((task_id, description, result))
            # Send task_complete message
            asyncio.get_event_loop().create_task(
                self._ws.send_json({"type": "task_complete", "task_id": task_id})
            )
        except asyncio.CancelledError:
            # Cancelled tasks are silent — no error message
            pass
        except Exception as e:
            # Send task_error message
            asyncio.get_event_loop().create_task(
                self._ws.send_json({"type": "task_error", "task_id": task_id, "error": str(e)})
            )

    def get_pending_result(self) -> tuple[str, str, str] | None:
        """Non-blocking check for completed task result. Returns (task_id, description, result) or None."""
        try:
            return self._results.get_nowait()
        except asyncio.QueueEmpty:
            return None

    @property
    def active_count(self) -> int:
        """Return number of running (not done) tasks."""
        return sum(1 for task in self._tasks.values() if not task.done())

    def cancel_all(self) -> None:
        """Cancel all running tasks."""
        for task in self._tasks.values():
            if not task.done():
                task.cancel()


async def call_delegated_agent(task: str) -> str:
    """Delegate a task to an external AI agent via ACP (preferred) or CLI fallback."""
    if DELEGATE_USE_ACP:
        try:
            return await call_delegated_agent_acp(task)
        except Exception as e:
            print(f"[Tool] ACP delegation failed, falling back to CLI: {e}")
    return await call_delegated_agent_cli(task)


class OpencodeACPClient:
    """Minimal ACP JSON-RPC client over stdio for a persistent opencode session."""

    def __init__(self, cmd_path: str, cwd: str):
        self._cmd_path = cmd_path
        self._cwd = cwd
        self._proc: asyncio.subprocess.Process | None = None
        self._reader_task: asyncio.Task | None = None
        self._stderr_task: asyncio.Task | None = None
        self._next_id = 1
        self._pending: dict[int, asyncio.Future] = {}
        self._session_id: str | None = None
        self._updates_by_session: dict[str, list[str]] = {}
        self._session_lock = asyncio.Lock()
        self._write_lock = asyncio.Lock()
        self._start_lock = asyncio.Lock()
        self._failed_until = 0.0

    async def _read_stdout(self):
        assert self._proc and self._proc.stdout
        while True:
            line = await self._proc.stdout.readline()
            if not line:
                break
            raw = line.decode("utf-8", errors="replace").strip()
            if not raw:
                continue
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                print(f"[ACP] Non-JSON line: {raw[:200]}")
                continue
            msg_id = msg.get("id")
            if isinstance(msg_id, int):
                fut = self._pending.get(msg_id)
                if fut and not fut.done():
                    fut.set_result(msg)
                continue
            method = msg.get("method")
            if method == "session/update":
                params = msg.get("params", {})
                session_id = params.get("sessionId")
                if isinstance(session_id, str):
                    text = self._extract_update_text(params.get("update"))
                    if text:
                        self._updates_by_session.setdefault(session_id, []).append(text)

    # Stderr noise patterns from ACP agents that retry multiple payload
    # formats — the JSON-RPC error response is already handled by the caller.
    _STDERR_NOISE_RE = re.compile(
        r"(Method not found|Background task failed|"
        r"RequestError\.method_not_found|"
        r"acp\.exceptions\.RequestError)",
    )

    async def _read_stderr(self):
        assert self._proc and self._proc.stderr
        squelch_traceback = False
        while True:
            line = await self._proc.stderr.readline()
            if not line:
                break
            text = line.decode("utf-8", errors="replace").rstrip()
            if not text:
                squelch_traceback = False
                continue
            # Suppress known noise: "Method not found" tracebacks from
            # unsupported ACP methods that we already handle via JSON-RPC.
            if self._STDERR_NOISE_RE.search(text):
                squelch_traceback = True
                continue
            if squelch_traceback:
                # Swallow continuation lines of a noisy traceback.
                if text.startswith((" ", "Traceback", "File ")):
                    continue
                squelch_traceback = False
            print(f"[ACP] {text}")

    def _extract_update_text(self, update: object) -> str:
        parts: list[str] = []

        def walk(obj: object):
            if isinstance(obj, dict):
                text_val = obj.get("text")
                if isinstance(text_val, str) and text_val.strip():
                    parts.append(text_val)
                for k, v in obj.items():
                    if k == "text":
                        continue
                    walk(v)
                return
            if isinstance(obj, list):
                for item in obj:
                    walk(item)

        walk(update)
        return "".join(parts).strip()

    def _extract_result_text(self, result: object) -> str:
        parts: list[str] = []

        def walk(obj: object):
            if isinstance(obj, dict):
                text_val = obj.get("text")
                if isinstance(text_val, str) and text_val.strip():
                    parts.append(text_val)
                content_val = obj.get("content")
                if isinstance(content_val, str) and content_val.strip():
                    parts.append(content_val)
                for k, v in obj.items():
                    if k in ("text", "content"):
                        continue
                    walk(v)
                return
            if isinstance(obj, list):
                for item in obj:
                    walk(item)

        walk(result)
        return "".join(parts).strip()

    async def _request(self, method: str, params: dict, timeout_sec: float) -> dict:
        if not self._proc or not self._proc.stdin:
            raise RuntimeError("ACP process not running")
        req_id = self._next_id
        self._next_id += 1
        fut = asyncio.get_event_loop().create_future()
        self._pending[req_id] = fut
        payload = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params,
        }
        data = (json.dumps(payload) + "\n").encode("utf-8")
        async with self._write_lock:
            self._proc.stdin.write(data)
            await self._proc.stdin.drain()
        try:
            msg = await asyncio.wait_for(fut, timeout=timeout_sec)
        finally:
            self._pending.pop(req_id, None)
        if "error" in msg:
            err = msg["error"]
            raise RuntimeError(f"ACP {method} error: {err}")
        return msg.get("result", {})

    async def ensure_started(self):
        async with self._start_lock:
            if self._proc and self._proc.returncode is None and self._session_id:
                return
            if time.time() < self._failed_until:
                raise RuntimeError("ACP startup is in cooldown after previous failure")

            # Build args: all ACP agents accept "acp"; --cwd is
            # opencode-specific — other agents receive cwd via session/new.
            cmd_name = Path(self._cmd_path).stem.lower()
            acp_args = [self._cmd_path, "acp"]
            if cmd_name == "opencode":
                acp_args += ["--cwd", self._cwd]
            self._proc = await asyncio.create_subprocess_exec(
                *acp_args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            self._reader_task = asyncio.create_task(self._read_stdout())
            self._stderr_task = asyncio.create_task(self._read_stderr())
            try:
                init_payloads = [
                    {"protocolVersion": 1},
                    {"protocolVersion": 1, "client": {"name": "kismet-voice-agent", "version": "0.1.0"}},
                    {"protocolVersion": 1, "clientInfo": {"name": "kismet-voice-agent", "version": "0.1.0"}},
                ]
                last_init_error: Exception | None = None
                for payload in init_payloads:
                    try:
                        await self._request("initialize", payload, timeout_sec=10.0)
                        last_init_error = None
                        break
                    except Exception as e:
                        last_init_error = e
                if last_init_error is not None:
                    raise last_init_error
                session_payloads = [
                    {"cwd": self._cwd, "mcpServers": []},
                    {"mcpServers": [], "cwd": self._cwd},
                    {"mcpServers": []},
                ]
                last_session_error: Exception | None = None
                session_result: dict = {}
                for payload in session_payloads:
                    try:
                        session_result = await self._request("session/new", payload, timeout_sec=10.0)
                        last_session_error = None
                        break
                    except Exception as e:
                        last_session_error = e
                if last_session_error is not None:
                    raise last_session_error
                session_id = session_result.get("sessionId")
                if not isinstance(session_id, str) or not session_id:
                    raise RuntimeError(f"ACP session/new missing sessionId: {session_result}")
                self._session_id = session_id
                print(f"[ACP] Connected session={session_id}")
            except Exception:
                await self.close()
                self._failed_until = time.time() + 60.0
                raise

    async def prompt(self, task: str, model: str, timeout_sec: float) -> str:
        await self.ensure_started()
        if not self._session_id:
            raise RuntimeError("ACP session unavailable")
        async with self._session_lock:
            session_id = self._session_id
            self._updates_by_session[session_id] = []

            # Best effort model switch — only supported by opencode.
            cmd_name = Path(self._cmd_path).stem.lower()
            if cmd_name == "opencode":
                try:
                    set_model_payloads = [
                        {"sessionId": session_id, "modelId": model},
                        {"sessionId": session_id, "model": model},
                    ]
                    set_model_error: Exception | None = None
                    for payload in set_model_payloads:
                        try:
                            await self._request(
                                "session/set_model",
                                payload,
                                timeout_sec=5.0,
                            )
                            set_model_error = None
                            break
                        except Exception as e:
                            set_model_error = e
                    if set_model_error is not None:
                        raise set_model_error
                except Exception as e:
                    print(f"[ACP] session/set_model not applied: {e}")

            prompt_payloads = [
                {"sessionId": session_id, "prompt": [{"type": "text", "text": task}]},
                {"sessionId": session_id, "content": [{"type": "text", "text": task}]},
                {"sessionId": session_id, "prompt": [{"type": "input_text", "text": task}]},
            ]
            result: dict = {}
            last_prompt_error: Exception | None = None
            for payload in prompt_payloads:
                try:
                    result = await self._request(
                        "session/prompt",
                        payload,
                        timeout_sec=timeout_sec,
                    )
                    last_prompt_error = None
                    break
                except Exception as e:
                    last_prompt_error = e
            if last_prompt_error is not None:
                raise last_prompt_error

            chunks = self._updates_by_session.pop(session_id, [])
            text = "".join(chunks).strip()
            if text:
                return text
            fallback_text = self._extract_result_text(result)
            return fallback_text

    async def close(self):
        for fut in self._pending.values():
            if not fut.done():
                fut.cancel()
        self._pending.clear()
        if self._proc:
            if self._proc.returncode is None:
                self._proc.terminate()
                try:
                    await asyncio.wait_for(self._proc.wait(), timeout=2.0)
                except Exception:
                    self._proc.kill()
            self._proc = None
        if self._reader_task:
            self._reader_task.cancel()
            self._reader_task = None
        if self._stderr_task:
            self._stderr_task.cancel()
            self._stderr_task = None
        self._session_id = None


def _get_delegate_acp_client(cmd_path: str) -> OpencodeACPClient:
    global _delegate_acp_client
    if _delegate_acp_client is None:
        _delegate_acp_client = OpencodeACPClient(
            cmd_path=cmd_path,
            cwd=str(Path(__file__).parent),
        )
    return _delegate_acp_client


async def call_delegated_agent_acp(task: str) -> str:
    import shutil

    cmd_path = shutil.which(DELEGATE_CMD)
    if not cmd_path:
        return f"Error: {DELEGATE_CMD} CLI not found. Install it or set DELEGATE_CMD env var."

    t0 = time.time()
    preview = task.replace("\n", "\\n")
    if len(preview) > 240:
        preview = preview[:240] + "..."
    print(f"[Tool] Delegating via ACP (task_chars={len(task)}): {preview!r}")

    client = _get_delegate_acp_client(cmd_path)
    result = await client.prompt(task=task, model=DELEGATE_MODEL, timeout_sec=float(DELEGATE_TIMEOUT))
    elapsed_ms = int((time.time() - t0) * 1000)
    print(f"[Tool] ACP responded ({elapsed_ms}ms, {len(result)} chars)")
    if len(result) > 4000:
        result = result[:4000] + "\n\n[Response truncated at 4000 chars]"
    return result


async def call_delegated_agent_cli(task: str) -> str:
    """Delegate a task to an external AI agent via opencode run."""
    import shutil

    cmd_path = shutil.which(DELEGATE_CMD)
    if not cmd_path:
        return f"Error: {DELEGATE_CMD} CLI not found. Install it or set DELEGATE_CMD env var."

    t0 = time.time()
    preview = task.replace("\n", "\\n")
    if len(preview) > 240:
        preview = preview[:240] + "..."
    print(f"[Tool] Delegating to {DELEGATE_CMD} (task_chars={len(task)}): {preview!r}")

    proc: asyncio.subprocess.Process | None = None
    try:
        # Build command: opencode run -m <model> "task"
        cmd_args = [cmd_path, "run", "-m", DELEGATE_MODEL, task]
        proc = await asyncio.create_subprocess_exec(
            *cmd_args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(),
            timeout=DELEGATE_TIMEOUT,
        )
        elapsed_ms = int((time.time() - t0) * 1000)
        result = stdout.decode("utf-8", errors="replace").strip()
        if proc.returncode != 0:
            err = stderr.decode("utf-8", errors="replace").strip()
            print(f"[Tool] {DELEGATE_CMD} failed (rc={proc.returncode}, {elapsed_ms}ms): {err[:200]}")
            return f"Error: {DELEGATE_CMD} returned exit code {proc.returncode}. {err[:500]}"
        print(f"[Tool] {DELEGATE_CMD} responded ({elapsed_ms}ms, {len(result)} chars)")
        # Truncate very long responses to avoid blowing context
        if len(result) > 4000:
            result = result[:4000] + "\n\n[Response truncated at 4000 chars]"
        return result
    except asyncio.TimeoutError:
        elapsed_ms = int((time.time() - t0) * 1000)
        print(f"[Tool] {DELEGATE_CMD} timed out after {elapsed_ms}ms")
        if proc and proc.returncode is None:
            proc.kill()
        return f"Error: {DELEGATE_CMD} timed out after {DELEGATE_TIMEOUT}s"
    except Exception as e:
        print(f"[Tool] {DELEGATE_CMD} error: {e}")
        return f"Error calling {DELEGATE_CMD}: {str(e)}"


def _coerce_delegate_task(raw: object) -> str:
    """Best-effort extraction of a delegate task from nested/malformed tool args."""
    value: object = raw
    for _ in range(3):
        if isinstance(value, dict):
            value = value.get("task", "")
            continue
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return ""
            # Handle cases where task is a JSON string containing {"task": "..."}.
            try:
                decoded = json.loads(text)
            except json.JSONDecodeError:
                return text
            value = decoded
            continue
        return str(value).strip()
    return str(value).strip()


def _execute_read_file(arguments: dict) -> str:
    """Read a file from the local filesystem. Returns contents truncated to 8000 chars."""
    path_str = arguments.get("path", "")
    if not path_str:
        return "Error: No path provided."
    target = Path(path_str).resolve()
    if not target.exists():
        return f"Error: {target} does not exist."
    if not target.is_file():
        return f"Error: {target} is not a file."
    try:
        content = target.read_text(errors="replace")
        if len(content) > 8000:
            return content[:8000] + f"\n\n[Truncated — file is {len(content)} chars total]"
        return content
    except Exception as e:
        return f"Error reading {target}: {e}"


def _execute_list_directory(arguments: dict) -> str:
    """List files and directories at a given path."""
    path_str = arguments.get("path", "")
    if not path_str:
        return "Error: No path provided."
    target = Path(path_str).resolve()
    if not target.exists():
        return f"Error: {target} does not exist."
    if not target.is_dir():
        return f"Error: {target} is not a directory."
    try:
        entries = sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        lines = []
        for entry in entries[:200]:
            name = entry.name + ("/" if entry.is_dir() else "")
            lines.append(name)
        result = "\n".join(lines)
        if len(entries) > 200:
            result += f"\n\n[Showing 200 of {len(entries)} entries]"
        return result
    except PermissionError:
        return f"Error: Permission denied for {target}"
    except Exception as e:
        return f"Error listing {target}: {e}"


async def execute_tool(
    tool_name: str,
    arguments: dict,
    bg_manager: BackgroundTaskManager | None = None,
) -> str:
    """Execute a Harmony tool call and return the result text."""
    if tool_name in ("functions.read_file", "read_file"):
        print(f"[Tool] read_file: {arguments.get('path', '?')}")
        return _execute_read_file(arguments)

    if tool_name in ("functions.list_directory", "list_directory"):
        print(f"[Tool] list_directory: {arguments.get('path', '?')}")
        return _execute_list_directory(arguments)

    if tool_name in ("functions.delegate", "delegate"):
        task = _coerce_delegate_task(arguments)
        preview = task.replace("\n", "\\n")
        if len(preview) > 240:
            preview = preview[:240] + "..."
        print(f"[Tool] Parsed delegate task chars={len(task)} from args keys={list(arguments.keys())}: {preview!r}")
        if not task:
            return "Error: No task provided for delegation."
        if not DELEGATE_ENABLED:
            return "Delegation is disabled. Answer based on your own knowledge."
        if bg_manager is not None:
            task_id = uuid.uuid4().hex[:8]
            description = task[:80].replace("\n", " ")
            bg_manager.start_delegate(task_id, description, call_delegated_agent(task))
            print(f"[Delegate] Fired background delegate {task_id}: {description[:60]}")
            return (
                f"Task '{description[:80]}' has been delegated to a background agent "
                f"(task {task_id}). It will run asynchronously. "
                f"Briefly acknowledge to the user that you are working on this in the "
                f"background and they can continue talking to you."
            )
        return await call_delegated_agent(task)

    print(f"[Tool] Unknown tool: {tool_name}")
    return f"Error: Unknown tool '{tool_name}'. Available tools: read_file, list_directory, delegate"




async def chat_stream(
    user_text: str,
    cancel_event: asyncio.Event,
    system_prompt: str | None = None,
    bg_manager: BackgroundTaskManager | None = None,
) -> AsyncIterator[tuple[str, str]]:
    """
    Stream chat response from LLM server (OpenAI-compatible API).
    Yields tuples of (event_type, data):
      - ("token", token_text) for each token
      - ("sentence", sentence_text) for each complete sentence
      - ("working", "") when tools are likely running (>2s before first token)
      - ("done", full_text) when finished
      - ("cancelled", partial_text) when interrupted
    """
    global conversation_history

    # Use pre-fetched memory context (from previous turn's background query)
    memory_context = consume_memory_context()
    if memory_context:
        print(f"[Memory] Injecting pre-fetched memories ({len(memory_context)} chars)")
    effective_prompt = _build_system_prompt(system_prompt or SYSTEM_PROMPT, memory_context)
    # Schedule background memory fetch for the NEXT turn (non-blocking)
    schedule_memory_fetch(user_text)

    # Build messages with conversation history (sliding window)
    messages = [{"role": "system", "content": effective_prompt}]
    async with _history_lock:
        messages.extend(conversation_history[-MAX_HISTORY_MESSAGES:])
    messages.append({"role": "user", "content": user_text})

    # Track in conversation history + persistent session memory
    async with _history_lock:
        conversation_history.append({"role": "user", "content": user_text})
        persist_session_message("user", user_text)
        # Trim history to sliding window
        if len(conversation_history) > MAX_HISTORY_MESSAGES:
            conversation_history = conversation_history[-MAX_HISTORY_MESSAGES:]

    MAX_TOOL_ITERATIONS = 3
    request_start = time.time()

    headers = {"Content-Type": "application/json"}
    if LLM_API_KEY:
        headers["Authorization"] = f"Bearer {LLM_API_KEY}"

    for iteration in range(MAX_TOOL_ITERATIONS + 1):
        full_text = ""
        raw_text = ""  # unfiltered LLM output for diagnostics
        buffer = ""
        cancelled = False
        got_first_token = False
        working_emitted = False
        harmony = HarmonyFilter()

        try:
            async with _http_client.stream(
                "POST",
                LLM_URL,
                headers=headers,
                json={
                    "model": LLM_MODEL,
                    "messages": messages,
                    "stream": True,
                },
                timeout=httpx.Timeout(10.0, read=120.0),
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    # Check for cancellation
                    if cancel_event.is_set():
                        cancelled = True
                        print(f"[LLM] Cancelled after: \"{full_text[:50]}...\"")
                        break

                    # Detect tool usage: >2s without content tokens
                    if not got_first_token and not working_emitted and time.time() - request_start > 2.0:
                        working_emitted = True
                        print("[LLM] Tools likely running (>2s before first token)")
                        yield ("working", "")

                    if not line.startswith("data: "):
                        continue

                    data = line[6:]
                    if data == "[DONE]":
                        break

                    try:
                        chunk = json.loads(data)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        token = delta.get("content", "")

                        if token:
                            if not got_first_token:
                                ttft_ms = int((time.time() - request_start) * 1000)
                                print(f"[LLM] TTFT: {ttft_ms}ms")
                            got_first_token = True
                            raw_text += token

                            # Filter Harmony control tokens — only keep 'final' channel
                            filtered = harmony.feed(token)
                            if not filtered:
                                continue
                            full_text += filtered
                            buffer += filtered
                            yield ("token", filtered)

                            # Check for sentence boundaries
                            while True:
                                match = SENTENCE_END_RE.search(buffer)
                                if not match:
                                    break

                                end_pos = match.end()
                                sentence = buffer[:end_pos].strip()
                                buffer = buffer[end_pos:].lstrip()

                                if sentence:
                                    yield ("sentence", sentence)

                    except json.JSONDecodeError:
                        continue

        except httpx.HTTPError as e:
            print(f"[LLM] HTTP error: {e}")
            cancelled = True

        # Flush any remaining content from the Harmony filter
        remaining = harmony.flush()
        if remaining:
            full_text += remaining
            buffer += remaining

        if cancelled:
            # Remove user message from history on cancel
            async with _history_lock:
                if conversation_history and conversation_history[-1]["role"] == "user":
                    conversation_history.pop()
            remove_last_session_message("user")
            yield ("cancelled", full_text)
            return

        # Tool-call iteration: no final channel content, but model requested a tool
        if harmony.tool_calls and not full_text.strip():
            if iteration >= MAX_TOOL_ITERATIONS:
                break

            tool_name, constrain, tool_args_raw = harmony.tool_calls[0]
            print(f"[Tool] Harmony tool call detected (iteration {iteration + 1}): {tool_name}")
            print(f"[Tool] Raw args constrain={constrain} chars={len(tool_args_raw or '')}")

            # Parse tool arguments
            tool_args = {}
            if constrain == "json":
                try:
                    parsed = json.loads(tool_args_raw) if tool_args_raw else {}
                    if isinstance(parsed, dict):
                        tool_args = parsed
                    else:
                        tool_args = {"task": str(parsed)}
                except json.JSONDecodeError:
                    tool_args = {"task": tool_args_raw}
            else:
                tool_args = {"task": tool_args_raw}

            if not working_emitted:
                yield ("working", "")

            # Execute tool, then continue loop with appended context
            tool_result = await execute_tool(tool_name, tool_args, bg_manager=bg_manager)
            messages.append({"role": "assistant", "content": raw_text})
            messages.append({
                "role": "user",
                "content": f"[Tool result from {tool_name}]:\n{tool_result}",
            })
            continue

        # Final response path
        if buffer.strip():
            yield ("sentence", buffer.strip())

        # Track assistant response in history (clean final text only)
        async with _history_lock:
            conversation_history.append({"role": "assistant", "content": full_text})
            persist_session_message("assistant", full_text)
            if len(conversation_history) > MAX_HISTORY_MESSAGES:
                conversation_history = conversation_history[-MAX_HISTORY_MESSAGES:]
        llm_total_ms = int((time.time() - request_start) * 1000)
        print(f"[LLM] → \"{full_text[:80]}...\" ({llm_total_ms}ms)" if len(full_text) > 80 else f"[LLM] → \"{full_text}\" ({llm_total_ms}ms)")
        # Diagnostic: log raw LLM output when it was filtered to empty or significantly reduced
        if not full_text and raw_text:
            print(f"[LLM] ⚠ Response filtered to empty! HarmonyFilter state: harmony={harmony._harmony}, passthrough={harmony._passthrough}, emitting={harmony._emitting}")
            print(f"[LLM] ⚠ Raw LLM output ({len(raw_text)} chars): \"{raw_text[:300]}\"")
        elif raw_text and len(full_text) < len(raw_text) * 0.5:
            print(f"[LLM] ⚠ Significant filtering: {len(raw_text)} raw chars → {len(full_text)} filtered chars")
            print(f"[LLM] ⚠ Raw LLM output (first 200): \"{raw_text[:200]}\"")
        yield ("done", full_text)
        return

    # Exhausted tool iterations without a final channel response
    print(f"[LLM] ⚠ Exhausted {MAX_TOOL_ITERATIONS} tool iterations without final response")
    fallback = "I tried to look that up but couldn't get a complete answer. Could you try asking differently?"
    async with _history_lock:
        conversation_history.append({"role": "assistant", "content": fallback})
        persist_session_message("assistant", fallback)
        if len(conversation_history) > MAX_HISTORY_MESSAGES:
            conversation_history = conversation_history[-MAX_HISTORY_MESSAGES:]
    yield ("sentence", fallback)
    yield ("done", fallback)


def synthesize(text: str) -> tuple[bytes, int]:
    """Convert text to speech, return (WAV bytes, sample_rate)."""
    model, sr = get_tts()

    if TTS_BACKEND == "mlx-audio":
        import mlx.core as mx
        # MLX Audio — works for both Chatterbox and Kokoro MLX models
        is_chatterbox = "chatterbox" in MLX_TTS_MODEL.lower()
        generate_kwargs = {"text": text}
        if not is_chatterbox:
            generate_kwargs["voice"] = MLX_TTS_VOICE
            generate_kwargs["speed"] = 1.0
            generate_kwargs["lang_code"] = "a"  # American English
        if is_chatterbox and CHATTERBOX_REF:
            generate_kwargs["ref_audio"] = CHATTERBOX_REF

        samples = None
        for result in model.generate(**generate_kwargs):
            audio_data = result.audio
            # Convert mx.array to numpy if needed
            if hasattr(audio_data, 'tolist'):
                audio_data = np.array(audio_data.tolist(), dtype=np.float32)
            elif not isinstance(audio_data, np.ndarray):
                audio_data = np.array(audio_data, dtype=np.float32)
            samples = audio_data if samples is None else np.concatenate([samples, audio_data])

        if hasattr(result, 'sample_rate') and result.sample_rate:
            sr = result.sample_rate

    elif TTS_BACKEND == "chatterbox-cuda" or TTS_ENGINE == "chatterbox":
        import torch
        # Chatterbox Turbo (CUDA)
        wav_tensor = model.generate(
            text,
            audio_prompt_path=CHATTERBOX_REF,
            exaggeration=0.0,
            cfg_weight=0.0,
        )
        # wav_tensor is (1, samples) float32 tensor
        samples = wav_tensor.squeeze(0).cpu().numpy()
    else:
        # Kokoro ONNX
        samples, sr = model.create(text, voice=KOKORO_VOICE, speed=1.0)

    # Convert to WAV bytes
    buf = io.BytesIO()
    with wave.open(buf, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        # Ensure float32 → int16
        if samples.dtype != np.int16:
            pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
        else:
            pcm = samples
        wf.writeframes(pcm.tobytes())

    return buf.getvalue(), sr


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Meeting Companion — speaker clustering + transcript
# ---------------------------------------------------------------------------
class MeetingSession:
    """Tracks meeting transcript and speaker embeddings for diarization."""

    def __init__(self, owner_embedding=None, similarity_threshold=0.75):
        self.transcript: list[dict] = []  # [{speaker, text, time}]
        self.owner_embedding = owner_embedding  # Ham's enrolled embedding
        self.speaker_embeddings: dict[str, np.ndarray] = {}  # "Speaker 1" -> avg embedding
        self.similarity_threshold = similarity_threshold
        self._next_speaker_id = 1

    def identify_speaker(self, embedding: np.ndarray) -> str:
        """Identify speaker from embedding. Returns 'Ham' or 'Speaker N'."""
        import speaker_verify

        # Check if it's the owner (Ham)
        if self.owner_embedding is not None:
            score = speaker_verify.compare(embedding, self.owner_embedding)
            if score >= speaker_verify.DEFAULT_THRESHOLD:
                return "Ham"

        # Check against known speakers
        best_match = None
        best_score = 0.0
        for name, spk_emb in self.speaker_embeddings.items():
            score = speaker_verify.compare(embedding, spk_emb)
            if score > best_score:
                best_score = score
                best_match = name

        if best_match and best_score >= self.similarity_threshold:
            # Update running average for this speaker
            old = self.speaker_embeddings[best_match]
            updated = (old + embedding) / 2
            self.speaker_embeddings[best_match] = updated / np.linalg.norm(updated)
            return best_match

        # New speaker
        name = f"Speaker {self._next_speaker_id}"
        self._next_speaker_id += 1
        norm_emb = embedding / np.linalg.norm(embedding)
        self.speaker_embeddings[name] = norm_emb
        print(f"[Meeting] New speaker detected: {name}")
        return name

    def add_entry(self, speaker: str, text: str):
        """Add a transcript entry."""
        entry = {
            "speaker": speaker,
            "text": text,
            "time": time.strftime("%H:%M:%S"),
        }
        self.transcript.append(entry)
        return entry

    def get_transcript_text(self, last_n: int = 0) -> str:
        """Format transcript as plain text for LLM context."""
        entries = self.transcript[-last_n:] if last_n else self.transcript
        lines = []
        for e in entries:
            lines.append(f"[{e['time']}] {e['speaker']}: {e['text']}")
        return "\n".join(lines)

    def clear(self):
        self.transcript.clear()
        self.speaker_embeddings.clear()
        self._next_speaker_id = 1


app = FastAPI(title="Kismet Voice Agent")

# --- WebAuthn authentication ---
mount_auth_routes(app)

@app.middleware("http")
async def _auth_middleware(request, call_next):
    # /push and /webhook/* use their own Bearer token auth — skip session cookie check
    if request.url.path == "/push" or request.url.path.startswith("/webhook/"):
        return await call_next(request)
    return await auth_middleware(request, call_next)


@app.on_event("startup")
async def startup_event():
    """Preload all models at server start, not per-connection."""
    global conversation_history
    init_session_db()
    async with _history_lock:
        conversation_history = load_recent_session_messages(MAX_HISTORY_MESSAGES)
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, preload_models)


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up Porcupine and HTTP client resources."""
    global wake_word_model, _delegate_acp_client
    if wake_word_model is not None:
        try:
            wake_word_model.delete()
            print("[WakeWord] Porcupine cleaned up.")
        except Exception:
            pass
        wake_word_model = None
    # Close persistent HTTP client
    await _http_client.aclose()
    print("[HTTP] Client closed.")
    if _delegate_acp_client is not None:
        try:
            await _delegate_acp_client.close()
            print("[ACP] Client closed.")
        except Exception:
            pass
        _delegate_acp_client = None


# Serve React build output (frontend/dist/)
_DIST_DIR = Path(__file__).parent / "frontend" / "dist"
_DIST_INDEX = _DIST_DIR / "index.html"
_FALLBACK_HTML = Path(__file__).parent / "index.html"

from fastapi.responses import FileResponse

# Serve /assets/ files — check dist/assets first, then fall back to dist root
# (onnxruntime-web requests .wasm/.mjs files relative to the JS bundle in /assets/)
@app.get("/assets/{filename:path}")
async def serve_assets(filename: str):
    # Try dist/assets first
    fpath = _DIST_DIR / "assets" / filename
    if fpath.exists() and fpath.is_file():
        return FileResponse(str(fpath))
    # Fall back to dist root (for wasm/onnx files copied by vite-plugin-static-copy)
    fpath = _DIST_DIR / filename
    if fpath.exists() and fpath.is_file():
        return FileResponse(str(fpath))
    from fastapi.responses import JSONResponse
    return JSONResponse({"error": "not found"}, status_code=404)

@app.get("/")
async def index():
    if _DIST_INDEX.exists():
        return HTMLResponse(_DIST_INDEX.read_text())
    return HTMLResponse(_FALLBACK_HTML.read_text())

CANVAS_PAGE = """<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Kismet Canvas</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #1a1a2e; color: #e0e0e0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif; padding: 24px; line-height: 1.6; min-height: 100vh; }
  h1, h2, h3 { color: #ffffff; margin-bottom: 12px; }
  table { border-collapse: collapse; width: 100%; margin: 16px 0; }
  th, td { border: 1px solid #333; padding: 10px 14px; text-align: left; }
  th { background: #16213e; color: #00d97e; }
  tr:nth-child(even) { background: rgba(255,255,255,0.03); }
  code, pre { background: #0f0f0f; padding: 2px 6px; border-radius: 4px; font-size: 0.9em; }
  pre { padding: 16px; overflow-x: auto; margin: 12px 0; }
  canvas { max-width: 100%; }
  #status { position: fixed; top: 12px; right: 16px; font-size: 0.75rem; color: #666; }
  #status.connected { color: #00d97e; }
  #content { max-width: 900px; margin: 0 auto; }
  .canvas-block { margin-bottom: 24px; animation: fadeIn 0.3s ease; }
  @keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
  .empty { display: flex; align-items: center; justify-content: center; height: 80vh; color: #555; font-size: 1.1rem; }
</style>
</head><body>
<div id="status">disconnected</div>
<div id="content">
  <div class="empty" id="placeholder">Waiting for canvas content...</div>
</div>
<script>
const content = document.getElementById('content');
const status = document.getElementById('status');
const placeholder = document.getElementById('placeholder');
let ws;

function connect() {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(`${proto}//${location.host}/ws/canvas`);
  ws.onopen = () => { status.textContent = 'connected'; status.className = 'connected'; };
  ws.onclose = () => { status.textContent = 'disconnected'; status.className = ''; setTimeout(connect, 2000); };
  ws.onerror = () => ws.close();
  ws.onmessage = (e) => {
    const msg = JSON.parse(e.data);
    if (msg.type === 'canvas_update') {
      if (placeholder) placeholder.remove();
      const block = msg.block;
      const div = document.createElement('div');
      div.className = 'canvas-block';
      if (block.type === 'html') {
        div.innerHTML = (block.title ? '<h2>' + block.title + '</h2>' : '') + block.content;
        // Execute any script tags in the HTML content
        div.querySelectorAll('script').forEach(old => {
          const s = document.createElement('script');
          s.textContent = old.textContent;
          old.replaceWith(s);
        });
      } else {
        div.innerHTML = (block.title ? '<h2>' + block.title + '</h2>' : '') + '<pre>' + block.content.replace(/</g,'&lt;') + '</pre>';
      }
      content.innerHTML = '';
      content.appendChild(div);
    }
  };
}
connect();
</script>
</body></html>"""


@app.post("/push")
async def push_endpoint(request: Request):
    """
    Allow sub-agents (e.g. Opus) to POST results back to the active voice UI.
    The result is spoken aloud and shown in the conversation as an assistant message.

    Auth: Bearer token (PUSH_SECRET env var).
    Body: { "text": "...", "skip_tts": false }

    Sub-agent usage:
        curl -k -X POST {PUSH_URL} \\
          -H "Authorization: Bearer <PUSH_SECRET>" \\
          -H "Content-Type: application/json" \\
          -d '{{"text": "Your answer here"}}'
    """
    from fastapi.responses import JSONResponse
    auth_header = request.headers.get("Authorization", "")
    token = auth_header.removeprefix("Bearer ").strip()
    if not PUSH_SECRET or token != PUSH_SECRET:
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)
    text = (body.get("text") or "").strip()
    if not text:
        return JSONResponse({"error": "text is required"}, status_code=400)
    skip_tts = bool(body.get("skip_tts", False))
    await _push_queue.put({"text": text, "skip_tts": skip_tts})
    print(f"[Push] Queued ({len(text)} chars): \"{text[:80]}...\"" if len(text) > 80 else f"[Push] Queued: \"{text}\"")
    return JSONResponse({"ok": True, "queued": True})


@app.post("/webhook/cron-result")
async def cron_result_webhook(request: Request):
    """
    Receive a cron finished-run event and speak the result.

    External systems POST here when a scheduled job completes.
    The response should be in plain spoken English (no markdown, no bullet lists)
    since the output will be read aloud.

    Auth: Bearer token (same as PUSH_SECRET).
    Body: Cron finished event JSON, e.g.:
        {
            "jobId": "...",
            "action": "finished",
            "status": "ok",
            "summary": "The agent's spoken response here",
            ...
        }
    """
    from fastapi.responses import JSONResponse
    auth_header = request.headers.get("Authorization", "")
    token = auth_header.removeprefix("Bearer ").strip()
    if not PUSH_SECRET or token != PUSH_SECRET:
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)

    job_id = body.get("jobId", "unknown")
    status = body.get("status", "unknown")
    summary = (body.get("summary") or "").strip()

    print(f"[CronWebhook] Received finished event: jobId={job_id} status={status} summary_len={len(summary)}")

    if status != "ok":
        error = body.get("error", "unknown error")
        print(f"[CronWebhook] Job failed: {error}")
        await _push_queue.put({"text": f"The background task didn't complete. {error}", "skip_tts": False})
        return JSONResponse({"ok": True, "note": "error forwarded to push queue"})

    if not summary:
        print(f"[CronWebhook] No summary in finished event, skipping")
        return JSONResponse({"ok": True, "note": "no summary, skipped"})

    await _push_queue.put({"text": summary, "skip_tts": False})
    print(f"[CronWebhook] Queued summary for TTS ({len(summary)} chars)")
    return JSONResponse({"ok": True, "queued": True})


@app.get("/canvas")
async def canvas_page():
    return HTMLResponse(CANVAS_PAGE)


@app.websocket("/ws/canvas")
async def canvas_ws(ws: WebSocket):
    if not is_ws_authenticated(ws):
        await ws.close(code=4401, reason="Unauthorized")
        return
    await ws.accept()
    _canvas_clients.add(ws)
    print(f"[Canvas] Display client connected ({len(_canvas_clients)} total)")
    try:
        while True:
            await ws.receive_text()  # keep alive, ignore messages
    except Exception:
        pass
    finally:
        _canvas_clients.discard(ws)
        print(f"[Canvas] Display client disconnected ({len(_canvas_clients)} total)")


@app.get("/{filename:path}")
async def serve_static(filename: str):
    """Serve static files from frontend/dist/ (e.g. .onnx, .wasm, .svg)"""
    if filename:
        # Try exact path first (e.g. /assets/foo.js → dist/assets/foo.js)
        fpath = _DIST_DIR / filename
        if fpath.exists() and fpath.is_file():
            return FileResponse(str(fpath))
        # Fallback: strip /assets/ prefix and check dist root
        # (onnxruntime-web requests wasm files relative to the JS bundle in /assets/)
        if filename.startswith("assets/"):
            fpath = _DIST_DIR / filename.removeprefix("assets/")
            if fpath.exists() and fpath.is_file():
                return FileResponse(str(fpath))
    from fastapi.responses import JSONResponse
    return JSONResponse({"error": "not found"}, status_code=404)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    if not is_ws_authenticated(ws):
        await ws.close(code=4401, reason="Unauthorized")
        return
    await ws.accept()
    print("[WS] Client connected")

    loop = asyncio.get_event_loop()

    # Models are preloaded at startup — no per-connection loading needed
    
    # Send ready with wake word config
    await ws.send_json({
        "type": "ready",
        "wake_word_enabled": WAKE_WORD_ENABLED,
        "wake_word": WAKE_WORD_KEYWORD if WAKE_WORD_ENABLED else None,
        "idle_timeout": IDLE_TIMEOUT_SEC,
        "noise_suppression": False,
        "smart_turn": SMART_TURN_ENABLED,
    })

    # Client state: sleeping (wake word mode) or awake (VAD mode)
    client_state = "sleeping" if WAKE_WORD_ENABLED else "awake"
    last_activity = time.time()
    # Reset Porcupine buffer for this connection
    global _porcupine_buffer
    _porcupine_buffer = np.array([], dtype=np.int16)
    
    # Cancellation event for current processing
    cancel_event = asyncio.Event()
    processing_task = None
    bg_manager = BackgroundTaskManager(ws)
    idle_check_task = None
    
    async def check_idle():
        """Background task to check for idle timeout and return to sleep."""
        nonlocal client_state, last_activity
        while True:
            await asyncio.sleep(5)  # Check every 5 seconds
            if client_state == "awake" and not processing_task:
                elapsed = time.time() - last_activity
                if elapsed > IDLE_TIMEOUT_SEC:
                    client_state = "sleeping"
                    print(f"[State] Idle timeout ({IDLE_TIMEOUT_SEC}s), going to sleep")
                    try:
                        await ws.send_json({"type": "sleep"})
                    except:
                        break

    # Meeting companion state
    meeting_session = None  # None = normal mode, MeetingSession = meeting mode

    # Enrollment session state
    enrollment_samples = []
    enrolling = False

    # Canvas state (per-connection)
    canvas_enabled = False

    # Noise suppression state (per-connection)
    noise_suppression_enabled = False

    # TTS skip state (per-connection) — when voice is disabled on client
    skip_tts = False

    async def process_audio(audio_bytes: bytes):
        nonlocal cancel_event, last_activity, client_state, skip_tts
        cancel_event.clear()
        last_activity = time.time()

        # 0. Speaker verification gate
        if _should_verify():
            try:
                import speaker_verify
                t_sv = time.time()
                enrolled = speaker_enrolled_embedding if speaker_enrolled_embedding is not None else _load_enrolled_embedding()
                if enrolled is not None:
                    embedding = await loop.run_in_executor(None, speaker_verify.extract_embedding, audio_bytes)
                    score = speaker_verify.compare(embedding, enrolled)
                    sv_time = time.time() - t_sv
                    if score < speaker_verify.DEFAULT_THRESHOLD:
                        print(f"[Speaker] Rejected: score={score:.3f} ({sv_time:.2f}s)")
                        await ws.send_json({"type": "rejected", "score": round(float(score), 3), "time": round(sv_time, 2)})
                        # Stay awake so user can retry without wake word
                        # Reset idle timer instead of going back to sleep
                        last_activity = time.time()
                        return
                    print(f"[Speaker] Verified: score={score:.3f} ({sv_time:.2f}s)")
                    # Pass score along for UI display
                    await ws.send_json({"type": "verified", "score": round(score, 3), "time": round(sv_time, 2)})
            except Exception as e:
                print(f"[Speaker] Verification error: {e}")
                await ws.send_json({"type": "error", "text": "Speaker verification failed — proceeding without it"})

        # 1. STT
        await ws.send_json({"type": "status", "text": "Transcribing..."})
        t0 = time.time()
        try:
            user_text = await loop.run_in_executor(None, lambda: transcribe(audio_bytes, apply_denoise=noise_suppression_enabled))
        except Exception as e:
            print(f"[STT] Error: {e}")
            await ws.send_json({"type": "error", "text": "Couldn't understand that, try again"})
            return
        stt_time = time.time() - t0

        if cancel_event.is_set():
            return

        if not user_text:
            await ws.send_json({"type": "error", "text": "Couldn't understand that, try again"})
            return

        await ws.send_json({"type": "transcript", "role": "user", "text": user_text, "time": round(stt_time, 2)})

        if cancel_event.is_set():
            return

        # Voice commands for meeting mode
        text_lower = user_text.strip().lower()
        if any(phrase in text_lower for phrase in ["start meeting mode", "activate meeting mode", "turn on meeting mode", "enable meeting mode"]):
            nonlocal meeting_session
            enrolled = speaker_enrolled_embedding if speaker_enrolled_embedding is not None else _load_enrolled_embedding()
            meeting_session = MeetingSession(owner_embedding=enrolled)
            print("[Meeting] Session started via voice command")
            await ws.send_json({"type": "meeting_started"})
            reply = "Meeting mode activated. I'll transcribe in the background. Hold the command button when you need me."
            await ws.send_json({"type": "stream_start"})
            await ws.send_json({"type": "token", "text": reply})
            try:
                tts_text = strip_markdown_for_tts(reply)
                audio_out, _ = await loop.run_in_executor(None, synthesize, tts_text)
                await ws.send_json({"type": "audio_chunk", "data": base64.b64encode(audio_out).decode(), "sentence": reply, "index": 0})
            except Exception as e:
                print(f"[TTS] Error: {e}")
            await ws.send_json({"type": "stream_end"})
            return
        elif any(phrase in text_lower for phrase in ["stop meeting mode", "deactivate meeting mode", "turn off meeting mode", "disable meeting mode", "end meeting mode", "end meeting"]):
            if meeting_session:
                transcript = meeting_session.get_transcript_text()
                entry_count = len(meeting_session.transcript)
                speakers = list(meeting_session.speaker_embeddings.keys())
                if meeting_session.owner_embedding is not None:
                    speakers = ["Ham"] + speakers
                meeting_session = None
                print(f"[Meeting] Session ended via voice command ({entry_count} entries)")
                await ws.send_json({
                    "type": "meeting_stopped",
                    "transcript": transcript,
                    "entries": entry_count,
                    "speakers": speakers,
                })
                reply = f"Meeting mode ended. I captured {entry_count} entries."
            else:
                reply = "Meeting mode isn't active right now."
            await ws.send_json({"type": "stream_start"})
            await ws.send_json({"type": "token", "text": reply})
            try:
                tts_text = strip_markdown_for_tts(reply)
                audio_out, _ = await loop.run_in_executor(None, synthesize, tts_text)
                await ws.send_json({"type": "audio_chunk", "data": base64.b64encode(audio_out).decode(), "sentence": reply, "index": 0})
            except Exception as e:
                print(f"[TTS] Error: {e}")
            await ws.send_json({"type": "stream_end"})
            return

        # 2. LLM streaming + TTS per sentence
        await ws.send_json({"type": "status", "text": "Thinking..."})
        await ws.send_json({"type": "stream_start"})

        llm_start = time.time()
        first_sentence_time = None
        sentence_count = 0
        total_tts_time = 0
        llm_error = False
        spoken_structures: set[str] = set()

        # If SYSTEM_PROMPT was overridden via env (already contains canvas instructions), skip appending CANVAS_INSTRUCTION
        if canvas_enabled and not os.getenv("SYSTEM_PROMPT"):
            effective_prompt = SYSTEM_PROMPT + CANVAS_INSTRUCTION
        else:
            effective_prompt = SYSTEM_PROMPT
        in_canvas_block = False    # Track if we're inside a <canvas> block
        canvas_token_buf = ""      # Buffer tokens while inside canvas block
        in_thinking_block = False  # Track if we're inside a <thinking> block
        thinking_token_buf = ""    # Buffer tokens while inside thinking block

        async for event_type, data in chat_stream(user_text, cancel_event, effective_prompt, bg_manager=bg_manager):
            if cancel_event.is_set() and event_type not in ("cancelled", "done"):
                continue

            if event_type == "working":
                await ws.send_json({"type": "working"})

            elif event_type == "token":
                # Always suppress <thinking> blocks — never spoken or displayed
                thinking_token_buf += data
                if not in_thinking_block and '<thinking>' in thinking_token_buf:
                    in_thinking_block = True
                if in_thinking_block and '</thinking>' in thinking_token_buf:
                    in_thinking_block = False
                    thinking_token_buf = ""
                    continue
                if in_thinking_block:
                    continue
                thinking_token_buf = ""

                if canvas_enabled:
                    canvas_token_buf += data
                    # Detect entering canvas block
                    if not in_canvas_block and '<canvas' in canvas_token_buf:
                        in_canvas_block = True
                    # Detect leaving canvas block
                    if in_canvas_block and '</canvas>' in canvas_token_buf:
                        in_canvas_block = False
                        canvas_token_buf = ""
                        continue
                    # Suppress tokens while inside canvas block
                    if in_canvas_block:
                        continue
                    # Not in canvas block — flush and send
                    canvas_token_buf = ""
                await ws.send_json({"type": "token", "text": data})

            elif event_type == "sentence":
                if first_sentence_time is None:
                    first_sentence_time = time.time() - llm_start

                # Check cancellation before TTS
                if cancel_event.is_set():
                    continue

                # Skip TTS for any sentence with thinking or canvas markup
                if '<thinking>' in data or '</thinking>' in data or in_thinking_block:
                    continue
                if canvas_enabled and ('<canvas' in data or '</canvas>' in data or in_canvas_block):
                    continue

                if skip_tts:
                    sentence_count += 1
                    continue

                await ws.send_json({"type": "status", "text": "Speaking..."})
                try:
                    tts_start = time.time()
                    tts_text = strip_markdown_for_tts(data, spoken_structures)
                    if not tts_text:
                        continue
                    audio_out, sr = await loop.run_in_executor(None, synthesize, tts_text)
                    tts_time = time.time() - tts_start
                    total_tts_time += tts_time

                    # Check cancellation after TTS
                    if cancel_event.is_set():
                        continue

                    audio_b64 = base64.b64encode(audio_out).decode()
                    await ws.send_json({
                        "type": "audio_chunk",
                        "data": audio_b64,
                        "sentence": data,
                        "index": sentence_count,
                    })
                    sentence_count += 1
                    print(f"[TTS] Sentence {sentence_count}: \"{data[:40]}...\" ({tts_time:.2f}s)" if len(data) > 40 else f"[TTS] Sentence {sentence_count}: \"{data}\" ({tts_time:.2f}s)")
                except Exception as e:
                    print(f"[TTS] Error: {e}")
                    await ws.send_json({"type": "error", "text": "Audio generation failed — see text response above"})

            elif event_type == "cancelled":
                # Check if this was an HTTP error (no text generated)
                if not data:
                    llm_error = True
                    await ws.send_json({"type": "error", "text": "Something went wrong, try again"})
                print(f"[WS] Response cancelled")
                await ws.send_json({"type": "cancelled"})
                return

            elif event_type == "done":
                # Strip thinking blocks from final text
                if '<thinking>' in data:
                    data = THINKING_BLOCK_RE.sub("", data).strip()
                    data = re.sub(r'\n{3,}', '\n\n', data)
                    async with _history_lock:
                        if conversation_history and conversation_history[-1]["role"] == "assistant":
                            conversation_history[-1]["content"] = data

                # Extract and push canvas blocks if enabled
                if canvas_enabled and '<canvas' in data:
                    cleaned_text, canvas_blocks = extract_canvas_blocks(data)
                    # Update display log with cleaned text (no canvas markup)
                    async with _history_lock:
                        if conversation_history and conversation_history[-1]["role"] == "assistant":
                            conversation_history[-1]["content"] = cleaned_text
                    if canvas_blocks:
                        asyncio.create_task(push_canvas(canvas_blocks, loop))
                        await ws.send_json({"type": "canvas_pushed", "count": len(canvas_blocks)})
                    data = cleaned_text

                llm_time = time.time() - llm_start
                last_activity = time.time()
                await ws.send_json({
                    "type": "stream_end",
                    "text": data,
                    "times": {
                        "stt": round(stt_time, 2),
                        "llm": round(llm_time, 2),
                        "tts": round(total_tts_time, 2),
                        "first_sentence": round(first_sentence_time, 2) if first_sentence_time else None,
                    },
                    "sentences": sentence_count,
                })

    async def process_text(user_text: str):
        """Process a text message — skip STT, go straight to LLM + TTS (unless skip_tts)."""
        nonlocal cancel_event, last_activity, skip_tts
        cancel_event.clear()
        last_activity = time.time()

        await ws.send_json({"type": "transcript", "role": "user", "text": user_text})

        if cancel_event.is_set():
            return

        # LLM streaming + TTS per sentence (same as process_audio)
        await ws.send_json({"type": "status", "text": "Thinking..."})
        await ws.send_json({"type": "stream_start"})

        llm_start = time.time()
        first_sentence_time = None
        sentence_count = 0
        total_tts_time = 0
        spoken_structures: set[str] = set()

        if canvas_enabled and not os.getenv("SYSTEM_PROMPT"):
            effective_prompt = SYSTEM_PROMPT + CANVAS_INSTRUCTION
        else:
            effective_prompt = SYSTEM_PROMPT
        in_canvas_block = False
        canvas_token_buf = ""
        in_thinking_block = False
        thinking_token_buf = ""

        async for event_type, data in chat_stream(user_text, cancel_event, effective_prompt, bg_manager=bg_manager):
            if cancel_event.is_set() and event_type not in ("cancelled", "done"):
                continue

            if event_type == "working":
                await ws.send_json({"type": "working"})

            elif event_type == "token":
                # Always suppress <thinking> blocks — never spoken or displayed
                thinking_token_buf += data
                if not in_thinking_block and '<thinking>' in thinking_token_buf:
                    in_thinking_block = True
                if in_thinking_block and '</thinking>' in thinking_token_buf:
                    in_thinking_block = False
                    thinking_token_buf = ""
                    continue
                if in_thinking_block:
                    continue
                thinking_token_buf = ""

                if canvas_enabled:
                    canvas_token_buf += data
                    if not in_canvas_block and '<canvas' in canvas_token_buf:
                        in_canvas_block = True
                    if in_canvas_block and '</canvas>' in canvas_token_buf:
                        in_canvas_block = False
                        canvas_token_buf = ""
                        continue
                    if in_canvas_block:
                        continue
                    canvas_token_buf = ""
                await ws.send_json({"type": "token", "text": data})

            elif event_type == "sentence":
                if first_sentence_time is None:
                    first_sentence_time = time.time() - llm_start
                if cancel_event.is_set():
                    continue
                if '<thinking>' in data or '</thinking>' in data or in_thinking_block:
                    continue
                if canvas_enabled and ('<canvas' in data or '</canvas>' in data or in_canvas_block):
                    continue
                if skip_tts:
                    sentence_count += 1
                    continue
                await ws.send_json({"type": "status", "text": "Speaking..."})
                try:
                    tts_start = time.time()
                    tts_text = strip_markdown_for_tts(data, spoken_structures)
                    if not tts_text:
                        continue
                    audio_out, sr = await loop.run_in_executor(None, synthesize, tts_text)
                    tts_time = time.time() - tts_start
                    total_tts_time += tts_time
                    if cancel_event.is_set():
                        continue
                    audio_b64 = base64.b64encode(audio_out).decode()
                    await ws.send_json({
                        "type": "audio_chunk",
                        "data": audio_b64,
                        "sentence": data,
                        "index": sentence_count,
                    })
                    sentence_count += 1
                except Exception as e:
                    print(f"[TTS] Error: {e}")
                    await ws.send_json({"type": "error", "text": "Audio generation failed — see text response above"})

            elif event_type == "cancelled":
                if not data:
                    await ws.send_json({"type": "error", "text": "Something went wrong, try again"})
                print(f"[WS] Response cancelled")
                await ws.send_json({"type": "cancelled"})
                return

            elif event_type == "done":
                # Strip thinking blocks from final text
                if '<thinking>' in data:
                    data = THINKING_BLOCK_RE.sub("", data).strip()
                    data = re.sub(r'\n{3,}', '\n\n', data)
                    async with _history_lock:
                        if conversation_history and conversation_history[-1]["role"] == "assistant":
                            conversation_history[-1]["content"] = data

                if canvas_enabled and '<canvas' in data:
                    cleaned_text, canvas_blocks = extract_canvas_blocks(data)
                    # Update display log with cleaned text (no canvas markup)
                    async with _history_lock:
                        if conversation_history and conversation_history[-1]["role"] == "assistant":
                            conversation_history[-1]["content"] = cleaned_text
                    if canvas_blocks:
                        asyncio.create_task(push_canvas(canvas_blocks, loop))
                        await ws.send_json({"type": "canvas_pushed", "count": len(canvas_blocks)})
                    data = cleaned_text

                llm_time = time.time() - llm_start
                last_activity = time.time()
                await ws.send_json({
                    "type": "stream_end",
                    "text": data,
                    "times": {
                        "stt": 0,
                        "llm": round(llm_time, 2),
                        "tts": round(total_tts_time, 2),
                        "first_sentence": round(first_sentence_time, 2) if first_sentence_time else None,
                    },
                    "sentences": sentence_count,
                })

    async def process_meeting_audio(audio_bytes: bytes):
        """Process audio in meeting mode: transcribe + identify speaker, no LLM."""
        nonlocal meeting_session, last_activity
        last_activity = time.time()

        if meeting_session is None:
            return

        # 1. Extract speaker embedding
        speaker_name = "Unknown"
        try:
            import speaker_verify
            embedding = await loop.run_in_executor(None, speaker_verify.extract_embedding, audio_bytes)
            speaker_name = meeting_session.identify_speaker(embedding)
        except Exception as e:
            print(f"[Meeting] Speaker ID error: {e}")

        # 2. Transcribe
        try:
            text = await loop.run_in_executor(None, lambda: transcribe(audio_bytes, apply_denoise=noise_suppression_enabled))
        except Exception as e:
            print(f"[Meeting] STT error: {e}")
            return

        if not text:
            return

        # 3. Add to transcript and send to client
        entry = meeting_session.add_entry(speaker_name, text)
        print(f"[Meeting] [{entry['time']}] {speaker_name}: {text}")
        await ws.send_json({
            "type": "meeting_transcript",
            "speaker": speaker_name,
            "text": text,
            "time": entry["time"],
            "is_owner": speaker_name == "Ham",
        })

    async def process_meeting_command(audio_bytes: bytes):
        """Process a direct command from Ham during meeting mode."""
        nonlocal cancel_event, meeting_session, last_activity
        cancel_event.clear()
        last_activity = time.time()

        if meeting_session is None:
            return

        # Skip speaker verification for meeting commands — Ham already identified at meeting start
        print(f"[Meeting] Processing command ({len(audio_bytes)} bytes)")

        # 2. Transcribe the command
        await ws.send_json({"type": "status", "text": "Transcribing command..."})
        try:
            user_text = await loop.run_in_executor(None, lambda: transcribe(audio_bytes, apply_denoise=noise_suppression_enabled))
        except Exception as e:
            await ws.send_json({"type": "error", "text": "Couldn't understand that"})
            return

        if not user_text:
            return

        await ws.send_json({"type": "transcript", "role": "user", "text": user_text})
        async with _history_lock:
            conversation_history.append({"role": "user", "content": user_text})
            persist_session_message("user", user_text)
            if len(conversation_history) > MAX_HISTORY_MESSAGES:
                conversation_history[:] = conversation_history[-MAX_HISTORY_MESSAGES:]

        # 3. Send to LLM with transcript context
        transcript_context = meeting_session.get_transcript_text(last_n=50)
        context_msg = f"=== MEETING TRANSCRIPT (last 50 entries) ===\n{transcript_context}\n=== END TRANSCRIPT ===\n\nHam's command: {user_text}"

        await ws.send_json({"type": "status", "text": "Thinking..."})
        await ws.send_json({"type": "stream_start"})

        llm_start = time.time()
        sentence_count = 0
        total_tts_time = 0
        spoken_structures: set[str] = set()

        # Use meeting system prompt + transcript context
        meeting_messages = [
            {"role": "system", "content": MEETING_SYSTEM_PROMPT},
            {"role": "user", "content": context_msg},
        ]

        headers = {"Content-Type": "application/json"}
        if LLM_API_KEY:
            headers["Authorization"] = f"Bearer {LLM_API_KEY}"
        payload = {
            "model": LLM_MODEL,
            "messages": meeting_messages,
            "stream": True,
        }

        full_text = ""
        buffer = ""

        async with _http_client.stream("POST", LLM_URL, json=payload, headers=headers, timeout=httpx.Timeout(10.0, read=60.0)) as resp:
            if resp.status_code != 200:
                await ws.send_json({"type": "error", "text": "LLM request failed"})
                return

            async for line in resp.aiter_lines():
                if cancel_event.is_set():
                    break
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    delta = chunk["choices"][0].get("delta", {})
                    token = delta.get("content", "")
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

                if not token:
                    continue

                full_text += token
                buffer += token
                await ws.send_json({"type": "token", "text": token})

                # Check for sentence boundaries
                while SENTENCE_END_RE.search(buffer):
                    match = SENTENCE_END_RE.search(buffer)
                    sentence = buffer[:match.end()].strip()
                    buffer = buffer[match.end():]

                    if sentence and not cancel_event.is_set():
                        try:
                            tts_start = time.time()
                            tts_text = strip_markdown_for_tts(sentence, spoken_structures)
                            if not tts_text:
                                continue
                            audio_out, sr = await loop.run_in_executor(None, synthesize, tts_text)
                            tts_time = time.time() - tts_start
                            total_tts_time += tts_time
                            audio_b64 = base64.b64encode(audio_out).decode()
                            await ws.send_json({"type": "audio_chunk", "data": audio_b64, "sentence": sentence, "index": sentence_count})
                            sentence_count += 1
                        except Exception as e:
                            print(f"[Meeting TTS] Error: {e}")

        # Handle remaining buffer
        if buffer.strip() and not cancel_event.is_set():
            try:
                tts_start = time.time()
                tts_text = strip_markdown_for_tts(buffer.strip(), spoken_structures)
                if tts_text:
                    audio_out, sr = await loop.run_in_executor(None, synthesize, tts_text)
                    tts_time = time.time() - tts_start
                    total_tts_time += tts_time
                    audio_b64 = base64.b64encode(audio_out).decode()
                    await ws.send_json({"type": "audio_chunk", "data": audio_b64, "sentence": buffer.strip(), "index": sentence_count})
                    sentence_count += 1
            except Exception:
                pass

        # Add response to meeting transcript + display log
        if full_text:
            meeting_session.add_entry("Kismet", full_text)
            async with _history_lock:
                conversation_history.append({"role": "assistant", "content": full_text})
                persist_session_message("assistant", full_text)
                if len(conversation_history) > MAX_HISTORY_MESSAGES:
                    conversation_history[:] = conversation_history[-MAX_HISTORY_MESSAGES:]

        await ws.send_json({
            "type": "stream_end",
            "text": full_text,
            "times": {"llm": round(time.time() - llm_start, 2), "tts": round(total_tts_time, 2)},
            "sentences": sentence_count,
        })

    async def deliver_result(task_id: str, description: str, result: str):
        """Deliver a completed background delegate result via LLM summarization."""
        nonlocal skip_tts
        if not (result or "").strip():
            await ws.send_json({
                "type": "transcript",
                "role": "system",
                "text": f"Background task '{description[:60]}' completed with no output.",
            })
            return

        print(f"[Delegate] Delivering result for {task_id}: {description[:60]}")
        await ws.send_json({
            "type": "transcript",
            "role": "system",
            "text": f"Background task completed: {description[:80]}",
        })
        await ws.send_json({"type": "stream_start"})

        wrapper_text = (
            f"[Background task completed]\n"
            f"You previously started a background task for the user: '{description}'\n\n"
            f"Here is the result:\n{result}\n\n"
            "Briefly summarize what was accomplished or found. Keep it concise for spoken delivery."
        )

        sentence_count = 0
        total_tts_time = 0.0
        full_text = ""
        llm_start = time.time()
        spoken_structures: set[str] = set()

        # Stream LLM summarization — pass bg_manager=None to prevent recursive delegates
        async for event_type, data in chat_stream(wrapper_text, cancel_event, bg_manager=None):
            if cancel_event.is_set() and event_type not in ("cancelled", "done"):
                break

            if event_type == "token":
                await ws.send_json({"type": "token", "text": data})

            elif event_type == "sentence":
                full_text += data + " "
                if skip_tts:
                    sentence_count += 1
                    continue
                try:
                    tts_start = time.time()
                    tts_text = strip_markdown_for_tts(data, spoken_structures)
                    if not tts_text:
                        continue
                    audio_out, sr = await loop.run_in_executor(None, synthesize, tts_text)
                    tts_time = time.time() - tts_start
                    total_tts_time += tts_time
                    audio_b64 = base64.b64encode(audio_out).decode()
                    await ws.send_json({
                        "type": "audio_chunk",
                        "data": audio_b64,
                        "sentence": data,
                        "index": sentence_count,
                    })
                    sentence_count += 1
                except Exception:
                    pass

            elif event_type == "done":
                full_text = data
                llm_time = time.time() - llm_start
                await ws.send_json({
                    "type": "stream_end",
                    "text": full_text,
                    "times": {
                        "llm": round(llm_time, 2),
                        "tts": round(total_tts_time, 2),
                    },
                    "sentences": sentence_count,
                })
                async with _history_lock:
                    conversation_history.append({"role": "assistant", "content": full_text})
                    persist_session_message("assistant", full_text)
                    if len(conversation_history) > MAX_HISTORY_MESSAGES:
                        conversation_history[:] = conversation_history[-MAX_HISTORY_MESSAGES:]
                print(f"[Delegate] Result delivery complete for {task_id}")
                break

            elif event_type == "cancelled":
                break

    async def deliver_delegate_results():
        """Background task: polls for completed delegate results and delivers them when idle."""
        while True:
            try:
                await asyncio.sleep(1.5)
                pending = bg_manager.get_pending_result()
                if pending is None:
                    continue
                task_id, description, result = pending
                if processing_task and not processing_task.done():
                    try:
                        await asyncio.wait_for(asyncio.shield(processing_task), timeout=60.0)
                    except (asyncio.TimeoutError, asyncio.CancelledError):
                        pass
                await deliver_result(task_id, description, result)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[Delegate] Error in deliver_delegate_results: {e}")

    async def listen_for_pushes():
        """
        Background task: picks up messages from POST /push and delivers them to
        this WebSocket client as assistant responses (TTS + transcript).
        Waits for any in-flight processing to complete before speaking.
        """
        nonlocal skip_tts
        while True:
            try:
                item = await _push_queue.get()
                # Wait for any current voice/text processing to finish (up to 60s)
                if processing_task and not processing_task.done():
                    try:
                        await asyncio.wait_for(asyncio.shield(processing_task), timeout=60.0)
                    except (asyncio.TimeoutError, asyncio.CancelledError):
                        pass
                pushed_text = item["text"]
                old_skip_tts = skip_tts
                skip_tts = item.get("skip_tts", False)
                print(f"[Push] Delivering to client: \"{pushed_text[:80]}\"")
                await process_text(pushed_text)
                skip_tts = old_skip_tts
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[Push] Error delivering message: {e}")

    # Start idle check task if wake word is enabled
    if WAKE_WORD_ENABLED:
        idle_check_task = asyncio.create_task(check_idle())
    push_listener_task = asyncio.create_task(listen_for_pushes())
    delegate_delivery_task = asyncio.create_task(deliver_delegate_results())

    try:
        while True:
            msg = await ws.receive_json()

            if msg["type"] == "audio":
                # Full audio utterance (from VAD or manual recording)
                audio_bytes = base64.b64decode(msg["data"])

                # SmartTurn: check if user is done speaking before processing
                if SMART_TURN_ENABLED and not msg.get("force", False):
                    try:
                        is_complete, prob = await loop.run_in_executor(
                            None, predict_turn_complete, audio_bytes
                        )
                        print(f"[SmartTurn] prob={prob:.3f} complete={is_complete}")
                        if not is_complete:
                            # User likely mid-thought — tell browser to keep listening
                            await ws.send_json({
                                "type": "keep_listening",
                                "probability": round(prob, 3),
                            })
                            continue
                    except Exception as e:
                        print(f"[SmartTurn] Error: {e}, proceeding anyway")

                # Cancel any existing processing
                if processing_task and not processing_task.done():
                    cancel_event.set()
                    try:
                        await asyncio.wait_for(processing_task, timeout=2.0)
                    except asyncio.TimeoutError:
                        processing_task.cancel()

                processing_task = asyncio.create_task(process_audio(audio_bytes))

            elif msg["type"] == "audio_stream":
                # Streaming audio chunk for wake word detection
                # Works in both sleep mode (normal) and meeting mode
                if meeting_session is None and client_state != "sleeping":
                    continue  # Normal mode: ignore if not sleeping
                
                audio_b64 = msg["data"]
                audio_bytes = base64.b64decode(audio_b64)
                audio_chunk = np.frombuffer(audio_bytes, dtype=np.int16)
                
                # Run wake word detection (on CPU, fast)
                detected, score = await loop.run_in_executor(None, detect_wake_word, audio_chunk)
                
                if detected:
                    last_activity = time.time()
                    if meeting_session is not None:
                        # Meeting mode: wake triggers command recording
                        print(f"[State] Wake word detected in meeting mode")
                        await ws.send_json({"type": "wake", "score": round(float(score), 3), "meeting": True})
                    else:
                        # Normal mode: transition from sleeping to awake
                        client_state = "awake"
                        print(f"[State] Wake word detected, now awake")
                        await ws.send_json({"type": "wake", "score": round(float(score), 3)})

            elif msg["type"] == "cancel":
                print("[WS] Cancel requested")
                cancel_event.set()
                await ws.send_json({"type": "cancelled"})

            elif msg["type"] == "enroll_start":
                enrolling = True
                enrollment_samples.clear()
                await ws.send_json({"type": "enroll_status", "status": "started", "samples": 0})

            elif msg["type"] == "enroll_sample":
                if enrolling:
                    sample_bytes = base64.b64decode(msg["data"])
                    enrollment_samples.append(sample_bytes)
                    await ws.send_json({"type": "enroll_status", "status": "sample_received", "samples": len(enrollment_samples)})

            elif msg["type"] == "enroll_complete":
                if enrolling and len(enrollment_samples) >= 1:
                    import speaker_verify
                    await ws.send_json({"type": "enroll_status", "status": "processing"})
                    avg_emb = await loop.run_in_executor(None, speaker_verify.enroll, enrollment_samples)
                    # Update cached embedding
                    global speaker_enrolled_embedding
                    speaker_enrolled_embedding = avg_emb
                    enrolling = False
                    enrollment_samples.clear()
                    await ws.send_json({"type": "enroll_status", "status": "complete"})
                else:
                    await ws.send_json({"type": "enroll_status", "status": "error", "message": "Not enough samples"})

            elif msg["type"] == "enroll_cancel":
                enrolling = False
                enrollment_samples.clear()
                await ws.send_json({"type": "enroll_status", "status": "cancelled"})

            elif msg["type"] == "check_enrollment":
                import speaker_verify
                await ws.send_json({"type": "enrollment_info", "enrolled": speaker_verify.has_enrollment(), "verify_enabled": _should_verify()})

            elif msg["type"] == "toggle_verify":
                global SPEAKER_VERIFY
                SPEAKER_VERIFY = "true" if msg.get("enabled") else "false"
                await ws.send_json({"type": "verify_toggled", "enabled": msg.get("enabled", False)})

            elif msg["type"] == "canvas_toggle":
                canvas_enabled = msg.get("enabled", False)
                print(f"[Canvas] {'Enabled' if canvas_enabled else 'Disabled'}")
                await ws.send_json({"type": "canvas_toggled", "enabled": canvas_enabled})

            elif msg["type"] == "noise_suppression_toggle":
                noise_suppression_enabled = msg.get("enabled", False)
                print(f"[Denoise] {'Enabled' if noise_suppression_enabled else 'Disabled'}")
                await ws.send_json({"type": "noise_suppression_toggled", "enabled": noise_suppression_enabled})

            elif msg["type"] == "voice_toggle":
                skip_tts = not msg.get("enabled", True)
                print(f"[TTS] {'Skipping' if skip_tts else 'Enabled'}")

            elif msg["type"] == "meeting_start":
                # Start meeting companion mode
                enrolled = speaker_enrolled_embedding if speaker_enrolled_embedding is not None else _load_enrolled_embedding()
                meeting_session = MeetingSession(owner_embedding=enrolled)
                print("[Meeting] Session started")
                await ws.send_json({"type": "meeting_started"})

            elif msg["type"] == "meeting_stop":
                # Stop meeting mode, return transcript
                if meeting_session:
                    transcript = meeting_session.get_transcript_text()
                    entry_count = len(meeting_session.transcript)
                    speakers = list(meeting_session.speaker_embeddings.keys())
                    if meeting_session.owner_embedding is not None:
                        speakers = ["Ham"] + speakers
                    meeting_session = None
                    print(f"[Meeting] Session ended ({entry_count} entries)")
                    await ws.send_json({
                        "type": "meeting_stopped",
                        "transcript": transcript,
                        "entries": entry_count,
                        "speakers": speakers,
                    })
                else:
                    await ws.send_json({"type": "meeting_stopped", "transcript": "", "entries": 0, "speakers": []})

            elif msg["type"] == "meeting_audio":
                # Passive transcription in meeting mode
                if meeting_session:
                    audio_bytes = base64.b64decode(msg["data"])
                    asyncio.create_task(process_meeting_audio(audio_bytes))

            elif msg["type"] == "meeting_command":
                # Ham's direct command during meeting
                if meeting_session:
                    if processing_task and not processing_task.done():
                        cancel_event.set()
                        try:
                            await asyncio.wait_for(processing_task, timeout=2.0)
                        except asyncio.TimeoutError:
                            processing_task.cancel()
                    audio_bytes = base64.b64decode(msg["data"])
                    processing_task = asyncio.create_task(process_meeting_command(audio_bytes))

            elif msg["type"] == "text_message":
                # Text chat — skip STT, go straight to LLM
                # Per-message skip_tts flag overrides connection state
                if msg.get("skip_tts") is not None:
                    skip_tts = msg["skip_tts"]
                if processing_task and not processing_task.done():
                    cancel_event.set()
                    try:
                        await asyncio.wait_for(processing_task, timeout=2.0)
                    except asyncio.TimeoutError:
                        processing_task.cancel()
                processing_task = asyncio.create_task(process_text(msg["text"]))

            elif msg["type"] == "clear":
                cancel_event.set()
                async with _history_lock:
                    conversation_history.clear()  # Clear conversation history
                clear_session_memory()
                if meeting_session:
                    meeting_session.clear()
                await ws.send_json({"type": "status", "text": "Conversation cleared."})

            elif msg["type"] == "set_state":
                # Client can request state change (for manual wake/sleep toggle)
                new_state = msg.get("state")
                if new_state in ("sleeping", "awake"):
                    client_state = new_state
                    last_activity = time.time()
                    print(f"[State] Client requested: {new_state}")

    except WebSocketDisconnect:
        cancel_event.set()
        if idle_check_task:
            idle_check_task.cancel()
        push_listener_task.cancel()
        delegate_delivery_task.cancel()
        print("[WS] Client disconnected")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import ssl
    cert_dir = Path(__file__).parent
    cert_file = cert_dir / "cert.pem"
    key_file = cert_dir / "key.pem"

    kwargs = {}
    if cert_file.exists() and key_file.exists():
        kwargs["ssl_certfile"] = str(cert_file)
        kwargs["ssl_keyfile"] = str(key_file)
        print(f"[SSL] HTTPS enabled")

    print_config()
    print(f"[LLM] Endpoint: {LLM_URL}")
    print(f"[LLM] Model: {LLM_MODEL}")
    print(f"[LLM] History: {MAX_HISTORY_MESSAGES} messages (sliding window)")
    print(f"[Memory] Forgetful RAG: {'enabled' if FORGETFUL_ENABLED else 'disabled'}" + (f" (top-{FORGETFUL_MAX_MEMORIES})" if FORGETFUL_ENABLED else ""))
    print(f"[TTS] Engine: {TTS_ENGINE} (backend: {TTS_BACKEND})")
    print(f"[Push] Endpoint: POST {PUSH_URL}")
    if WAKE_WORD_ENABLED:
        print(f"[WakeWord] Porcupine keyword: {WAKE_WORD_KEYWORD}, Sensitivity: {WAKE_WORD_SENSITIVITY}")
    uvicorn.run(app, host="0.0.0.0", port=8765, **kwargs)
