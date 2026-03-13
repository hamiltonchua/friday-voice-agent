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

# OpenClaw gateway
OPENCLAW_URL = os.getenv("OPENCLAW_URL", "http://127.0.0.1:18789/v1/chat/completions")

def _read_openclaw_token() -> str:
    """Read token from OpenClaw config file, fallback to env var."""
    token = os.getenv("OPENCLAW_TOKEN")
    if token:
        return token
    try:
        import json
        config_path = Path.home() / ".openclaw" / "openclaw.json"
        with open(config_path) as f:
            config = json.load(f)
        return config["gateway"]["auth"]["token"]
    except Exception:
        return ""

OPENCLAW_TOKEN = _read_openclaw_token()
OPENCLAW_AGENT = os.getenv("OPENCLAW_AGENT", "main")

SPEAKER_VERIFY = os.getenv("SPEAKER_VERIFY", "auto")  # "true", "false", or "auto" (verify if enrolled)

# Push endpoint — allows sub-agents to POST results back to the voice UI
PUSH_SECRET = os.getenv("PUSH_SECRET", OPENCLAW_TOKEN)  # Defaults to OpenClaw token
PUSH_URL = os.getenv("PUSH_URL", "https://prodigy.skunk-shark.ts.net:8765/push")

SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", (
    "You are Kismet, a voice assistant. Your response will be spoken aloud via TTS. "
    "STRICT RULES: No emoji. No emoticons. No markdown. No bullet lists. No code blocks. No asterisks. No special characters. "
    "Keep responses concise and conversational. Just plain spoken English, like you're talking to someone. "
    "Be natural, warm, and to the point. "
    "You have a voice agent canvas at /canvas for visual output. There is also a separate 'prodigy node canvas' "
    "managed by Friday. If the user asks to open or show something on the prodigy canvas, tell them to ask Friday."
))

MEETING_SYSTEM_PROMPT = os.getenv("MEETING_SYSTEM_PROMPT", (
    "You are Kismet, a meeting companion assistant. You have access to the meeting transcript so far. "
    "When the user (Ham) asks you something, answer based on the transcript context. "
    "STRICT RULES: No emoji. No emoticons. No markdown. No bullet lists. No code blocks. No asterisks. "
    "Keep responses concise and conversational. Plain spoken English only."
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
deepfilter_model = None  # DeepFilterNet model + state
conversation_history = []  # Local display log only — OpenClaw session manages LLM context
_model_lock = threading.Lock()  # Prevent concurrent MLX/GPU model loads
_models_ready = False  # True once startup preload completes
_deepfilter_available = True  # Set to False if import fails

# Persistent HTTP client — reuses TCP connections to OpenClaw gateway (avoids
# per-request connection setup overhead of ~50-200ms). Created at module level,
# closed on shutdown via lifespan handler.
_http_client = httpx.AsyncClient(
    limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
    timeout=httpx.Timeout(10.0, read=120.0),
)


def get_whisper():
    global whisper_model
    with _model_lock:
        if whisper_model is None:
            if STT_BACKEND == "mlx-audio":
                from mlx_audio.stt.utils import load_model
                print(f"[STT] Loading MLX model: {MLX_STT_MODEL}...")
                whisper_model = load_model(MLX_STT_MODEL)
                print("[STT] MLX Whisper ready.")
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
    """Lazy-load DeepFilterNet model."""
    global deepfilter_model, _deepfilter_available
    if not _deepfilter_available:
        return None
    with _model_lock:
        if deepfilter_model is None:
            try:
                from df.enhance import init_df
                print("[Denoise] Loading DeepFilterNet model...")
                model, df_state, _ = init_df()
                deepfilter_model = (model, df_state)
                print("[Denoise] DeepFilterNet ready.")
            except Exception as e:
                print(f"[Denoise] WARNING: Failed to load DeepFilterNet: {e}")
                _deepfilter_available = False
                return None
    return deepfilter_model


def denoise(audio_bytes: bytes) -> bytes:
    """Apply DeepFilterNet noise suppression to raw PCM bytes (16-bit 16kHz mono)."""
    df = get_deepfilter()
    if df is None:
        return audio_bytes

    t0 = time.time()
    model, df_state = df

    from df.enhance import enhance
    from scipy.signal import resample_poly

    # PCM 16-bit → float32
    pcm = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    # Resample 16kHz → 48kHz (DeepFilterNet native rate)
    audio_48k = resample_poly(pcm, up=3, down=1).astype(np.float32)

    # DeepFilterNet expects torch tensor [1, samples]
    import torch
    tensor_48k = torch.from_numpy(audio_48k).unsqueeze(0)
    enhanced = enhance(model, df_state, tensor_48k)

    # enhanced is a numpy array or tensor — ensure numpy
    if isinstance(enhanced, torch.Tensor):
        enhanced = enhanced.numpy()
    enhanced = enhanced.squeeze()

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

# Delegate block extraction — Haiku emits this to hand off to Opus
DELEGATE_RE = re.compile(r'<delegate>(.*?)</delegate>', re.DOTALL)

# Thinking block extraction — strip model reasoning from output
THINKING_BLOCK_RE = re.compile(r'<thinking>.*?</thinking>', re.DOTALL)

# OpenClaw silent reply tokens — suppress these from voice output
SILENT_REPLY_TOKEN = "NO_REPLY"
HEARTBEAT_TOKEN = "HEARTBEAT_OK"


def _is_silent_reply(text: str) -> bool:
    """Check if text is an OpenClaw silent reply token (NO_REPLY or HEARTBEAT_OK)."""
    normalized = text.strip().upper()
    return normalized in (SILENT_REPLY_TOKEN, HEARTBEAT_TOKEN)


def _is_silent_reply_prefix(text: str) -> bool:
    """Check if text is a streaming prefix of a silent reply token (e.g. 'NO', 'NO_')."""
    normalized = re.sub(r'[^A-Z_]', '', text.strip().upper())
    if not normalized:
        return False
    return SILENT_REPLY_TOKEN.startswith(normalized) or HEARTBEAT_TOKEN.startswith(normalized)


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


async def run_delegate(task: str):
    """
    Call Opus with the delegated task (non-streaming), then push the result
    to the active voice UI via the push queue.
    """
    print(f"[Delegate] Spawning Opus for: \"{task[:80]}\"")
    full_text = ""
    try:
        response = await _http_client.post(
            OPENCLAW_URL,
            headers={
                "Authorization": f"Bearer {OPENCLAW_TOKEN}",
                "Content-Type": "application/json",
                "x-openclaw-agent-id": OPENCLAW_AGENT,
            },
            json={
                "model": "anthropic/claude-opus-4-6",
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"/think:off {task}"},
                ],
                "user": "voice-delegate",
                "stream": False,
                "thinking": {"type": "disabled"},
            },
            timeout=httpx.Timeout(10.0, read=180.0),
        )
        response.raise_for_status()
        data = response.json()
        full_text = data["choices"][0]["message"]["content"].strip()
        # Suppress silent replies from delegate too
        if _is_silent_reply(full_text):
            print(f"[Delegate] Silent reply from Opus — suppressing")
            return
        print(f"[Delegate] Opus complete ({len(full_text)} chars)")
    except Exception as e:
        print(f"[Delegate] Error calling Opus: {e}")
        full_text = "Sorry, I ran into a problem getting that answer."
    await _push_queue.put({"text": full_text, "skip_tts": False})


async def chat_stream(user_text: str, cancel_event: asyncio.Event, system_prompt: str | None = None) -> AsyncIterator[tuple[str, str]]:
    """
    Stream chat response from OpenClaw.
    Yields tuples of (event_type, data):
      - ("token", token_text) for each token
      - ("sentence", sentence_text) for each complete sentence
      - ("working", "") when tools are likely running (>2s before first token)
      - ("done", full_text) when finished
      - ("cancelled", partial_text) when interrupted
    """
    global conversation_history

    # Prepend /think:off directive to disable extended thinking (reduces latency for voice)
    prefixed_text = f"/think:off {user_text}"

    # Send only the latest message — OpenClaw's session manages conversation
    # context (transcript + LCM compaction) server-side via stable "voice-chat" session key
    messages = [{"role": "system", "content": system_prompt or SYSTEM_PROMPT}, {"role": "user", "content": prefixed_text}]

    # Track locally for UI display
    conversation_history.append({"role": "user", "content": user_text})

    full_text = ""
    buffer = ""
    cancelled = False
    got_first_token = False
    working_emitted = False
    silent_reply = False
    initial_buf = ""  # Buffer initial tokens to detect NO_REPLY before yielding
    initial_phase = True  # True until we confirm this isn't a silent reply
    request_start = time.time()

    try:
        async with _http_client.stream(
            "POST",
            OPENCLAW_URL,
            headers={
                "Authorization": f"Bearer {OPENCLAW_TOKEN}",
                "Content-Type": "application/json",
                "x-openclaw-agent-id": OPENCLAW_AGENT,
            },
            json={
                "model": os.getenv("OPENCLAW_MODEL", "anthropic/claude-haiku-3.5-20241022"),
                "messages": messages,
                "user": "voice-chat",
                "stream": True,
                "thinking": {"type": "disabled"},
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
                        full_text += token

                        # Buffer initial tokens to detect NO_REPLY before sending to TTS
                        if initial_phase:
                            initial_buf += token
                            # Check if still a possible silent reply prefix
                            if _is_silent_reply_prefix(initial_buf):
                                continue  # Keep buffering
                            elif _is_silent_reply(initial_buf):
                                silent_reply = True
                                print(f"[LLM] Silent reply detected: \"{initial_buf.strip()}\" — suppressing")
                                continue
                            else:
                                # Not a silent reply — flush buffered tokens
                                initial_phase = False
                                buffer += initial_buf
                                for ch in initial_buf:
                                    yield ("token", ch)
                        else:
                            buffer += token
                            yield ("token", token)

                        # Check for sentence boundaries (only after initial phase)
                        if not initial_phase:
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

    # Check if the complete response is a silent reply (catches edge cases)
    if not cancelled and _is_silent_reply(full_text):
        silent_reply = True
        print(f"[LLM] Silent reply (complete): \"{full_text.strip()}\" — suppressing")

    if silent_reply:
        # Remove user message from display log — this was a system housekeeping turn
        if conversation_history and conversation_history[-1]["role"] == "user":
            conversation_history.pop()
        yield ("cancelled", "")
    elif cancelled:
        # Remove user message from display log on cancel
        if conversation_history and conversation_history[-1]["role"] == "user":
            conversation_history.pop()
        yield ("cancelled", full_text)
    else:
        # Emit remaining text
        if buffer.strip():
            yield ("sentence", buffer.strip())

        # Track assistant response in display log
        conversation_history.append({"role": "assistant", "content": full_text})
        llm_total_ms = int((time.time() - request_start) * 1000)
        print(f"[LLM] → \"{full_text[:80]}...\" ({llm_total_ms}ms)" if len(full_text) > 80 else f"[LLM] → \"{full_text}\" ({llm_total_ms}ms)")
        yield ("done", full_text)


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
    # /push uses its own Bearer token auth — skip session cookie check
    if request.url.path == "/push":
        return await call_next(request)
    return await auth_middleware(request, call_next)


@app.on_event("startup")
async def startup_event():
    """Preload all models at server start, not per-connection."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, preload_models)


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up Porcupine and HTTP client resources."""
    global wake_word_model
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

    Auth: Bearer token (same as OPENCLAW_TOKEN / PUSH_SECRET).
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
                audio_out, _ = await loop.run_in_executor(None, synthesize, reply)
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
                audio_out, _ = await loop.run_in_executor(None, synthesize, reply)
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

        # If SYSTEM_PROMPT was overridden via env (already contains canvas instructions), skip appending CANVAS_INSTRUCTION
        if canvas_enabled and not os.getenv("SYSTEM_PROMPT"):
            effective_prompt = SYSTEM_PROMPT + CANVAS_INSTRUCTION
        else:
            effective_prompt = SYSTEM_PROMPT
        in_canvas_block = False    # Track if we're inside a <canvas> block
        canvas_token_buf = ""      # Buffer tokens while inside canvas block
        in_delegate_block = False  # Track if we're inside a <delegate> block
        delegate_token_buf = ""    # Buffer tokens while inside delegate block
        in_thinking_block = False  # Track if we're inside a <thinking> block
        thinking_token_buf = ""    # Buffer tokens while inside thinking block

        async for event_type, data in chat_stream(user_text, cancel_event, effective_prompt):
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

                # Always suppress <delegate> blocks — never spoken or displayed
                delegate_token_buf += data
                if not in_delegate_block and '<delegate>' in delegate_token_buf:
                    in_delegate_block = True
                if in_delegate_block and '</delegate>' in delegate_token_buf:
                    in_delegate_block = False
                    delegate_token_buf = ""
                    continue
                if in_delegate_block:
                    continue
                delegate_token_buf = ""

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

                # Skip TTS for any sentence with thinking, delegate, or canvas markup
                if '<thinking>' in data or '</thinking>' in data or in_thinking_block:
                    continue
                if '<delegate>' in data or '</delegate>' in data or in_delegate_block:
                    continue
                if canvas_enabled and ('<canvas' in data or '</canvas>' in data or in_canvas_block):
                    continue

                if skip_tts:
                    sentence_count += 1
                    continue

                await ws.send_json({"type": "status", "text": "Speaking..."})
                try:
                    tts_start = time.time()
                    audio_out, sr = await loop.run_in_executor(None, synthesize, data)
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
                    if conversation_history and conversation_history[-1]["role"] == "assistant":
                        conversation_history[-1]["content"] = data

                # Extract and push canvas blocks if enabled
                if canvas_enabled and '<canvas' in data:
                    cleaned_text, canvas_blocks = extract_canvas_blocks(data)
                    # Update display log with cleaned text (no canvas markup)
                    if conversation_history and conversation_history[-1]["role"] == "assistant":
                        conversation_history[-1]["content"] = cleaned_text
                    if canvas_blocks:
                        asyncio.create_task(push_canvas(canvas_blocks, loop))
                        await ws.send_json({"type": "canvas_pushed", "count": len(canvas_blocks)})
                    data = cleaned_text

                # Detect and spawn delegate tasks
                if '<delegate>' in data:
                    delegate_match = DELEGATE_RE.search(data)
                    if delegate_match:
                        delegate_body = delegate_match.group(1).strip()
                        task_match = re.search(r'TASK:\s*(.+?)(?:\nPUSH_URL:|$)', delegate_body, re.DOTALL)
                        if task_match:
                            task_text = task_match.group(1).strip()
                            print(f"[Delegate] Detected, spawning Opus: \"{task_text[:80]}\"")
                            asyncio.create_task(run_delegate(task_text))
                    # Strip delegate block from display text
                    data = DELEGATE_RE.sub("", data).strip()
                    data = re.sub(r'\n{3,}', '\n\n', data)
                    if conversation_history and conversation_history[-1]["role"] == "assistant":
                        conversation_history[-1]["content"] = data

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

        if canvas_enabled and not os.getenv("SYSTEM_PROMPT"):
            effective_prompt = SYSTEM_PROMPT + CANVAS_INSTRUCTION
        else:
            effective_prompt = SYSTEM_PROMPT
        in_canvas_block = False
        canvas_token_buf = ""
        in_delegate_block = False
        delegate_token_buf = ""
        in_thinking_block = False
        thinking_token_buf = ""

        async for event_type, data in chat_stream(user_text, cancel_event, effective_prompt):
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

                # Always suppress <delegate> blocks
                delegate_token_buf += data
                if not in_delegate_block and '<delegate>' in delegate_token_buf:
                    in_delegate_block = True
                if in_delegate_block and '</delegate>' in delegate_token_buf:
                    in_delegate_block = False
                    delegate_token_buf = ""
                    continue
                if in_delegate_block:
                    continue
                delegate_token_buf = ""

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
                if '<delegate>' in data or '</delegate>' in data or in_delegate_block:
                    continue
                if canvas_enabled and ('<canvas' in data or '</canvas>' in data or in_canvas_block):
                    continue
                if skip_tts:
                    sentence_count += 1
                    continue
                await ws.send_json({"type": "status", "text": "Speaking..."})
                try:
                    tts_start = time.time()
                    audio_out, sr = await loop.run_in_executor(None, synthesize, data)
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
                    if conversation_history and conversation_history[-1]["role"] == "assistant":
                        conversation_history[-1]["content"] = data

                if canvas_enabled and '<canvas' in data:
                    cleaned_text, canvas_blocks = extract_canvas_blocks(data)
                    # Update display log with cleaned text (no canvas markup)
                    if conversation_history and conversation_history[-1]["role"] == "assistant":
                        conversation_history[-1]["content"] = cleaned_text
                    if canvas_blocks:
                        asyncio.create_task(push_canvas(canvas_blocks, loop))
                        await ws.send_json({"type": "canvas_pushed", "count": len(canvas_blocks)})
                    data = cleaned_text

                # Detect and spawn delegate tasks
                if '<delegate>' in data:
                    delegate_match = DELEGATE_RE.search(data)
                    if delegate_match:
                        delegate_body = delegate_match.group(1).strip()
                        task_match = re.search(r'TASK:\s*(.+?)(?:\nPUSH_URL:|$)', delegate_body, re.DOTALL)
                        if task_match:
                            task_text = task_match.group(1).strip()
                            print(f"[Delegate] Detected, spawning Opus: \"{task_text[:80]}\"")
                            asyncio.create_task(run_delegate(task_text))
                    data = DELEGATE_RE.sub("", data).strip()
                    data = re.sub(r'\n{3,}', '\n\n', data)
                    if conversation_history and conversation_history[-1]["role"] == "assistant":
                        conversation_history[-1]["content"] = data

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

        # 3. Send to LLM with transcript context
        transcript_context = meeting_session.get_transcript_text(last_n=50)
        context_msg = f"=== MEETING TRANSCRIPT (last 50 entries) ===\n{transcript_context}\n=== END TRANSCRIPT ===\n\nHam's command: {user_text}"

        await ws.send_json({"type": "status", "text": "Thinking..."})
        await ws.send_json({"type": "stream_start"})

        llm_start = time.time()
        sentence_count = 0
        total_tts_time = 0

        # Use meeting system prompt + transcript context
        meeting_messages = [
            {"role": "system", "content": MEETING_SYSTEM_PROMPT},
            {"role": "user", "content": context_msg},
        ]

        headers = {"Authorization": f"Bearer {OPENCLAW_TOKEN}"} if OPENCLAW_TOKEN else {}
        payload = {
            "model": OPENCLAW_AGENT,
            "messages": meeting_messages,
            "stream": True,
        }

        full_text = ""
        buffer = ""

        async with _http_client.stream("POST", OPENCLAW_URL, json=payload, headers=headers, timeout=httpx.Timeout(10.0, read=60.0)) as resp:
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
                            audio_out, sr = await loop.run_in_executor(None, synthesize, sentence)
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
                audio_out, sr = await loop.run_in_executor(None, synthesize, buffer.strip())
                tts_time = time.time() - tts_start
                total_tts_time += tts_time
                audio_b64 = base64.b64encode(audio_out).decode()
                await ws.send_json({"type": "audio_chunk", "data": audio_b64, "sentence": buffer.strip(), "index": sentence_count})
                sentence_count += 1
            except Exception:
                pass

        # Suppress silent replies from meeting mode
        if _is_silent_reply(full_text):
            print(f"[Meeting] Silent reply — suppressing")
            await ws.send_json({"type": "stream_end", "text": "", "times": {"llm": 0, "tts": 0}, "sentences": 0})
            return

        # Add response to meeting transcript + display log
        if full_text:
            meeting_session.add_entry("Kismet", full_text)
            conversation_history.append({"role": "assistant", "content": full_text})  # display log only

        await ws.send_json({
            "type": "stream_end",
            "text": full_text,
            "times": {"llm": round(time.time() - llm_start, 2), "tts": round(total_tts_time, 2)},
            "sentences": sentence_count,
        })

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

    try:
        while True:
            msg = await ws.receive_json()

            if msg["type"] == "audio":
                # Full audio utterance (from VAD or manual recording)
                # Cancel any existing processing
                if processing_task and not processing_task.done():
                    cancel_event.set()
                    try:
                        await asyncio.wait_for(processing_task, timeout=2.0)
                    except asyncio.TimeoutError:
                        processing_task.cancel()

                audio_bytes = base64.b64decode(msg["data"])
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
                conversation_history.clear()  # Clear display log (OpenClaw session persists server-side)
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
    print(f"[LLM] OpenClaw endpoint: {OPENCLAW_URL}")
    print(f"[TTS] Engine: {TTS_ENGINE} (backend: {TTS_BACKEND})")
    print(f"[Push] Endpoint: POST {PUSH_URL}  (Bearer token = PUSH_SECRET or OPENCLAW_TOKEN)")
    if WAKE_WORD_ENABLED:
        print(f"[WakeWord] Porcupine keyword: {WAKE_WORD_KEYWORD}, Sensitivity: {WAKE_WORD_SENSITIVITY}")
    uvicorn.run(app, host="0.0.0.0", port=8765, **kwargs)
