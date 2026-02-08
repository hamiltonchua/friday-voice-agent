#!/usr/bin/env python3
"""
Kismet Voice Agent - Real-time voice chat with streaming TTS and interruption support.
STT: faster-whisper (GPU)  |  LLM: OpenClaw  |  TTS: Chatterbox Turbo (GPU)

v0.4: Chatterbox TTS integration
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
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cuda")
WHISPER_COMPUTE = os.getenv("WHISPER_COMPUTE", "float16")
TTS_ENGINE = os.getenv("TTS_ENGINE", "chatterbox")  # "chatterbox" or "kokoro"
KOKORO_VOICE = os.getenv("KOKORO_VOICE", "af_heart")
CHATTERBOX_REF = os.getenv("CHATTERBOX_REF", None)  # Optional reference audio for voice cloning
STT_SAMPLE_RATE = 16000

# OpenClaw gateway
OPENCLAW_URL = os.getenv("OPENCLAW_URL", "http://127.0.0.1:18789/v1/chat/completions")
OPENCLAW_TOKEN = os.getenv("OPENCLAW_TOKEN", "92c0ca8eeb7054cd6587b7368e83f25673e41c7b0cf9985b")
OPENCLAW_AGENT = os.getenv("OPENCLAW_AGENT", "main")

SPEAKER_VERIFY = os.getenv("SPEAKER_VERIFY", "auto")  # "true", "false", or "auto" (verify if enrolled)

SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", (
    "You are Kismet, a voice assistant. Your response will be spoken aloud via TTS. "
    "STRICT RULES: No emoji. No emoticons. No markdown. No bullet lists. No code blocks. No asterisks. No special characters. "
    "Keep responses concise and conversational. Just plain spoken English, like you're talking to someone. "
    "Be natural, warm, and to the point."
))

# ---------------------------------------------------------------------------
# Global singletons
# ---------------------------------------------------------------------------
whisper_model = None
speaker_enrolled_embedding = None  # Cached enrollment embedding
tts_model = None
tts_sample_rate = None
conversation_history = []

def get_whisper():
    global whisper_model
    if whisper_model is None:
        from faster_whisper import WhisperModel
        print(f"[STT] Loading {WHISPER_MODEL} on {WHISPER_DEVICE}...")
        whisper_model = WhisperModel(WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE)
        print("[STT] Ready.")
    return whisper_model

def get_tts():
    global tts_model, tts_sample_rate
    if tts_model is None:
        if TTS_ENGINE == "chatterbox":
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


# ---------------------------------------------------------------------------
# Pipeline functions
# ---------------------------------------------------------------------------
def transcribe(audio_bytes: bytes) -> str:
    """Convert raw PCM 16-bit 16kHz mono audio to text."""
    model = get_whisper()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name
        with wave.open(f, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(STT_SAMPLE_RATE)
            wf.writeframes(audio_bytes)

    try:
        segments, info = model.transcribe(tmp_path, language=None, vad_filter=True, beam_size=5)
        text = " ".join(seg.text.strip() for seg in segments).strip()
        print(f"[STT] ({info.language}, {info.duration:.1f}s) → \"{text}\"")
        return text
    finally:
        os.unlink(tmp_path)


# Sentence boundary pattern: ends with .!? followed by space or end
SENTENCE_END_RE = re.compile(r'[.!?](?:\s|$)')


async def chat_stream(user_text: str, cancel_event: asyncio.Event) -> AsyncIterator[tuple[str, str]]:
    """
    Stream chat response from OpenClaw.
    Yields tuples of (event_type, data):
      - ("token", token_text) for each token
      - ("sentence", sentence_text) for each complete sentence
      - ("done", full_text) when finished
      - ("cancelled", partial_text) when interrupted
    """
    global conversation_history

    conversation_history.append({"role": "user", "content": user_text})

    # Keep last 40 messages
    if len(conversation_history) > 40:
        conversation_history = conversation_history[-40:]

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + conversation_history

    full_text = ""
    buffer = ""
    cancelled = False

    try:
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                OPENCLAW_URL,
                headers={
                    "Authorization": f"Bearer {OPENCLAW_TOKEN}",
                    "Content-Type": "application/json",
                    "x-openclaw-agent-id": OPENCLAW_AGENT,
                },
                json={
                    "model": "openclaw",
                    "messages": messages,
                    "user": "voice-chat",
                    "stream": True,
                },
                timeout=120.0,
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    # Check for cancellation
                    if cancel_event.is_set():
                        cancelled = True
                        print(f"[LLM] Cancelled after: \"{full_text[:50]}...\"")
                        break

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
                            full_text += token
                            buffer += token
                            yield ("token", token)

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

    if cancelled:
        # Don't add incomplete response to history, remove user message too
        if conversation_history and conversation_history[-1]["role"] == "user":
            conversation_history.pop()
        yield ("cancelled", full_text)
    else:
        # Emit remaining text
        if buffer.strip():
            yield ("sentence", buffer.strip())

        # Record full response
        conversation_history.append({"role": "assistant", "content": full_text})
        print(f"[LLM] → \"{full_text[:80]}...\"" if len(full_text) > 80 else f"[LLM] → \"{full_text}\"")
        yield ("done", full_text)


def synthesize(text: str) -> tuple[bytes, int]:
    """Convert text to speech, return (WAV bytes, sample_rate)."""
    model, sr = get_tts()

    if TTS_ENGINE == "chatterbox":
        import torch
        # Chatterbox Turbo
        wav_tensor = model.generate(
            text,
            audio_prompt_path=CHATTERBOX_REF,
            exaggeration=0.0,
            cfg_weight=0.0,
        )
        # wav_tensor is (1, samples) float32 tensor
        samples = wav_tensor.squeeze(0).cpu().numpy()
    else:
        # Kokoro
        samples, sr = model.create(text, voice=KOKORO_VOICE, speed=1.0)

    # Convert to WAV bytes
    buf = io.BytesIO()
    with wave.open(buf, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        # Convert float32 to int16
        pcm = (samples * 32767).astype(np.int16)
        wf.writeframes(pcm.tobytes())

    return buf.getvalue(), sr


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="Kismet Voice Agent")

@app.get("/")
async def index():
    html_path = Path(__file__).parent / "index.html"
    return HTMLResponse(html_path.read_text())

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    print("[WS] Client connected")

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, get_whisper)
    await loop.run_in_executor(None, get_tts)
    await ws.send_json({"type": "ready"})

    # Cancellation event for current processing
    cancel_event = asyncio.Event()
    processing_task = None

    # Enrollment session state
    enrollment_samples = []
    enrolling = False

    async def process_audio(audio_bytes: bytes):
        nonlocal cancel_event
        cancel_event.clear()

        # 0. Speaker verification gate
        if _should_verify():
            import speaker_verify
            t_sv = time.time()
            enrolled = speaker_enrolled_embedding if speaker_enrolled_embedding is not None else _load_enrolled_embedding()
            if enrolled is not None:
                embedding = await loop.run_in_executor(None, speaker_verify.extract_embedding, audio_bytes)
                score = speaker_verify.compare(embedding, enrolled)
                sv_time = time.time() - t_sv
                if score < speaker_verify.DEFAULT_THRESHOLD:
                    print(f"[Speaker] Rejected: score={score:.3f} ({sv_time:.2f}s)")
                    await ws.send_json({"type": "rejected", "score": round(score, 3), "time": round(sv_time, 2)})
                    return
                print(f"[Speaker] Verified: score={score:.3f} ({sv_time:.2f}s)")
                # Pass score along for UI display
                await ws.send_json({"type": "verified", "score": round(score, 3), "time": round(sv_time, 2)})

        # 1. STT
        await ws.send_json({"type": "status", "text": "Transcribing..."})
        t0 = time.time()
        user_text = await loop.run_in_executor(None, transcribe, audio_bytes)
        stt_time = time.time() - t0

        if cancel_event.is_set():
            return

        if not user_text:
            await ws.send_json({"type": "status", "text": "Didn't catch that. Try again?"})
            return

        await ws.send_json({"type": "transcript", "role": "user", "text": user_text, "time": round(stt_time, 2)})

        if cancel_event.is_set():
            return

        # 2. LLM streaming + TTS per sentence
        await ws.send_json({"type": "status", "text": "Thinking..."})
        await ws.send_json({"type": "stream_start"})

        llm_start = time.time()
        first_sentence_time = None
        sentence_count = 0
        total_tts_time = 0

        async for event_type, data in chat_stream(user_text, cancel_event):
            if cancel_event.is_set() and event_type not in ("cancelled", "done"):
                continue

            if event_type == "token":
                await ws.send_json({"type": "token", "text": data})

            elif event_type == "sentence":
                if first_sentence_time is None:
                    first_sentence_time = time.time() - llm_start

                # Check cancellation before TTS
                if cancel_event.is_set():
                    continue

                await ws.send_json({"type": "status", "text": "Speaking..."})
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

            elif event_type == "cancelled":
                print(f"[WS] Response cancelled")
                await ws.send_json({"type": "cancelled"})
                return

            elif event_type == "done":
                llm_time = time.time() - llm_start
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

    try:
        while True:
            msg = await ws.receive_json()

            if msg["type"] == "audio":
                # Cancel any existing processing
                if processing_task and not processing_task.done():
                    cancel_event.set()
                    try:
                        await asyncio.wait_for(processing_task, timeout=2.0)
                    except asyncio.TimeoutError:
                        processing_task.cancel()

                audio_bytes = base64.b64decode(msg["data"])
                processing_task = asyncio.create_task(process_audio(audio_bytes))

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

            elif msg["type"] == "clear":
                cancel_event.set()
                conversation_history.clear()
                await ws.send_json({"type": "status", "text": "Conversation cleared."})

    except WebSocketDisconnect:
        cancel_event.set()
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

    print(f"[LLM] OpenClaw endpoint: {OPENCLAW_URL}")
    print(f"[TTS] Engine: {TTS_ENGINE}")
    uvicorn.run(app, host="0.0.0.0", port=8765, **kwargs)
