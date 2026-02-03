#!/usr/bin/env python3
"""
Real-time voice chat server with streaming TTS.
STT: faster-whisper (GPU)  |  LLM: OpenClaw (Friday)  |  TTS: Kokoro ONNX (local)

v0.2: Streaming TTS — sentences are spoken as they arrive from the LLM.
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
KOKORO_VOICE = os.getenv("KOKORO_VOICE", "af_heart")
SAMPLE_RATE = 16000

# OpenClaw gateway
OPENCLAW_URL = os.getenv("OPENCLAW_URL", "http://127.0.0.1:18789/v1/chat/completions")
OPENCLAW_TOKEN = os.getenv("OPENCLAW_TOKEN", "92c0ca8eeb7054cd6587b7368e83f25673e41c7b0cf9985b")
OPENCLAW_AGENT = os.getenv("OPENCLAW_AGENT", "main")

SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", (
    "You are Friday, responding via voice chat. Your response will be spoken aloud via TTS. "
    "STRICT RULES: No emoji. No emoticons. No markdown. No bullet lists. No code blocks. No asterisks. No special characters. "
    "Keep responses concise and conversational. Just plain spoken English, like you're talking to someone. "
    "Be natural, warm, and to the point."
))

# ---------------------------------------------------------------------------
# Global singletons
# ---------------------------------------------------------------------------
whisper_model = None
kokoro_tts = None
conversation_history = []

def get_whisper():
    global whisper_model
    if whisper_model is None:
        from faster_whisper import WhisperModel
        print(f"[STT] Loading {WHISPER_MODEL} on {WHISPER_DEVICE}...")
        whisper_model = WhisperModel(WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE)
        print("[STT] Ready.")
    return whisper_model

def get_kokoro():
    global kokoro_tts
    if kokoro_tts is None:
        import kokoro_onnx
        print("[TTS] Loading Kokoro...")
        kokoro_tts = kokoro_onnx.Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")
        print(f"[TTS] Ready. Voice: {KOKORO_VOICE}")
    return kokoro_tts

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
            wf.setframerate(SAMPLE_RATE)
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


async def chat_stream(user_text: str) -> AsyncIterator[tuple[str, str]]:
    """
    Stream chat response from OpenClaw.
    Yields tuples of (event_type, data):
      - ("token", token_text) for each token
      - ("sentence", sentence_text) for each complete sentence
      - ("done", full_text) when finished
    """
    global conversation_history

    conversation_history.append({"role": "user", "content": user_text})

    # Keep last 40 messages
    if len(conversation_history) > 40:
        conversation_history = conversation_history[-40:]

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + conversation_history

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

            full_text = ""
            buffer = ""  # Accumulates text until we find a sentence boundary

            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue

                data = line[6:]  # Strip "data: " prefix
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

                        # Check for sentence boundaries in buffer
                        while True:
                            match = SENTENCE_END_RE.search(buffer)
                            if not match:
                                break

                            # Found a sentence end
                            end_pos = match.end()
                            sentence = buffer[:end_pos].strip()
                            buffer = buffer[end_pos:].lstrip()

                            if sentence:
                                yield ("sentence", sentence)

                except json.JSONDecodeError:
                    continue

            # Emit any remaining text as final sentence
            if buffer.strip():
                yield ("sentence", buffer.strip())

            # Record full response in history
            conversation_history.append({"role": "assistant", "content": full_text})
            print(f"[LLM] → \"{full_text[:80]}...\"" if len(full_text) > 80 else f"[LLM] → \"{full_text}\"")

            yield ("done", full_text)


def synthesize(text: str) -> bytes:
    """Convert text to speech, return WAV bytes."""
    tts = get_kokoro()
    samples, sr = tts.create(text, voice=KOKORO_VOICE, speed=1.0)

    buf = io.BytesIO()
    with wave.open(buf, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        pcm = (samples * 32767).astype(np.int16)
        wf.writeframes(pcm.tobytes())

    return buf.getvalue()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="Friday Voice Chat")

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
    await loop.run_in_executor(None, get_kokoro)
    await ws.send_json({"type": "ready"})

    try:
        while True:
            msg = await ws.receive_json()

            if msg["type"] == "audio":
                audio_bytes = base64.b64decode(msg["data"])

                # 1. STT
                await ws.send_json({"type": "status", "text": "Transcribing..."})
                t0 = time.time()
                user_text = await loop.run_in_executor(None, transcribe, audio_bytes)
                stt_time = time.time() - t0

                if not user_text:
                    await ws.send_json({"type": "status", "text": "Didn't catch that. Try again?"})
                    continue

                await ws.send_json({"type": "transcript", "role": "user", "text": user_text, "time": round(stt_time, 2)})

                # 2. LLM streaming + TTS per sentence
                await ws.send_json({"type": "status", "text": "Thinking..."})
                await ws.send_json({"type": "stream_start"})

                llm_start = time.time()
                first_sentence_time = None
                sentence_count = 0
                total_tts_time = 0

                async for event_type, data in chat_stream(user_text):
                    if event_type == "token":
                        # Send token for real-time transcript display
                        await ws.send_json({"type": "token", "text": data})

                    elif event_type == "sentence":
                        if first_sentence_time is None:
                            first_sentence_time = time.time() - llm_start

                        # Generate TTS for this sentence
                        await ws.send_json({"type": "status", "text": "Speaking..."})
                        tts_start = time.time()
                        audio_out = await loop.run_in_executor(None, synthesize, data)
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
                        print(f"[TTS] Sentence {sentence_count}: \"{data[:40]}...\" ({tts_time:.2f}s)" if len(data) > 40 else f"[TTS] Sentence {sentence_count}: \"{data}\" ({tts_time:.2f}s)")

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

            elif msg["type"] == "clear":
                conversation_history.clear()
                await ws.send_json({"type": "status", "text": "Conversation cleared."})

    except WebSocketDisconnect:
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
    uvicorn.run(app, host="0.0.0.0", port=8765, **kwargs)
