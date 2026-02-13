#!/usr/bin/env python3
"""
Speaker Verification Module — SpeechBrain ECAPA-TDNN

Provides speaker embedding extraction and verification against an enrolled voice.
Runs on CPU to keep GPU free for whisper + chatterbox.
"""

import io
import os
import time
import wave
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
VOICES_DIR = Path(__file__).parent / "voices"
_PERSISTENT_DIR = Path.home() / ".kismet" / "voices"
# Prefer persistent path in home directory, fall back to repo-local
ENROLLMENT_PATH = (
    _PERSISTENT_DIR / "ham_embedding.npy"
    if (_PERSISTENT_DIR / "ham_embedding.npy").exists()
    else VOICES_DIR / "ham_embedding.npy"
)
DEFAULT_THRESHOLD = float(os.getenv("SPEAKER_VERIFY_THRESHOLD", "0.50"))

# ---------------------------------------------------------------------------
# Model singleton
# ---------------------------------------------------------------------------
_model = None


def _get_model():
    """Load ECAPA-TDNN model (lazy, CPU-only)."""
    global _model
    if _model is None:
        from speechbrain.inference.speaker import EncoderClassifier
        print("[Speaker] Loading ECAPA-TDNN on CPU...")
        t0 = time.time()
        _model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": "cpu"},
        )
        print(f"[Speaker] Model loaded in {time.time() - t0:.1f}s")
    return _model


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------
def extract_embedding(audio_bytes: bytes) -> np.ndarray:
    """
    Extract speaker embedding from raw PCM 16-bit 16kHz mono audio bytes.
    Returns a 1D numpy array (192-dim embedding).
    """
    model = _get_model()

    # Write PCM bytes to a temporary WAV file (SpeechBrain needs a file path)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name
        with wave.open(f, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(audio_bytes)

    try:
        # SpeechBrain returns shape (1, 1, 192) — squeeze to (192,)
        embedding = model.encode_batch(
            model.load_audio(tmp_path).unsqueeze(0)
        )
        return embedding.squeeze().cpu().numpy()
    finally:
        os.unlink(tmp_path)


def compare(embedding: np.ndarray, enrolled_embedding: np.ndarray) -> float:
    """Compute cosine similarity between two embeddings."""
    a = embedding.flatten()
    b = enrolled_embedding.flatten()
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(dot / norm)


def enroll(audio_samples: list[bytes], save_path: Path = _PERSISTENT_DIR / "ham_embedding.npy") -> np.ndarray:
    """
    Enroll a speaker from multiple audio samples.
    Extracts embeddings from each, averages them, saves to disk.
    Returns the averaged embedding.
    """
    print(f"[Speaker] Enrolling from {len(audio_samples)} samples...")
    embeddings = []
    for i, audio in enumerate(audio_samples):
        t0 = time.time()
        emb = extract_embedding(audio)
        print(f"[Speaker] Sample {i+1}/{len(audio_samples)} embedded in {time.time()-t0:.2f}s")
        embeddings.append(emb)

    avg_embedding = np.mean(embeddings, axis=0)
    # Normalize
    avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(save_path), avg_embedding)
    print(f"[Speaker] Enrollment saved to {save_path}")
    return avg_embedding


def has_enrollment(path: Path = ENROLLMENT_PATH) -> bool:
    """Check if an enrollment file exists."""
    return path.exists()


def load_enrollment(path: Path = ENROLLMENT_PATH) -> Optional[np.ndarray]:
    """Load enrolled embedding from disk. Returns None if not found."""
    if not path.exists():
        return None
    return np.load(str(path))


def verify(audio_bytes: bytes, threshold: float = DEFAULT_THRESHOLD) -> tuple[bool, float]:
    """
    Verify audio against enrolled speaker.
    Returns (is_verified, similarity_score).
    If no enrollment exists, returns (True, 1.0) — pass-through.
    """
    enrolled = load_enrollment()
    if enrolled is None:
        return True, 1.0

    embedding = extract_embedding(audio_bytes)
    score = compare(embedding, enrolled)
    is_verified = score >= threshold
    print(f"[Speaker] Verify: score={score:.3f} threshold={threshold} → {'PASS' if is_verified else 'REJECT'}")
    return is_verified, score
