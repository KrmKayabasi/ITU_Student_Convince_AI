"""
Audio format helpers shared across the orchestrator.

These are lifted verbatim from the OpenAI realtime bridge on the
`origin/gpt-realtime` branch (backend/speech_backend/openai_realtime_bridge.py)
so the Gemini bridge reuses the exact, already-tested conversions:

  - resample_mono_float32   : mono float32 resample (scipy poly, linear fallback)
  - float32_to_pcm16_bytes  : float32 [-1,1] -> little-endian PCM16 bytes
  - pcm16_bytes_to_float32_bytes : PCM16 bytes -> float32 bytes

Gemini Live wants 16 kHz PCM16 input and emits 24 kHz PCM16 output, so the same
helpers cover both directions.
"""

from __future__ import annotations

import math

import numpy as np


def resample_mono_float32(
    audio: np.ndarray,
    source_rate: int,
    target_rate: int,
) -> np.ndarray:
    """Return mono float32 audio resampled to target_rate."""
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    if source_rate == target_rate or len(samples) == 0:
        return samples

    try:
        from scipy.signal import resample_poly

        gcd = math.gcd(source_rate, target_rate)
        resampled = resample_poly(samples, target_rate // gcd, source_rate // gcd)
        return np.asarray(resampled, dtype=np.float32)
    except Exception:
        duration = len(samples) / float(source_rate)
        target_len = max(1, int(round(duration * target_rate)))
        src_x = np.linspace(0.0, duration, num=len(samples), endpoint=False)
        dst_x = np.linspace(0.0, duration, num=target_len, endpoint=False)
        return np.interp(dst_x, src_x, samples).astype(np.float32)


def float32_to_pcm16_bytes(audio: np.ndarray) -> bytes:
    """Convert float32 [-1, 1] audio to little-endian PCM16 bytes."""
    clipped = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    return pcm.tobytes()


def pcm16_bytes_to_float32_bytes(audio_bytes: bytes) -> bytes:
    """Convert little-endian PCM16 bytes to float32 bytes (mono)."""
    if not audio_bytes:
        return b""
    pcm = np.frombuffer(audio_bytes, dtype="<i2")
    float_audio = (pcm.astype(np.float32) / 32768.0).astype(np.float32)
    return float_audio.tobytes()


def pcm16_bytes_to_float32(audio_bytes: bytes) -> np.ndarray:
    """Convert little-endian PCM16 bytes to a float32 numpy array (mono)."""
    if not audio_bytes:
        return np.zeros(0, dtype=np.float32)
    pcm = np.frombuffer(audio_bytes, dtype="<i2")
    return (pcm.astype(np.float32) / 32768.0).astype(np.float32)
