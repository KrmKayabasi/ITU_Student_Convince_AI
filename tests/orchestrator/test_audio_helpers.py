"""Audio format helper round-trips."""

import pytest

np = pytest.importorskip("numpy")
import audio_helpers as ah  # noqa: E402


def test_resample_length_16k_to_24k():
    sig = np.zeros(16000, dtype=np.float32)
    out = ah.resample_mono_float32(sig, 16000, 24000)
    assert abs(len(out) - 24000) <= 2


def test_resample_noop_same_rate():
    sig = np.linspace(-1, 1, 100, dtype=np.float32)
    out = ah.resample_mono_float32(sig, 16000, 16000)
    assert np.allclose(out, sig)


def test_pcm16_round_trip_precision():
    t = np.linspace(0, 1, 24000, endpoint=False, dtype=np.float32)
    sig = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    pcm = ah.float32_to_pcm16_bytes(sig)
    back = np.frombuffer(ah.pcm16_bytes_to_float32_bytes(pcm), dtype="<f4")
    assert len(back) == len(sig)
    assert float(np.max(np.abs(back - sig))) < 1e-3


def test_pcm16_clipping():
    sig = np.array([2.0, -2.0, 0.0], dtype=np.float32)
    back = np.frombuffer(ah.pcm16_bytes_to_float32_bytes(ah.float32_to_pcm16_bytes(sig)),
                         dtype="<f4")
    assert back[0] <= 1.0 and back[1] >= -1.0


def test_pcm16_to_float32_array():
    arr = ah.pcm16_bytes_to_float32(ah.float32_to_pcm16_bytes(np.zeros(10, dtype=np.float32)))
    assert arr.shape == (10,) and arr.dtype == np.float32


def test_empty_inputs():
    assert ah.pcm16_bytes_to_float32_bytes(b"") == b""
    assert len(ah.resample_mono_float32(np.zeros(0, dtype=np.float32), 16000, 24000)) == 0
