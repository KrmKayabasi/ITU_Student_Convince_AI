import numpy as np
import pytest

from backend.speech_backend.openai_realtime_bridge import (
    OpenAIRealtimeBridge,
    float32_to_pcm16_bytes,
    pcm16_bytes_to_float32_bytes,
    resample_mono_float32,
)


def test_resample_16khz_to_24khz_preserves_duration():
    audio = np.linspace(-0.5, 0.5, 1600, dtype=np.float32)

    resampled = resample_mono_float32(audio, source_rate=16000, target_rate=24000)

    assert resampled.dtype == np.float32
    assert len(resampled) == 2400


def test_pcm16_conversion_returns_float32_stream_bytes():
    audio = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)

    pcm_bytes = float32_to_pcm16_bytes(audio)
    float_bytes = pcm16_bytes_to_float32_bytes(pcm_bytes)
    roundtrip = np.frombuffer(float_bytes, dtype=np.float32)

    assert roundtrip.dtype == np.float32
    assert roundtrip.shape == audio.shape
    assert np.allclose(roundtrip, audio, atol=1 / 32768)


def test_bridge_requires_openai_api_key():
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        OpenAIRealtimeBridge(
            api_key="",
            model="gpt-realtime-2.1",
            voice="marin",
            instructions="test",
        )
