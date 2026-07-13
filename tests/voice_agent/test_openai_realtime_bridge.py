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


@pytest.mark.asyncio
async def test_speaker_id_is_added_to_response_instructions(monkeypatch):
    bridge = OpenAIRealtimeBridge(
        api_key="test-key",
        model="gpt-realtime-2.1",
        voice="marin",
        instructions="Base instructions",
    )
    bridge._ws = object()
    sent = []

    async def fake_ensure_connected():
        return None

    async def fake_send_json(event):
        sent.append(event)

    async def fake_recv_json():
        return {
            "type": "response.done",
            "response": {"status": "completed", "output": []},
        }

    monkeypatch.setattr(bridge, "_ensure_connected", fake_ensure_connected)
    monkeypatch.setattr(bridge, "_send_json", fake_send_json)
    monkeypatch.setattr(bridge, "_recv_json", fake_recv_json)

    chunks = [
        chunk
        async for chunk in bridge.stream_turn(
            np.ones(320, dtype=np.float32) * 0.1,
            speaker_id=1,
        )
    ]

    response_create = next(event for event in sent if event["type"] == "response.create")
    instructions = response_create["response"]["instructions"]
    assert chunks == []
    assert "Base instructions" in instructions
    assert "Visitor 2" in instructions
