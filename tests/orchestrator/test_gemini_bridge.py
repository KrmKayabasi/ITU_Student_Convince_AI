"""Gemini bridge: key guard, config assembly, and event normalization."""

from types import SimpleNamespace

import pytest

import gemini_live_bridge as glb


def _bridge():
    return glb.GeminiLiveBridge(
        api_key="dummy",
        model="gemini-3.1-flash-live-preview",
        voice="Aoede",
        instructions="Türkçe konuş.",
    )


def test_missing_key_raises():
    with pytest.raises(RuntimeError):
        glb.GeminiLiveBridge(api_key="", model="m", voice="v", instructions="i")


def test_build_config_assembles():
    pytest.importorskip("google.genai")
    from google.genai import types

    cfg = _bridge()._build_config()
    assert cfg.enable_affective_dialog is True
    assert cfg.proactivity.proactive_audio is True
    aad = cfg.realtime_input_config.automatic_activity_detection
    assert aad.disabled is False
    assert (
        aad.start_of_speech_sensitivity
        == types.StartSensitivity.START_SENSITIVITY_LOW
    )
    assert (
        aad.end_of_speech_sensitivity
        == types.EndSensitivity.END_SENSITIVITY_HIGH
    )
    assert aad.prefix_padding_ms == 160
    assert aad.silence_duration_ms == 500
    assert cfg.speech_config.voice_config.prebuilt_voice_config.voice_name == "Aoede"
    declaration = cfg.tools[0].function_declarations[0]
    assert declaration.name == "search_itu_professors"
    assert declaration.behavior == types.Behavior.NON_BLOCKING


def _resp(*, audio=None, in_tx=None, out_tx=None, interrupted=None, turn_complete=None):
    parts = []
    if audio is not None:
        parts.append(SimpleNamespace(inline_data=SimpleNamespace(data=audio)))
    model_turn = SimpleNamespace(parts=parts) if parts else None
    sc = SimpleNamespace(
        model_turn=model_turn,
        input_transcription=SimpleNamespace(text=in_tx) if in_tx else None,
        output_transcription=SimpleNamespace(text=out_tx) if out_tx else None,
        interrupted=interrupted,
        turn_complete=turn_complete,
    )
    return SimpleNamespace(server_content=sc, data=None)


def test_normalize_audio_event():
    events = _bridge()._normalize(_resp(audio=b"\x01\x02"))
    assert events == [{"type": "audio", "pcm16": b"\x01\x02"}]


def test_normalize_transcripts_and_commit():
    b = _bridge()
    b._normalize(_resp(in_tx="merhaba"))
    b._normalize(_resp(out_tx="hoş geldin"))
    events = b._normalize(_resp(turn_complete=True))
    assert {"type": "turn_complete"} in events
    turn = b.get_last_turn()
    assert turn["user"] == "merhaba"
    assert turn["assistant"] == "hoş geldin"


def test_normalize_interrupt_drops_partial_assistant():
    b = _bridge()
    b._normalize(_resp(out_tx="yarım cümle"))
    events = b._normalize(_resp(interrupted=True))
    # The bridge emits the internal "interrupted"; the server maps it to the
    # browser-facing "interrupt".
    assert {"type": "interrupted"} in events
    # After a barge-in, the partial assistant text is discarded on commit.
    b._normalize(_resp(turn_complete=True))
    assert b.get_last_turn()["assistant"] == ""


def test_receive_continues_across_turns():
    """Regression: the SDK's session.receive() iterator ENDS at each
    turn_complete. The bridge must keep listening for subsequent turns instead
    of dying after the first answer."""
    import asyncio

    b = _bridge()

    class FakeSession:
        """Yields two separate one-turn iterators, like the real SDK."""

        def __init__(self):
            self.calls = 0

        def receive(self):
            self.calls += 1
            call = self.calls

            async def gen():
                if call == 1:
                    yield _resp(audio=b"turn1")
                    yield _resp(turn_complete=True)
                elif call == 2:
                    yield _resp(audio=b"turn2")
                    yield _resp(turn_complete=True)
                else:
                    b._closed = True  # stop the bridge's outer loop

            return gen()

    b._session = FakeSession()

    async def collect():
        events = []
        async for ev in b.receive():
            events.append(ev)
        return events

    events = asyncio.run(collect())
    audio = [e["pcm16"] for e in events if e["type"] == "audio"]
    turns = [e for e in events if e["type"] == "turn_complete"]
    assert audio == [b"turn1", b"turn2"]   # BOTH turns were forwarded
    assert len(turns) == 2
    assert b._session.calls >= 3           # receive() was re-entered after each turn


def test_normalize_fallback_to_response_data():
    b = _bridge()
    resp = SimpleNamespace(server_content=SimpleNamespace(
        model_turn=None, input_transcription=None, output_transcription=None,
        interrupted=None, turn_complete=None), data=b"\x09\x09")
    assert b._normalize(resp) == [{"type": "audio", "pcm16": b"\x09\x09"}]


def test_normalize_tool_call_and_cancellation():
    b = _bridge()
    call = SimpleNamespace(id="call-1", name="search_itu_professors", args={"topic": "Robotik"})
    response = SimpleNamespace(
        server_content=None,
        data=None,
        tool_call=SimpleNamespace(function_calls=[call]),
        tool_call_cancellation=SimpleNamespace(ids=["call-2"]),
    )
    assert b._normalize(response) == [
        {
            "type": "tool_call",
            "calls": [
                {
                    "id": "call-1",
                    "name": "search_itu_professors",
                    "args": {"topic": "Robotik"},
                }
            ],
        },
        {"type": "tool_cancel", "ids": ["call-2"]},
    ]


def test_send_tool_response_is_scheduled_when_idle():
    import asyncio

    pytest.importorskip("google.genai")
    from google.genai import types

    b = _bridge()

    class FakeSession:
        response = None

        async def send_tool_response(self, *, function_responses):
            self.response = function_responses

    session = FakeSession()
    b._session = session
    asyncio.run(
        b.send_tool_response(
            call_id="call-1",
            name="search_itu_professors",
            response={"results": []},
        )
    )
    assert session.response.id == "call-1"
    assert session.response.scheduling == types.FunctionResponseScheduling.WHEN_IDLE
