"""Focus debounce/cooldown + profile-driven opener logic (no real CV/network)."""

import asyncio

import pytest

pytest.importorskip("websockets")
import config  # noqa: E402
import cv_injector  # noqa: E402


class FakeBridge:
    def __init__(self):
        self.steers = []
        self.contexts = []

    async def steer(self, text):
        self.steers.append(text)

    async def inject_context(self, text, turn_complete=False):
        self.contexts.append((text, turn_complete))


def _make_injector():
    seek = []

    async def send_json(obj):
        seek.append(obj)

    inj = cv_injector.CvInjector(session_id="s1", bridge=FakeBridge(), send_json=send_json)
    return inj, seek


def test_focus_debounce_and_cooldown(monkeypatch):
    monkeypatch.setattr(config, "FOCUS_LOSS_SECONDS", 5.0)
    monkeypatch.setattr(config, "NUDGE_COOLDOWN_SECONDS", 20.0)
    clock = {"t": 100.0}
    monkeypatch.setattr(cv_injector.time, "monotonic", lambda: clock["t"])

    inj, seek = _make_injector()
    inj._opened = True  # opener already ran

    async def scenario():
        await inj._on_focus({"is_focused": True})
        assert inj._nudge_count == 0

        clock["t"] = 100.0
        await inj._on_focus({"is_focused": False})   # timer starts
        clock["t"] = 102.0
        await inj._on_focus({"is_focused": False})   # 2s < 5s -> no nudge
        assert inj._nudge_count == 0

        clock["t"] = 106.0
        await inj._on_focus({"is_focused": False})   # 6s >= 5s -> nudge #1
        assert inj._nudge_count == 1

        clock["t"] = 110.0
        await inj._on_focus({"is_focused": False})   # within cooldown -> no nudge
        assert inj._nudge_count == 1

        clock["t"] = 130.0
        await inj._on_focus({"is_focused": False})   # cooldown elapsed -> nudge #2
        assert inj._nudge_count == 2

        clock["t"] = 131.0
        await inj._on_focus({"is_focused": True})    # refocus resets timer
        assert inj._distracted_since is None

    asyncio.run(scenario())
    assert len(seek) == 2                 # one seekAttention per nudge
    assert len(inj.bridge.steers) == 2    # one verbal steer per nudge


def test_no_nudge_before_opener(monkeypatch):
    monkeypatch.setattr(config, "FOCUS_LOSS_SECONDS", 1.0)
    clock = {"t": 0.0}
    monkeypatch.setattr(cv_injector.time, "monotonic", lambda: clock["t"])
    inj, seek = _make_injector()
    inj._opened = False

    async def scenario():
        clock["t"] = 0.0
        await inj._on_focus({"is_focused": False})
        clock["t"] = 10.0
        await inj._on_focus({"is_focused": False})

    asyncio.run(scenario())
    assert inj._nudge_count == 0
    assert seek == []


def test_opener_uses_profile_when_available(monkeypatch):
    monkeypatch.setattr(config, "PROFILE_WAIT_SECONDS", 1.0)
    inj, _ = _make_injector()

    async def scenario():
        inj._profile_future = asyncio.get_running_loop().create_future()
        inj._profile_future.set_result({
            "scores": {"attention": 0.8, "openness": 0.8, "energy": 0.8},
            "signals": {"emotion": {"dominant": "happy"}},
        })
        await inj._opener()

    asyncio.run(scenario())
    assert len(inj.bridge.contexts) == 1
    text, turn_complete = inj.bridge.contexts[0]
    assert turn_complete is True
    assert "açık ve rahat" in text
    assert inj._opened is True


def test_opener_falls_back_to_generic_on_timeout(monkeypatch):
    monkeypatch.setattr(config, "PROFILE_WAIT_SECONDS", 0.05)
    inj, _ = _make_injector()

    async def scenario():
        inj._profile_future = asyncio.get_running_loop().create_future()  # never resolved
        await inj._opener()

    asyncio.run(scenario())
    assert len(inj.bridge.contexts) == 1
    text, turn_complete = inj.bridge.contexts[0]
    assert turn_complete is True
    assert "tanıtım standına" in text  # generic opener
