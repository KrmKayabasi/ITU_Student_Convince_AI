"""
Realtime orchestrator — FastAPI WebSocket bridge between the browser kiosk and
Gemini Live, with server-side CV context injection.

Transport (browser <-> orchestrator), one held-open WebSocket per visitor:
  Uplink  (browser -> here):
    - binary frame  = raw PCM16 mono 16 kHz  (mic)
    - text frame    = JSON control: {"type": "interrupt" | "session.stop"}
  Downlink (here -> browser):
    - binary frame  = raw PCM16 mono 24 kHz  (model audio)
     - text frame    = JSON: {"type":"ready","sample_rate":24000}
                             {"type":"transcript","role":"user|assistant","text":...}
                             {"type":"assistant_audio_start"}
                             {"type":"tool_activity",...} / {"type":"tool_result",...}
                             {"type":"interrupt"}            (barge-in; flush playback)
                            {"type":"seekAttention"}        (avatar attention-grab)
                            {"type":"turn_complete"} / {"type":"error","message":...}

All outbound frames go through a single queue + sender task, so the Gemini
forwarder and the CV injector never write to the socket concurrently.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import secrets
import sys

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from starlette.websockets import WebSocketState

sys.path.insert(0, os.path.dirname(__file__))

import config  # noqa: E402
from gemini_live_bridge import GeminiLiveBridge  # noqa: E402
from professor_search import ItuProfessorSearch  # noqa: E402
from tool_coordinator import ToolCoordinator  # noqa: E402

try:
    from cv_injector import CvInjector  # noqa: E402
except Exception:  # pragma: no cover - injector optional until P5 lands
    CvInjector = None  # type: ignore

try:
    from emotion import get_classifier  # noqa: E402
except Exception:  # pragma: no cover - emotion is optional (heavy deps)
    get_classifier = None  # type: ignore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("orchestrator")

app = FastAPI(title="İTÜ Convince AI — Realtime Orchestrator")

_SESSION_RE = re.compile(r"[A-Za-z0-9_.-]{1,64}")
# session_id -> the currently-connected SessionRunner (one active visitor per id)
_active: dict[str, "SessionRunner"] = {}


def _valid_session_id(session_id: str) -> bool:
    return bool(_SESSION_RE.fullmatch(session_id or ""))


def _authorized(token: str | None) -> bool:
    if not config.ORCH_TOKEN:
        return True
    return token is not None and secrets.compare_digest(token, config.ORCH_TOKEN)


class SessionRunner:
    """Owns one browser WebSocket, its Gemini bridge, and the CV injector."""

    def __init__(self, websocket: WebSocket, session_id: str) -> None:
        self.ws = websocket
        self.session_id = session_id
        self.out: asyncio.Queue = asyncio.Queue(maxsize=512)
        self.bridge = GeminiLiveBridge(
            api_key=config.GOOGLE_API_KEY,
            model=config.GEMINI_LIVE_MODEL,
            voice=config.GEMINI_VOICE,
            instructions=config.SYSTEM_INSTRUCTION,
            api_version=config.GEMINI_API_VERSION,
            input_sample_rate=config.INPUT_SAMPLE_RATE,
            output_sample_rate=config.OUTPUT_SAMPLE_RATE,
            enable_affective_dialog=config.ENABLE_AFFECTIVE_DIALOG,
            enable_proactive_audio=config.ENABLE_PROACTIVE_AUDIO,
        )
        self.injector = None
        self.tools = ToolCoordinator(
            bridge=self.bridge,
            send_json=self.send_json,
            professor_search=ItuProfessorSearch(
                timeout_s=config.PROFESSOR_SEARCH_TIMEOUT_S,
                max_results=config.PROFESSOR_SEARCH_MAX_RESULTS,
            ),
        )
        # ── avatar emotion state (per-session) ─────────────────────────────────
        # Shared classifier instance (lazy-loaded on first transcript).
        self._loop = asyncio.get_event_loop()
        self._emotion = get_classifier() if get_classifier is not None else None
        self._emotion_buf = ""           # accumulated assistant transcript for classification
        self._emotion_last_emit = 0.0    # monotonic time of last emitted emotion
        self._emotion_last_label = "neutral"
        self._emotion_task: asyncio.Task | None = None
        self._emotion_started = False
        self._assistant_audio_started = False

    # ── outbound helpers (only the sender task touches the socket) ────────────
    async def send_json(self, obj: dict) -> None:
        await self.out.put(("json", obj))

    async def send_bytes(self, data: bytes) -> None:
        await self.out.put(("bytes", data))

    async def _sender(self) -> None:
        while True:
            kind, payload = await self.out.get()
            if kind == "bytes":
                await self.ws.send_bytes(payload)
            else:
                await self.ws.send_json(payload)

    # ── uplink: browser -> Gemini ─────────────────────────────────────────────
    async def _browser_reader(self) -> None:
        while True:
            msg = await self.ws.receive()
            if msg["type"] == "websocket.disconnect":
                raise WebSocketDisconnect()
            data = msg.get("bytes")
            if data is not None:
                await self.bridge.send_audio(data)
                continue
            text = msg.get("text")
            if text:
                await self._handle_control(text)

    async def _handle_control(self, text: str) -> None:
        import json

        try:
            evt = json.loads(text)
        except Exception:
            return
        t = evt.get("type")
        if t == "session.stop":
            raise WebSocketDisconnect()
        # "interrupt" is handled natively by Gemini VAD; nothing to forward.
        logger.debug("control from browser: %s", t)

    # ── downlink: Gemini -> browser ───────────────────────────────────────────
    async def _gemini_forwarder(self) -> None:
        async for ev in self.bridge.receive():
            et = ev["type"]
            if et == "audio":
                if not self._assistant_audio_started:
                    # Ordered before the first binary frame. The browser uses
                    # this boundary to distinguish old in-flight PCM after a
                    # local barge-in from the next legitimate response.
                    await self.send_json({"type": "assistant_audio_start"})
                    self._assistant_audio_started = True
                await self.send_bytes(ev["pcm16"])
            elif et == "output_transcript":
                await self.send_json({"type": "transcript", "role": "assistant", "text": ev["text"]})
                self._maybe_classify_emotion(ev["text"])
            elif et == "input_transcript":
                await self.send_json({"type": "transcript", "role": "user", "text": ev["text"]})
            elif et == "interrupted":
                self._assistant_audio_started = False
                self._reset_emotion("neutral")
                await self.send_json({"type": "interrupt"})
            elif et == "turn_complete":
                self._assistant_audio_started = False
                self._reset_emotion("neutral")
                await self.send_json({"type": "turn_complete"})
            elif et == "tool_call":
                for call in ev["calls"]:
                    self.tools.start(call)
            elif et == "tool_cancel":
                await self.tools.cancel(ev["ids"])

    # ── avatar emotion helpers ─────────────────────────────────────────────────
    def _maybe_classify_emotion(self, chunk: str) -> None:
        """Accumulate assistant text and (debounced) classify on sentence end.

        Non-blocking: spawns a task so the realtime loop never waits on the
        model. All failures are logged and swallowed — emotion is cosmetic.
        """
        clf = self._emotion
        if clf is None:
            return
        self._emotion_buf += chunk or ""
        buf = self._emotion_buf

        # Wait for enough text AND a plausible sentence boundary.
        ends_sentence = bool(buf) and buf[-1] in ".!?…\n"
        if not ends_sentence:
            # Also classify on long unbroken runs so slow streams still react.
            if len(buf) < config.EMOTION_MIN_CHARS * 3:
                return
        if len(buf) < config.EMOTION_MIN_CHARS:
            return

        now = self._loop.time()
        if now - self._emotion_last_emit < config.EMOTION_DEBOUNCE_S:
            return

        # Lazy-load the model on first qualifying chunk.
        if not self._emotion_started:
            self._emotion_started = True

        # Snapshot + clear the buffer so concurrent chunks start fresh.
        text, self._emotion_buf = buf, ""

        # Coalesce: skip if a classification is already in flight.
        if self._emotion_task is not None and not self._emotion_task.done():
            return

        self._emotion_task = asyncio.create_task(self._classify_and_emit(text, now))

    async def _classify_and_emit(self, text: str, queued_at: float) -> None:
        clf = self._emotion
        if clf is None:
            return
        try:
            if not clf._pipe:  # model not loaded yet
                await clf.start()
            label = await clf.classify(text)
        except Exception:
            logger.exception("emotion classification failed; continuing")
            return
        if not label:
            return
        # Only emit when the label actually changes (cut WS chatter).
        if label == self._emotion_last_label:
            self._emotion_last_emit = queued_at
            return
        self._emotion_last_label = label
        self._emotion_last_emit = self._loop.time()
        await self.send_json({"type": "emotion", "emotion": label})

    def _reset_emotion(self, label: str = "neutral") -> None:
        """Clear the buffer (barge-in / turn boundary)."""
        self._emotion_buf = ""
        if self._emotion_last_label != label:
            self._emotion_last_label = label
            asyncio.create_task(self.send_json({"type": "emotion", "emotion": label}))

    # ── orchestration ─────────────────────────────────────────────────────────
    async def run(self) -> None:
        await self.bridge.connect()
        await self.send_json({"type": "ready", "sample_rate": self.bridge.sample_rate})

        tasks = [
            asyncio.create_task(self._sender(), name="sender"),
            asyncio.create_task(self._browser_reader(), name="reader"),
            asyncio.create_task(self._gemini_forwarder(), name="forwarder"),
        ]

        if CvInjector is not None:
            self.injector = CvInjector(
                session_id=self.session_id,
                bridge=self.bridge,
                send_json=self.send_json,
            )
            tasks.append(asyncio.create_task(self.injector.run(), name="cv_injector"))

        try:
            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED
            )
            for t in pending:
                t.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            # Surface the first real error (ignore normal disconnects).
            for t in done:
                exc = t.exception()
                if exc and not isinstance(exc, WebSocketDisconnect):
                    raise exc
        finally:
            await self.close()

    async def close(self) -> None:
        await self.tools.close()
        if self.injector is not None:
            await self.injector.close()
        await self.bridge.aclose()


@app.websocket("/v1/realtime")
async def realtime(websocket: WebSocket, session_id: str = Query(...), token: str = Query(None)):
    await websocket.accept()
    if not _valid_session_id(session_id):
        await websocket.close(code=4000, reason="Invalid session_id")
        return
    if not _authorized(token):
        await websocket.close(code=4001, reason="Invalid token")
        return
    if not config.GOOGLE_API_KEY:
        await websocket.send_json({"type": "error", "message": "orchestrator missing GOOGLE_API_KEY"})
        await websocket.close(code=1011, reason="server misconfigured")
        return

    # Replace any stale runner for this session_id.
    old = _active.pop(session_id, None)
    if old is not None:
        await old.close()

    runner = SessionRunner(websocket, session_id)
    _active[session_id] = runner
    logger.info("realtime connected: %s", session_id)
    try:
        await runner.run()
    except WebSocketDisconnect:
        logger.info("realtime disconnected: %s", session_id)
    except Exception:
        logger.exception("realtime session error: %s", session_id)
        if websocket.application_state == WebSocketState.CONNECTED:
            try:
                await websocket.send_json({"type": "error", "message": "session error"})
            except Exception:
                pass
    finally:
        if _active.get(session_id) is runner:
            _active.pop(session_id, None)
        if websocket.application_state == WebSocketState.CONNECTED:
            await websocket.close()


@app.get("/v1/health")
async def health():
    return {
        "status": "ok",
        "provider": "gemini_live",
        "model": config.GEMINI_LIVE_MODEL,
        "api_version": config.GEMINI_API_VERSION,
        "affective_dialog": config.ENABLE_AFFECTIVE_DIALOG,
        "proactive_audio": config.ENABLE_PROACTIVE_AUDIO,
        "auth_enabled": bool(config.ORCH_TOKEN),
        "key_present": bool(config.GOOGLE_API_KEY),
        "active_sessions": len(_active),
    }


if __name__ == "__main__":
    import uvicorn

    print(f"[Orchestrator] key {'present' if config.GOOGLE_API_KEY else 'MISSING'}; "
          f"model={config.GEMINI_LIVE_MODEL} port={config.SERVER_PORT}", flush=True)
    uvicorn.run("server:app", host=config.SERVER_HOST, port=config.SERVER_PORT, log_level="info")
