"""
CV -> LLM injection.

Per visitor, the orchestrator runs one CvInjector that subscribes *server-side*
to the CV pipeline's WebSocket channels for this session_id and feeds two kinds
of context into the live Gemini session:

  (a) one-shot  /profile  -> a Turkish opening hint, delivered once so the
      assistant greets in a way that fits the student's first impression.
  (b) continuous /focus    -> when the student's attention is lost for a
      sustained window, a debounced re-engage nudge (voice via bridge.steer +
      an avatar {"type":"seekAttention"} downlink so face and voice act together).

The CV pipeline itself is unchanged; multiple subscribers per session_id are
supported there (subscriber sets), so the browser and this injector can both
listen to the same session.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Awaitable, Callable, Optional

import websockets

import config
from cv_hints import build_opening_hint  # opener text formatting

logger = logging.getLogger("orchestrator.cv")

SendJson = Callable[[dict], Awaitable[None]]


def _cv_url(channel: str, session_id: str) -> str:
    base = config.CV_PIPELINE_WS_URL.rstrip("/")
    url = f"{base}/{channel}/{session_id}"
    if config.CV_PIPELINE_TOKEN:
        url += f"?token={config.CV_PIPELINE_TOKEN}"
    return url


# Escalating Turkish re-engage nudges (plain text; steer, not a user turn).
_NUDGE_TEXTS = [
    "Öğrenci dikkatini biraz kaybetmiş görünüyor; sıcak ve kısa bir soruyla "
    "yeniden ilgisini çek, enerjini yükselt.",
    "Öğrencinin dikkati hâlâ dağınık; onu ismiyle ya da ilgi alanına dokunan "
    "merak uyandıran kısa bir cümleyle geri kazan.",
    "Öğrenci uzaklaşıyor olabilir; konuyu onun hedefiyle bağlayan çok kısa, "
    "canlı bir cümle söyle ve bir soru sor.",
]


class CvInjector:
    def __init__(self, *, session_id: str, bridge: Any, send_json: SendJson) -> None:
        self.session_id = session_id
        self.bridge = bridge
        self.send_json = send_json

        self._profile_future: "Optional[asyncio.Future[dict]]" = None
        self._opened = False
        self._closing = False

        # focus debounce state
        self._distracted_since: Optional[float] = None
        self._last_nudge_at: float = 0.0
        self._nudge_count = 0

        self._tasks: list[asyncio.Task] = []

    async def run(self) -> None:
        self._profile_future = asyncio.get_running_loop().create_future()
        self._tasks = [
            asyncio.create_task(self._profile_loop(), name="cv_profile"),
            asyncio.create_task(self._focus_loop(), name="cv_focus"),
            asyncio.create_task(self._opener(), name="cv_opener"),
        ]
        try:
            await asyncio.gather(*self._tasks)
        except asyncio.CancelledError:
            pass

    async def close(self) -> None:
        self._closing = True
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

    # ── (a) one-shot profile -> opening hint ──────────────────────────────────
    async def _profile_loop(self) -> None:
        url = _cv_url("profile", self.session_id)
        while not self._closing:
            try:
                async with websockets.connect(url, ping_interval=20) as ws:
                    async for msg in ws:
                        try:
                            profile = json.loads(msg)
                        except Exception:
                            continue
                        if not self._profile_future.done():
                            self._profile_future.set_result(profile)
                        return  # one-shot; stop after the first profile
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.debug("profile subscribe retry (%s): %s", self.session_id, e)
                await asyncio.sleep(1.0)

    async def _opener(self) -> None:
        """Open the conversation once: profile-informed if it arrives in time,
        otherwise a generic warm greeting after PROFILE_WAIT_SECONDS."""
        profile: Optional[dict] = None
        try:
            profile = await asyncio.wait_for(
                asyncio.shield(self._profile_future), timeout=config.PROFILE_WAIT_SECONDS
            )
        except (asyncio.TimeoutError, TimeoutError):
            profile = None
        except asyncio.CancelledError:
            return

        if self._opened or self._closing:
            return
        self._opened = True

        hint = build_opening_hint(profile)
        hint = config.strip_markdown_for_llm(hint)
        logger.info("opener (%s): %s", self.session_id, hint)
        try:
            # Force exactly one opening turn so we don't double-greet with
            # proactive audio.
            await self.bridge.inject_context(hint, turn_complete=True)
        except Exception:
            logger.exception("failed to inject opening hint")

    # ── (b) continuous focus -> debounced re-engage nudge ─────────────────────
    async def _focus_loop(self) -> None:
        url = _cv_url("focus", self.session_id)
        while not self._closing:
            try:
                async with websockets.connect(url, ping_interval=20) as ws:
                    async for msg in ws:
                        try:
                            payload = json.loads(msg)
                        except Exception:
                            continue
                        await self._on_focus(payload)
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.debug("focus subscribe retry (%s): %s", self.session_id, e)
                await asyncio.sleep(1.0)

    async def _on_focus(self, payload: dict) -> None:
        is_focused = bool(payload.get("is_focused"))
        now = time.monotonic()

        if is_focused:
            self._distracted_since = None
            return

        # Don't nudge before the assistant has even opened.
        if not self._opened:
            return

        if self._distracted_since is None:
            self._distracted_since = now
            return

        distracted_for = now - self._distracted_since
        if distracted_for < config.FOCUS_LOSS_SECONDS:
            return
        if (now - self._last_nudge_at) < config.NUDGE_COOLDOWN_SECONDS:
            return

        await self._fire_nudge()
        self._last_nudge_at = now

    async def _fire_nudge(self) -> None:
        text = _NUDGE_TEXTS[min(self._nudge_count, len(_NUDGE_TEXTS) - 1)]
        self._nudge_count += 1
        logger.info("focus nudge #%d (%s)", self._nudge_count, self.session_id)
        # Fire the avatar attention-grab and the verbal steer together.
        try:
            await self.send_json({"type": "seekAttention"})
        except Exception:
            logger.debug("failed to send seekAttention", exc_info=True)
        try:
            await self.bridge.steer(config.strip_markdown_for_llm(text))
        except Exception:
            logger.exception("failed to steer for focus nudge")
