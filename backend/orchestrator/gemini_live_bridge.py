"""
Gemini Live (native-audio) bridge.

Owns ONE persistent, full-duplex Gemini Live session for a single kiosk visitor.
Unlike the turn-based OpenAI bridge on `origin/gpt-realtime` (which this mirrors
in spirit), this is continuous: the orchestrator pumps microphone PCM in and
consumes model audio out concurrently, letting Gemini's native VAD + barge-in
drive turn-taking. Affective dialog + proactive audio are enabled via the
v1alpha API surface.

Contract:
  in  : 16 kHz little-endian PCM16 mono  -> session.send_realtime_input(audio=Blob)
  out : 24 kHz little-endian PCM16 mono  <- server_content.model_turn.parts[].inline_data.data

The bridge exposes provider-agnostic hooks the CV injector uses:
  - inject_context(text) : seed one-shot context (the CV profile opener)
  - steer(text)          : mid-session nudge (the focus re-engage hint)
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator, Optional

logger = logging.getLogger("orchestrator.gemini")


class GeminiLiveBridge:
    """Persistent Gemini Live native-audio session for one visitor."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        voice: str,
        instructions: str,
        api_version: str = "v1alpha",
        input_sample_rate: int = 16000,
        output_sample_rate: int = 24000,
        enable_affective_dialog: bool = True,
        enable_proactive_audio: bool = True,
    ) -> None:
        if not api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY is required to start the Gemini Live orchestrator"
            )
        self.api_key = api_key
        self.model = model
        self.voice = voice
        self.instructions = instructions
        self.api_version = api_version
        self.input_sample_rate = input_sample_rate
        # `sample_rate` = output rate, mirroring the OpenAI bridge attribute the
        # server reports to the client so the playback stream is sized correctly.
        self.sample_rate = output_sample_rate

        self.enable_affective_dialog = enable_affective_dialog
        self.enable_proactive_audio = enable_proactive_audio

        self._client: Any = None
        self._cm: Any = None            # the live.connect() async context manager
        self._session: Any = None
        self._lock = asyncio.Lock()
        self._closed = False
        # Session resumption: Gemini live sessions have hard duration limits;
        # the server sends periodic resumption handles (and a GoAway shortly
        # before disconnect) so we can transparently reconnect mid-conversation.
        self._resumption_handle: Optional[str] = None

        self._last_user_text = ""
        self._last_assistant_text = ""
        self._user_parts: list[str] = []
        self._assistant_parts: list[str] = []

    # ── config ────────────────────────────────────────────────────────────────
    def _build_config(self) -> Any:
        from google.genai import types

        speech_config = types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=self.voice)
            )
        )

        kwargs: dict[str, Any] = dict(
            response_modalities=["AUDIO"],
            system_instruction=self.instructions,
            speech_config=speech_config,
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
            realtime_input_config=types.RealtimeInputConfig(
                automatic_activity_detection=types.AutomaticActivityDetection(
                    disabled=False,
                    # Local VAD is the first line of defense. Keep Gemini's
                    # detector conservative as well so brief leaked background
                    # speech does not start a turn or interrupt model audio.
                    start_of_speech_sensitivity=(
                        types.StartSensitivity.START_SENSITIVITY_LOW
                    ),
                    end_of_speech_sensitivity=(
                        types.EndSensitivity.END_SENSITIVITY_HIGH
                    ),
                    prefix_padding_ms=160,
                    silence_duration_ms=500,
                )
            ),
            # Long-conversation sustainability: sliding-window compression lifts
            # the session context limit, and resumption lets us reconnect when
            # the server rotates the connection (GoAway / duration limit).
            context_window_compression=types.ContextWindowCompressionConfig(
                sliding_window=types.SlidingWindow(),
            ),
            session_resumption=types.SessionResumptionConfig(
                handle=self._resumption_handle,
            ),
        )
        # v1alpha-only knobs — guarded so a stricter SDK/model still connects.
        if self.enable_affective_dialog:
            kwargs["enable_affective_dialog"] = True
        if self.enable_proactive_audio:
            kwargs["proactivity"] = types.ProactivityConfig(proactive_audio=True)

        return types.LiveConnectConfig(**kwargs)

    # ── lifecycle ─────────────────────────────────────────────────────────────
    async def connect(self) -> None:
        """Open the Gemini Live session (idempotent)."""
        async with self._lock:
            if self._session is not None:
                return
            self._closed = False
            from google import genai

            if self._client is None:
                self._client = genai.Client(
                    api_key=self.api_key,
                    http_options={"api_version": self.api_version},
                )
            config = self._build_config()
            self._cm = self._client.aio.live.connect(model=self.model, config=config)
            self._session = await self._cm.__aenter__()
            logger.info("Gemini Live session opened (model=%s)", self.model)

    async def _reconnect(self) -> bool:
        """Re-open the session after a drop, resuming via the last handle."""
        async with self._lock:
            await self._close_locked()
            if self._closed:
                return False
            try:
                config = self._build_config()
                self._cm = self._client.aio.live.connect(model=self.model, config=config)
                self._session = await self._cm.__aenter__()
                logger.info(
                    "Gemini Live session resumed (handle=%s)",
                    "yes" if self._resumption_handle else "none",
                )
                return True
            except Exception:
                logger.exception("Gemini Live reconnect failed")
                return False

    async def ensure_ready(self) -> None:
        await self.connect()

    async def aclose(self) -> None:
        async with self._lock:
            self._closed = True
            await self._close_locked()

    async def _close_locked(self) -> None:
        if self._cm is not None:
            try:
                await self._cm.__aexit__(None, None, None)
            except Exception:
                logger.debug("error closing Gemini session", exc_info=True)
        self._cm = None
        self._session = None

    async def reset(self) -> None:
        self._last_user_text = ""
        self._last_assistant_text = ""
        self._user_parts.clear()
        self._assistant_parts.clear()
        await self.aclose()

    def get_last_turn(self) -> dict[str, str]:
        return {"user": self._last_user_text, "assistant": self._last_assistant_text}

    # ── uplink (browser -> Gemini) ────────────────────────────────────────────
    async def send_audio(self, pcm16: bytes) -> None:
        """Feed one chunk of 16 kHz PCM16 mic audio to the model."""
        if self._session is None or not pcm16:
            return
        from google.genai import types

        await self._session.send_realtime_input(
            audio=types.Blob(
                data=pcm16,
                mime_type=f"audio/pcm;rate={self.input_sample_rate}",
            )
        )

    async def steer(self, text: str) -> None:
        """Inject a mid-session text nudge that steers the live audio stream.

        Used for the continuous focus re-engagement hint. This is *not* a user
        turn; it biases the model's next spoken output.
        """
        if self._session is None or not text:
            return
        await self._session.send_realtime_input(text=text)

    async def inject_context(self, text: str, *, turn_complete: bool = False) -> None:
        """Seed one-shot conversational context (the CV profile opener).

        Delivered as structured client content so it becomes part of the session
        history rather than a transient bias. With turn_complete=False it seeds
        context; proactive audio / a follow-up steer prompts the actual greeting.
        """
        if self._session is None or not text:
            return
        from google.genai import types

        await self._session.send_client_content(
            turns=types.Content(role="user", parts=[types.Part(text=text)]),
            turn_complete=turn_complete,
        )

    # ── downlink (Gemini -> browser) ──────────────────────────────────────────
    async def receive(self) -> AsyncIterator[dict[str, Any]]:
        """Yield normalized events from the live session, across ALL turns.

        IMPORTANT: the SDK's session.receive() iterator ENDS at each
        turn_complete (it represents one model turn), so we loop it forever —
        otherwise the conversation dies after the first answer. On transport
        drops (GoAway / duration limit) we transparently reconnect using the
        last session-resumption handle and keep listening.

        Event shapes:
          {"type": "audio", "pcm16": bytes}            # 24 kHz PCM16 model audio
          {"type": "input_transcript", "text": str}    # what the user said
          {"type": "output_transcript", "text": str}   # what the model said
          {"type": "interrupted"}                       # barge-in
          {"type": "turn_complete"}                     # model finished a turn
        """
        if self._session is None:
            raise RuntimeError("Gemini Live session is not connected")

        while not self._closed:
            try:
                async for response in self._session.receive():
                    for event in self._normalize(response):
                        yield event
                # Iterator ended = one turn finished; loop to await the next.
            except asyncio.CancelledError:
                raise
            except Exception as e:
                if self._closed:
                    break
                logger.warning("live stream dropped (%s: %s); reconnecting...",
                               type(e).__name__, e)
                if not await self._reconnect():
                    raise

    def _normalize(self, response: Any) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []

        # Track the resumption handle so reconnects keep the conversation.
        sru = getattr(response, "session_resumption_update", None)
        if sru is not None and getattr(sru, "resumable", False):
            handle = getattr(sru, "new_handle", None)
            if handle:
                self._resumption_handle = handle

        go_away = getattr(response, "go_away", None)
        if go_away is not None:
            logger.info("GoAway received (time_left=%s); will resume on drop",
                        getattr(go_away, "time_left", "?"))

        sc = getattr(response, "server_content", None)

        # Audio: prefer the explicit inline_data path; fall back to response.data.
        emitted_audio = False
        model_turn = getattr(sc, "model_turn", None) if sc else None
        if model_turn is not None:
            for part in getattr(model_turn, "parts", None) or []:
                inline = getattr(part, "inline_data", None)
                data = getattr(inline, "data", None) if inline else None
                if data:
                    events.append({"type": "audio", "pcm16": data})
                    emitted_audio = True
        if not emitted_audio:
            data = getattr(response, "data", None)
            if data:
                events.append({"type": "audio", "pcm16": data})

        if sc is not None:
            in_tx = getattr(sc, "input_transcription", None)
            in_text = getattr(in_tx, "text", None) if in_tx else None
            if in_text:
                self._user_parts.append(in_text)
                events.append({"type": "input_transcript", "text": in_text})

            out_tx = getattr(sc, "output_transcription", None)
            out_text = getattr(out_tx, "text", None) if out_tx else None
            if out_text:
                self._assistant_parts.append(out_text)
                events.append({"type": "output_transcript", "text": out_text})

            if getattr(sc, "interrupted", None):
                # Barge-in: drop the assistant text accumulated for this turn.
                self._assistant_parts.clear()
                events.append({"type": "interrupted"})

            if getattr(sc, "turn_complete", None):
                self._commit_turn()
                events.append({"type": "turn_complete"})

        return events

    def _commit_turn(self) -> None:
        if self._user_parts:
            self._last_user_text = "".join(self._user_parts).strip()
            self._user_parts.clear()
        if self._assistant_parts:
            self._last_assistant_text = "".join(self._assistant_parts).strip()
            self._assistant_parts.clear()

    @property
    def connected(self) -> bool:
        return self._session is not None
