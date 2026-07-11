from __future__ import annotations

import asyncio
import base64
import json
import math
from typing import Any, AsyncIterator

import numpy as np


OPENAI_REALTIME_URL = "wss://api.openai.com/v1/realtime"


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
    clipped = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    return pcm.tobytes()


def pcm16_bytes_to_float32_bytes(audio_bytes: bytes) -> bytes:
    if not audio_bytes:
        return b""
    pcm = np.frombuffer(audio_bytes, dtype="<i2")
    float_audio = (pcm.astype(np.float32) / 32768.0).astype(np.float32)
    return float_audio.tobytes()


def _extract_response_transcript(event: dict[str, Any]) -> str:
    response = event.get("response") or {}
    output = response.get("output") or []
    parts: list[str] = []
    for item in output:
        for content in item.get("content") or []:
            text = content.get("transcript") or content.get("text")
            if text:
                parts.append(str(text))
    return " ".join(parts).strip()


class OpenAIRealtimeBridge:
    """Persistent Realtime API bridge for the existing HTTP speech contract."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        voice: str,
        instructions: str,
        transcription_model: str = "gpt-realtime-whisper",
        language: str = "tr",
        input_sample_rate: int = 16000,
        realtime_sample_rate: int = 24000,
    ) -> None:
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required when SPEECH_PROVIDER=openai_realtime")

        self.api_key = api_key
        self.model = model
        self.voice = voice
        self.instructions = instructions
        self.transcription_model = transcription_model
        self.language = language
        self.input_sample_rate = input_sample_rate
        self.sample_rate = realtime_sample_rate

        self._ws: Any | None = None
        self._lock = asyncio.Lock()
        self._last_user_text = ""
        self._last_assistant_text = ""

    async def ensure_ready(self) -> None:
        async with self._lock:
            await self._ensure_connected()

    def get_last_turn(self) -> dict[str, str]:
        return {"user": self._last_user_text, "assistant": self._last_assistant_text}

    async def reset(self) -> None:
        async with self._lock:
            self._last_user_text = ""
            self._last_assistant_text = ""
            await self._close_ws()

    async def stream_turn(self, audio_data: np.ndarray) -> AsyncIterator[bytes]:
        async with self._lock:
            await self._ensure_connected()
            user_text = ""
            assistant_parts: list[str] = []

            audio_24k = resample_mono_float32(
                audio_data,
                self.input_sample_rate,
                self.sample_rate,
            )
            pcm16 = float32_to_pcm16_bytes(audio_24k)
            audio_b64 = base64.b64encode(pcm16).decode("ascii")

            await self._send_json({"type": "input_audio_buffer.append", "audio": audio_b64})
            await self._send_json({"type": "input_audio_buffer.commit"})
            await self._send_json({"type": "response.create"})

            response_complete = False
            try:
                while True:
                    event = await self._recv_json()
                    event_type = event.get("type", "")

                    if event_type == "error":
                        raise RuntimeError(f"OpenAI Realtime error: {event.get('error', event)}")

                    if event_type == "conversation.item.input_audio_transcription.completed":
                        user_text = str(event.get("transcript") or "").strip()
                        continue

                    if event_type in {
                        "response.audio_transcript.delta",
                        "response.output_audio_transcript.delta",
                        "response.output_text.delta",
                    }:
                        delta = event.get("delta")
                        if delta:
                            assistant_parts.append(str(delta))
                        continue

                    if event_type in {
                        "response.audio_transcript.done",
                        "response.output_audio_transcript.done",
                        "response.output_text.done",
                    }:
                        transcript = event.get("transcript") or event.get("text")
                        if transcript:
                            assistant_parts = [str(transcript)]
                        continue

                    if event_type in {
                        "response.audio.delta",
                        "response.output_audio.delta",
                        "session.output_audio.delta",
                    }:
                        delta = event.get("delta")
                        if delta:
                            yield pcm16_bytes_to_float32_bytes(base64.b64decode(delta))
                        continue

                    if event_type == "response.done":
                        done_transcript = _extract_response_transcript(event)
                        if done_transcript:
                            assistant_parts = [done_transcript]
                        response_complete = True
                        break

            except asyncio.CancelledError:
                await self._cancel_response()
                raise
            finally:
                if response_complete:
                    self._last_user_text = user_text
                    self._last_assistant_text = "".join(assistant_parts).strip()

    async def _ensure_connected(self) -> None:
        if self._ws_is_open():
            return

        import websockets

        url = f"{OPENAI_REALTIME_URL}?model={self.model}"
        headers = [
            ("Authorization", f"Bearer {self.api_key}"),
        ]

        try:
            self._ws = await websockets.connect(
                url,
                additional_headers=headers,
                ping_interval=20,
                max_size=32 * 1024 * 1024,
            )
        except TypeError:
            self._ws = await websockets.connect(
                url,
                extra_headers=headers,
                ping_interval=20,
                max_size=32 * 1024 * 1024,
            )

        try:
            await self._configure_session()
        except Exception:
            await self._close_ws()
            raise

    async def _configure_session(self) -> None:
        input_audio: dict[str, Any] = {
            "format": {"type": "audio/pcm", "rate": self.sample_rate},
            "turn_detection": None,
        }
        if self.transcription_model:
            input_audio["transcription"] = {
                "model": self.transcription_model,
                "language": self.language,
            }

        await self._send_json(
            {
                "type": "session.update",
                "session": {
                    "type": "realtime",
                    "model": self.model,
                    "output_modalities": ["audio"],
                    "instructions": self.instructions,
                    "audio": {
                        "input": input_audio,
                        "output": {
                            "format": {"type": "audio/pcm", "rate": self.sample_rate},
                            "voice": self.voice,
                        },
                    },
                },
            }
        )

        while True:
            event = await asyncio.wait_for(self._recv_json(), timeout=10.0)
            event_type = event.get("type", "")
            if event_type == "error":
                raise RuntimeError(f"OpenAI Realtime session error: {event.get('error', event)}")
            if event_type == "session.updated":
                return

    async def _send_json(self, event: dict[str, Any]) -> None:
        if self._ws is None:
            raise RuntimeError("OpenAI Realtime websocket is not connected")
        await self._ws.send(json.dumps(event))

    async def _recv_json(self) -> dict[str, Any]:
        if self._ws is None:
            raise RuntimeError("OpenAI Realtime websocket is not connected")
        message = await self._ws.recv()
        return json.loads(message)

    async def _cancel_response(self) -> None:
        try:
            if self._ws_is_open():
                await self._send_json({"type": "response.cancel"})
        except Exception:
            pass

    def _ws_is_open(self) -> bool:
        if self._ws is None:
            return False
        if getattr(self._ws, "closed", False):
            return False
        if getattr(self._ws, "close_code", None) is not None:
            return False
        return True

    async def _close_ws(self) -> None:
        if self._ws is None:
            return
        try:
            await self._ws.close()
        finally:
            self._ws = None
