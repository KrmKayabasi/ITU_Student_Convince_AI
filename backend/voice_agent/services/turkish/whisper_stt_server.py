import asyncio
import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass

import msgpack
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from unmute.kyutai_constants import FRAME_TIME_SEC, SAMPLE_RATE

logger = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))

SPEECH_TO_TEXT_PATH = "/api/asr-streaming"
TARGET_SAMPLE_RATE = 16000

app = FastAPI()


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def pack_message(message: dict) -> bytes:
    return msgpack.packb(message, use_bin_type=True, use_single_float=True)


def resample_linear(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate:
        return audio.astype(np.float32, copy=False)
    if audio.size == 0:
        return audio.astype(np.float32, copy=False)

    duration = audio.size / src_rate
    dst_size = max(1, int(round(duration * dst_rate)))
    src_x = np.linspace(0.0, duration, audio.size, endpoint=False)
    dst_x = np.linspace(0.0, duration, dst_size, endpoint=False)
    return np.interp(dst_x, src_x, audio).astype(np.float32)


@dataclass
class TranscribedWord:
    text: str
    start_time: float


class FasterWhisperTranscriber:
    def __init__(self) -> None:
        from faster_whisper import WhisperModel  # pyright: ignore[reportMissingImports]

        self.whisper_model_cls = WhisperModel
        self.model_name = os.environ.get("WHISPER_MODEL", "openai/whisper-large-v3-turbo")
        import torch
        default_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = os.environ.get("WHISPER_DEVICE", default_device)
        if self.device == "cpu":
            self.compute_type = os.environ.get("WHISPER_CPU_COMPUTE_TYPE", "int8")
        else:
            self.compute_type = os.environ.get("WHISPER_COMPUTE_TYPE", "int8_float16")
        self.fallback_to_cpu = os.environ.get("WHISPER_FALLBACK_TO_CPU", "true").lower() in {
            "1",
            "true",
            "yes",
        }
        self.cpu_threads = _env_int("WHISPER_CPU_THREADS", 4)
        self.num_workers = _env_int("WHISPER_NUM_WORKERS", 1)
        self.model_lock = threading.Lock()
        self.language = os.environ.get("WHISPER_LANGUAGE", "tr")
        self.beam_size = _env_int("WHISPER_BEAM_SIZE", 1)
        self.temperature = _env_float("WHISPER_TEMPERATURE", 0.0)

        try:
            self.model = self._load_model(self.device, self.compute_type)
        except RuntimeError:
            if self.device == "cpu" or not self.fallback_to_cpu:
                raise
            logger.exception("CUDA Whisper initialization failed; falling back to CPU")
            self._fallback_to_cpu()
        self._warmup()

    def _load_model(self, device: str, compute_type: str):
        logger.info(
            "Loading faster-whisper model=%s device=%s compute_type=%s",
            self.model_name,
            device,
            compute_type,
        )
        return self.whisper_model_cls(
            self.model_name,
            device=device,
            compute_type=compute_type,
            cpu_threads=self.cpu_threads,
            num_workers=self.num_workers,
        )

    def _fallback_to_cpu(self) -> None:
        self.device = "cpu"
        self.compute_type = os.environ.get("WHISPER_CPU_COMPUTE_TYPE", "int8")
        self.model = self._load_model(self.device, self.compute_type)

    def _warmup(self) -> None:
        warmup_sec = _env_float("WHISPER_WARMUP_SEC", 1.0)
        if warmup_sec <= 0:
            return
        audio = np.zeros(int(TARGET_SAMPLE_RATE * warmup_sec), dtype=np.float32)
        t0 = time.perf_counter()
        segments, _info = self.model.transcribe(
            audio,
            language=self.language,
            beam_size=1,
            temperature=0.0,
            vad_filter=False,
            word_timestamps=False,
            condition_on_previous_text=False,
            without_timestamps=True,
        )
        list(segments)
        logger.info("Whisper warmup finished in %.3fs", time.perf_counter() - t0)

    def transcribe(self, audio_24k: np.ndarray, start_time: float) -> list[TranscribedWord]:
        try:
            return self._transcribe_once(audio_24k, start_time)
        except RuntimeError:
            if self.device == "cpu" or not self.fallback_to_cpu:
                raise
            logger.exception("CUDA Whisper transcription failed; falling back to CPU")
            with self.model_lock:
                if self.device != "cpu":
                    self._fallback_to_cpu()
            return self._transcribe_once(audio_24k, start_time)

    def _transcribe_once(self, audio_24k: np.ndarray, start_time: float) -> list[TranscribedWord]:
        audio_16k = resample_linear(audio_24k, SAMPLE_RATE, TARGET_SAMPLE_RATE)
        t0 = time.perf_counter()
        segments, info = self.model.transcribe(
            audio_16k,
            language=self.language,
            beam_size=self.beam_size,
            temperature=self.temperature,
            vad_filter=False,
            word_timestamps=True,
            condition_on_previous_text=False,
            without_timestamps=False,
        )

        words: list[TranscribedWord] = []
        fallback_text_parts: list[str] = []
        for segment in segments:
            fallback_text_parts.append(segment.text.strip())
            segment_words = getattr(segment, "words", None) or []
            for word in segment_words:
                text = word.word.strip()
                if text:
                    words.append(
                        TranscribedWord(
                            text=text,
                            start_time=start_time + max(0.0, float(word.start or 0.0)),
                        )
                    )

        if not words:
            # Some faster-whisper builds/models may not return word-level timestamps.
            text = " ".join(part for part in fallback_text_parts if part).strip()
            words = [
                TranscribedWord(text=word, start_time=start_time)
                for word in text.split()
            ]

        elapsed = time.perf_counter() - t0
        audio_duration = audio_24k.size / SAMPLE_RATE
        logger.info(
            "Transcribed %.2fs audio in %.3fs, rtf=%.3f, language=%s, prob=%.2f, text=%r",
            audio_duration,
            elapsed,
            elapsed / max(audio_duration, 1e-6),
            getattr(info, "language", None),
            getattr(info, "language_probability", 0.0),
            " ".join(word.text for word in words),
        )
        return words


class UtteranceBuffer:
    """Energy-based VAD with an adaptive noise floor and hysteresis.

    Speech detection uses two thresholds derived from a running estimate of the
    background noise level (``noise_floor``):

      * enter speech when ``rms >= max(speech_rms_threshold, noise_floor * speech_factor)``
      * exit speech  when ``rms <  max(speech_rms_threshold, noise_floor * speech_off_factor)``

    The ``speech_off_factor`` is lower than ``speech_factor``, so once speech has
    started brief dips in energy don't fragment the utterance, and brief noise
    spikes in silence don't start a false one. The noise floor is updated only
    while we believe we're in silence, so sustained speech doesn't raise it.

    A small pre-roll ring of recent silence frames is prepended to each captured
    utterance so word onsets reach Whisper intact. Pre-roll never changes when an
    utterance ends; it only adds lead-in audio.
    """

    def __init__(self) -> None:
        # Fixed floor (absolute RMS). Effective threshold is max of this and the
        # adaptive floor, so this acts as a minimum in a very quiet room.
        self.speech_threshold = _env_float("STT_SPEECH_RMS_THRESHOLD", 0.008)
        self.end_silence_sec = _env_float("STT_END_SILENCE_SEC", 0.55)
        self.min_speech_sec = _env_float("STT_MIN_SPEECH_SEC", 0.25)
        self.max_utterance_sec = _env_float("STT_MAX_UTTERANCE_SEC", 20.0)

        # Adaptive noise floor: EMA of RMS over silence frames.
        # alpha is the weight given to the new sample (small = slow tracking).
        self.noise_floor_alpha = _env_float("STT_NOISE_FLOOR_ALPHA", 0.02)
        self.speech_factor = _env_float("STT_SPEECH_FACTOR", 3.0)
        self.speech_off_factor = _env_float("STT_SPEECH_OFF_FACTOR", 1.5)

        # Pre-roll: keep the last N seconds of frames to prepend on speech onset.
        preroll_sec = _env_float("STT_PREROLL_SEC", 0.2)
        preroll_frames = int(round(preroll_sec / FRAME_TIME_SEC)) if preroll_sec > 0 else 0
        self.preroll: deque[np.ndarray] = deque(maxlen=max(preroll_frames, 0))

        # Noise floor estimate. Initialized to the fixed threshold so the first
        # frames in a quiet room don't instantly count as speech; it adapts down
        # to the real background level as silence frames arrive.
        self.noise_floor = self.speech_threshold

        self.in_speech = False
        self.start_time = 0.0
        self.speech_sec = 0.0
        self.silence_sec = 0.0
        self.frames: list[np.ndarray] = []

    def _enter_threshold(self) -> float:
        return max(self.speech_threshold, self.noise_floor * self.speech_factor)

    def _exit_threshold(self) -> float:
        return max(self.speech_threshold, self.noise_floor * self.speech_off_factor)

    def accept_frame(self, frame: np.ndarray, frame_start_time: float) -> tuple[np.ndarray | None, float, float]:
        rms = float(np.sqrt(np.mean(frame * frame))) if frame.size else 0.0

        if not self.in_speech:
            # Track background noise only in silence. Use a floor so a single
            # loud transient doesn't yank the estimate up too far.
            self.noise_floor = (1.0 - self.noise_floor_alpha) * self.noise_floor + (
                self.noise_floor_alpha * rms
            )
            is_speech = rms >= self._enter_threshold()
        else:
            is_speech = rms >= self._exit_threshold()

        if is_speech and not self.in_speech:
            self.in_speech = True
            self.start_time = frame_start_time
            self.speech_sec = 0.0
            self.silence_sec = 0.0
            # Seed with pre-roll so onsets are preserved. Copy out and clear so
            # we don't double-count if accept_frame is ever re-entered.
            self.frames = list(self.preroll)

        utterance: np.ndarray | None = None
        utterance_start = self.start_time

        if self.in_speech:
            self.frames.append(frame.copy())
            if is_speech:
                self.speech_sec += FRAME_TIME_SEC
                self.silence_sec = 0.0
            else:
                self.silence_sec += FRAME_TIME_SEC

            too_much_silence = self.silence_sec >= self.end_silence_sec
            too_long = (len(self.frames) * FRAME_TIME_SEC) >= self.max_utterance_sec
            if too_much_silence or too_long:
                if self.speech_sec >= self.min_speech_sec:
                    utterance = np.concatenate(self.frames)
                self.in_speech = False
                self.speech_sec = 0.0
                self.silence_sec = 0.0
                self.frames = []

        # Maintain the pre-roll ring (only meaningful while in silence).
        if not self.in_speech and self.preroll.maxlen:
            self.preroll.append(frame.copy())

        pause_probability = 1.0 if not self.in_speech or self.silence_sec >= self.end_silence_sec else 0.0
        return utterance, utterance_start, pause_probability


@app.get("/api/build_info")
def get_build_info() -> dict[str, str]:
    return {
        "service": "turkish-whisper-stt",
        "model": os.environ.get("WHISPER_MODEL", "openai/whisper-large-v3-turbo"),
    }


@app.websocket(SPEECH_TO_TEXT_PATH)
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    await websocket.send_bytes(pack_message({"type": "Ready"}))

    transcriber = app.state.transcriber
    buffer = UtteranceBuffer()
    send_lock = asyncio.Lock()
    transcription_tasks: set[asyncio.Task[None]] = set()
    sent_samples = 0

    async def send_words(audio: np.ndarray, utterance_start: float) -> None:
        try:
            words = await asyncio.to_thread(transcriber.transcribe, audio, utterance_start)
        except RuntimeError:
            logger.exception("STT transcription failed")
            return
        async with send_lock:
            for word in words:
                await websocket.send_bytes(
                    pack_message(
                        {
                            "type": "Word",
                            "text": word.text,
                            "start_time": word.start_time,
                        }
                    )
                )

    try:
        while True:
            data = await websocket.receive_bytes()
            message = msgpack.unpackb(data, raw=False)
            message_type = message.get("type")

            if message_type == "Audio":
                frame = np.asarray(message["pcm"], dtype=np.float32)
                frame_start_time = sent_samples / SAMPLE_RATE
                sent_samples += frame.size

                utterance, utterance_start, pause_probability = buffer.accept_frame(
                    frame, frame_start_time
                )

                async with send_lock:
                    await websocket.send_bytes(
                        pack_message(
                            {
                                "type": "Step",
                                "step_idx": int(round(frame_start_time / FRAME_TIME_SEC)),
                                "prs": [0.0, 0.0, pause_probability],
                            }
                        )
                    )

                if utterance is not None:
                    task = asyncio.create_task(send_words(utterance, utterance_start))
                    transcription_tasks.add(task)
                    task.add_done_callback(transcription_tasks.discard)

            elif message_type == "Marker":
                async with send_lock:
                    await websocket.send_bytes(
                        pack_message({"type": "Marker", "id": message.get("id", 0)})
                    )
            else:
                logger.warning("Ignoring unknown STT message: %s", message_type)
    except WebSocketDisconnect:
        logger.info("STT client disconnected")
    finally:
        for task in transcription_tasks:
            task.cancel()


@app.on_event("startup")  # pyright: ignore[reportDeprecated]
def load_model() -> None:
    app.state.transcriber = FasterWhisperTranscriber()
