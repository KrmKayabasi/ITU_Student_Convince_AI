import asyncio
import logging
import math
import os
import tarfile
import time
import urllib.request
from pathlib import Path
from typing import Protocol

import msgpack
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from unmute.kyutai_constants import SAMPLE_RATE, SAMPLES_PER_FRAME

logger = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))

TEXT_TO_SPEECH_PATH = "/api/tts_streaming"
DEFAULT_SHERPA_MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/"
    "vits-piper-tr_TR-dfki-medium.tar.bz2"
)

app = FastAPI()


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


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


def chunk_audio(audio: np.ndarray, chunk_size: int = SAMPLES_PER_FRAME) -> list[np.ndarray]:
    if audio.size == 0:
        return []
    return [audio[i : i + chunk_size] for i in range(0, audio.size, chunk_size)]


def safe_extract_tar(tar: tarfile.TarFile, destination: Path) -> None:
    destination_resolved = destination.resolve()
    for member in tar.getmembers():
        target = (destination / member.name).resolve()
        if not str(target).startswith(str(destination_resolved)):
            raise RuntimeError(f"Unsafe tar member path: {member.name}")
    tar.extractall(destination)


def ensure_sherpa_model() -> Path:
    model_dir = Path(os.environ.get("SHERPA_TTS_MODEL_DIR", "/models/sherpa-tts"))
    model_dir.mkdir(parents=True, exist_ok=True)
    if list(model_dir.rglob("*.onnx")):
        return model_dir

    url = os.environ.get("SHERPA_TTS_MODEL_URL", DEFAULT_SHERPA_MODEL_URL)
    archive_path = model_dir / Path(url).name
    logger.info("Downloading sherpa-onnx TTS model from %s", url)
    urllib.request.urlretrieve(url, archive_path)
    with tarfile.open(archive_path) as tar:
        safe_extract_tar(tar, model_dir)
    archive_path.unlink(missing_ok=True)
    return model_dir


def optional_path_env(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value) if value else None


class TtsBackend(Protocol):
    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        ...


class SherpaOnnxBackend:
    def __init__(self) -> None:
        import sherpa_onnx  # pyright: ignore[reportMissingImports]

        model_root = ensure_sherpa_model()
        onnx_files = sorted(model_root.rglob("*.onnx"), key=lambda path: path.stat().st_size)
        if not onnx_files:
            raise FileNotFoundError(f"No .onnx model found in {model_root}")

        model = optional_path_env("SHERPA_TTS_MODEL") or onnx_files[-1]
        tokens = optional_path_env("SHERPA_TTS_TOKENS")
        if tokens is None:
            token_matches = list(model_root.rglob("tokens.txt"))
            if not token_matches:
                raise FileNotFoundError(f"No tokens.txt found in {model_root}")
            tokens = token_matches[0]

        data_dir = optional_path_env("SHERPA_TTS_DATA_DIR")
        data_dir_value = str(data_dir) if data_dir is not None else ""
        if not data_dir_value:
            data_dirs = [path for path in model_root.rglob("espeak-ng-data") if path.is_dir()]
            data_dir_value = str(data_dirs[0]) if data_dirs else ""

        lexicon = optional_path_env("SHERPA_TTS_LEXICON")
        lexicon_value = str(lexicon) if lexicon is not None else ""
        if not lexicon_value:
            lexicons = list(model_root.rglob("lexicon.txt"))
            lexicon_value = str(lexicons[0]) if lexicons else ""

        rule_fsts = os.environ.get("SHERPA_TTS_RULE_FSTS", "")
        provider = os.environ.get("SHERPA_TTS_PROVIDER", "cpu")
        num_threads = _env_int("SHERPA_TTS_NUM_THREADS", 2)

        logger.info(
            "Loading sherpa-onnx TTS model=%s tokens=%s data_dir=%s provider=%s",
            model,
            tokens,
            data_dir_value,
            provider,
        )
        config = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                    model=str(model),
                    lexicon=lexicon_value,
                    data_dir=data_dir_value,
                    tokens=str(tokens),
                ),
                provider=provider,
                debug=False,
                num_threads=num_threads,
            ),
            rule_fsts=rule_fsts,
            max_num_sentences=_env_int("SHERPA_TTS_MAX_NUM_SENTENCES", 1),
        )
        if not config.validate():
            raise ValueError("Invalid sherpa-onnx TTS config")

        self.sherpa_onnx = sherpa_onnx
        self.tts = sherpa_onnx.OfflineTts(config)
        self.sid = _env_int("SHERPA_TTS_SID", 0)
        self.speed = _env_float("SHERPA_TTS_SPEED", 1.0)
        self.silence_scale = _env_float("SHERPA_TTS_SILENCE_SCALE", 0.2)

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        gen_config = self.sherpa_onnx.GenerationConfig()
        gen_config.sid = self.sid
        gen_config.speed = self.speed
        gen_config.silence_scale = self.silence_scale
        audio = self.tts.generate(text, gen_config)
        return np.asarray(audio.samples, dtype=np.float32), int(audio.sample_rate)


class SupertonicBackend:
    def __init__(self) -> None:
        from supertonic import TTS  # pyright: ignore[reportMissingImports]

        self.tts = TTS(auto_download=True)
        voice_name = os.environ.get("SUPERTONIC_VOICE", "M1")
        voice_style_path = os.environ.get("SUPERTONIC_VOICE_STYLE_PATH")
        if voice_style_path:
            self.voice_style = self.tts.get_voice_style_from_path(voice_style_path)
        else:
            self.voice_style = self.tts.get_voice_style(voice_name=voice_name)
        self.lang = os.environ.get("SUPERTONIC_LANG", "tr")
        self.steps = _env_int("SUPERTONIC_STEPS", 5)
        self.speed = _env_float("SUPERTONIC_SPEED", 1.1)
        self.max_chunk_length = _env_int("SUPERTONIC_MAX_CHUNK_LENGTH", 160)

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        wav, _duration = self.tts.synthesize(
            text=text,
            lang=self.lang,
            voice_style=self.voice_style,
            total_steps=self.steps,
            speed=self.speed,
            max_chunk_length=self.max_chunk_length,
            silence_duration=0.05,
            verbose=False,
        )
        return np.asarray(wav).reshape(-1).astype(np.float32), 44100


class DummyBackend:
    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        duration = max(0.2, 0.06 * len(text))
        t = np.linspace(0.0, duration, int(SAMPLE_RATE * duration), endpoint=False)
        audio = 0.15 * np.sin(2.0 * math.pi * 220.0 * t)
        return audio.astype(np.float32), SAMPLE_RATE


def make_backend() -> TtsBackend:
    backend = os.environ.get("TURKISH_TTS_BACKEND", "sherpa").lower()
    if backend == "sherpa":
        return SherpaOnnxBackend()
    if backend == "supertonic":
        return SupertonicBackend()
    if backend == "dummy":
        return DummyBackend()
    raise ValueError(f"Unknown TURKISH_TTS_BACKEND={backend!r}")


class TextChunker:
    def __init__(self) -> None:
        self.max_words = _env_int("TTS_CHUNK_WORDS", 4)
        self.max_chars = _env_int("TTS_CHUNK_CHARS", 90)
        self.buffer: list[str] = []

    def add(self, text: str) -> str | None:
        text = text.strip()
        if not text:
            return None
        self.buffer.append(text)
        candidate = " ".join(self.buffer)
        word_count = len(candidate.split())
        if (
            word_count >= self.max_words
            or len(candidate) >= self.max_chars
            or candidate.endswith(('.', '!', '?', ',', ';', ':'))
        ):
            return self.flush()
        return None

    def flush(self) -> str | None:
        if not self.buffer:
            return None
        text = " ".join(self.buffer).strip()
        self.buffer = []
        return text or None


@app.get("/api/build_info")
def get_build_info() -> dict[str, str]:
    return {
        "service": "turkish-tts",
        "backend": os.environ.get("TURKISH_TTS_BACKEND", "sherpa"),
    }


@app.websocket(TEXT_TO_SPEECH_PATH)
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    await websocket.send_bytes(pack_message({"type": "Ready"}))

    backend = app.state.backend
    chunker = TextChunker()
    audio_cursor_s = 0.0

    async def synthesize_and_send(text: str) -> None:
        nonlocal audio_cursor_s
        t0 = time.perf_counter()
        audio, sample_rate = await asyncio.to_thread(backend.synthesize, text)
        synth_elapsed = time.perf_counter() - t0
        audio = resample_linear(audio, sample_rate, SAMPLE_RATE)
        if audio.size:
            peak = float(np.max(np.abs(audio)))
            if peak > 1.0:
                audio = audio / peak

        duration = audio.size / SAMPLE_RATE
        rtf = synth_elapsed / max(duration, 1e-6)
        logger.info(
            "TTS synthesized %r in %.3fs, audio=%.3fs, rtf=%.3f",
            text,
            synth_elapsed,
            duration,
            rtf,
        )

        await websocket.send_bytes(
            pack_message(
                {
                    "type": "Text",
                    "text": text,
                    "start_s": audio_cursor_s,
                    "stop_s": audio_cursor_s + duration,
                }
            )
        )
        for chunk in chunk_audio(audio):
            await websocket.send_bytes(
                pack_message({"type": "Audio", "pcm": chunk.astype(np.float32).tolist()})
            )
        audio_cursor_s += duration

    try:
        while True:
            raw = await websocket.receive_bytes()
            message = msgpack.unpackb(raw, raw=False)
            message_type = message.get("type")
            if message_type == "Text":
                chunk = chunker.add(str(message.get("text", "")))
                if chunk:
                    await synthesize_and_send(chunk)
            elif message_type == "Eos":
                chunk = chunker.flush()
                if chunk:
                    await synthesize_and_send(chunk)
                await websocket.close()
                return
            elif message_type == "Voice":
                # Piper/Sherpa/Supertonic adapters do not consume Kyutai voice embeddings.
                continue
            else:
                logger.warning("Ignoring unknown TTS message: %s", message_type)
    except WebSocketDisconnect:
        logger.info("TTS client disconnected")


@app.on_event("startup")  # pyright: ignore[reportDeprecated]
def load_backend() -> None:
    app.state.backend = make_backend()
