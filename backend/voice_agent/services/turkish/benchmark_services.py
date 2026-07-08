import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

import msgpack
import numpy as np
import sphn
import websockets
from websockets.typing import Data

from unmute.kyutai_constants import SAMPLE_RATE, SAMPLES_PER_FRAME


def pack_message(message: dict[str, Any]) -> bytes:
    return msgpack.packb(message, use_bin_type=True, use_single_float=True)


def data_to_bytes(data: Data) -> bytes:
    return data.encode("utf-8") if isinstance(data, str) else data


async def benchmark_tts(url: str, text: str) -> dict[str, Any]:
    first_audio_time: float | None = None
    audio_samples = 0
    start = time.perf_counter()

    async with websockets.connect(url) as websocket:
        ready = msgpack.unpackb(data_to_bytes(await websocket.recv()), raw=False)
        if ready.get("type") != "Ready":
            raise RuntimeError(f"Expected Ready, got {ready}")

        await websocket.send(pack_message({"type": "Text", "text": text}))
        await websocket.send(pack_message({"type": "Eos"}))

        try:
            async for raw in websocket:
                message = msgpack.unpackb(data_to_bytes(raw), raw=False)
                if message.get("type") == "Audio":
                    if first_audio_time is None:
                        first_audio_time = time.perf_counter()
                    audio_samples += len(message["pcm"])
        except websockets.ConnectionClosedOK:
            pass

    end = time.perf_counter()
    audio_duration = audio_samples / SAMPLE_RATE
    return {
        "kind": "tts",
        "text": text,
        "first_audio_latency_sec": None
        if first_audio_time is None
        else first_audio_time - start,
        "total_latency_sec": end - start,
        "audio_duration_sec": audio_duration,
        "rtf": (end - start) / audio_duration if audio_duration else None,
    }


async def benchmark_stt(url: str, audio_path: Path, realtime: bool, post_wait: float) -> dict[str, Any]:
    audio, _sr = sphn.read(audio_path, sample_rate=SAMPLE_RATE)
    audio = audio[0].astype(np.float32)
    trailing_silence = np.zeros(int(SAMPLE_RATE * 0.9), dtype=np.float32)
    audio_to_send = np.concatenate([audio, trailing_silence])

    words: list[str] = []
    first_word_time: float | None = None
    send_done_time: float
    start = time.perf_counter()

    async with websockets.connect(url) as websocket:
        ready = msgpack.unpackb(data_to_bytes(await websocket.recv()), raw=False)
        if ready.get("type") != "Ready":
            raise RuntimeError(f"Expected Ready, got {ready}")

        async def receive_loop() -> None:
            nonlocal first_word_time
            async for raw in websocket:
                message = msgpack.unpackb(data_to_bytes(raw), raw=False)
                if message.get("type") == "Word":
                    if first_word_time is None:
                        first_word_time = time.perf_counter()
                    words.append(message["text"])

        receive_task = asyncio.create_task(receive_loop())

        for i in range(0, len(audio_to_send), SAMPLES_PER_FRAME):
            chunk = audio_to_send[i : i + SAMPLES_PER_FRAME]
            await websocket.send(pack_message({"type": "Audio", "pcm": chunk.tolist()}))
            if realtime:
                await asyncio.sleep(len(chunk) / SAMPLE_RATE)

        send_done_time = time.perf_counter()
        await asyncio.sleep(post_wait)
        receive_task.cancel()

    end = time.perf_counter()
    audio_duration = len(audio) / SAMPLE_RATE
    return {
        "kind": "stt",
        "audio_path": str(audio_path),
        "audio_duration_sec": audio_duration,
        "first_word_latency_from_start_sec": None
        if first_word_time is None
        else first_word_time - start,
        "first_word_latency_after_audio_sent_sec": None
        if first_word_time is None
        else first_word_time - send_done_time,
        "total_wall_time_sec": end - start,
        "text": " ".join(words),
        "realtime_send": realtime,
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="kind", required=True)

    tts_parser = subparsers.add_parser("tts")
    tts_parser.add_argument("--url", default="ws://localhost:8089/api/tts_streaming")
    tts_parser.add_argument(
        "--text",
        default="Merhaba, bu Türkçe ses sentezi için kısa bir gecikme testidir.",
    )

    stt_parser = subparsers.add_parser("stt")
    stt_parser.add_argument("--url", default="ws://localhost:8090/api/asr-streaming")
    stt_parser.add_argument("--audio", type=Path, required=True)
    stt_parser.add_argument("--realtime", action="store_true")
    stt_parser.add_argument("--post-wait", type=float, default=3.0)

    args = parser.parse_args()
    if args.kind == "tts":
        result = await benchmark_tts(args.url, args.text)
    else:
        result = await benchmark_stt(args.url, args.audio, args.realtime, args.post_wait)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
