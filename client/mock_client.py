"""
Basit test istemcisi: bir kareyi (varsayilan: sahte daire; --image ile
gercek bir yuz/govde fotografi de verilebilir) JPEG'e encode edip belirli
bir session_id ile /stream'e 15fps push eder, ayni anda /profile ve /focus'a
baglanip gelen JSON'lari basar. Gercek robot istemcisinin yerini tutar.

Kullanim:
  python test_client.py <session_id> <duration_seconds> [image_path]

image_path verilirse (gercek yuz/govde iceren bir fotograf), pipeline'in
gercek MediaPipe + duygu modelleriyle uctan uca calistigi deterministik
sekilde dogrulanabilir (bkz. sprint Bolum 6 test protokolu).
"""
import asyncio
import sys
import time

import cv2
import numpy as np
import websockets


def _load_frame(image_path: str | None) -> bytes:
    if image_path:
        frame = cv2.imread(image_path)
        if frame is None:
            raise FileNotFoundError(f"goruntu okunamadi: {image_path}")
    else:
        # yuz benzeri bir daire iceren sahte bir kare (model hicbir yuz bulamaz;
        # yalnizca ingest/state-machine iskeletini duman testi icin kullanislidir)
        frame = np.full((480, 640, 3), 40, dtype=np.uint8)
        cv2.circle(frame, (320, 240), 80, (200, 180, 160), -1)
    ok, buf = cv2.imencode(".jpg", frame)
    return buf.tobytes()


async def push_frames(session_id: str, duration: float, image_path: str | None):
    uri = f"ws://localhost:8000/stream/{session_id}"
    async with websockets.connect(uri) as ws:
        jpg_bytes = _load_frame(image_path)

        end = time.time() + duration
        n = 0
        while time.time() < end:
            await ws.send(jpg_bytes)
            n += 1
            await asyncio.sleep(1 / 15)
        print(f"[{session_id}] {n} kare gonderildi")


async def listen_profile(session_id: str, duration: float):
    uri = f"ws://localhost:8000/profile/{session_id}"
    async with websockets.connect(uri) as ws:
        end = time.time() + duration + 2
        while time.time() < end:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=duration + 2)
            except asyncio.TimeoutError:
                break
            print(f"[{session_id}] profile: {msg[:300]}")


async def listen_focus(session_id: str, duration: float):
    uri = f"ws://localhost:8000/focus/{session_id}"
    async with websockets.connect(uri) as ws:
        end = time.time() + duration + 2
        while time.time() < end:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=duration + 2)
            except asyncio.TimeoutError:
                break
            print(f"[{session_id}] focus: {msg[:200]}")


async def main():
    session_id = sys.argv[1] if len(sys.argv) > 1 else "test-kiosk-1"
    duration = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
    image_path = sys.argv[3] if len(sys.argv) > 3 else None
    await asyncio.gather(
        push_frames(session_id, duration, image_path),
        listen_profile(session_id, duration),
        listen_focus(session_id, duration),
    )


if __name__ == "__main__":
    asyncio.run(main())
