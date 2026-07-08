"""
Cihazin gercek kamerasini kullanan test istemcisi.

Mimari kararlar geregi (bkz. docs/SPRINT.md Bolum 2), cv-pipeline modulunun
kendisi KAMERAYA DOKUNMAZ — yalnizca network uzerinden JPEG kare alir. Bu
script, o "robot/kiosk" istemcisinin yerini tutar: yerel webcam'i acar,
kareleri encode edip /stream'e gonderir; /profile (tek seferlik), /focus
(periyodik) ve /debug (SADECE test icin, ~0.3sn'de tum ham+turetilmis
degerler) kanallarindan gelenleri canli olarak onizleme penceresine ve
terminale basar.

Kullanim:
  python camera_client.py [session_id] [--camera-index 0] [--server ws://localhost:8000]

Cikmak icin onizleme penceresindeyken 'q' tusuna basin (veya Ctrl+C).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import threading

import cv2
import websockets


def parse_args():
    p = argparse.ArgumentParser(description="Gercek kamera ile cv-pipeline test istemcisi")
    p.add_argument("session_id", nargs="?", default="camera-test-1")
    p.add_argument("--camera-index", type=int, default=0, help="cv2.VideoCapture cihaz indeksi")
    p.add_argument("--server", default="ws://localhost:8000", help="cv-pipeline sunucu adresi")
    p.add_argument("--fps", type=float, default=15.0)
    return p.parse_args()


# Ana thread'te goruntulenen en guncel /profile, /focus, /debug payload'lari.
_latest_profile: dict | None = None
_latest_focus: dict | None = None
_latest_debug: dict | None = None
_lock = threading.Lock()


def _fmt(x, nd=3):
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def _draw_overlay(preview) -> None:
    with _lock:
        debug, profile = _latest_debug, _latest_profile

    lines: list[str] = []
    if debug is not None:
        raw = debug.get("raw", {})
        smoothed = debug.get("smoothed", {})
        live = debug.get("live_score_preview", {})
        emotion_scores = raw.get("emotion_scores") or {}
        top_emotion = ", ".join(
            f"{k}={_fmt(v, 2)}" for k, v in sorted(emotion_scores.items(), key=lambda kv: -kv[1])[:3]
        )
        lines = [
            (f"state={debug.get('state')}  face={raw.get('face_present')}", (255, 255, 255)),
            (f"focused={debug.get('is_focused')}  focus_time={_fmt(debug.get('focus_time'))}", (0, 255, 0)),
            (
                f"lean(raw)={_fmt(raw.get('lean'))}  delta_lean={_fmt(smoothed.get('delta_lean'))}  baseline={_fmt(smoothed.get('baseline_lean'))}",
                (0, 200, 255),
            ),
            (
                f"eye_contact(raw)={_fmt(raw.get('eye_contact'))}  avg={_fmt(smoothed.get('avg_eye_contact'))}  head_yaw={_fmt(raw.get('head_yaw_deg'), 1)}",
                (0, 200, 255),
            ),
            (
                f"spine_ratio={_fmt(raw.get('spine_ratio'))}  shoulder_tilt={_fmt(raw.get('shoulder_tilt'))}  arms_crossed={raw.get('arms_crossed')}",
                (0, 200, 255),
            ),
            (f"emotion={raw.get('emotion_label')}  ({top_emotion})", (255, 200, 0)),
            (
                f"live attn={_fmt(live.get('attention'))}  open={_fmt(live.get('openness'))}  energy={_fmt(live.get('energy'))}",
                (200, 255, 200),
            ),
        ]
    else:
        lines = [("/debug baglantisi bekleniyor...", (255, 255, 255))]

    if profile is not None:
        scores = profile.get("scores", {})
        lines.append(
            (
                f"[/profile TEK SEFERLIK] attn={scores.get('attention')} open={scores.get('openness')} energy={scores.get('energy')}",
                (0, 165, 255),
            )
        )

    y = 22
    for text, color in lines:
        cv2.putText(preview, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        y += 22


async def push_camera_frames(server: str, session_id: str, camera_index: int, fps: float, stop_event: threading.Event):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"kamera acilamadi (index={camera_index})")

    uri = f"{server}/stream/{session_id}"
    interval = 1.0 / fps
    try:
        async with websockets.connect(uri) as ws:
            print(f"[{session_id}] kameradan {uri} adresine akis basladi")
            while not stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    await asyncio.sleep(0.05)
                    continue

                preview = frame.copy()
                _draw_overlay(preview)
                cv2.imshow("cv-pipeline camera test (q ile cik)", preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    stop_event.set()
                    break

                ok, buf = cv2.imencode(".jpg", frame)
                if ok:
                    await ws.send(buf.tobytes())
                await asyncio.sleep(interval)
    finally:
        cap.release()
        cv2.destroyAllWindows()


async def listen_profile(server: str, session_id: str, stop_event: threading.Event):
    uri = f"{server}/profile/{session_id}"
    async with websockets.connect(uri) as ws:
        print(f"[{session_id}] /profile dinleniyor")
        while not stop_event.is_set():
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            data = json.loads(msg)
            with _lock:
                global _latest_profile
                _latest_profile = data
            print(f"[{session_id}] PROFILE (tek seferlik): {json.dumps(data, ensure_ascii=False)}")


async def listen_focus(server: str, session_id: str, stop_event: threading.Event):
    uri = f"{server}/focus/{session_id}"
    async with websockets.connect(uri) as ws:
        print(f"[{session_id}] /focus dinleniyor")
        while not stop_event.is_set():
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            data = json.loads(msg)
            with _lock:
                global _latest_focus
                _latest_focus = data
            print(f"[{session_id}] FOCUS: {data}")


async def listen_debug(server: str, session_id: str, stop_event: threading.Event):
    """SADECE test/dogrulama icin: /debug'dan gelen anlik ham+turetilmis
    degerleri hem overlay'e hem (spam olmasin diye seyreltilmis) terminale basar."""
    uri = f"{server}/debug/{session_id}"
    tick = 0
    async with websockets.connect(uri) as ws:
        print(f"[{session_id}] /debug dinleniyor (yalnizca test amacli kanal)")
        while not stop_event.is_set():
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            data = json.loads(msg)
            with _lock:
                global _latest_debug
                _latest_debug = data
            tick += 1
            if tick % 3 == 0:  # ~1sn'de bir terminale bas, overlay her karede zaten guncel
                print(f"[{session_id}] DEBUG: {json.dumps(data, ensure_ascii=False)}")


async def main():
    args = parse_args()
    stop_event = threading.Event()
    try:
        await asyncio.gather(
            push_camera_frames(args.server, args.session_id, args.camera_index, args.fps, stop_event),
            listen_profile(args.server, args.session_id, stop_event),
            listen_focus(args.server, args.session_id, stop_event),
            listen_debug(args.server, args.session_id, stop_event),
        )
    except KeyboardInterrupt:
        stop_event.set()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\ncikiliyor...")
