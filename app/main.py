"""
CV -> Description pipeline'inin ingest + profil/focus servisi.

Coklu kiosk destegi: her kiosk kendi session_id'siyle baglanir.
  - WS /stream/{session_id}   : istemci (robot/kamera) JPEG kareleri push eder
  - WS /profile/{session_id}  : kisi basina TEK SEFERLIK zengin profil JSON'u
                                 push eder (yeterli veri toplanir toplanmaz;
                                 yeni kisi gelince otomatik yeniden tetiklenir)
  - WS /focus/{session_id}    : her ~2.5sn is_focused + focus_time push eder
  - WS /debug/{session_id}    : SADECE gelistirme/test icin, her ~0.3sn tum
                                 ham+turetilmis degerleri push eder (uretim
                                 kontratinin parcasi degildir)
  - GET /health                : liveness/readiness

Mimari:
  ingest (async, IO)  ->  FrameSlot (drop-stale)  ->  worker thread (CPU: MediaPipe)
                                                          |
                                                          v
                                                  SessionData (state)
                                                    |              |
                                        profile trigger loop   focus emit loop
                                        (tek seferlik push)   (periyodik push)
"""

from __future__ import annotations

import asyncio
import logging
from typing import Dict

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from app import config
from app.manager import session_manager
from app.processing import FrameSlot, SignalExtractor
from app.scoring import build_debug_payload, build_focus_payload, update_session

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cv-pipeline")

app = FastAPI(title="ITU CV Description Pipeline")

# session_id -> FrameSlot  (her kiosk'un kendi "son kare" kutusu)
frame_slots: Dict[str, FrameSlot] = {}
# session_id -> SignalExtractor (Face/Pose Landmarker + emotion worker; VIDEO
# modu monoton timestamp gerektirdiginden ornek basina, paylasilmaz)
extractors: Dict[str, SignalExtractor] = {}
# session_id -> set of profile-subscriber websocket'leri
profile_subscribers: Dict[str, set] = {}
# session_id -> set of focus-subscriber websocket'leri
focus_subscribers: Dict[str, set] = {}
# session_id -> set of debug-subscriber websocket'leri (SADECE gelistirme/test)
debug_subscribers: Dict[str, set] = {}
# session_id -> worker/emit gorevlerinin calisip calismadigini takip eder
active_workers: set[str] = set()


def _get_frame_slot(session_id: str) -> FrameSlot:
    slot = frame_slots.get(session_id)
    if slot is None:
        slot = FrameSlot()
        frame_slots[session_id] = slot
    return slot


def _get_extractor(session_id: str) -> SignalExtractor:
    extractor = extractors.get(session_id)
    if extractor is None:
        extractor = SignalExtractor(session_id)
        extractors[session_id] = extractor
    return extractor


def _teardown_stream(session_id: str) -> None:
    """Stream baglantisi kesildiginde (T11): oturumu hizla IDLE'a al, worker
    gorevlerini durdur ve CV modeli kaynaklarini serbest birak. 120sn'lik
    stale-GC'yi beklemeden, "kisi ayrildi"yi aninda yansitir."""
    active_workers.discard(session_id)
    session = session_manager.get(session_id)
    if session is not None:
        session.reset_to_idle()
    extractor = extractors.pop(session_id, None)
    if extractor is not None:
        extractor.close()
    frame_slots.pop(session_id, None)


async def _processing_worker(session_id: str) -> None:
    """Her kiosk icin surekli calisan islem dongusu: en guncel kareyi al,
    sinyal cikar, oturuma isle. CPU-bound MediaPipe cagrisi executor'da
    calistirilir ki event loop bloklanmasin."""
    slot = _get_frame_slot(session_id)
    extractor = _get_extractor(session_id)
    loop = asyncio.get_running_loop()
    logger.info("processing worker started: %s", session_id)
    try:
        while session_id in active_workers:
            frame, frame_ts = slot.get_latest()
            if frame is None:
                await asyncio.sleep(0.01)
                continue

            raw = await loop.run_in_executor(None, extractor.extract, frame)

            session = session_manager.get_or_create(session_id)
            update_session(session, raw)
    except asyncio.CancelledError:
        pass
    finally:
        logger.info("processing worker stopped: %s", session_id)


async def _profile_trigger_loop(session_id: str) -> None:
    """Kisi basina TEK SEFERLIK zengin profili, hazir olur olmaz push eder.
    scoring.update_session, yeterli veri toplaninca session.pending_profile'i
    doldurur; burada yalnizca poll edip abonelere iletiyoruz."""
    try:
        while session_id in active_workers:
            await asyncio.sleep(config.PROFILE_TRIGGER_POLL_SECONDS)
            session = session_manager.get(session_id)
            if session is None or session.pending_profile is None:
                continue

            profile = session.pending_profile
            session.pending_profile = None

            subscribers = profile_subscribers.get(session_id, set())
            dead = []
            for ws in subscribers:
                try:
                    await ws.send_json(profile)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                subscribers.discard(ws)
    except asyncio.CancelledError:
        pass


async def _focus_emit_loop(session_id: str) -> None:
    """Her ~2.5 saniyede bir is_focused + focus_time'i o kiosk'a abone olan
    tum /focus baglantilarina push eder."""
    try:
        while session_id in active_workers:
            await asyncio.sleep(config.FOCUS_EMIT_INTERVAL_SECONDS)
            session = session_manager.get(session_id)
            if session is None:
                continue

            payload = build_focus_payload(session)

            subscribers = focus_subscribers.get(session_id, set())
            dead = []
            for ws in subscribers:
                try:
                    await ws.send_json(payload)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                subscribers.discard(ws)
    except asyncio.CancelledError:
        pass


async def _debug_emit_loop(session_id: str) -> None:
    """SADECE gelistirme/test amacli: her ~0.3sn'de anlik ham+turetilmis tum
    degerleri /debug abonelerine push eder. Uretim kontratinin (/profile,
    /focus) parcasi degildir."""
    try:
        while session_id in active_workers:
            await asyncio.sleep(config.DEBUG_EMIT_INTERVAL_SECONDS)
            session = session_manager.get(session_id)
            if session is None:
                continue

            payload = build_debug_payload(session)

            subscribers = debug_subscribers.get(session_id, set())
            dead = []
            for ws in subscribers:
                try:
                    await ws.send_json(payload)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                subscribers.discard(ws)
    except asyncio.CancelledError:
        pass


def _ensure_worker_running(session_id: str) -> None:
    if session_id in active_workers:
        return
    active_workers.add(session_id)
    asyncio.create_task(_processing_worker(session_id))
    asyncio.create_task(_profile_trigger_loop(session_id))
    asyncio.create_task(_focus_emit_loop(session_id))
    asyncio.create_task(_debug_emit_loop(session_id))


@app.websocket("/stream/{session_id}")
async def stream_ingest(websocket: WebSocket, session_id: str):
    """Istemci (robot) buraya JPEG-encoded binary frame'ler push eder.
    Servis her zaman en guncel kareyi tutar; islemeye yetismeyen eski
    kareler bilerek dusurulur (drop-stale), kuyruklama yapilmaz."""
    await websocket.accept()
    session_manager.get_or_create(session_id)
    slot = _get_frame_slot(session_id)
    _ensure_worker_running(session_id)
    logger.info("stream connected: %s", session_id)

    try:
        while True:
            data = await websocket.receive_bytes()
            jpg_array = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(jpg_array, cv2.IMREAD_COLOR)
            if frame is not None:
                slot.put(frame)
                session = session_manager.get(session_id)
                if session is not None:
                    session.touch_frame()
    except WebSocketDisconnect:
        logger.info("stream disconnected: %s", session_id)
    except Exception:
        logger.exception("stream error: %s", session_id)
    finally:
        _teardown_stream(session_id)


@app.websocket("/profile/{session_id}")
async def profile_push(websocket: WebSocket, session_id: str):
    """LLM tarafi buraya baglanir; kisi basina tek seferlik zengin profil
    JSON'u, yeterli veri toplanir toplanmaz otomatik push edilir."""
    await websocket.accept()
    profile_subscribers.setdefault(session_id, set()).add(websocket)
    logger.info("profile subscriber connected: %s", session_id)
    try:
        while True:
            # Bu kanal tek yonlu (server->client); baglanti acik kalsin diye bekliyoruz.
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        profile_subscribers.get(session_id, set()).discard(websocket)
        logger.info("profile subscriber disconnected: %s", session_id)


@app.websocket("/focus/{session_id}")
async def focus_push(websocket: WebSocket, session_id: str):
    """Her ~2.5sn'de guncel is_focused + focus_time push edilir."""
    await websocket.accept()
    focus_subscribers.setdefault(session_id, set()).add(websocket)
    logger.info("focus subscriber connected: %s", session_id)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        focus_subscribers.get(session_id, set()).discard(websocket)
        logger.info("focus subscriber disconnected: %s", session_id)


@app.websocket("/debug/{session_id}")
async def debug_push(websocket: WebSocket, session_id: str):
    """SADECE gelistirme/test amacli: her ~0.3sn'de anlik ham+turetilmis tum
    degerleri push eder (uretim kontratinin disinda, bkz. build_debug_payload)."""
    await websocket.accept()
    debug_subscribers.setdefault(session_id, set()).add(websocket)
    logger.info("debug subscriber connected: %s", session_id)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        debug_subscribers.get(session_id, set()).discard(websocket)
        logger.info("debug subscriber disconnected: %s", session_id)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "active_sessions": session_manager.all_session_ids(),
    }


@app.on_event("startup")
async def _start_gc_loop():
    async def gc_loop():
        while True:
            await asyncio.sleep(30)
            removed = session_manager.gc_stale_sessions()
            for sid in removed:
                active_workers.discard(sid)
                frame_slots.pop(sid, None)
                profile_subscribers.pop(sid, None)
                focus_subscribers.pop(sid, None)
                debug_subscribers.pop(sid, None)
                extractor = extractors.pop(sid, None)
                if extractor is not None:
                    extractor.close()
                logger.info("session GC'd (stale): %s", sid)

    asyncio.create_task(gc_loop())
