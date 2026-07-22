"""
CV -> Description pipeline'inin ingest + profil/focus servisi.

Coklu kiosk destegi: her kiosk kendi session_id'siyle baglanir.
  - WS /stream/{session_id}   : istemci (robot/kamera) JPEG kareleri push eder
  - WS /profile/{session_id}  : kisi basina TEK SEFERLIK zengin profil JSON'u
                                 push eder (yeterli veri toplanir toplanmaz;
                                 yeni kisi gelince otomatik yeniden tetiklenir)
  - WS /focus/{session_id}    : her ~2.5sn is_focused + focus_time push eder
  - WS /tracking/{session_id} : her ~0.2sn yuz varligi + normalize konum push eder
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
from dataclasses import dataclass, field
import itertools
import logging
import os
import secrets
import sys
from typing import Dict, Optional

import cv2
from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
import numpy as np

# Ensure the project root is on sys.path so that `backend.cv_pipeline.*`
# imports resolve regardless of the working directory.  Prefer running with
# `PYTHONPATH=.` or `pip install -e .`, but this fallback keeps things
# working when launched from any directory inside the repo.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from backend.cv_pipeline import config  # noqa: E402
from backend.cv_pipeline.manager import session_manager  # noqa: E402
from backend.cv_pipeline.processing import FrameSlot, SignalExtractor  # noqa: E402
from backend.cv_pipeline.scoring import (  # noqa: E402
    build_debug_payload,
    build_focus_payload,
    build_tracking_payload,
    update_session,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cv-pipeline")

app = FastAPI(title="ITU CV Description Pipeline")

# ── Auth ─────────────────────────────────────────────────────────────────────
# Optional token-based auth for WebSocket endpoints.  Set CV_PIPELINE_TOKEN in
# the environment to enable; if unset all endpoints are open (dev mode).
_AUTH_TOKEN = os.environ.get("CV_PIPELINE_TOKEN", "")
_AUTH_ENABLED = bool(_AUTH_TOKEN)


def _verify_ws_token(websocket: WebSocket, token: Optional[str]) -> bool:
    """WebSocket-safe auth check.  Returns True if the token is valid or auth is
    disabled.  When False, the caller should close the socket with code 4001."""
    if not _AUTH_ENABLED:
        return True
    if token is None:
        return False
    return secrets.compare_digest(token, _AUTH_TOKEN)


# session_id -> FrameSlot  (her kiosk'un kendi "son kare" kutusu)
frame_slots: Dict[str, FrameSlot] = {}
# session_id -> SignalExtractor (Face/Pose Landmarker + emotion worker; VIDEO
# modu monoton timestamp gerektirdiginden ornek basina, paylasilmaz)
extractors: Dict[str, SignalExtractor] = {}
# session_id -> set of profile-subscriber websocket'leri
profile_subscribers: Dict[str, set] = {}
# session_id -> set of focus-subscriber websocket'leri
focus_subscribers: Dict[str, set] = {}
# session_id -> set of tracking-subscriber websocket'leri
tracking_subscribers: Dict[str, set] = {}
# session_id -> set of debug-subscriber websocket'leri (SADECE gelistirme/test)
debug_subscribers: Dict[str, set] = {}
# session_id -> worker/emit gorevlerinin calisip calismadigini takip eder
active_workers: set[str] = set()
# Generation guards prevent a slow old worker/stream from writing into a
# replacement connection that reuses the same session id.
worker_generations: Dict[str, int] = {}
stream_generations: Dict[str, int] = {}
stream_connections: Dict[str, WebSocket] = {}
_generation_ids = itertools.count(1)


@dataclass
class _WorkerGroup:
    generation: int
    slot: FrameSlot
    extractor: SignalExtractor
    tasks: set[asyncio.Task] = field(default_factory=set)
    processing_task: asyncio.Task | None = None


worker_tasks: Dict[str, _WorkerGroup] = {}
stream_locks: Dict[str, asyncio.Lock] = {}
# Tracking, stream kesildikten sonra abonelere unknown gonderebilmeli.
active_tracking_emitters: set[str] = set()
tracking_generations: Dict[str, int] = {}
tracking_tasks: Dict[str, asyncio.Task] = {}


def _next_generation() -> int:
    """Return a process-unique generation; IDs are never reset or reused."""
    return next(_generation_ids)


def _get_stream_lock(session_id: str) -> asyncio.Lock:
    lock = stream_locks.get(session_id)
    if lock is None:
        lock = asyncio.Lock()
        stream_locks[session_id] = lock
    return lock


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


def _retire_worker(session_id: str) -> _WorkerGroup | None:
    """Synchronously supersede a worker before any cleanup await can race."""
    group = worker_tasks.get(session_id)
    if group is not None and worker_generations.get(session_id) == group.generation:
        worker_generations[session_id] = _next_generation()
    active_workers.discard(session_id)
    return group


async def _await_worker(group: _WorkerGroup | None) -> None:
    if group is None or group.processing_task is None:
        return
    await asyncio.gather(group.processing_task, return_exceptions=True)


async def _teardown_stream(
    session_id: str,
    stream_generation: int,
    websocket: WebSocket | None = None,
) -> None:
    """Stream baglantisi kesildiginde (T11): oturumu hizla IDLE'a al, worker
    gorevlerini durdur ve CV modeli kaynaklarini serbest birak. 120sn'lik
    stale-GC'yi beklemeden, "kisi ayrildi"yi aninda yansitir."""
    async with _get_stream_lock(session_id):
        if stream_generations.get(session_id) != stream_generation:
            return
        current_websocket = stream_connections.get(session_id)
        if websocket is not None and current_websocket is not websocket:
            return
        if current_websocket is not None:
            stream_connections.pop(session_id, None)
        group = _retire_worker(session_id)
        session = session_manager.get(session_id)
        if session is not None:
            session.reset_to_idle()
            session.invalidate_observation()
        await _await_worker(group)


def _worker_is_current(session_id: str, generation: int) -> bool:
    return (
        session_id in active_workers
        and worker_generations.get(session_id) == generation
    )


async def _processing_worker(session_id: str, group: _WorkerGroup) -> None:
    """Her kiosk icin surekli calisan islem dongusu: en guncel kareyi al,
    sinyal cikar, oturuma isle. CPU-bound MediaPipe cagrisi executor'da
    calistirilir ki event loop bloklanmasin."""
    generation = group.generation
    slot = group.slot
    extractor = group.extractor
    logger.info("processing worker started: %s", session_id)
    try:
        while _worker_is_current(session_id, generation):
            frame, frame_ts, frame_monotonic = slot.get_latest_with_timestamps()
            if frame is None:
                await asyncio.sleep(0.01)
                continue

            inference = asyncio.create_task(asyncio.to_thread(extractor.extract, frame))
            try:
                raw = await asyncio.shield(inference)
            except asyncio.CancelledError:
                # asyncio cancellation cannot stop a running executor call.
                # Wait for ownership to return before close() in finally.
                await inference
                raise
            if not _worker_is_current(session_id, generation):
                break
            # FrameSlot zamani kamera karesinin sisteme girdigi ani temsil eder;
            # yavas isleme tamamlanma zamanini tazelik olarak gostermeyelim.
            raw.observation_ts = frame_ts

            session = session_manager.get_or_create(session_id)
            update_session(session, raw, observation_monotonic=frame_monotonic)
    except asyncio.CancelledError:
        pass
    except Exception:
        logger.exception("processing worker failed: %s", session_id)
    finally:
        current_task = asyncio.current_task()
        for task in group.tasks:
            if task is not current_task:
                task.cancel()
        await asyncio.gather(
            *(task for task in group.tasks if task is not current_task),
            return_exceptions=True,
        )
        if worker_tasks.get(session_id) is group:
            worker_tasks.pop(session_id, None)
        if worker_generations.get(session_id) == generation:
            active_workers.discard(session_id)
        if frame_slots.get(session_id) is slot:
            frame_slots.pop(session_id, None)
        if extractors.get(session_id) is extractor:
            extractors.pop(session_id, None)
        try:
            await asyncio.to_thread(extractor.close)
        except Exception:
            logger.exception("failed to close extractor: %s", session_id)
        logger.info("processing worker stopped: %s", session_id)


async def _profile_trigger_loop(session_id: str, generation: int) -> None:
    """Kisi basina TEK SEFERLIK zengin profili, hazir olur olmaz push eder.
    scoring.update_session, yeterli veri toplaninca session.pending_profile'i
    doldurur; burada yalnizca poll edip abonelere iletiyoruz."""
    try:
        while _worker_is_current(session_id, generation):
            await asyncio.sleep(config.PROFILE_TRIGGER_POLL_SECONDS)
            session = session_manager.get(session_id)
            if session is None or session.pending_profile is None:
                continue

            profile = session.pending_profile
            session.pending_profile = None

            subscribers = profile_subscribers.get(session_id, set())
            dead = []
            for ws in list(subscribers):
                try:
                    await ws.send_json(profile)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                subscribers.discard(ws)
    except asyncio.CancelledError:
        pass


async def _focus_emit_loop(session_id: str, generation: int) -> None:
    """Her ~2.5 saniyede bir is_focused + focus_time'i o kiosk'a abone olan
    tum /focus baglantilarina push eder."""
    try:
        while _worker_is_current(session_id, generation):
            await asyncio.sleep(config.FOCUS_EMIT_INTERVAL_SECONDS)
            session = session_manager.get(session_id)
            if session is None:
                continue

            payload = build_focus_payload(session)

            subscribers = focus_subscribers.get(session_id, set())
            dead = []
            for ws in list(subscribers):
                try:
                    await ws.send_json(payload)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                subscribers.discard(ws)
    except asyncio.CancelledError:
        pass


async def _tracking_emit_loop(session_id: str, generation: int) -> None:
    """Taze gozlemleri ~5 Hz push eder; stream yoksa unknown yaymaya devam eder."""
    try:
        while tracking_generations.get(session_id) == generation and (
            session_id in active_workers or tracking_subscribers.get(session_id)
        ):
            await asyncio.sleep(config.TRACKING_EMIT_INTERVAL_SECONDS)
            session = session_manager.get(session_id)
            if session is None:
                continue

            payload = build_tracking_payload(session)
            subscribers = tracking_subscribers.get(session_id, set())
            dead = []
            for ws in list(subscribers):
                try:
                    await ws.send_json(payload)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                subscribers.discard(ws)
            if not subscribers:
                tracking_subscribers.pop(session_id, None)
    except asyncio.CancelledError:
        pass
    finally:
        if tracking_generations.get(session_id) == generation:
            active_tracking_emitters.discard(session_id)
        if tracking_tasks.get(session_id) is asyncio.current_task():
            tracking_tasks.pop(session_id, None)


async def _debug_emit_loop(session_id: str, generation: int) -> None:
    """SADECE gelistirme/test amacli: her ~0.3sn'de anlik ham+turetilmis tum
    degerleri /debug abonelerine push eder. Uretim kontratinin (/profile,
    /focus) parcasi degildir."""
    try:
        while _worker_is_current(session_id, generation):
            await asyncio.sleep(config.DEBUG_EMIT_INTERVAL_SECONDS)
            session = session_manager.get(session_id)
            if session is None:
                continue

            payload = build_debug_payload(session)

            subscribers = debug_subscribers.get(session_id, set())
            dead = []
            for ws in list(subscribers):
                try:
                    await ws.send_json(payload)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                subscribers.discard(ws)
    except asyncio.CancelledError:
        pass


async def _ensure_worker_running(session_id: str) -> None:
    existing = worker_tasks.get(session_id)
    if (
        session_id in active_workers
        and existing is not None
        and existing.processing_task is not None
        and not existing.processing_task.done()
    ):
        return
    await _await_worker(existing)
    generation = _next_generation()
    worker_generations[session_id] = generation
    slot = FrameSlot()
    extractor = SignalExtractor(session_id)
    group = _WorkerGroup(generation, slot, extractor)
    frame_slots[session_id] = slot
    extractors[session_id] = extractor
    worker_tasks[session_id] = group
    active_workers.add(session_id)
    group.processing_task = asyncio.create_task(
        _processing_worker(session_id, group), name=f"cv_processing:{session_id}"
    )
    group.tasks.update(
        {
            group.processing_task,
            asyncio.create_task(_profile_trigger_loop(session_id, generation)),
            asyncio.create_task(_focus_emit_loop(session_id, generation)),
            asyncio.create_task(_debug_emit_loop(session_id, generation)),
        }
    )
    _ensure_tracking_emitter_running(session_id)


def _ensure_tracking_emitter_running(session_id: str) -> None:
    task = tracking_tasks.get(session_id)
    if session_id in active_tracking_emitters and task is not None and not task.done():
        return
    generation = _next_generation()
    tracking_generations[session_id] = generation
    active_tracking_emitters.add(session_id)
    tracking_tasks[session_id] = asyncio.create_task(
        _tracking_emit_loop(session_id, generation)
    )


@app.websocket("/stream/{session_id}")
async def stream_ingest(
    websocket: WebSocket, session_id: str, token: str = Query(None)
):
    """Istemci (robot) buraya JPEG-encoded binary frame'ler push eder.
    Servis her zaman en guncel kareyi tutar; islemeye yetismeyen eski
    kareler bilerek dusurulur (drop-stale), kuyruklama yapilmaz."""
    await websocket.accept()
    if not _verify_ws_token(websocket, token):
        await websocket.close(code=4001, reason="Invalid token")
        return
    async with _get_stream_lock(session_id):
        stream_generation = _next_generation()
        previous_websocket = stream_connections.get(session_id)
        stream_generations[session_id] = stream_generation
        stream_connections[session_id] = websocket
        session = session_manager.get_or_create(session_id)
        # Refresh before closing the superseded socket: close() awaits and the
        # GC loop must not classify this replacement connection as stale then.
        session.touch_frame()
        if previous_websocket is not None and previous_websocket is not websocket:
            try:
                await previous_websocket.close(code=1001, reason="Stream replaced")
            except Exception:
                pass
        await _ensure_worker_running(session_id)
        slot = frame_slots[session_id]
    logger.info("stream connected: %s", session_id)

    try:
        while True:
            data = await websocket.receive_bytes()
            if stream_generations.get(session_id) != stream_generation:
                await websocket.close(code=1001, reason="Stream replaced")
                break
            jpg_array = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(jpg_array, cv2.IMREAD_COLOR)
            if frame is not None:
                if session_id not in active_workers:
                    async with _get_stream_lock(session_id):
                        if (
                            stream_generations.get(session_id) != stream_generation
                            or stream_connections.get(session_id) is not websocket
                        ):
                            continue
                        await _ensure_worker_running(session_id)
                        slot = frame_slots[session_id]
                slot.put(frame)
                session = session_manager.get(session_id)
                if session is not None:
                    session.touch_frame()
    except WebSocketDisconnect:
        logger.info("stream disconnected: %s", session_id)
    except Exception:
        logger.exception("stream error: %s", session_id)
    finally:
        await _teardown_stream(session_id, stream_generation, websocket)


@app.websocket("/profile/{session_id}")
async def profile_push(websocket: WebSocket, session_id: str, token: str = Query(None)):
    """LLM tarafi buraya baglanir; kisi basina tek seferlik zengin profil
    JSON'u, yeterli veri toplanir toplanmaz otomatik push edilir."""
    await websocket.accept()
    if not _verify_ws_token(websocket, token):
        await websocket.close(code=4001, reason="Invalid token")
        return
    profile_subscribers.setdefault(session_id, set()).add(websocket)
    logger.info("profile subscriber connected: %s", session_id)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        profile_subscribers.get(session_id, set()).discard(websocket)
        logger.info("profile subscriber disconnected: %s", session_id)


@app.websocket("/focus/{session_id}")
async def focus_push(websocket: WebSocket, session_id: str, token: str = Query(None)):
    """Her ~2.5sn'de guncel is_focused + focus_time push edilir."""
    await websocket.accept()
    if not _verify_ws_token(websocket, token):
        await websocket.close(code=4001, reason="Invalid token")
        return
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


@app.websocket("/tracking/{session_id}")
async def tracking_push(
    websocket: WebSocket, session_id: str, token: str = Query(None)
):
    """~5 Hz yuz varligi, normalize bbox konumu ve gozlem tazeligi push eder."""
    await websocket.accept()
    if not _verify_ws_token(websocket, token):
        await websocket.close(code=4001, reason="Invalid token")
        return
    session_manager.get_or_create(session_id)
    tracking_subscribers.setdefault(session_id, set()).add(websocket)
    _ensure_tracking_emitter_running(session_id)
    logger.info("tracking subscriber connected: %s", session_id)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        subscribers = tracking_subscribers.get(session_id)
        if subscribers is not None:
            subscribers.discard(websocket)
            if not subscribers:
                tracking_subscribers.pop(session_id, None)
        logger.info("tracking subscriber disconnected: %s", session_id)


@app.websocket("/debug/{session_id}")
async def debug_push(websocket: WebSocket, session_id: str, token: str = Query(None)):
    """SADECE gelistirme/test amacli: her ~0.3sn'de anlik ham+turetilmis tum
    degerleri push eder (uretim kontratinin disinda, bkz. build_debug_payload)."""
    await websocket.accept()
    if not _verify_ws_token(websocket, token):
        await websocket.close(code=4001, reason="Invalid token")
        return
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


async def _cleanup_removed_session(session_id: str) -> None:
    """Detach a GC'd session's resources before awaiting any slow cleanup.

    A reconnect may happen while sockets close or inference finishes. All map
    removals therefore happen up front; subsequent work uses captured object
    identities and cannot remove a replacement session's resources.
    """
    group = _retire_worker(session_id)
    stream_connection = stream_connections.pop(session_id, None)
    if session_id in stream_generations:
        stream_generations[session_id] = _next_generation()
    profile_subscribers.pop(session_id, None)
    focus_subscribers.pop(session_id, None)
    tracking_connections = tracking_subscribers.pop(session_id, set())
    debug_subscribers.pop(session_id, None)

    tracking_task = tracking_tasks.pop(session_id, None)
    if tracking_task is not None:
        tracking_generations[session_id] = _next_generation()
        active_tracking_emitters.discard(session_id)
        tracking_task.cancel()

    orphan_slot = frame_slots.get(session_id)
    if group is None or orphan_slot is not group.slot:
        frame_slots.pop(session_id, None)
    orphan_extractor = extractors.get(session_id)
    if group is None or orphan_extractor is not group.extractor:
        extractors.pop(session_id, None)
    else:
        orphan_extractor = None

    await _await_worker(group)
    if orphan_extractor is not None:
        try:
            await asyncio.to_thread(orphan_extractor.close)
        except Exception:
            logger.exception("failed to close orphan extractor: %s", session_id)
    if tracking_task is not None:
        await asyncio.gather(tracking_task, return_exceptions=True)
    if stream_connection is not None:
        try:
            await stream_connection.close(code=1001, reason="Session expired")
        except Exception:
            pass
    for websocket in tracking_connections:
        try:
            await websocket.close(code=1001, reason="Session expired")
        except Exception:
            pass
    lock = _get_stream_lock(session_id)
    async with lock:
        if (
            session_manager.get(session_id) is None
            and session_id not in stream_connections
            and session_id not in worker_tasks
            and not tracking_subscribers.get(session_id)
        ):
            stream_generations.pop(session_id, None)
            worker_generations.pop(session_id, None)
            tracking_generations.pop(session_id, None)
            stream_locks.pop(session_id, None)
    logger.info("session GC'd (stale): %s", session_id)


@app.on_event("startup")
async def _start_gc_loop():
    async def gc_loop():
        while True:
            await asyncio.sleep(30)
            removed = session_manager.gc_stale_sessions()
            for sid in removed:
                await _cleanup_removed_session(sid)

    asyncio.create_task(gc_loop())
