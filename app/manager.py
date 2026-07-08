"""
SessionManager: session_id -> SessionData eşlemesini tutan tek merkezi nokta.

Çoklu kiosk senaryosu için tasarlandı: her kiosk kendi session_id'siyle
bağlanır, kendi ring buffer / baseline / EMA durumunu alır. Thread-safe
erişim için basit bir lock kullanılıyor (asyncio tek thread'de çalışsa da,
işleme worker'ları ayrı thread'de state'e dokunacağı için gerekli).
"""

from __future__ import annotations

import threading
import time
from typing import Dict, Optional

from app import config
from app.session import SessionData


class SessionManager:
    def __init__(self) -> None:
        self._sessions: Dict[str, SessionData] = {}
        self._lock = threading.Lock()

    def get_or_create(self, session_id: str) -> SessionData:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                session = SessionData(session_id=session_id)
                self._sessions[session_id] = session
            return session

    def get(self, session_id: str) -> Optional[SessionData]:
        with self._lock:
            return self._sessions.get(session_id)

    def remove(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def all_session_ids(self) -> list[str]:
        with self._lock:
            return list(self._sessions.keys())

    def gc_stale_sessions(self) -> list[str]:
        """Uzun süredir frame almayan oturumları temizler. Silinen id'leri döner."""
        now = time.time()
        removed = []
        with self._lock:
            stale = [
                sid
                for sid, s in self._sessions.items()
                if now - s.last_frame_at > config.SESSION_GC_TIMEOUT_SECONDS
            ]
            for sid in stale:
                del self._sessions[sid]
                removed.append(sid)
        return removed


# Süreç genelinde tek bir manager örneği paylaşılır.
session_manager = SessionManager()
