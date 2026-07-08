"""Coklu kiosk izolasyonu: session_id'ler birbirinin state'ine dokunmaz (T12)."""
from app import config
from app.manager import SessionManager
from app.scoring import update_session
from app.session import RawSignals, SessionState


def test_sessions_are_isolated(monkeypatch):
    monkeypatch.setattr(config, "CALIBRATION_SECONDS", 0.0)
    manager = SessionManager()
    s1 = manager.get_or_create("kiosk-a")
    s2 = manager.get_or_create("kiosk-b")

    raw_present = RawSignals(face_present=True, lean=0.05, eye_contact=0.9, spine_ratio=0.9)
    for _ in range(10):
        update_session(s1, raw_present)

    assert s2.state is SessionState.IDLE
    assert len(s2.eye_history) == 0
    assert len(s1.eye_history) > 0
    assert manager.get("kiosk-a") is s1
    assert manager.get("kiosk-b") is s2
    assert set(manager.all_session_ids()) == {"kiosk-a", "kiosk-b"}
