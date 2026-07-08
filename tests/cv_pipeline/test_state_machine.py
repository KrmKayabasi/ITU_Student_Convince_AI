"""IDLE -> CALIBRATING -> ACTIVE -> IDLE gecisleri (T12)."""
import time

from backend.cv_pipeline import config
from backend.cv_pipeline.scoring import update_session
from backend.cv_pipeline.session import RawSignals, SessionData, SessionState


def test_new_person_starts_calibrating(monkeypatch):
    monkeypatch.setattr(config, "CALIBRATION_SECONDS", 0.05)
    session = SessionData(session_id="s-calib")
    assert session.state is SessionState.IDLE

    raw = RawSignals(face_present=True, lean=0.0, eye_contact=0.6, spine_ratio=0.9)
    update_session(session, raw)
    assert session.state is SessionState.CALIBRATING


def test_calibration_completes_into_active(monkeypatch):
    monkeypatch.setattr(config, "CALIBRATION_SECONDS", 0.02)
    session = SessionData(session_id="s-active")
    raw = RawSignals(face_present=True, lean=0.0, eye_contact=0.6, spine_ratio=0.9)

    for _ in range(10):
        update_session(session, raw)
        if session.state is SessionState.ACTIVE:
            break
        time.sleep(0.01)

    assert session.state is SessionState.ACTIVE
    assert session.baseline_lean is not None


def test_face_lost_resets_to_idle_after_timeout(monkeypatch):
    monkeypatch.setattr(config, "CALIBRATION_SECONDS", 0.0)
    monkeypatch.setattr(config, "NO_FACE_TIMEOUT_SECONDS", 0.03)
    session = SessionData(session_id="s-idle")
    raw_present = RawSignals(face_present=True, lean=0.0, eye_contact=0.6, spine_ratio=0.9)

    for _ in range(10):
        update_session(session, raw_present)
        if session.state is SessionState.ACTIVE:
            break
    assert session.state is SessionState.ACTIVE

    raw_absent = RawSignals(face_present=False)
    update_session(session, raw_absent)
    assert session.state is SessionState.ACTIVE  # timeout henuz dolmadi

    time.sleep(0.05)
    update_session(session, raw_absent)
    assert session.state is SessionState.IDLE
