"""/profile kisi basina TEK SEFERLIK uretilir; yeni kisi gelince yeniden tetiklenir (T9/T11/T12)."""
import time

from app import config
from app.scoring import update_session
from app.session import RawSignals, SessionData, SessionState


def _feed(session, raw, n=10):
    for _ in range(n):
        update_session(session, raw)


def test_profile_sent_exactly_once_per_person(monkeypatch):
    monkeypatch.setattr(config, "CALIBRATION_SECONDS", 0.0)
    monkeypatch.setattr(config, "PROFILE_MIN_SAMPLES", 3)
    session = SessionData(session_id="profile-once")
    raw = RawSignals(
        face_present=True,
        lean=0.0,
        eye_contact=0.7,
        spine_ratio=0.9,
        emotion_label="neutral",
    )

    _feed(session, raw)

    assert session.profile_sent is True
    assert session.pending_profile is not None

    # main.py bu noktada pending_profile'i tuketip None'a cekiyor
    session.pending_profile = None

    # ayni kisi icin daha fazla kare -> yeni bir profil UretilMEZ
    _feed(session, raw)
    assert session.pending_profile is None


def test_profile_retriggers_for_new_person(monkeypatch):
    monkeypatch.setattr(config, "CALIBRATION_SECONDS", 0.0)
    monkeypatch.setattr(config, "PROFILE_MIN_SAMPLES", 3)
    monkeypatch.setattr(config, "NO_FACE_TIMEOUT_SECONDS", 0.01)
    session = SessionData(session_id="profile-retrigger")
    raw = RawSignals(
        face_present=True, lean=0.0, eye_contact=0.7, spine_ratio=0.9, emotion_label="neutral"
    )

    _feed(session, raw)
    assert session.profile_sent is True
    session.pending_profile = None

    # kisi ayrildi -> IDLE (timeout dolana kadar bekle)
    time.sleep(0.02)
    update_session(session, RawSignals(face_present=False))
    assert session.state is SessionState.IDLE
    assert session.profile_sent is False

    # yeni kisi geldi -> tek seferlik profil yeniden uretilir
    _feed(session, raw)
    assert session.profile_sent is True
    assert session.pending_profile is not None
