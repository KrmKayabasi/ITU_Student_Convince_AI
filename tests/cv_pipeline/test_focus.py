"""Focus streak: kesintisiz surdukce artar, dikkat dagilinca aninda sifirlanir (T12)."""
from backend.cv_pipeline import config
from backend.cv_pipeline.scoring import update_session
from backend.cv_pipeline.session import RawSignals, SessionData


def test_focus_streak_accumulates_while_eye_contact_high(monkeypatch):
    monkeypatch.setattr(config, "FOCUS_EYE_CONTACT_THRESHOLD", 0.5)
    session = SessionData(session_id="focus-accum")

    raw_focused = RawSignals(face_present=True, eye_contact=0.9, lean=0.0)
    update_session(session, raw_focused)
    assert session.is_focused is True
    first = session.focus_time

    update_session(session, raw_focused)
    assert session.focus_time >= first


def test_focus_resets_immediately_when_attention_drops(monkeypatch):
    monkeypatch.setattr(config, "FOCUS_EYE_CONTACT_THRESHOLD", 0.5)
    session = SessionData(session_id="focus-reset")

    update_session(session, RawSignals(face_present=True, eye_contact=0.9, lean=0.0))
    assert session.is_focused is True

    update_session(session, RawSignals(face_present=True, eye_contact=0.1, lean=0.0))
    assert session.is_focused is False
    assert session.focus_time == 0.0


def test_focus_resets_when_face_disappears(monkeypatch):
    monkeypatch.setattr(config, "FOCUS_EYE_CONTACT_THRESHOLD", 0.5)
    session = SessionData(session_id="focus-noface")

    update_session(session, RawSignals(face_present=True, eye_contact=0.9, lean=0.0))
    assert session.is_focused is True

    update_session(session, RawSignals(face_present=False))
    assert session.is_focused is False
    assert session.focus_time == 0.0
