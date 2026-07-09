"""
Ham sinyalleri oturuma isleyen (baseline guncelleme, focus streak) ve iki
farkli JSON'u ureten katman:
  - build_initial_profile: kisi basina TEK SEFERLIK zengin profil (/profile).
  - build_focus_payload: ~2.5sn'de bir push edilen hafif focus payload'i (/focus).

Agirlik tablosu ve formuller, teknik spesifikasyon dokumanindaki Skor Modeli
(Bolum 4) temel alinarak yazildi; ancak lean/spine artik world-landmark
tabanli, gaze artik bas pozu ile duzeltilmis sinyaller uzerinden calisiyor
(bkz. cv_pipeline/detectors/).
"""

from __future__ import annotations

import time

import numpy as np

from backend.cv_pipeline import config
from backend.cv_pipeline.session import RawSignals, SessionData, SessionState

_EMOTION_ENERGY = {
    "happy": 0.9,
    "surprise": 0.8,
    "neutral": 0.5,
    "sad": 0.3,
    "fear": 0.2,
    "angry": 0.6,
    "disgust": 0.4,
    "contempt": 0.45,
}


def _update_focus(session: SessionData, raw: RawSignals, now: float) -> None:
    """Odaklanma, her karede guncellenir (T-focus): dikkat dagilinca aninda
    0'a sifirlanir, kesintisiz surdukce focus_time artar."""
    focused = bool(raw.face_present) and (raw.eye_contact or 0.0) >= config.FOCUS_EYE_CONTACT_THRESHOLD
    if focused:
        if session.focus_streak_started_at is None:
            session.focus_streak_started_at = now
        session.focus_time = now - session.focus_streak_started_at
    else:
        session.focus_streak_started_at = None
        session.focus_time = 0.0
    session.is_focused = focused


def update_session(session: SessionData, raw: RawSignals) -> None:
    """Tek bir islenmis kareden gelen ham sinyali oturum durumuna isler.

    State machine gecisleri (IDLE / CALIBRATING / ACTIVE), baseline/ring-buffer
    guncellemeleri ve focus streak burada olur. Yeterli ornek toplanir
    toplanmaz (PROFILE_MIN_SAMPLES) tek seferlik zengin profil bu fonksiyon
    icinde uretilip session.pending_profile'a yazilir; main.py'daki asyncio
    dongusu bunu poll edip /profile abonelerine push eder.
    """
    now = time.time()
    session.last_raw = raw

    # NOT: _update_focus, reset_for_new_person/reset_to_idle SONRASINDA
    # cagrilir; aksi halde bu resetler az once hesaplanan focus durumunu
    # ezer (reset_for_new_person/reset_to_idle is_focused'i False'a ceker).
    if not raw.face_present:
        if (
            session.state is not SessionState.IDLE
            and now - session.last_face_seen_at > config.NO_FACE_TIMEOUT_SECONDS
        ):
            session.reset_to_idle()
        _update_focus(session, raw, now)
        return

    session.last_face_seen_at = now

    if session.state is SessionState.IDLE:
        session.reset_for_new_person()

    if session.state is SessionState.CALIBRATING:
        if raw.lean is not None:
            session.calibration_lean_samples.append(raw.lean)
        elapsed = now - (session.calibration_started_at or now)
        if elapsed >= config.CALIBRATION_SECONDS:
            samples = session.calibration_lean_samples or [0.0]
            session.baseline_lean = float(sum(samples) / len(samples))
            session.state = SessionState.ACTIVE
        else:
            _update_focus(session, raw, now)
            return  # kalibrasyon bitmeden ring buffer'lara yazmayalim

    # state == ACTIVE
    _update_focus(session, raw, now)
    if raw.lean is not None:
        session.lean_history.append(raw.lean)
    if raw.eye_contact is not None:
        session.eye_history.append(raw.eye_contact)

    if not session.profile_sent and len(session.eye_history) >= config.PROFILE_MIN_SAMPLES:
        session.pending_profile = build_initial_profile(session)
        session.profile_sent = True


def _score_components(session: SessionData, raw: RawSignals) -> tuple[float, float, float, float, float]:
    """Anlik (attention, openness, energy) skoru + ara degerler."""
    smoothed_lean = float(sum(session.lean_history) / len(session.lean_history)) if session.lean_history else 0.0
    baseline = session.baseline_lean if session.baseline_lean is not None else 0.0
    delta_lean = smoothed_lean - baseline

    avg_eye_contact = float(sum(session.eye_history) / len(session.eye_history)) if session.eye_history else 0.5

    spine_ratio = raw.spine_ratio if raw.spine_ratio is not None else 0.75
    spine_score = 1.0 if spine_ratio > config.SPINE_UPRIGHT_THRESHOLD else 0.5

    lean_score = 1.0 if delta_lean < -0.03 else (0.0 if delta_lean > 0.03 else 0.5)

    emotion_energy = _EMOTION_ENERGY.get(raw.emotion_label or "neutral", 0.5)

    attention = (
        avg_eye_contact * 0.45
        + lean_score * 0.30
        + emotion_energy * 0.15
        + spine_score * 0.10
    )
    openness = (
        avg_eye_contact * 0.25
        + lean_score * 0.20
        + emotion_energy * 0.20
        + spine_score * 0.25
        + (0.0 if raw.arms_crossed else 1.0 if raw.arms_crossed is not None else 0.5) * 0.10
    )
    energy = (
        emotion_energy * 0.50
        + lean_score * 0.20
        + spine_score * 0.20
        + avg_eye_contact * 0.10
    )
    return attention, openness, energy, delta_lean, avg_eye_contact


def build_initial_profile(session: SessionData) -> dict:
    """Kisi basina TEK SEFERLIK cagrilir: yeterli veri toplanir toplanmaz
    zengin profili uretir (/profile)."""
    raw = session.last_raw
    attention, openness, energy, delta_lean, avg_eye_contact = _score_components(session, raw)

    arms_valid = raw.arms_crossed is not None

    return {
        "session_id": session.session_id,
        "ts": time.time(),
        "state": session.state.value,
        "person": {"present": raw.face_present},
        "signals": {
            "lean": {
                "value": round(delta_lean, 4),
                "baseline": round(session.baseline_lean or 0.0, 4),
                "confidence": 0.8 if session.lean_history else 0.0,
            },
            "eye_contact": {
                "value": round(avg_eye_contact, 4),
                "head_yaw_deg": raw.head_yaw_deg,
                "confidence": 0.9 if raw.head_yaw_deg is not None else 0.5,
            },
            "spine": {
                "ratio": raw.spine_ratio,
                "tilt": raw.shoulder_tilt,
                "confidence": 0.7,
            },
            "arms_crossed": {
                "value": bool(raw.arms_crossed) if arms_valid else False,
                "valid": arms_valid,
            },
            "emotion": {
                "dominant": raw.emotion_label,
                "scores": raw.emotion_scores,
                "confidence": 0.55,
            },
        },
        "scores": {
            "attention": round(attention, 4),
            "openness": round(openness, 4),
            "energy": round(energy, 4),
        },
        "schema_version": "1.0",
    }


def build_focus_payload(session: SessionData) -> dict:
    """~2.5sn'de bir cagrilir: hafif is_focused + focus_time payload'i (/focus)."""
    return {
        "session_id": session.session_id,
        "ts": time.time(),
        "is_focused": session.is_focused,
        "focus_time": round(session.focus_time, 2),
    }


def build_debug_payload(session: SessionData) -> dict:
    """SADECE gelistirme/test amacli (/debug): /profile'in aksine TEK SEFERLIK
    degil, her cagrida session.last_raw'dan anlik tum ham + turetilmis
    degerleri doner. Uretim kontratinin (/profile, /focus) bir parcasi
    DEGILDIR; canli kamera testinde olculen degerleri dogrulamak icindir."""
    raw = session.last_raw
    attention, openness, energy, delta_lean, avg_eye_contact = _score_components(session, raw)
    return {
        "session_id": session.session_id,
        "ts": time.time(),
        "state": session.state.value,
        "is_focused": session.is_focused,
        "focus_time": round(session.focus_time, 2),
        "raw": {
            "face_present": raw.face_present,
            "lean": raw.lean,
            "eye_contact": raw.eye_contact,
            "head_yaw_deg": raw.head_yaw_deg,
            "spine_ratio": raw.spine_ratio,
            "shoulder_tilt": raw.shoulder_tilt,
            "arms_crossed": raw.arms_crossed,
            "emotion_label": raw.emotion_label,
            "emotion_scores": raw.emotion_scores,
        },
        "smoothed": {
            "delta_lean": round(delta_lean, 4),
            "baseline_lean": round(session.baseline_lean, 4) if session.baseline_lean is not None else None,
            "avg_eye_contact": round(avg_eye_contact, 4),
        },
        "live_score_preview": {
            "attention": round(attention, 4),
            "openness": round(openness, 4),
            "energy": round(energy, 4),
        },
    }
