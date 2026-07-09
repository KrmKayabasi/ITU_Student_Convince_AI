"""Tests for scoring formulas: update_session, build_profile, build_focus, build_debug."""

from __future__ import annotations

import time
import pytest
import numpy as np
from backend.cv_pipeline.scoring import (
    update_session,
    build_initial_profile,
    build_focus_payload,
    build_debug_payload,
    _score_components,
    _lean_confidence,
    _eye_contact_confidence,
    _spine_confidence,
    _emotion_confidence,
)
from backend.cv_pipeline.session import SessionData, SessionState, RawSignals


class TestUpdateSession:
    """Test the session update logic including state transitions and focus tracking."""

    def test_face_present_transitions_idle_to_calibrating(self):
        s = SessionData(session_id="test")
        raw = RawSignals(face_present=True, lean=0.1, eye_contact=0.8)
        update_session(s, raw)
        assert s.state is SessionState.CALIBRATING

    def test_no_face_triggers_idle_after_timeout(self):
        s = SessionData(session_id="test")
        s.state = SessionState.ACTIVE
        s.last_face_seen_at = time.time() - 10.0  # well past timeout
        raw = RawSignals(face_present=False)
        update_session(s, raw)
        assert s.state is SessionState.IDLE

    def test_no_face_before_timeout_keeps_state(self):
        s = SessionData(session_id="test")
        s.state = SessionState.ACTIVE
        s.last_face_seen_at = time.time()  # just now
        raw = RawSignals(face_present=False)
        update_session(s, raw)
        assert s.state is SessionState.ACTIVE

    def test_focus_starts_when_eye_contact_above_threshold(self):
        import time
        s = SessionData(session_id="test")
        s.state = SessionState.ACTIVE
        raw = RawSignals(face_present=True, eye_contact=0.9)
        update_session(s, raw)
        assert s.is_focused is True
        assert s.focus_streak_started_at is not None
        # focus_time may be 0.0 within the same timestamp, so verify started
        assert s.focus_time >= 0.0

    def test_focus_resets_when_eye_contact_drops(self):
        s = SessionData(session_id="test")
        s.state = SessionState.ACTIVE
        # First frame: focused
        update_session(s, RawSignals(face_present=True, eye_contact=0.9))
        assert s.is_focused is True
        # Second frame: lost focus
        update_session(s, RawSignals(face_present=True, eye_contact=0.1))
        assert s.is_focused is False
        assert s.focus_time == 0.0

    def test_profile_generated_once_per_person(self):
        s = SessionData(session_id="test")
        s.state = SessionState.ACTIVE
        s.baseline_lean = 0.1
        # Feed enough samples to trigger profile
        for _ in range(60):
            update_session(s, RawSignals(face_present=True, lean=0.1, eye_contact=0.8))
        assert s.profile_sent is True
        assert s.pending_profile is not None
        assert "scores" in s.pending_profile


class TestScoreComponents:
    """Test the internal scoring calculation."""

    def test_returns_tuple_of_5(self):
        s = SessionData(session_id="test")
        s.baseline_lean = 0.1
        for _ in range(10):
            s.lean_history.append(0.1)
            s.eye_history.append(0.8)
        raw = RawSignals(spine_ratio=0.9, shoulder_tilt=0.02,
                         arms_crossed=False, emotion_label="neutral")
        result = _score_components(s, raw)
        assert len(result) == 5
        attention, openness, energy, delta_lean, avg_eye = result
        assert 0.0 <= attention <= 1.0

    def test_all_outputs_in_range(self):
        s = SessionData(session_id="test")
        s.baseline_lean = 0.1
        for _ in range(30):
            s.lean_history.append(0.1)
            s.eye_history.append(0.8)
        raw = RawSignals(spine_ratio=0.9, shoulder_tilt=0.02,
                         arms_crossed=False, emotion_label="neutral")
        attn, opn, en, dl, ae = _score_components(s, raw)
        for val, name in [(attn, "attention"), (opn, "openness"), (en, "energy")]:
            assert 0.0 <= val <= 1.0, f"{name}={val} out of [0,1]"

    def test_empty_history_defaults_sensibly(self):
        s = SessionData(session_id="test")
        s.baseline_lean = 0.0
        raw = RawSignals()
        attn, opn, en, dl, ae = _score_components(s, raw)
        # Should not crash and should return finite values
        assert all(np.isfinite(x) for x in (attn, opn, en, dl, ae))


class TestBuildProfile:
    """Test profile JSON construction."""

    def test_profile_has_required_keys(self, sample_session, sample_raw_signals):
        sample_session.last_raw = sample_raw_signals
        profile = build_initial_profile(sample_session)
        for key in ("session_id", "ts", "state", "person", "signals", "scores", "schema_version"):
            assert key in profile, f"Missing key: {key}"

    def test_profile_score_keys(self, sample_session, sample_raw_signals):
        sample_session.last_raw = sample_raw_signals
        profile = build_initial_profile(sample_session)
        scores = profile["scores"]
        for key in ("attention", "openness", "energy"):
            assert key in scores
            assert 0.0 <= scores[key] <= 1.0

    def test_profile_schema_version(self, sample_session, sample_raw_signals):
        sample_session.last_raw = sample_raw_signals
        profile = build_initial_profile(sample_session)
        assert profile["schema_version"] == "1.0"


class TestBuildFocusPayload:
    def test_focus_payload_keys(self, sample_session):
        payload = build_focus_payload(sample_session)
        assert "session_id" in payload
        assert "ts" in payload
        assert "is_focused" in payload
        assert "focus_time" in payload
        assert isinstance(payload["is_focused"], bool)


class TestBuildDebugPayload:
    def test_debug_payload_keys(self, sample_session, sample_raw_signals):
        sample_session.last_raw = sample_raw_signals
        payload = build_debug_payload(sample_session)
        assert "raw" in payload
        assert "smoothed" in payload
        assert "live_score_preview" in payload


class TestDynamicConfidence:
    """Test that confidence functions return dynamic values, not hardcoded constants."""

    def test_lean_confidence_zero_for_empty(self):
        assert _lean_confidence([]) == 0.0

    def test_lean_confidence_grows_with_samples(self):
        c10 = _lean_confidence([0.1] * 10)
        c50 = _lean_confidence([0.1] * 50)
        assert c50 > c10, f"Confidence should grow: c10={c10}, c50={c50}"

    def test_lean_confidence_capped_at_one(self):
        c = _lean_confidence([0.1] * 1000)
        assert c <= 1.0

    def test_eye_contact_confidence_no_data(self):
        assert _eye_contact_confidence(None) == 0.5

    def test_eye_contact_confidence_with_data(self):
        assert _eye_contact_confidence(5.0) == 0.9

    def test_spine_confidence_plausible(self):
        c = _spine_confidence(0.9, 0.02)
        assert c > 0.5

    def test_spine_confidence_implausible(self):
        c = _spine_confidence(1.5, 0.1)
        assert c < 0.6  # should be penalized

    def test_emotion_confidence_high_margin(self):
        c = _emotion_confidence({"happy": 0.9, "neutral": 0.05, "sad": 0.05})
        assert c > 0.6, f"High margin should give high confidence, got {c}"

    def test_emotion_confidence_low_margin(self):
        c = _emotion_confidence({"happy": 0.35, "neutral": 0.33, "sad": 0.32})
        assert c < 0.5, f"Low margin should give low confidence, got {c}"

    def test_emotion_confidence_empty(self):
        assert _emotion_confidence({}) == 0.0
