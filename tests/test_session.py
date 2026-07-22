"""Tests for SessionData state machine and SessionManager."""

from __future__ import annotations

import time
import pytest
from backend.cv_pipeline.session import SessionData, SessionState, RawSignals
from backend.cv_pipeline import config


class TestSessionStateMachine:
    """Test state transitions: IDLE -> CALIBRATING -> ACTIVE -> IDLE."""

    def test_new_session_starts_idle(self):
        s = SessionData(session_id="test")
        assert s.state is SessionState.IDLE

    def test_reset_for_new_person_goes_to_calibrating(self):
        s = SessionData(session_id="test")
        s.reset_for_new_person()
        assert s.state is SessionState.CALIBRATING
        assert s.calibration_started_at is not None
        assert s.profile_sent is False
        assert len(s.calibration_lean_samples) == 0

    def test_reset_for_new_person_clears_history(self):
        s = SessionData(session_id="test")
        s.lean_history.append(0.1)
        s.eye_history.append(0.5)
        s.baseline_lean = 0.15
        s.is_focused = True
        s.focus_time = 10.0

        s.reset_for_new_person()
        assert len(s.lean_history) == 0
        assert len(s.eye_history) == 0
        assert s.baseline_lean is None
        assert s.is_focused is False
        assert s.focus_time == 0.0

    def test_reset_to_idle_clears_state(self):
        s = SessionData(session_id="test")
        s.state = SessionState.ACTIVE
        s.baseline_lean = 0.2
        s.is_focused = True
        s.focus_time = 5.0

        s.reset_to_idle()
        assert s.state is SessionState.IDLE
        assert s.baseline_lean is None
        assert s.is_focused is False
        assert s.focus_time == 0.0
        assert s.profile_sent is False


class TestSessionDataDefaultValues:
    """Ensure sensible defaults on creation."""

    def test_lean_history_maxlen_from_config(self):
        s = SessionData(session_id="test")
        assert s.lean_history.maxlen == config.LEAN_HISTORY_MAXLEN

    def test_eye_history_maxlen_from_config(self):
        s = SessionData(session_id="test")
        assert s.eye_history.maxlen == config.EYE_HISTORY_MAXLEN

    def test_default_emotion_is_empty_dict(self):
        raw = RawSignals()
        assert raw.emotion_scores == {}
        assert raw.emotion_label is None

    def test_touch_frame_updates_timestamp(self):
        s = SessionData(session_id="test")
        original = s.last_frame_at
        time.sleep(0.01)
        s.touch_frame()
        assert s.last_frame_at > original

    def test_ring_buffer_enforces_maxlen(self):
        s = SessionData(session_id="test")
        maxlen = config.LEAN_HISTORY_MAXLEN
        for i in range(maxlen + 10):
            s.lean_history.append(float(i))
        assert len(s.lean_history) == maxlen


class TestRawSignals:
    """RawSignals is a simple dataclass — verify defaults."""

    def test_default_face_not_present(self):
        raw = RawSignals()
        assert raw.face_present is False

    def test_all_optionals_default_to_none(self):
        raw = RawSignals()
        assert raw.lean is None
        assert raw.eye_contact is None
        assert raw.head_yaw_deg is None
        assert raw.spine_ratio is None
        assert raw.shoulder_tilt is None
        assert raw.arms_crossed is None
        assert raw.face_center_x is None
        assert raw.face_center_y is None
        assert raw.face_bbox_width is None
        assert raw.face_bbox_height is None
        assert raw.observation_ts is None

    def test_can_set_all_fields(self):
        raw = RawSignals(
            face_present=True,
            lean=0.15,
            eye_contact=0.85,
            head_yaw_deg=3.0,
            spine_ratio=0.92,
            shoulder_tilt=0.01,
            arms_crossed=False,
            emotion_label="happy",
            emotion_scores={"happy": 0.8, "neutral": 0.2},
        )
        assert raw.face_present is True
        assert raw.lean == 0.15
        assert raw.emotion_label == "happy"
