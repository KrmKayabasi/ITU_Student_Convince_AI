"""
Shared fixtures for CV pipeline tests.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
import numpy as np


@pytest.fixture
def mock_mediapipe_image():
    """Returns a mock MediaPipe Image that won't actually load model files."""
    with patch("mediapipe.tasks.python.vision.FaceLandmarker"), \
         patch("mediapipe.tasks.python.vision.PoseLandmarker"), \
         patch("mediapipe.tasks.python.vision.FaceLandmarkerOptions"), \
         patch("mediapipe.tasks.python.vision.PoseLandmarkerOptions"), \
         patch("mediapipe.tasks.python.core.base_options.BaseOptions"):
        yield


@pytest.fixture
def face_blendshapes_center():
    """Blendshapes for a person looking straight at the camera (centered gaze)."""
    return {
        "eyeLookInLeft": 0.0, "eyeLookInRight": 0.0,
        "eyeLookOutLeft": 0.0, "eyeLookOutRight": 0.0,
        "eyeLookUpLeft": 0.0, "eyeLookUpRight": 0.0,
        "eyeLookDownLeft": 0.0, "eyeLookDownRight": 0.0,
        "eyeBlinkLeft": 0.0, "eyeBlinkRight": 0.0,
    }


@pytest.fixture
def face_blendshapes_looking_away():
    """Blendshapes for someone looking far to the right and down."""
    return {
        "eyeLookInLeft": 0.0, "eyeLookInRight": 0.6,
        "eyeLookOutLeft": 0.0, "eyeLookOutRight": 0.4,
        "eyeLookUpLeft": 0.0, "eyeLookUpRight": 0.0,
        "eyeLookDownLeft": 0.0, "eyeLookDownRight": 0.5,
        "eyeBlinkLeft": 0.0, "eyeBlinkRight": 0.0,
    }


@pytest.fixture
def face_blendshapes_blinking():
    """Blendshapes during a full blink (eyes closed)."""
    return {
        "eyeLookInLeft": 0.0, "eyeLookInRight": 0.0,
        "eyeLookOutLeft": 0.0, "eyeLookOutRight": 0.0,
        "eyeLookUpLeft": 0.2, "eyeLookUpRight": 0.2,   # artifactual spike
        "eyeLookDownLeft": 0.3, "eyeLookDownRight": 0.3,  # artifactual spike
        "eyeBlinkLeft": 1.0, "eyeBlinkRight": 1.0,
    }


@pytest.fixture
def head_pose_center():
    """Head facing straight at camera."""
    from backend.cv_pipeline.detectors.face import HeadPose
    return HeadPose(yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0)


@pytest.fixture
def head_pose_turned():
    """Head turned 35 degrees right, 10 up."""
    from backend.cv_pipeline.detectors.face import HeadPose
    return HeadPose(yaw_deg=35.0, pitch_deg=10.0, roll_deg=2.0)


@pytest.fixture
def head_pose_extreme():
    """Head at extreme angle (80 degrees yaw)."""
    from backend.cv_pipeline.detectors.face import HeadPose
    return HeadPose(yaw_deg=80.0, pitch_deg=5.0, roll_deg=0.0)


@pytest.fixture
def sample_session():
    """A fresh ACTIVE session with some eye/lean history populated."""
    from backend.cv_pipeline.session import SessionData, SessionState
    s = SessionData(session_id="test-session-1")
    s.state = SessionState.ACTIVE
    s.baseline_lean = 0.15
    for _ in range(50):
        s.lean_history.append(0.14)
        s.eye_history.append(0.82)
    return s


@pytest.fixture
def sample_raw_signals():
    """A typical RawSignals with a face looking at camera, neutral emotion."""
    from backend.cv_pipeline.session import RawSignals
    return RawSignals(
        face_present=True,
        lean=0.14,
        eye_contact=0.85,
        head_yaw_deg=5.0,
        spine_ratio=0.90,
        shoulder_tilt=0.02,
        arms_crossed=False,
        emotion_label="neutral",
        emotion_scores={"neutral": 0.65, "happy": 0.10, "sad": 0.05,
                        "surprise": 0.05, "angry": 0.05, "fear": 0.05,
                        "disgust": 0.03, "contempt": 0.02},
    )


@pytest.fixture
def sample_pose_world_landmarks():
    """Mock pose world landmarks for a typical upright seated person."""
    from backend.cv_pipeline.detectors.pose import LandmarkPoint
    return {
        11: LandmarkPoint(x=0.3, y=0.4, z=-0.5, visibility=0.99, presence=0.99),   # L shoulder
        12: LandmarkPoint(x=0.1, y=0.4, z=-0.5, visibility=0.99, presence=0.99),   # R shoulder
        13: LandmarkPoint(x=0.35, y=0.55, z=-0.4, visibility=0.95, presence=0.95), # L elbow
        14: LandmarkPoint(x=0.05, y=0.55, z=-0.4, visibility=0.95, presence=0.95), # R elbow
        15: LandmarkPoint(x=0.40, y=0.70, z=-0.35, visibility=0.90, presence=0.90),# L wrist
        16: LandmarkPoint(x=0.00, y=0.70, z=-0.35, visibility=0.90, presence=0.90),# R wrist
        23: LandmarkPoint(x=0.28, y=0.75, z=-0.6, visibility=0.99, presence=0.99),  # L hip
        24: LandmarkPoint(x=0.12, y=0.75, z=-0.6, visibility=0.99, presence=0.99),  # R hip
    }
