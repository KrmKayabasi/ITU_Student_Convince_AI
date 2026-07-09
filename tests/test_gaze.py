"""Tests for gaze computation (eye contact from blendshapes + head pose)."""

from __future__ import annotations

import pytest
from backend.cv_pipeline import config
from backend.cv_pipeline.detectors.gaze import compute_eye_contact, GazeResult
from backend.cv_pipeline.detectors.face import HeadPose


class TestEyeContactCompute:
    """Test the core eye contact computation."""

    def test_centered_gaze_yields_high_contact(
        self, face_blendshapes_center, head_pose_center
    ):
        result = compute_eye_contact(face_blendshapes_center, head_pose_center)
        assert result.eye_contact > 0.8, \
            f"Expected eye_contact > 0.8 for centered gaze, got {result.eye_contact}"
        assert not result.gated, "Centered head should not trigger gating"

    def test_looking_away_yields_low_contact(
        self, face_blendshapes_looking_away, head_pose_center
    ):
        result = compute_eye_contact(face_blendshapes_looking_away, head_pose_center)
        assert result.eye_contact < 0.7, \
            f"Expected eye_contact < 0.7 when looking away, got {result.eye_contact}"

    def test_head_turned_triggers_gating_and_penalty(
        self, face_blendshapes_center, head_pose_turned
    ):
        result = compute_eye_contact(face_blendshapes_center, head_pose_turned)
        assert result.gated, "Head turned 35deg should trigger gating"
        # Even with centered eyes, the gate penalty should reduce contact
        assert result.eye_contact < 0.9, \
            f"Gating penalty should reduce contact, got {result.eye_contact}"

    def test_extreme_head_angle_near_zero(
        self, face_blendshapes_center, head_pose_extreme
    ):
        result = compute_eye_contact(face_blendshapes_center, head_pose_extreme)
        assert result.eye_contact < 0.3, \
            f"Extreme head angle should give near-zero contact, got {result.eye_contact}"

    def test_blink_does_not_penalize_contact(
        self, face_blendshapes_blinking, head_pose_center
    ):
        """Blinking should NOT reduce eye contact — the blink softening must work."""
        result = compute_eye_contact(face_blendshapes_blinking, head_pose_center)
        # During a full blink, eyeLook* artifacts spike but blink softening
        # should completely cancel the deviation penalty.
        assert result.eye_contact > 0.7, \
            f"Blink should not tank eye_contact, got {result.eye_contact}"

    def test_output_is_clamped_zero_to_one(
        self, face_blendshapes_center, head_pose_center
    ):
        result = compute_eye_contact(face_blendshapes_center, head_pose_center)
        assert 0.0 <= result.eye_contact <= 1.0, \
            f"eye_contact must be in [0,1], got {result.eye_contact}"

    def test_returns_head_pose_angles(
        self, face_blendshapes_center, head_pose_turned
    ):
        result = compute_eye_contact(face_blendshapes_center, head_pose_turned)
        assert result.head_yaw_deg == 35.0
        assert result.head_pitch_deg == 10.0

    def test_missing_blendshapes_default_to_zero(self, head_pose_center):
        """Missing keys in blendshapes dict should default to 0.0."""
        result = compute_eye_contact({}, head_pose_center)
        assert result.eye_contact > 0.9, \
            f"Empty blendshapes (all zeros) + centered head = high contact, got {result.eye_contact}"


class TestGazeConfig:
    """Verify config thresholds are reasonable."""

    def test_yaw_gate_is_positive(self):
        assert config.GAZE_YAW_GATE_DEG > 0

    def test_pitch_gate_is_positive(self):
        assert config.GAZE_PITCH_GATE_DEG > 0

    def test_gate_penalty_between_zero_and_one(self):
        assert 0.0 < config.GAZE_GATE_PENALTY < 1.0

    def test_eye_deviation_scale_positive(self):
        assert config.GAZE_EYE_DEVIATION_SCALE > 0

    def test_head_alignment_max_positive(self):
        assert config.GAZE_HEAD_ALIGNMENT_MAX_DEG > 0


class TestHeadPoseFromMatrix:
    """Test rotation matrix to Euler angle conversion."""

    def test_identity_matrix_gives_zero_angles(self):
        import numpy as np
        from backend.cv_pipeline.detectors.face import head_pose_from_matrix
        hp = head_pose_from_matrix(np.eye(4))
        assert abs(hp.yaw_deg) < 1e-6
        assert abs(hp.pitch_deg) < 1e-6
        assert abs(hp.roll_deg) < 1e-6

    def test_yaw_rotation_detected(self):
        import numpy as np
        import math
        from backend.cv_pipeline.detectors.face import head_pose_from_matrix

        # 30-degree yaw rotation around Y axis
        angle = math.radians(30)
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        R = np.eye(4)
        R[0, 0] = cos_a
        R[0, 2] = sin_a
        R[2, 0] = -sin_a
        R[2, 2] = cos_a

        hp = head_pose_from_matrix(R)
        assert 25 < hp.yaw_deg < 35, f"Expected ~30deg yaw, got {hp.yaw_deg}"
