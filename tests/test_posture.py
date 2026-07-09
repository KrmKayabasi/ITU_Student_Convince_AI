"""Tests for posture calculations (lean, spine, arms_crossed)."""

from __future__ import annotations

import pytest
from backend.cv_pipeline.detectors.posture import compute_lean, compute_spine, compute_arms_crossed
from backend.cv_pipeline.detectors.pose import PoseResult, LandmarkPoint


def _make_pose(shoulder_z, hip_z, shoulder_dy=0.0, wrist_vis=0.95, wrist_distance=0.3,
               torso_z_mean=0.0, wrist_z=0.0):
    """Factory for PoseResult with configurable world landmark positions."""
    world = {
        11: LandmarkPoint(x=0.3, y=0.4, z=-0.5 if shoulder_z is None else shoulder_z,
                          visibility=0.99, presence=0.99),
        12: LandmarkPoint(x=0.1, y=0.4 + shoulder_dy, z=-0.5 if shoulder_z is None else shoulder_z,
                          visibility=0.99, presence=0.99),
        13: LandmarkPoint(x=0.35, y=0.55, z=-0.4, visibility=0.95, presence=0.95),
        14: LandmarkPoint(x=0.05, y=0.55, z=-0.4, visibility=0.95, presence=0.95),
        15: LandmarkPoint(x=wrist_distance, y=0.70, z=wrist_z, visibility=wrist_vis, presence=wrist_vis),
        16: LandmarkPoint(x=-wrist_distance, y=0.70, z=wrist_z, visibility=wrist_vis, presence=wrist_vis),
        23: LandmarkPoint(x=0.28, y=0.75, z=-0.6 if hip_z is None else hip_z,
                          visibility=0.99, presence=0.99),
        24: LandmarkPoint(x=0.12, y=0.75, z=-0.6 if hip_z is None else hip_z,
                          visibility=0.99, presence=0.99),
    }
    from backend.cv_pipeline.detectors.pose import RELEVANT_INDICES
    image = {idx: LandmarkPoint(x=0.5, y=0.5, z=0, visibility=1, presence=1)
             for idx in RELEVANT_INDICES}
    return PoseResult(world_landmarks=world, image_landmarks=image, bbox=(0, 0, 1, 1))


class TestComputeLean:
    """Lean = shoulder_z - hip_z (world coords, meters)."""

    def test_upright_person_lean_near_zero(self):
        pose = _make_pose(shoulder_z=-0.5, hip_z=-0.5)
        lean = compute_lean(pose)
        assert abs(lean) < 0.01, f"Upright person should have lean ~0, got {lean}"

    def test_leaning_forward_positive_lean(self):
        """Leaning forward moves shoulders closer to camera (less negative z)."""
        pose = _make_pose(shoulder_z=-0.3, hip_z=-0.6)
        lean = compute_lean(pose)
        assert lean > 0.15, f"Forward lean should give positive lean, got {lean}"

    def test_leaning_back_negative_lean(self):
        pose = _make_pose(shoulder_z=-0.7, hip_z=-0.5)
        lean = compute_lean(pose)
        assert lean < -0.15, f"Backward lean should give negative lean, got {lean}"


class TestComputeSpine:
    """Spine ratio and shoulder tilt from world landmarks."""

    def test_upright_torso_spine_near_one(self):
        pose = _make_pose(shoulder_z=-0.5, hip_z=-0.6)
        ratio, tilt = compute_spine(pose, head_pitch_deg=0.0)
        assert ratio > 0.7, f"Upright torso should have high spine_ratio, got {ratio}"

    def test_head_pitch_reduces_spine_ratio(self):
        pose = _make_pose(shoulder_z=-0.5, hip_z=-0.6)
        ratio_neutral, _ = compute_spine(pose, head_pitch_deg=0.0)
        ratio_pitched, _ = compute_spine(pose, head_pitch_deg=40.0)
        assert ratio_pitched < ratio_neutral, \
            f"Head pitch should reduce spine_ratio ({ratio_pitched} >= {ratio_neutral})"

    def test_shoulder_tilt_zero_when_level(self):
        pose = _make_pose(shoulder_z=-0.5, hip_z=-0.6, shoulder_dy=0.0)
        _, tilt = compute_spine(pose, head_pitch_deg=0.0)
        assert tilt < 0.02, f"Level shoulders should give near-zero tilt, got {tilt}"

    def test_shoulder_tilt_nonzero_when_uneven(self):
        pose = _make_pose(shoulder_z=-0.5, hip_z=-0.6, shoulder_dy=0.1)
        _, tilt = compute_spine(pose, head_pitch_deg=0.0)
        assert tilt > 0.02, f"Uneven shoulders should give non-zero tilt, got {tilt}"


class TestComputeArmsCrossed:
    def test_arms_apart_not_crossed(self):
        pose = _make_pose(shoulder_z=-0.5, hip_z=-0.6, wrist_vis=0.95, wrist_distance=0.3, wrist_z=-0.35)
        result = compute_arms_crossed(pose)
        assert result is False or result is None, \
            f"Arms apart should not be crossed, got {result}"

    def test_low_visibility_returns_none(self):
        pose = _make_pose(shoulder_z=-0.5, hip_z=-0.6, wrist_vis=0.3)
        result = compute_arms_crossed(pose)
        assert result is None, \
            f"Low wrist visibility should return None, got {result}"

    def test_result_is_bool_or_none(self):
        pose = _make_pose(shoulder_z=-0.5, hip_z=-0.6, wrist_vis=0.95, wrist_distance=0.3, wrist_z=-0.35)
        result = compute_arms_crossed(pose)
        assert result is None or isinstance(result, bool)
