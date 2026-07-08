"""Postur: lean (T5), spine_ratio yaw-invariance + pitch (T6), arms_crossed gating (T7)."""
import copy
import math

from app.detectors.pose import (
    LEFT_HIP,
    LEFT_SHOULDER,
    LEFT_WRIST,
    LandmarkPoint,
    PoseResult,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
)
from app.detectors.posture import compute_arms_crossed, compute_lean, compute_spine


def _upright_pose(wrist_visibility=0.95, wrists_crossed=False):
    lm = {
        LEFT_SHOULDER: LandmarkPoint(x=-0.2, y=0.5, z=-0.1, visibility=0.99, presence=0.99),
        RIGHT_SHOULDER: LandmarkPoint(x=0.2, y=0.5, z=-0.1, visibility=0.99, presence=0.99),
        LEFT_HIP: LandmarkPoint(x=-0.15, y=0.0, z=-0.05, visibility=0.99, presence=0.99),
        RIGHT_HIP: LandmarkPoint(x=0.15, y=0.0, z=-0.05, visibility=0.99, presence=0.99),
    }
    if wrists_crossed:
        lm[LEFT_WRIST] = LandmarkPoint(x=0.02, y=0.3, z=-0.3, visibility=wrist_visibility, presence=0.9)
        lm[RIGHT_WRIST] = LandmarkPoint(x=-0.02, y=0.3, z=-0.3, visibility=wrist_visibility, presence=0.9)
    else:
        lm[LEFT_WRIST] = LandmarkPoint(x=-0.4, y=-0.2, z=-0.1, visibility=wrist_visibility, presence=0.9)
        lm[RIGHT_WRIST] = LandmarkPoint(x=0.4, y=-0.2, z=-0.1, visibility=wrist_visibility, presence=0.9)
    return PoseResult(world_landmarks=lm, image_landmarks=lm, bbox=(0.2, 0.1, 0.8, 0.9))


def _rotate_around_vertical_axis(pose: PoseResult, angle_deg: float) -> PoseResult:
    angle = math.radians(angle_deg)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    rotated = copy.deepcopy(pose)
    for idx in (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP):
        lm = rotated.world_landmarks[idx]
        x, z = lm.x, lm.z
        lm.x = x * cos_a - z * sin_a
        lm.z = x * sin_a + z * cos_a
    return rotated


def test_lean_is_shoulder_minus_hip_world_z():
    pose = _upright_pose()
    lean = compute_lean(pose)
    assert math.isclose(lean, -0.1 - (-0.05), rel_tol=1e-6)


def test_spine_ratio_invariant_to_sideways_yaw_rotation():
    """T6: yana donme spine_ratio'yu bozmamali (eski shoulder/hip-width oraninin aksine)."""
    pose = _upright_pose()
    ratio_before, _ = compute_spine(pose, head_pitch_deg=0.0)
    rotated = _rotate_around_vertical_axis(pose, 20.0)
    ratio_after, _ = compute_spine(rotated, head_pitch_deg=0.0)
    assert math.isclose(ratio_before, ratio_after, rel_tol=1e-6)


def test_spine_ratio_drops_with_real_head_droop():
    pose = _upright_pose()
    ratio_flat, _ = compute_spine(pose, head_pitch_deg=0.0)
    ratio_drooped, _ = compute_spine(pose, head_pitch_deg=30.0)
    assert ratio_drooped < ratio_flat


def test_arms_crossed_is_none_when_wrist_visibility_low():
    pose = _upright_pose(wrist_visibility=0.1)
    assert compute_arms_crossed(pose) is None


def test_arms_crossed_true_when_wrists_close_and_in_front():
    pose = _upright_pose(wrists_crossed=True)
    assert compute_arms_crossed(pose) is True


def test_arms_not_crossed_when_wrists_apart():
    pose = _upright_pose(wrists_crossed=False)
    assert compute_arms_crossed(pose) is False
