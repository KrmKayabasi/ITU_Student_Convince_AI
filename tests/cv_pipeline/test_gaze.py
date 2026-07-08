"""Gaze/eye_contact: bas pozu ile duzeltme + gating (T4)."""
from backend.cv_pipeline.detectors.face import HeadPose
from backend.cv_pipeline.detectors.gaze import compute_eye_contact


def test_frontal_centered_gaze_is_high_confidence():
    result = compute_eye_contact({}, HeadPose(yaw_deg=0, pitch_deg=0, roll_deg=0))
    assert result.eye_contact > 0.75
    assert result.gated is False


def test_head_turned_40deg_gates_eye_contact_even_if_eyes_centered():
    result = compute_eye_contact({}, HeadPose(yaw_deg=40, pitch_deg=0, roll_deg=0))
    assert result.eye_contact < 0.45
    assert result.gated is True


def test_eyes_deviated_lower_eye_contact_even_when_frontal():
    centered = compute_eye_contact({}, HeadPose(yaw_deg=0, pitch_deg=0, roll_deg=0))
    deviated = compute_eye_contact(
        {
            "eyeLookOutLeft": 0.8,
            "eyeLookOutRight": 0.8,
        },
        HeadPose(yaw_deg=0, pitch_deg=0, roll_deg=0),
    )
    assert deviated.eye_contact < centered.eye_contact
