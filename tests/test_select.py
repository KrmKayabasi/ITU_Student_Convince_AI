"""Birincil kisi secimi: en buyuk bbox + kare-kare zipla(t)mama hysteresis (T10)."""
from app.detectors.face import FaceResult
from app.detectors.select import select_primary_face


def _face(bbox):
    return FaceResult(landmarks=[], blendshapes={}, transform_matrix=None, bbox=bbox)


def test_selects_largest_face_with_no_history():
    small = _face((0.1, 0.1, 0.2, 0.2))
    big = _face((0.5, 0.5, 0.8, 0.8))
    assert select_primary_face([small, big]) is big


def test_continuity_prefers_previous_person_on_near_tie():
    a = _face((0.10, 0.10, 0.30, 0.30))  # area 0.04
    b = _face((0.50, 0.50, 0.71, 0.71))  # area 0.0441 (marjinal olarak buyuk)
    picked = select_primary_face([a, b], previous_center=(0.20, 0.20))
    assert picked is a


def test_continuity_yields_to_much_larger_face():
    small = _face((0.1, 0.1, 0.2, 0.2))  # area 0.01
    big = _face((0.5, 0.5, 0.8, 0.8))  # area 0.09
    picked = select_primary_face([small, big], previous_center=(0.15, 0.15))
    assert picked is big


def test_empty_faces_returns_none():
    assert select_primary_face([]) is None
