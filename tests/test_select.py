"""Tests for primary person selection logic (select_primary_face, match_pose_to_face)."""

from __future__ import annotations

import numpy as np
from backend.cv_pipeline.detectors.select import select_primary_face, select_primary_person, match_pose_to_face
from backend.cv_pipeline.detectors.face import FaceResult, HeadPose
from backend.cv_pipeline.detectors.pose import PoseResult, LandmarkPoint


def _make_face_result(bbox_center_x=0.5, bbox_center_y=0.5, bbox_w=0.2, bbox_h=0.25):
    """Factory for a FaceResult with a specific bbox."""
    xmin = bbox_center_x - bbox_w / 2
    xmax = bbox_center_x + bbox_w / 2
    ymin = bbox_center_y - bbox_h / 2
    ymax = bbox_center_y + bbox_h / 2
    return FaceResult(
        landmarks=[(0.5, 0.5, 0.0)],
        blendshapes={},
        transform_matrix=np.eye(4),
        bbox=(xmin, ymin, xmax, ymax),
        head_pose=HeadPose(0, 0, 0),
    )


def _make_pose_result(xmin=0.4, ymin=0.35, xmax=0.6, ymax=0.7):
    """Factory for a PoseResult with a specific bbox."""
    from backend.cv_pipeline.detectors.pose import RELEVANT_INDICES
    world = {idx: LandmarkPoint(x=0.5, y=0.5, z=0, visibility=1, presence=1)
             for idx in RELEVANT_INDICES}
    image = {idx: LandmarkPoint(x=0.5, y=0.5, z=0, visibility=1, presence=1)
             for idx in RELEVANT_INDICES}
    return PoseResult(world_landmarks=world, image_landmarks=image, bbox=(xmin, ymin, xmax, ymax))


class TestSelectPrimaryFace:
    """Tests for the face selection heuristic."""

    def test_empty_list_returns_none(self):
        assert select_primary_face([]) is None

    def test_single_face_returns_it(self):
        face = _make_face_result()
        result = select_primary_face([face])
        assert result is face

    def test_largest_face_selected(self):
        small = _make_face_result(bbox_w=0.1, bbox_h=0.1)
        large = _make_face_result(bbox_w=0.3, bbox_h=0.4)
        result = select_primary_face([small, large])
        assert result is large

    def test_continuity_prefers_previous_position(self):
        """When a face is near the previous center, it should be preferred."""
        prev_face = _make_face_result(bbox_center_x=0.5, bbox_center_y=0.5,
                                      bbox_w=0.23, bbox_h=0.28)  # area = 0.0644
        far_large = _make_face_result(bbox_center_x=0.8, bbox_center_y=0.5,
                                      bbox_w=0.25, bbox_h=0.30)  # area = 0.075
        # prev_face area (0.0644) is >= 70% of far_large area (0.075 * 0.7 = 0.0525)
        # and prev_face center (0.5,0.5) is at previous_center, so continuity wins.
        result = select_primary_face([prev_face, far_large], previous_center=(0.5, 0.5))
        assert result is prev_face, \
            f"Continuity should prefer face at previous position, got bbox={result.bbox}"

    def test_continuity_ignored_when_much_smaller(self):
        """Tiny face at previous position should lose to much larger face elsewhere."""
        tiny = _make_face_result(bbox_center_x=0.5, bbox_center_y=0.5,
                                 bbox_w=0.05, bbox_h=0.05)
        huge = _make_face_result(bbox_center_x=0.8, bbox_center_y=0.5,
                                 bbox_w=0.4, bbox_h=0.5)
        result = select_primary_face([tiny, huge], previous_center=(0.5, 0.5))
        assert result is huge, \
            f"Much larger face should win despite continuity, got bbox={result.bbox}"


class TestMatchPoseToFace:
    """Tests for matching a face to a body pose."""

    def test_empty_poses_returns_none(self):
        face = _make_face_result()
        assert match_pose_to_face(face, []) is None

    def test_pose_containing_face_center_is_preferred(self):
        face = _make_face_result(bbox_center_x=0.5, bbox_center_y=0.5)
        containing = _make_pose_result(xmin=0.4, ymin=0.3, xmax=0.6, ymax=0.7)
        outside = _make_pose_result(xmin=0.7, ymin=0.7, xmax=0.9, ymax=0.9)
        result = match_pose_to_face(face, [outside, containing])
        assert result is containing, \
            "Pose whose bbox contains face center should be selected"


class TestSelectPrimaryPerson:
    """Integration test for face + pose selection."""

    def test_no_faces_returns_none_none(self):
        result = select_primary_person([], [])
        assert result == (None, None)

    def test_face_without_pose_returns_face_none(self):
        face = _make_face_result()
        result = select_primary_person([face], [])
        assert result == (face, None)

    def test_face_with_matching_pose(self):
        face = _make_face_result(bbox_center_x=0.5, bbox_center_y=0.5)
        pose = _make_pose_result(xmin=0.4, ymin=0.3, xmax=0.6, ymax=0.7)
        result_face, result_pose = select_primary_person([face], [pose])
        assert result_face is face
        assert result_pose is pose
