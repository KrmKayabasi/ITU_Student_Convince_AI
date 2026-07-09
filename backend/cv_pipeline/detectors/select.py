"""
Birden fazla yüz/gövde olduğunda tek "birincil kişi" seçimi (sprint T10).

Kural (tek yerde, config'ten yapılandırılabilir): en büyük yüz bbox'ı
birincil kişi olarak seçilir. Kare kare zıplamayı önlemek için bir önceki
karede seçilen yüzün bbox merkezine yakın bir yüz varsa ve alanı çok daha
küçük değilse süreklilik tercih edilir (hysteresis).

Pose eşleme: seçilen yüzün bbox merkezinin, hangi kişinin gövde bbox'ı
içine düştüğüne (yoksa en yakın merkeze) bakılarak yapılır.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

from backend.cv_pipeline import config
from backend.cv_pipeline.detectors.face import FaceResult
from backend.cv_pipeline.detectors.pose import PoseResult

BBox = Tuple[float, float, float, float]


def _bbox_area(bbox: BBox) -> float:
    xmin, ymin, xmax, ymax = bbox
    return max(0.0, xmax - xmin) * max(0.0, ymax - ymin)


def _bbox_center(bbox: BBox) -> Tuple[float, float]:
    xmin, ymin, xmax, ymax = bbox
    return (xmin + xmax) / 2, (ymin + ymax) / 2


def select_primary_face(
    faces: List[FaceResult], previous_center: Optional[Tuple[float, float]] = None
) -> Optional[FaceResult]:
    """En buyuk bbox'a sahip yuzu secer; sureklilik icin hysteresis uygular."""
    if not faces:
        return None

    # Cache area and center calculations to avoid multiple redundant recalculations per frame.
    # Use id(f) as the key: FaceResult is a dataclass containing a numpy ndarray
    # (transform_matrix), which makes it unhashable.  id(f) is safe here because the
    # objects are created fresh each frame and CPython guarantees unique non-recycled ids
    # within the lifetime of a single frame's processing.
    face_metrics = {}
    for f in faces:
        area = _bbox_area(f.bbox)
        center = _bbox_center(f.bbox)
        face_metrics[id(f)] = (area, center)

    largest = max(faces, key=lambda f: face_metrics[id(f)][0])
    if previous_center is None:
        return largest

    largest_area = face_metrics[id(largest)][0]
    candidates = [
        f
        for f in faces
        if math.dist(face_metrics[id(f)][1], previous_center) < config.PRIMARY_PERSON_CONTINUITY_RADIUS
    ]
    if not candidates:
        return largest

    continuity_pick = max(candidates, key=lambda f: face_metrics[id(f)][0])
    if face_metrics[id(continuity_pick)][0] >= largest_area * config.PRIMARY_PERSON_CONTINUITY_AREA_RATIO:
        return continuity_pick
    return largest


def match_pose_to_face(face: FaceResult, poses: List[PoseResult]) -> Optional[PoseResult]:
    """Secilen yuze en iyi eslesen govdeyi bulur (bbox icerme, sonra merkez mesafesi)."""
    if not poses:
        return None

    face_center = _bbox_center(face.bbox)

    def score(pose: PoseResult) -> Tuple[int, float]:
        xmin, ymin, xmax, ymax = pose.bbox
        inside = xmin <= face_center[0] <= xmax and ymin <= face_center[1] <= ymax
        pose_center = _bbox_center(pose.bbox)
        distance = math.dist(pose_center, face_center)
        return (0 if inside else 1, distance)

    return min(poses, key=score)


def select_primary_person(
    faces: List[FaceResult],
    poses: List[PoseResult],
    previous_center: Optional[Tuple[float, float]] = None,
) -> Tuple[Optional[FaceResult], Optional[PoseResult]]:
    face = select_primary_face(faces, previous_center)
    if face is None:
        return None, None
    pose = match_pose_to_face(face, poses)
    return face, pose
