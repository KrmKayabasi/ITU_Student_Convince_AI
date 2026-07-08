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

    largest = max(faces, key=lambda f: _bbox_area(f.bbox))
    if previous_center is None:
        return largest

    largest_area = _bbox_area(largest.bbox)
    candidates = [
        f
        for f in faces
        if math.dist(_bbox_center(f.bbox), previous_center) < config.PRIMARY_PERSON_CONTINUITY_RADIUS
    ]
    if not candidates:
        return largest

    continuity_pick = max(candidates, key=lambda f: _bbox_area(f.bbox))
    if _bbox_area(continuity_pick.bbox) >= largest_area * config.PRIMARY_PERSON_CONTINUITY_AREA_RATIO:
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
