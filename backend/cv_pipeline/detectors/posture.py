"""
Postür sinyalleri: lean (T5), spine_ratio/shoulder_tilt (T6), arms_crossed (T7).

Hepsi Pose Landmarker'ın **world landmarks** çıktısı üzerinden hesaplanır
(metre ölçekli, kameraya göre 3D) — image-space landmarks'in aksine yana
dönme (yaw) ile karışmaz. Bu da sprint'in temel metod kararıdır (Bölüm 3).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from backend.cv_pipeline import config
from backend.cv_pipeline.detectors.pose import (
    LEFT_HIP,
    LEFT_SHOULDER,
    LEFT_WRIST,
    PoseResult,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
)


@dataclass
class PostureResult:
    lean: float
    spine_ratio: float
    shoulder_tilt: float
    arms_crossed: Optional[bool]  # None = ölçülemedi (bilek visibility düşük)


def _mid(a, b) -> tuple[float, float, float]:
    return ((a.x + b.x) / 2, (a.y + b.y) / 2, (a.z + b.z) / 2)


def compute_lean(pose: PoseResult) -> float:
    """lean = shoulder_z_world - hip_z_world (T5)."""
    ls, rs = pose.world_landmarks[LEFT_SHOULDER], pose.world_landmarks[RIGHT_SHOULDER]
    lh, rh = pose.world_landmarks[LEFT_HIP], pose.world_landmarks[RIGHT_HIP]
    shoulder_z = (ls.z + rs.z) / 2
    hip_z = (lh.z + rh.z) / 2
    return float(shoulder_z - hip_z)


def compute_spine(pose: PoseResult, head_pitch_deg: float) -> tuple[float, float]:
    """world omuz-kalça geometrisi + baş pitch'ten spine_ratio ve shoulder_tilt (T6).

    spine_ratio: gövdenin ne kadar dikey olduğu (1.0 = tam dikey). Yalnızca
    dikey (y) ayrışma / toplam omurga uzunluğu oranına dayandığı için,
    kameraya göre yana dönme (yaw) bu oranı bozmaz — yalnızca gerçek
    öne-eğilme/çökme oranı düşürür.
    shoulder_tilt: omuzlar arasi dikey fark (bir omuz digerinden yuksekse).
    """
    ls, rs = pose.world_landmarks[LEFT_SHOULDER], pose.world_landmarks[RIGHT_SHOULDER]
    lh, rh = pose.world_landmarks[LEFT_HIP], pose.world_landmarks[RIGHT_HIP]

    shoulder_mid = _mid(ls, rs)
    hip_mid = _mid(lh, rh)

    dx = shoulder_mid[0] - hip_mid[0]
    dy = shoulder_mid[1] - hip_mid[1]
    dz = shoulder_mid[2] - hip_mid[2]
    spine_length = math.sqrt(dx * dx + dy * dy + dz * dz)
    torso_upright = min(1.0, abs(dy) / spine_length) if spine_length > 1e-6 else 0.0

    head_pitch_factor = max(0.0, 1.0 - abs(head_pitch_deg) / config.HEAD_PITCH_MAX_DEG)
    spine_ratio = torso_upright * (0.7 + 0.3 * head_pitch_factor)

    shoulder_width = math.dist((ls.x, ls.y, ls.z), (rs.x, rs.y, rs.z))
    shoulder_tilt = abs(ls.y - rs.y) / shoulder_width if shoulder_width > 1e-6 else 0.0

    return float(spine_ratio), float(shoulder_tilt)


def compute_arms_crossed(pose: PoseResult) -> Optional[bool]:
    """Bilek visibility dusukse None (olculemedi) - T7."""
    lw = pose.world_landmarks[LEFT_WRIST]
    rw = pose.world_landmarks[RIGHT_WRIST]
    if (
        lw.visibility < config.ARMS_VISIBILITY_THRESHOLD
        or rw.visibility < config.ARMS_VISIBILITY_THRESHOLD
    ):
        return None

    ls, rs = pose.world_landmarks[LEFT_SHOULDER], pose.world_landmarks[RIGHT_SHOULDER]
    lh, rh = pose.world_landmarks[LEFT_HIP], pose.world_landmarks[RIGHT_HIP]
    shoulder_width = math.dist((ls.x, ls.y, ls.z), (rs.x, rs.y, rs.z))
    if shoulder_width < 1e-6:
        return False

    wrist_distance = math.dist((lw.x, lw.y, lw.z), (rw.x, rw.y, rw.z))
    wrists_close = wrist_distance < config.ARMS_CROSSED_DISTANCE_RATIO * shoulder_width

    torso_z_mean = (ls.z + rs.z + lh.z + rh.z) / 4
    left_in_front = (torso_z_mean - lw.z) > config.ARMS_CROSSED_Z_FRONT_THRESHOLD
    right_in_front = (torso_z_mean - rw.z) > config.ARMS_CROSSED_Z_FRONT_THRESHOLD

    return bool(wrists_close and left_in_front and right_in_front)
