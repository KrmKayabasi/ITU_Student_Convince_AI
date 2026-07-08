"""
MediaPipe Tasks — Face Landmarker sarmalayıcısı.

VIDEO running mode kullanılır (senkron `detect_for_video`, monoton artan
ms timestamp gerektirir). Her çağrı: landmark'lar + 52 blendshape + baş
pozu transformation matrix'i birlikte döner (T2).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

from app import config


@dataclass
class HeadPose:
    yaw_deg: float
    pitch_deg: float
    roll_deg: float


@dataclass
class FaceResult:
    landmarks: List[Tuple[float, float, float]]  # normalized (x, y, z)
    blendshapes: Dict[str, float]
    transform_matrix: np.ndarray  # 4x4
    bbox: Tuple[float, float, float, float]  # (xmin, ymin, xmax, ymax), normalized
    head_pose: HeadPose = field(default_factory=lambda: HeadPose(0.0, 0.0, 0.0))


def _bbox_from_landmarks(landmarks) -> Tuple[float, float, float, float]:
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    return (min(xs), min(ys), max(xs), max(ys))


def head_pose_from_matrix(matrix: np.ndarray) -> HeadPose:
    """4x4 facial transformation matrix'ten yaw/pitch/roll (derece) çıkarır.

    Üst-sol 3x3 blok, MediaPipe'ın kanonik yüz modelinden kameraya olan
    rotasyonu temsil eder (X: sağ, Y: yukarı, Z: yüzden dışarı/kameraya
    doğru). Standart XYZ Euler ayrıştırması kullanılır.
    """
    r = np.asarray(matrix)[:3, :3]
    pitch = math.atan2(r[2, 1], r[2, 2])
    yaw = math.atan2(-r[2, 0], math.sqrt(r[2, 1] ** 2 + r[2, 2] ** 2))
    roll = math.atan2(r[1, 0], r[0, 0])
    return HeadPose(
        yaw_deg=math.degrees(yaw),
        pitch_deg=math.degrees(pitch),
        roll_deg=math.degrees(roll),
    )


class FaceLandmarkerWrapper:
    def __init__(self, model_path: Optional[str] = None, num_faces: Optional[int] = None) -> None:
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path or config.FACE_MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=num_faces or config.MAX_NUM_FACES,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

    def detect(self, mp_image, timestamp_ms: int) -> List[FaceResult]:
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        if not result.face_landmarks:
            return []

        faces: List[FaceResult] = []
        num_faces = len(result.face_landmarks)
        blendshapes_all = result.face_blendshapes or []
        matrixes_all = result.facial_transformation_matrixes or []

        for i in range(num_faces):
            landmarks = result.face_landmarks[i]
            blendshapes = {}
            if i < len(blendshapes_all):
                blendshapes = {c.category_name: c.score for c in blendshapes_all[i]}

            matrix = (
                np.asarray(matrixes_all[i])
                if i < len(matrixes_all)
                else np.eye(4)
            )
            faces.append(
                FaceResult(
                    landmarks=[(lm.x, lm.y, lm.z) for lm in landmarks],
                    blendshapes=blendshapes,
                    transform_matrix=matrix,
                    bbox=_bbox_from_landmarks(landmarks),
                    head_pose=head_pose_from_matrix(matrix),
                )
            )
        return faces

    def close(self) -> None:
        self._landmarker.close()
