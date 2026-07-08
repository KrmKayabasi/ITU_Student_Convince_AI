"""
MediaPipe Tasks — Pose Landmarker sarmalayıcısı.

VIDEO running mode; **world landmarks** (metre ölçekli, kameraya göre 3D)
kullanılır — image-space landmarks yerine bunlar tercih edilir çünkü
monoküler image `z` gürültülü ve perspektifle karışır (bkz. sprint T3/T5/T6).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

from app import config

# İlgilenilen landmark indeksleri (BlazePose topolojisi).
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16

RELEVANT_INDICES = [
    LEFT_SHOULDER,
    RIGHT_SHOULDER,
    LEFT_HIP,
    RIGHT_HIP,
    LEFT_ELBOW,
    RIGHT_ELBOW,
    LEFT_WRIST,
    RIGHT_WRIST,
]


@dataclass
class LandmarkPoint:
    x: float
    y: float
    z: float
    visibility: float
    presence: float


@dataclass
class PoseResult:
    world_landmarks: dict  # index -> LandmarkPoint (world, meter scale)
    image_landmarks: dict  # index -> LandmarkPoint (normalized image space, for bbox/matching)
    bbox: Tuple[float, float, float, float]  # normalized image-space bbox


def _to_point(lm) -> LandmarkPoint:
    return LandmarkPoint(
        x=lm.x,
        y=lm.y,
        z=lm.z,
        visibility=getattr(lm, "visibility", 0.0) or 0.0,
        presence=getattr(lm, "presence", 0.0) or 0.0,
    )


def _bbox_from_image_landmarks(landmarks) -> Tuple[float, float, float, float]:
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    return (min(xs), min(ys), max(xs), max(ys))


class PoseLandmarkerWrapper:
    def __init__(self, model_path: Optional[str] = None, num_poses: Optional[int] = None) -> None:
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path or config.POSE_MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=num_poses or config.MAX_NUM_POSES,
            output_segmentation_masks=False,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)

    def detect(self, mp_image, timestamp_ms: int) -> List[Optional[PoseResult]]:
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        if not result.pose_world_landmarks:
            return []

        poses: List[Optional[PoseResult]] = []
        for i, world_lms in enumerate(result.pose_world_landmarks):
            image_lms = result.pose_landmarks[i] if i < len(result.pose_landmarks) else None
            if image_lms is None:
                continue
            world_dict = {idx: _to_point(world_lms[idx]) for idx in RELEVANT_INDICES}
            image_dict = {idx: _to_point(image_lms[idx]) for idx in RELEVANT_INDICES}
            poses.append(
                PoseResult(
                    world_landmarks=world_dict,
                    image_landmarks=image_dict,
                    bbox=_bbox_from_image_landmarks(image_lms),
                )
            )
        return poses

    def close(self) -> None:
        self._landmarker.close()
