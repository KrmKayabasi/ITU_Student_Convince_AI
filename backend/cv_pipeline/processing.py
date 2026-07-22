"""
İşleme katmanı.

FrameSlot: WebSocket ingest'in yazdığı, worker'ın okuduğu tek-elemanlı
"son kare" kutusu. Kuyruk DEĞİL — kasıtlı olarak yalnızca en güncel kareyi
tutar; işleme yetişemezse eski kareler sessizce ezilir (drop-stale).

SignalExtractor: her kiosk (session_id) için TEK bir örnek yaşar (Face/Pose
Landmarker VIDEO modu monoton artan timestamp gerektirir; bu da örnek
başına state demektir — instance'lar arası paylaşılamaz). Bir karadan
gerçek sinyalleri çıkarır: Face + Pose Landmarker -> birincil kişi seçimi
-> gaze/lean/spine/arms -> duygu worker'ına yüz kırpımı verme (T9).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from backend.cv_pipeline import config
from backend.cv_pipeline.detectors.emotion import EmotionWorker
from backend.cv_pipeline.detectors.face import FaceLandmarkerWrapper
from backend.cv_pipeline.detectors.gaze import compute_eye_contact
from backend.cv_pipeline.detectors.pose import PoseLandmarkerWrapper
from backend.cv_pipeline.detectors.posture import (
    compute_arms_crossed,
    compute_lean,
    compute_spine,
)
from backend.cv_pipeline.detectors.select import select_primary_person
from backend.cv_pipeline.session import RawSignals


@dataclass
class FrameSlot:
    """Tek kiosk için en güncel kareyi tutan, thread-safe drop-stale kutu."""

    _frame: Optional[np.ndarray] = None
    _frame_ts: float = 0.0
    _frame_monotonic: float = 0.0
    _lock: threading.Lock = None  # __post_init__'te doldurulur

    def __post_init__(self) -> None:
        self._lock = threading.Lock()

    def put(self, frame: np.ndarray) -> None:
        """Yeni kare geldi; eskisi ne olursa olsun üzerine yazılır."""
        with self._lock:
            self._frame = frame
            self._frame_ts = time.time()
            self._frame_monotonic = time.monotonic()

    def get_latest(self) -> tuple[Optional[np.ndarray], float]:
        """En güncel kareyi (varsa) al. Alındıktan sonra tekrar dönmez."""
        frame, ts, _ = self.get_latest_with_timestamps()
        return frame, ts

    def get_latest_with_timestamps(
        self,
    ) -> tuple[Optional[np.ndarray], float, float]:
        """Return wall and monotonic capture times for the newest frame."""
        with self._lock:
            frame = self._frame
            ts = self._frame_ts
            monotonic_ts = self._frame_monotonic
            self._frame = None
            return frame, ts, monotonic_ts


def _bbox_center(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    xmin, ymin, xmax, ymax = bbox
    return (xmin + xmax) / 2, (ymin + ymax) / 2


def crop_from_bbox(
    frame: np.ndarray,
    bbox: Tuple[float, float, float, float],
    padding_ratio: Optional[float] = None,
) -> Optional[np.ndarray]:
    """Normalize (xmin,ymin,xmax,ymax) bbox'tan, etrafina dolgu eklenmis
    KARE piksel kirpimi doner. Kirpim bossa None doner.

    Duygu modeli 224x224 kare girdi bekler; yuz bbox'i dikdortgen kaldiginda
    resize sirasinda goruntu yatay/dikey gerilip siniflandirmayi bozar. Kisa
    kenar, merkez korunarak uzun kenara esitlenir; kare kare bulunmasi
    gereken alan kadranin disina tasarsa kenar piksel tekrariyla doldurulur.
    """
    if padding_ratio is None:
        padding_ratio = config.FACE_CROP_PADDING_RATIO
    h, w = frame.shape[:2]
    xmin, ymin, xmax, ymax = bbox
    box_w, box_h = xmax - xmin, ymax - ymin
    xmin -= box_w * padding_ratio
    xmax += box_w * padding_ratio
    ymin -= box_h * padding_ratio
    ymax += box_h * padding_ratio

    x0, y0 = xmin * w, ymin * h
    x1, y1 = xmax * w, ymax * h
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = max(x1 - x0, y1 - y0)
    x0, x1 = int(round(cx - side / 2)), int(round(cx + side / 2))
    y0, y1 = int(round(cy - side / 2)), int(round(cy + side / 2))
    if x1 <= x0 or y1 <= y0:
        return None

    pad_left, pad_top = max(0, -x0), max(0, -y0)
    pad_right, pad_bottom = max(0, x1 - w), max(0, y1 - h)
    x0c, y0c = max(0, x0), max(0, y0)
    x1c, y1c = min(w, x1), min(h, y1)
    if x1c <= x0c or y1c <= y0c:
        return None
    crop = frame[y0c:y1c, x0c:x1c]

    if pad_left or pad_top or pad_right or pad_bottom:
        crop = cv2.copyMakeBorder(
            crop, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REPLICATE
        )
    return crop


class SignalExtractor:
    """Bir kiosk (session_id) icin gercek CV modelleriyle sinyal cikarma.

    VIDEO running mode monoton artan ms timestamp gerektirir; istemci
    saatine guvenilmez, her cagride kendi ic sayacimizi ilerletiriz.
    """

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self._face = FaceLandmarkerWrapper()
        self._pose = PoseLandmarkerWrapper()
        self._emotion_worker = EmotionWorker(session_id)
        self._emotion_worker.start()
        self._timestamp_counter = 0
        self._previous_face_center: Optional[Tuple[float, float]] = None

    def _next_timestamp_ms(self) -> int:
        """Monotonically increasing millisecond counter for MediaPipe VIDEO mode.

        MediaPipe requires strictly increasing timestamps per detection call.
        We use a simple counter rather than time.monotonic() because the latter
        has an undefined reference point and may not advance on system sleep.
        """
        self._timestamp_counter += 1
        return self._timestamp_counter

    def extract(self, frame: Optional[np.ndarray]) -> RawSignals:
        observation_ts = time.time()
        if frame is None or frame.size == 0:
            return RawSignals(face_present=False, observation_ts=observation_ts)

        timestamp_ms = self._next_timestamp_ms()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        faces = self._face.detect(mp_image, timestamp_ms)
        poses = self._pose.detect(mp_image, timestamp_ms)

        face, pose = select_primary_person(faces, poses, self._previous_face_center)
        if face is None:
            self._previous_face_center = None
            return RawSignals(
                face_present=False,
                person_present=bool(poses),
                observation_ts=observation_ts,
            )
        face_center_x, face_center_y = _bbox_center(face.bbox)
        self._previous_face_center = (face_center_x, face_center_y)
        xmin, ymin, xmax, ymax = face.bbox

        gaze = compute_eye_contact(face.blendshapes, face.head_pose)

        lean = spine_ratio = shoulder_tilt = None
        arms_crossed = None
        if pose is not None:
            lean = compute_lean(pose)
            spine_ratio, shoulder_tilt = compute_spine(pose, face.head_pose.pitch_deg)
            arms_crossed = compute_arms_crossed(pose)

        crop = crop_from_bbox(frame, face.bbox)
        if crop is not None:
            self._emotion_worker.submit(crop)
        emotion_label, emotion_scores = self._emotion_worker.latest()

        return RawSignals(
            face_present=True,
            person_present=True,
            face_center_x=face_center_x,
            face_center_y=face_center_y,
            face_bbox_width=xmax - xmin,
            face_bbox_height=ymax - ymin,
            observation_ts=observation_ts,
            lean=lean,
            eye_contact=gaze.eye_contact,
            head_yaw_deg=gaze.head_yaw_deg,
            spine_ratio=spine_ratio,
            shoulder_tilt=shoulder_tilt,
            arms_crossed=arms_crossed,
            emotion_label=emotion_label,
            emotion_scores=emotion_scores,
        )

    def close(self) -> None:
        self._face.close()
        self._pose.close()
        self._emotion_worker.stop()
