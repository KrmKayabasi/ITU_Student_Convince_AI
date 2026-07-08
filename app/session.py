"""
Tek bir kiosk/robot oturumuna ait tüm durumu tutar.
Her session_id için bu sınıftan bir örnek yaşar; container ayakta kaldığı
sürece bellekte durur (ring buffer, baseline, focus streak, tek-seferlik
profil bayrağı).
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Optional

from app import config


class SessionState(Enum):
    IDLE = "IDLE"                  # kadrajda kimse yok
    CALIBRATING = "CALIBRATING"    # yeni kişi geldi, baseline yakalanıyor
    ACTIVE = "ACTIVE"              # normal analiz


# Geriye dönük uyumluluk için eski isimlerle de erişilebilir sabitler.
# Gerçek değerler app.config'ten (env ile ayarlanabilir) okunur (T12).
CALIBRATION_SECONDS = config.CALIBRATION_SECONDS
NO_FACE_TIMEOUT_SECONDS = config.NO_FACE_TIMEOUT_SECONDS
LEAN_HISTORY_MAXLEN = config.LEAN_HISTORY_MAXLEN
EYE_HISTORY_MAXLEN = config.EYE_HISTORY_MAXLEN
EMA_ALPHA = config.EMA_ALPHA


@dataclass
class RawSignals:
    """Tek bir işlenmiş kareden çıkan ham, henüz smoothing uygulanmamış değerler."""

    face_present: bool = False
    lean: Optional[float] = None            # shoulder_z - hip_z (world landmarks)
    eye_contact: Optional[float] = None      # baş pozu + göz blendshape'leri birleşik
    head_yaw_deg: Optional[float] = None
    spine_ratio: Optional[float] = None
    shoulder_tilt: Optional[float] = None
    arms_crossed: Optional[bool] = None      # None = ölçülemedi (valid=False)
    emotion_label: Optional[str] = None
    emotion_scores: dict = field(default_factory=dict)


@dataclass
class SessionData:
    session_id: str
    state: SessionState = SessionState.IDLE
    created_at: float = field(default_factory=time.time)
    last_frame_at: float = field(default_factory=time.time)
    last_face_seen_at: float = field(default_factory=time.time)
    calibration_started_at: Optional[float] = None

    # Ring buffer'lar
    lean_history: Deque[float] = field(
        default_factory=lambda: deque(maxlen=config.LEAN_HISTORY_MAXLEN)
    )
    eye_history: Deque[float] = field(
        default_factory=lambda: deque(maxlen=config.EYE_HISTORY_MAXLEN)
    )
    calibration_lean_samples: list = field(default_factory=list)

    # Baseline
    baseline_lean: Optional[float] = None

    # /profile: kişi başına tek seferlik zengin profil.
    profile_sent: bool = False
    pending_profile: Optional[dict] = None

    # /focus: ~2.5sn'de bir push edilen kesintisiz odaklanma streak'i.
    is_focused: bool = False
    focus_streak_started_at: Optional[float] = None
    focus_time: float = 0.0

    # Son ham sinyal (gerekirse debug/JSON için)
    last_raw: RawSignals = field(default_factory=RawSignals)

    def touch_frame(self) -> None:
        self.last_frame_at = time.time()

    def reset_for_new_person(self) -> None:
        """Yeni bir kişi kadraja girdiğinde baseline, buffer ve tek-seferlik
        profil/focus durumunu sıfırla."""
        self.state = SessionState.CALIBRATING
        self.calibration_started_at = time.time()
        self.calibration_lean_samples.clear()
        self.lean_history.clear()
        self.eye_history.clear()
        self.baseline_lean = None
        self.profile_sent = False
        self.pending_profile = None
        self.is_focused = False
        self.focus_streak_started_at = None
        self.focus_time = 0.0

    def reset_to_idle(self) -> None:
        self.state = SessionState.IDLE
        self.calibration_started_at = None
        self.baseline_lean = None
        self.profile_sent = False
        self.pending_profile = None
        self.is_focused = False
        self.focus_streak_started_at = None
        self.focus_time = 0.0
