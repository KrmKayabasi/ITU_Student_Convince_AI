"""
Göz teması: baş pozu (yaw/pitch) + göz blendshape'leri birleşik (sprint T4).

Eski yöntem (yalnızca iris'in göz-içi yatay konumu) baş dönüklüğünü
ayırt edemiyordu. Burada iki bileşen çarpılır:
  1) eye_centeredness: eyeLookIn/Out/Up/Down blendshape ortalaması düşükse
     göz göz-yuvası içinde ortada demektir.
  2) head_alignment: baş yaw/pitch arttıkça yumuşak biçimde düşer.
Ayrıca bir eşik üstünde (gating) baş dönüklüğü, göz ortada olsa dahi
eye_contact'ı ek bir ceza çarpanıyla bastırır.

Göz kırpma sırasında kapanan göz kapağı iris takibini bozar (ör. eyeLookDown
sahte biçimde yükselir); bu da normalde eyeLookIn/Out/Up/Down ortalamasını
gercek bir bakis sapmasi gibi gösterip eye_contact'i haksiz yere düşürür. Bu
yüzden eyeBlinkLeft/Right skoru (blink), deviation cezasini orantili olarak
yumusatir — tam kirpma (blink=1) cezayi tamamen kaldirir, yari kapali gecis
kareleri de (blink=0.3-0.7) kismen korunur. Kirpma anlik bir dikkat kaybi
degildir.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from backend.cv_pipeline import config
from backend.cv_pipeline.detectors.face import HeadPose

_EYE_LOOK_KEYS = (
    "eyeLookInLeft",
    "eyeLookInRight",
    "eyeLookOutLeft",
    "eyeLookOutRight",
    "eyeLookUpLeft",
    "eyeLookUpRight",
    "eyeLookDownLeft",
    "eyeLookDownRight",
)
_EYE_BLINK_KEYS = ("eyeBlinkLeft", "eyeBlinkRight")


@dataclass
class GazeResult:
    eye_contact: float  # [0, 1]
    head_yaw_deg: float
    head_pitch_deg: float
    gated: bool  # baş dönüklüğü eşiği aşıldı mı


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def compute_eye_contact(blendshapes: Dict[str, float], head_pose: HeadPose) -> GazeResult:
    # Unrolled lookups to eliminate generator/iteration overhead in hot VAD/eye contact calculation loops
    blink = _clip01((blendshapes.get("eyeBlinkLeft", 0.0) + blendshapes.get("eyeBlinkRight", 0.0)) * 0.5)
    deviation = (
        blendshapes.get("eyeLookInLeft", 0.0)
        + blendshapes.get("eyeLookInRight", 0.0)
        + blendshapes.get("eyeLookOutLeft", 0.0)
        + blendshapes.get("eyeLookOutRight", 0.0)
        + blendshapes.get("eyeLookUpLeft", 0.0)
        + blendshapes.get("eyeLookUpRight", 0.0)
        + blendshapes.get("eyeLookDownLeft", 0.0)
        + blendshapes.get("eyeLookDownRight", 0.0)
    ) * 0.125
    deviation_centeredness = _clip01(1.0 - deviation / config.GAZE_EYE_DEVIATION_SCALE)
    # Kirpma yari kapali/gecis karelerinde de eyeLook* sapmasini bozar; sert
    # bir esik yerine blink skoruyla orantili yumusatma yapariz (blink=1 ->
    # ceza tamamen kalkar, blink=0 -> normal deviation cezasi).
    eye_centeredness = deviation_centeredness + (1.0 - deviation_centeredness) * blink

    max_deg = config.GAZE_HEAD_ALIGNMENT_MAX_DEG
    yaw_alignment = _clip01(1.0 - abs(head_pose.yaw_deg) / max_deg)
    pitch_alignment = _clip01(1.0 - abs(head_pose.pitch_deg) / max_deg)
    head_alignment = yaw_alignment * pitch_alignment

    eye_contact = eye_centeredness * head_alignment

    gated = (
        abs(head_pose.yaw_deg) > config.GAZE_YAW_GATE_DEG
        or abs(head_pose.pitch_deg) > config.GAZE_PITCH_GATE_DEG
    )
    if gated:
        eye_contact *= config.GAZE_GATE_PENALTY

    return GazeResult(
        eye_contact=_clip01(eye_contact),
        head_yaw_deg=head_pose.yaw_deg,
        head_pitch_deg=head_pose.pitch_deg,
        gated=gated,
    )
