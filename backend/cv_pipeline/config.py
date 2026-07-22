"""
Tek merkezi konfigürasyon noktası. Tüm eşik/pencere sabitleri buradan, env
değişkenleriyle override edilebilir şekilde okunur (bkz. sprint T12).
"""

from __future__ import annotations

import os


def _float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _str(name: str, default: str) -> str:
    return os.environ.get(name, default)


# --- Model varlıkları --------------------------------------------------------
MODELS_DIR = _str("MODELS_DIR", "models")
FACE_MODEL_PATH = _str("FACE_MODEL_PATH", os.path.join(MODELS_DIR, "face_landmarker.task"))
POSE_MODEL_PATH = _str("POSE_MODEL_PATH", os.path.join(MODELS_DIR, "pose_landmarker_full.task"))
EMOTION_MODEL_PATH = _str(
    "EMOTION_MODEL_PATH", os.path.join(MODELS_DIR, "emotion_enet_b0_8_best_vgaf.onnx")
)

# --- State machine / zamanlama -----------------------------------------------
CALIBRATION_SECONDS = _float("CALIBRATION_SECONDS", 3.0)
NO_FACE_TIMEOUT_SECONDS = _float("NO_FACE_TIMEOUT_SECONDS", 2.0)
SESSION_GC_TIMEOUT_SECONDS = _float("SESSION_GC_TIMEOUT_SECONDS", 120.0)

LEAN_HISTORY_MAXLEN = _int("LEAN_HISTORY_MAXLEN", 15)
EYE_HISTORY_MAXLEN = _int("EYE_HISTORY_MAXLEN", 450)
EMA_ALPHA = _float("EMA_ALPHA", 0.3)

# --- Çıktı kanalları ----------------------------------------------------------
# /profile: kişi başına tek seferlik; yeterli örnek toplanınca tetiklenir.
PROFILE_MIN_SAMPLES = _int("PROFILE_MIN_SAMPLES", 30)
PROFILE_TRIGGER_POLL_SECONDS = _float("PROFILE_TRIGGER_POLL_SECONDS", 0.5)
# /focus: periyodik push aralığı.
FOCUS_EMIT_INTERVAL_SECONDS = _float("FOCUS_EMIT_INTERVAL_SECONDS", 2.5)
# /tracking: secilen yuzun konumunu dusuk gecikmeyle push eder. Son islenmis
# gozlem bu esikten eskiyse yuz yok denmez; gozlem durumu unknown olur.
TRACKING_EMIT_INTERVAL_SECONDS = _float("TRACKING_EMIT_INTERVAL_SECONDS", 0.2)
TRACKING_STALE_AFTER_SECONDS = _float("TRACKING_STALE_AFTER_SECONDS", 1.0)
# /debug: SADECE gelistirme/test amacli, uretim kontratinin disinda kanal.
DEBUG_EMIT_INTERVAL_SECONDS = _float("DEBUG_EMIT_INTERVAL_SECONDS", 0.3)
FOCUS_EYE_CONTACT_THRESHOLD = _float("FOCUS_EYE_CONTACT_THRESHOLD", 0.5)

# --- Göz teması (gaze) ---------------------------------------------------------
GAZE_YAW_GATE_DEG = _float("GAZE_YAW_GATE_DEG", 25.0)
GAZE_PITCH_GATE_DEG = _float("GAZE_PITCH_GATE_DEG", 20.0)
# eyeLookIn/Out/Up/Down blendshape ortalaması bu değere ulaştığında göz
# tamamen kenara kaymış sayılır (eye_centeredness -> 0).
GAZE_EYE_DEVIATION_SCALE = _float("GAZE_EYE_DEVIATION_SCALE", 0.4)
# Baş yaw/pitch eşiği aşıldığında (gating) eye_contact'a uygulanan ek ceza çarpanı.
GAZE_GATE_PENALTY = _float("GAZE_GATE_PENALTY", 0.4)
# Baş açısı bu dereceye ulaştığında head_alignment sıfırlanır (yumuşak düşüş).
GAZE_HEAD_ALIGNMENT_MAX_DEG = _float("GAZE_HEAD_ALIGNMENT_MAX_DEG", 90.0)

# --- Postür ---------------------------------------------------------------
ARMS_VISIBILITY_THRESHOLD = _float("ARMS_VISIBILITY_THRESHOLD", 0.5)
# Bilekler arası mesafe, omuz genişliğinin bu oranından küçükse "yakın" sayılır.
ARMS_CROSSED_DISTANCE_RATIO = _float("ARMS_CROSSED_DISTANCE_RATIO", 0.55)
# Bilek, gövde ortalamasından bu kadar (metre) kameraya daha yakınsa "önde" sayılır.
ARMS_CROSSED_Z_FRONT_THRESHOLD = _float("ARMS_CROSSED_Z_FRONT_THRESHOLD", 0.02)
SPINE_UPRIGHT_THRESHOLD = _float("SPINE_UPRIGHT_THRESHOLD", 0.85)
# Baş pitch'in spine_ratio'yu ne kadar modüle edeceği (geri kalanı torso geometrisi).
HEAD_PITCH_MAX_DEG = _float("HEAD_PITCH_MAX_DEG", 45.0)

# --- Birincil kişi seçimi ---------------------------------------------------
MAX_NUM_FACES = _int("MAX_NUM_FACES", 3)
MAX_NUM_POSES = _int("MAX_NUM_POSES", 3)
PRIMARY_PERSON_STRATEGY = _str("PRIMARY_PERSON_STRATEGY", "largest_bbox")
# Onceki karede secilen yuzun bbox merkezine bu normalize mesafeden yakinsa
# "ayni kisi" sayilir ve kucuk alan farklarinda zipla(t)mamak icin tercih edilir.
PRIMARY_PERSON_CONTINUITY_RADIUS = _float("PRIMARY_PERSON_CONTINUITY_RADIUS", 0.15)
# Sureklilik icin onceki kisiyi, alani en buyuk yuzun bu oranindan kucuk
# olmadigi surece tercih etmeye devam et.
PRIMARY_PERSON_CONTINUITY_AREA_RATIO = _float("PRIMARY_PERSON_CONTINUITY_AREA_RATIO", 0.7)

# --- Duygu tanıma -----------------------------------------------------------
EMOTION_INFER_HZ = _float("EMOTION_INFER_HZ", 1.0)
# Yuz bbox'i kirpilirken her yonde eklenen normalize dolgu (kafanin tamamini
# kadraja almak icin, ONNX modeli tüm yüzü görmek ister).
FACE_CROP_PADDING_RATIO = _float("FACE_CROP_PADDING_RATIO", 0.25)
