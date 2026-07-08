"""
Duygu tanıma: EmotiEffLib (enet_b0_8_best_vgaf) ONNX ağırlığı, CPUExecutionProvider
ile (T8). Ana işleme yolunu (Face/Pose Landmarker, ~15fps hedefi) bloklamamak
için her oturum kendi arka plan thread'inde, drop-stale tek-elemanlı kutudan
~1 Hz'de en güncel yüz kırpımını işler. Sonuç (dominant etiket + skor
sözlüğü) session'a değil, worker'ın kendi belleğine yazılır ve ana yol
`latest()` ile okur — böylece yüz kaybolduğunda veya worker henüz ilk
sonucu üretmeden önce son bilinen değer korunur, crash olmaz.
"""

from __future__ import annotations

import threading
import time
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort

from app import config

# 8-sinif AffectNet siniflandirmasi (enet_b0_8_*): sirasi ONNX cikis
# indeksleriyle birebir eslesir (bkz. EmotiEffLib facial_analysis.py).
_LABELS = ("Anger", "Contempt", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise")
_RAW_TO_CANONICAL = {
    "Anger": "angry",
    "Contempt": "contempt",
    "Disgust": "disgust",
    "Fear": "fear",
    "Happiness": "happy",
    "Neutral": "neutral",
    "Sadness": "sad",
    "Surprise": "surprise",
}
_IMG_SIZE = 224
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)
_INPUT_NAME = "input"

_session_lock = threading.Lock()
_ort_session: Optional[ort.InferenceSession] = None


def _get_ort_session() -> ort.InferenceSession:
    """Sürecin tamamında tek bir ONNX InferenceSession paylaşılır (model
    ağırlıkları salt-okunur, thread-safe inference destekler)."""
    global _ort_session
    with _session_lock:
        if _ort_session is None:
            ort.set_default_logger_severity(3)
            _ort_session = ort.InferenceSession(
                config.EMOTION_MODEL_PATH, providers=["CPUExecutionProvider"]
            )
        return _ort_session


def _preprocess(face_bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (_IMG_SIZE, _IMG_SIZE)).astype(np.float32) / 255.0
    for i in range(3):
        resized[..., i] = (resized[..., i] - _MEAN[i]) / _STD[i]
    return resized.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)


def predict_emotion(face_bgr: np.ndarray) -> Tuple[str, Dict[str, float]]:
    """Tek bir BGR yuz kirpimindan (dominant_etiket, skor_sozlugu) uretir."""
    session = _get_ort_session()
    x = _preprocess(face_bgr)
    logits = session.run(None, {_INPUT_NAME: x})[0][0]
    exp = np.exp(logits - np.max(logits))
    probs = exp / exp.sum()
    scores = {_RAW_TO_CANONICAL[label]: float(p) for label, p in zip(_LABELS, probs)}
    dominant = max(scores, key=scores.get)
    return dominant, scores


class EmotionWorker:
    """Tek bir oturum icin drop-stale yuz-kirpimi kutusu + ~1Hz arka plan thread'i."""

    def __init__(self, session_id: str, infer_hz: Optional[float] = None) -> None:
        self.session_id = session_id
        self._interval = 1.0 / (infer_hz or config.EMOTION_INFER_HZ)
        self._lock = threading.Lock()
        self._latest_crop: Optional[np.ndarray] = None
        self._label = "neutral"
        self._scores: Dict[str, float] = {"neutral": 1.0}
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run, name=f"emotion-{self.session_id}", daemon=True
        )
        self._thread.start()

    def submit(self, face_crop: np.ndarray) -> None:
        with self._lock:
            self._latest_crop = face_crop

    def latest(self) -> Tuple[str, Dict[str, float]]:
        with self._lock:
            return self._label, dict(self._scores)

    def _run(self) -> None:
        while self._running:
            with self._lock:
                crop = self._latest_crop
                self._latest_crop = None
            if crop is None or crop.size == 0:
                time.sleep(self._interval)
                continue
            try:
                label, scores = predict_emotion(crop)
                with self._lock:
                    self._label = label
                    self._scores = scores
            except Exception:
                pass  # son bilinen deger korunur; worker crash olup dusmez
            time.sleep(self._interval)

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
