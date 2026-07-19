"""
Turn a one-shot CV /profile payload into a short, plain-text Turkish opening
hint for the assistant. Pure/deterministic and network-free, so it is unit
tested directly (see tests/orchestrator).

Profile shape (backend/cv_pipeline/scoring.py:build_initial_profile):
  scores: {attention, openness, energy}            # 0..1
  signals.emotion.dominant                          # str | None
  signals.arms_crossed: {value: bool, valid: bool}
  signals.lean.value                                # delta; < 0 = leaning in
"""

from __future__ import annotations

from typing import Any, Optional

# CV emotion label -> Turkish descriptor.
_EMOTION_TR = {
    "happy": "mutlu ve pozitif",
    "surprise": "meraklı",
    "neutral": "sakin",
    "sad": "biraz durgun",
    "fear": "biraz gergin",
    "angry": "gergin",
    "disgust": "mesafeli",
    "contempt": "temkinli",
}

_GENERIC_OPENER = (
    "Bir öğrenci tanıtım standına yeni geldi. Onu çok kısa, sıcak ve enerjik "
    "bir cümleyle Türkçe selamla, kendini kısaca tanıt ve nasıl yardımcı "
    "olabileceğini sor. Konuşmayı doğal başlat."
)


def _num(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def build_opening_hint(profile: Optional[dict]) -> str:
    """Return a plain-text Turkish instruction that primes the assistant's
    opening turn. Falls back to a generic warm greeting when no profile."""
    if not profile:
        return _GENERIC_OPENER

    scores = profile.get("scores") or {}
    signals = profile.get("signals") or {}
    attention = _num(scores.get("attention"), 0.5)
    openness = _num(scores.get("openness"), 0.5)
    energy = _num(scores.get("energy"), 0.5)

    traits: list[str] = []

    if openness >= 0.6:
        traits.append("açık ve rahat")
    elif openness <= 0.35:
        traits.append("biraz çekingen")

    if attention >= 0.6:
        traits.append("ilgili görünüyor")
    elif attention <= 0.4:
        traits.append("dikkati biraz dağınık")

    if energy >= 0.6:
        traits.append("enerjik")
    elif energy <= 0.35:
        traits.append("sakin bir enerjide")

    emotion = (signals.get("emotion") or {}).get("dominant")
    if emotion in _EMOTION_TR:
        traits.append(_EMOTION_TR[emotion])

    arms = signals.get("arms_crossed") or {}
    if arms.get("valid") and arms.get("value"):
        traits.append("kolları kapalı, biraz temkinli")

    lean_value = _num((signals.get("lean") or {}).get("value"), 0.0)
    if lean_value < -0.03:
        traits.append("öne eğilmiş, meraklı")

    # Choose an opening tone from attention/openness.
    if attention <= 0.4 or openness <= 0.35:
        tone = "önce ilgisini çekecek, merak uyandıran kısa bir açılış yap"
    elif energy >= 0.6 and openness >= 0.6:
        tone = "sıcak, enerjik ve samimi bir açılış yap"
    else:
        tone = "sıcak ve sakin bir açılış yap"

    trait_text = ", ".join(traits) if traits else "nötr"

    return (
        f"Öğrencinin ilk izlenimi: {trait_text}. "
        f"Bu izlenime uygun şekilde {tone}. "
        "Onu Türkçe selamla, kendini çok kısa tanıt ve nasıl yardımcı "
        "olabileceğini sor. Tek ve kısa bir açılış cümlesiyle başla; "
        "gözlemlediğin bu özellikleri asla açıkça söyleme."
    )
