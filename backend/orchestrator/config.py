"""
Orchestrator configuration — all overridable via environment variables.

Self-contained on purpose: the persona loader + markdown stripper are copied
from backend/speech_backend/config.py so the orchestrator does not import the
heavy/fragile cascaded speech backend. Injected LLM text (profile hints, focus
nudges) is also passed through the same stripper for consistency.
"""

from __future__ import annotations

import os
import re


# ── env helpers ───────────────────────────────────────────────────────────────
def _str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# ── Gemini Live ───────────────────────────────────────────────────────────────
GOOGLE_API_KEY = _str("GOOGLE_API_KEY", "")
# Native-audio model (supports affective dialog + proactive audio under v1alpha).
GEMINI_LIVE_MODEL = _str("GEMINI_LIVE_MODEL", "gemini-3.1-flash-live-preview")
# affective_dialog + proactive_audio require the v1alpha API surface.
GEMINI_API_VERSION = _str("GEMINI_API_VERSION", "v1alpha")
# Prebuilt voice name. Turkish availability is verified at startup (Phase 1);
# override via env if the chosen voice is unavailable.
GEMINI_VOICE = _str("GEMINI_VOICE", "Aoede")
GEMINI_LANGUAGE = _str("GEMINI_LANGUAGE", "tr-TR")
ENABLE_AFFECTIVE_DIALOG = _bool("GEMINI_AFFECTIVE_DIALOG", False)
ENABLE_PROACTIVE_AUDIO = _bool("GEMINI_PROACTIVE_AUDIO", False)

# Audio rates (Gemini Live contract): 16 kHz PCM16 in, 24 kHz PCM16 out.
INPUT_SAMPLE_RATE = _int("ORCH_INPUT_SAMPLE_RATE", 16000)
OUTPUT_SAMPLE_RATE = _int("ORCH_OUTPUT_SAMPLE_RATE", 24000)


# ── CV pipeline (unchanged, :8000) ────────────────────────────────────────────
# Base WS URL of the CV pipeline the orchestrator subscribes to server-side.
CV_PIPELINE_WS_URL = _str("CV_PIPELINE_WS_URL", "ws://localhost:8000")
# Optional CV token (query param) if CV_PIPELINE_TOKEN auth is enabled there.
CV_PIPELINE_TOKEN = _str("CV_PIPELINE_TOKEN", "")

# Focus-steering behaviour.
FOCUS_LOSS_SECONDS = _float("FOCUS_LOSS_SECONDS", 5.0)       # sustained !focused before a nudge
NUDGE_COOLDOWN_SECONDS = _float("NUDGE_COOLDOWN_SECONDS", 20.0)  # min gap between nudges
# How long to wait for the one-shot /profile before opening with a generic greeting.
PROFILE_WAIT_SECONDS = _float("PROFILE_WAIT_SECONDS", 3.0)


# ── Orchestrator server ───────────────────────────────────────────────────────
SERVER_HOST = _str("ORCH_HOST", "0.0.0.0")
SERVER_PORT = _int("ORCH_PORT", 8001)
# Optional bearer/query token to protect the browser WS (off by default in dev).
ORCH_TOKEN = _str("ORCH_TOKEN", "")


# ── System persona (SYSTEM_PROMPT.md) ─────────────────────────────────────────
_sys_prompt_local = os.path.join(os.path.dirname(__file__), "SYSTEM_PROMPT.md")
_sys_prompt_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "SYSTEM_PROMPT.md")
)
_sys_prompt_path = _sys_prompt_local if os.path.exists(_sys_prompt_local) else _sys_prompt_root


def strip_markdown_for_llm(text: str) -> str:
    """Strip Markdown syntax so the model never voices formatting characters.

    Copied from backend/speech_backend/config.py. Applied to the system prompt
    AND to every injected profile/focus hint so nothing like ``*`` or ``#`` is
    spoken aloud.
    """
    text = re.sub(r"\*{1,3}", "", text)
    text = re.sub(r"(?<!\w)_{1,2}(?!\w)", "", text)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = text.replace("`", "")
    text = text.replace("~~", "")
    text = re.sub(r"^>\s?", "", text, flags=re.MULTILINE)
    text = re.sub(r"\|", " ", text)
    text = re.sub(r"^-{3,}$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _load_system_instruction() -> str:
    if os.path.exists(_sys_prompt_path):
        with open(_sys_prompt_path, "r", encoding="utf-8") as f:
            return strip_markdown_for_llm(f.read().strip())
    return (
        "Sen İTÜ Tanıtım Günleri'nde görev yapan, deneyimli ve sıcak bir "
        "üniversite tercih danışmanısın. Öğrenciyle doğal, samimi ve kısa "
        "cümlelerle Türkçe konuş. Amacın öğrenciyi anlamak ve doğru kararı "
        "vermesine yardımcı olarak İTÜ'yü tercih etme olasılığını artırmak."
    )


SYSTEM_INSTRUCTION = _load_system_instruction()
