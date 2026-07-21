"""
Emotion classifier for the avatar expressions.

A slim, lazy port of jaison-core's ``emotion_roberta`` filter
(``src/utils/operations/filter_text/emotion_roberta.py``) — without the
operations-framework scaffolding, which is overkill for the orchestrator.

Loads ``SamLowe/roberta-base-go_emotions`` on first use (so the orchestrator
boots fast and never imports torch unless emotion is enabled), runs inference
off the event loop via :func:`asyncio.to_thread`, and returns the top
go_emotions label (one of 28, e.g. ``"joy"``, ``"neutral"``, ``"surprise"``).

The orchestrator owns a single shared instance (see ``get_classifier``) and
must treat any failure here as non-fatal: emotion is purely cosmetic and must
never break the realtime audio loop.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

import config

logger = logging.getLogger("orchestrator.emotion")

# Lazily-imported heavy deps so a disabled orchestrator never pulls torch.
_pipeline = None  # type: ignore[var-annotated]
_torch = None


class EmotionClassifier:
    """Async wrapper around the go_emotions roberta pipeline."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._pipe = None
        self._device = "cpu"
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        """Lazily load the model on first use (off the event loop)."""
        if self._pipe is not None:
            return

        def _load():
            global _pipeline, _torch
            if _pipeline is None:
                from transformers import pipeline as _hf_pipeline  # type: ignore

                _pipeline = _hf_pipeline
            if _torch is None:
                import torch  # type: ignore

                _torch = torch
            device = "cuda" if _torch.cuda.is_available() else "cpu"
            pipe = _pipeline(
                task="text-classification",
                model=self.model_name,
                top_k=1,
                device=device,
            )
            return pipe, device

        try:
            self._pipe, self._device = await asyncio.to_thread(_load)
            logger.info(
                "emotion classifier loaded: model=%s device=%s",
                self.model_name,
                self._device,
            )
        except Exception:
            logger.exception("failed to load emotion classifier; expressions disabled")
            self._pipe = None
            raise

    async def classify(self, text: str) -> Optional[str]:
        """Return the top emotion label for ``text``, or ``None`` on failure."""
        if self._pipe is None:
            return None
        text = (text or "").strip()
        if len(text) < config.EMOTION_MIN_CHARS:
            return None

        def _infer():
            # pipeline returns [[{"label": ..., "score": ...}]] with top_k=1
            return self._pipe(text)[0][0]["label"]

        async with self._lock:
            try:
                return await asyncio.to_thread(_infer)
            except Exception:
                logger.exception("emotion inference failed")
                return None

    async def aclose(self) -> None:
        """Drop the model and free CUDA cache if present."""
        if self._pipe is None:
            return
        self._pipe = None
        try:
            if _torch is not None and _torch.cuda.is_available():
                _torch.cuda.empty_cache()
        except Exception:
            pass


# ── shared singleton (module-scoped) ──────────────────────────────────────────
_classifier: Optional[EmotionClassifier] = None


def get_classifier() -> Optional[EmotionClassifier]:
    """Return the shared classifier, or ``None`` if emotion is disabled.

    The instance is created lazily but NOT pre-loaded — :meth:`start` runs on
    the first assistant transcript so boot time stays fast.
    """
    global _classifier
    if not config.ENABLE_EMOTION:
        return None
    if _classifier is None:
        _classifier = EmotionClassifier(config.EMOTION_MODEL)
    return _classifier


async def close_classifier() -> None:
    """Tear down the shared classifier, if any (process shutdown)."""
    global _classifier
    if _classifier is not None:
        await _classifier.aclose()
        _classifier = None
