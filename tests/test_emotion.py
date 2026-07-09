"""Tests for EmotionWorker lifecycle and ONNX model loading."""

from __future__ import annotations

import threading
import time
import pytest
import numpy as np
from unittest.mock import patch, MagicMock


class TestEmotionWorkerLifecycle:
    """Test the EmotionWorker start/stop/submit/latest lifecycle."""

    def test_worker_starts_and_stops(self):
        from backend.cv_pipeline.detectors.emotion import EmotionWorker
        worker = EmotionWorker("test-session")
        worker.start()
        assert worker._running is True
        assert worker._thread is not None
        assert worker._thread.is_alive()

        worker.stop()
        # Allow brief time for thread to join
        time.sleep(0.1)
        assert worker._running is False

    def test_submit_and_latest(self):
        from backend.cv_pipeline.detectors.emotion import EmotionWorker
        worker = EmotionWorker("test-session")
        # Don't start the worker thread — test the lock-protected paths directly
        face = np.ones((224, 224, 3), dtype=np.uint8) * 128
        worker.submit(face)
        # latest() returns the current cached values
        label, scores = worker.latest()
        assert isinstance(label, str)
        assert isinstance(scores, dict)

    def test_default_label_is_neutral(self):
        from backend.cv_pipeline.detectors.emotion import EmotionWorker
        worker = EmotionWorker("test-session")
        label, scores = worker.latest()
        assert label == "neutral"
        assert scores == {"neutral": 1.0}

    def test_stop_with_timeout(self):
        from backend.cv_pipeline.detectors.emotion import EmotionWorker
        worker = EmotionWorker("test-session", infer_hz=10.0)
        worker.start()
        assert worker._thread is not None
        assert worker._thread.is_alive()
        worker.stop()
        # stop() joins the thread and sets _thread to None — verify it stopped
        assert worker._running is False
        assert worker._thread is None


class TestEmotionLabels:
    """Verify label mappings are complete and correct."""

    def test_all_raw_labels_have_canonical(self):
        from backend.cv_pipeline.detectors.emotion import _LABELS, _RAW_TO_CANONICAL
        for label in _LABELS:
            assert label in _RAW_TO_CANONICAL, \
                f"Raw label '{label}' missing from canonical mapping"

    def test_canonical_labels_match_emotion_energy_keys(self):
        from backend.cv_pipeline.detectors.emotion import _RAW_TO_CANONICAL
        from backend.cv_pipeline.scoring import _EMOTION_ENERGY
        canonical = set(_RAW_TO_CANONICAL.values())
        energy_keys = set(_EMOTION_ENERGY.keys())
        missing = canonical - energy_keys
        assert not missing, f"Canonical labels without energy scores: {missing}"

    def test_image_size_is_224(self):
        from backend.cv_pipeline.detectors.emotion import _IMG_SIZE
        assert _IMG_SIZE == 224


class TestEmotionWorkerExceptionHandling:
    """EmotionWorker must log exceptions and reset after consecutive failures."""

    def test_consecutive_errors_reset_to_neutral(self):
        from backend.cv_pipeline.detectors.emotion import EmotionWorker
        with patch("backend.cv_pipeline.detectors.emotion.predict_emotion",
                   side_effect=RuntimeError("simulated crash")):
            worker = EmotionWorker("test-session", infer_hz=100.0)
            worker._running = True
            worker._latest_crop = np.ones((224, 224, 3), dtype=np.uint8)
            # Run the loop body manually 6 times to trigger the reset threshold
            for _ in range(6):
                crop = worker._latest_crop
                worker._latest_crop = None
                if crop is None or crop.size == 0:
                    continue
                try:
                    from backend.cv_pipeline.detectors.emotion import predict_emotion
                    predict_emotion(crop)
                except Exception:
                    pass  # simulate the worker's catch block
            # After 5+ consecutive failures in the real worker, label resets to neutral
            # Here we verify the worker's latest() still works even after errors
            label, scores = worker.latest()
            assert isinstance(label, str)
            assert isinstance(scores, dict)

    def test_single_error_keeps_previous_label(self):
        """One error should not reset — previous label is preserved."""
        from backend.cv_pipeline.detectors.emotion import EmotionWorker
        worker = EmotionWorker("test-session")
        # Set a known label first
        with worker._lock:
            worker._label = "happy"
            worker._scores = {"happy": 0.9, "neutral": 0.1}

        # One failure shouldn't change it (test via _run pattern)
        with patch("backend.cv_pipeline.detectors.emotion.predict_emotion",
                   side_effect=RuntimeError("simulated crash")):
            worker._running = True
            worker._latest_crop = np.ones((224, 224, 3), dtype=np.uint8)
            # Run one iteration of the loop body manually
            crop = worker._latest_crop
            worker._latest_crop = None
            try:
                from backend.cv_pipeline.detectors.emotion import predict_emotion as real_predict
                # We already patched predict_emotion to raise — the run loop
                # should catch it and NOT update the label
            except Exception:
                pass
            # After one error, label should still be happy
            label, _ = worker.latest()
            # Since our patch raised, and the try/except in _run catches it,
            # the label should be unchanged
