"""Tests for FrameSlot thread safety and SignalExtractor."""

from __future__ import annotations

import threading
import time
import pytest
import numpy as np
from backend.cv_pipeline.processing import FrameSlot, crop_from_bbox


class TestFrameSlot:
    """Thread-safe drop-stale frame container."""

    def test_initial_state_returns_none(self):
        slot = FrameSlot()
        frame, ts = slot.get_latest()
        assert frame is None

    def test_put_and_get_returns_same_frame(self):
        slot = FrameSlot()
        test_frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        slot.put(test_frame)
        frame, ts = slot.get_latest()
        assert frame is not None
        assert frame.shape == test_frame.shape
        assert np.array_equal(frame, test_frame)

    def test_get_latest_consumes_frame(self):
        """After get_latest, the frame is consumed — second call returns None."""
        slot = FrameSlot()
        slot.put(np.zeros((100, 100, 3), dtype=np.uint8))
        frame1, _ = slot.get_latest()
        assert frame1 is not None
        frame2, _ = slot.get_latest()
        assert frame2 is None

    def test_put_overwrites_unconsumed_frame(self):
        """Second put overwrites the first before it's consumed."""
        slot = FrameSlot()
        slot.put(np.zeros((100, 100, 3), dtype=np.uint8))
        slot.put(np.ones((100, 100, 3), dtype=np.uint8))  # overwrite
        frame, _ = slot.get_latest()
        assert frame is not None
        assert frame.mean() > 0.9  # all ones, overwrote zeros

    def test_put_records_timestamp(self):
        slot = FrameSlot()
        before = time.time()
        slot.put(np.zeros((10, 10, 3), dtype=np.uint8))
        _, ts = slot.get_latest()
        after = time.time()
        assert before <= ts <= after + 0.01

    def test_thread_safety_concurrent_put_get(self):
        """Multiple threads putting and getting should not crash or corrupt data."""
        slot = FrameSlot()
        errors = []
        iterations = 500

        def producer():
            for i in range(iterations):
                try:
                    slot.put(np.full((10, 10, 3), i % 256, dtype=np.uint8))
                except Exception as e:
                    errors.append(e)

        def consumer():
            for _ in range(iterations):
                try:
                    slot.get_latest()
                    time.sleep(0.0001)
                except Exception as e:
                    errors.append(e)

        threads = [
            threading.Thread(target=producer),
            threading.Thread(target=consumer),
            threading.Thread(target=producer),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert len(errors) == 0, f"Thread safety errors: {errors}"


class TestCropFromBBox:
    """Test the face cropping utility."""

    def test_crop_returns_array_for_valid_bbox(self):
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        bbox = (0.3, 0.3, 0.7, 0.7)  # center 40% of frame
        crop = crop_from_bbox(frame, bbox)
        assert crop is not None
        assert crop.ndim == 3

    def test_crop_is_square(self):
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        bbox = (0.3, 0.3, 0.7, 0.5)  # rectangular bbox
        crop = crop_from_bbox(frame, bbox)
        assert crop is not None
        h, w = crop.shape[:2]
        assert h == w, f"Expected square crop, got {h}x{w}"

    def test_empty_bbox_returns_none(self):
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        bbox = (0.0, 0.0, 0.0, 0.0)  # degenerate
        crop = crop_from_bbox(frame, bbox)
        assert crop is None

    def test_padding_added_when_bbox_at_edge(self):
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        bbox = (-0.1, -0.1, 0.2, 0.2)  # extends past top-left
        crop = crop_from_bbox(frame, bbox)
        assert crop is not None
        # The resulting square should include replicated border pixels
        assert crop.shape[0] == crop.shape[1]
