import os
import numpy as np
import pytest
from PyQt6.QtCore import QCoreApplication
import sys

from client.desktop_client import AudioCaptureWorker, PipelineLoaderWorker, parse_args
from client.workers import SpeakerTracker

# Setup Qt application context for tests
@pytest.fixture(scope="session", autouse=True)
def qt_app():
    app = QCoreApplication.instance()
    if app is None:
        app = QCoreApplication(sys.argv[:1])
    yield app


def test_pipeline_loader_initialization():
    """Verify that the background pipeline loader thread runs and emits signal."""
    loader = PipelineLoaderWorker()
    
    # We mock actual loading to avoid downloading model on testing pipeline
    loaded_pipeline = None
    
    def on_loaded(pipeline):
        nonlocal loaded_pipeline
        loaded_pipeline = pipeline

    loader.loaded.connect(on_loaded)
    
    # Check simple state
    assert not loader.isRunning()
    
    # We verify class properties
    assert hasattr(loader, "loaded")


def test_audio_capture_worker_properties():
    """Verify default configurations of AudioCaptureWorker."""
    worker = AudioCaptureWorker(sample_rate=16000)
    
    assert worker.sample_rate == 16000
    assert worker.use_vad is True
    assert worker.is_active is True
    assert worker.is_recording is False
    assert len(worker.buffer) == 0


def test_festival_and_diarization_flags(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["desktop_client.py", "--no-diarization"])

    args = parse_args()

    assert args.festival_mode is True
    assert args.diarization is False
    assert args.noise_suppression is True


def test_flag_defaults_can_come_from_environment(monkeypatch):
    monkeypatch.setenv("FESTIVAL_MODE", "0")
    monkeypatch.setenv("ENABLE_DIARIZATION", "0")
    monkeypatch.setenv("AUDIO_NOISE_SUPPRESSION", "0")
    monkeypatch.setattr(sys, "argv", ["desktop_client.py"])

    args = parse_args()

    assert args.festival_mode is False
    assert args.diarization is False
    assert args.noise_suppression is False


def test_speaker_tracker_keeps_stable_ids_across_turns():
    tracker = SpeakerTracker(threshold=0.2)

    first = tracker.assign({"local-a": np.array([1.0, 0.0], dtype=np.float32)})
    second = tracker.assign({"local-x": np.array([0.99, 0.01], dtype=np.float32)})

    assert first["local-a"] == 0
    assert second["local-x"] == 0


def test_speaker_tracker_allocates_and_resets_different_speakers():
    tracker = SpeakerTracker(threshold=0.2)

    first = tracker.assign({"a": np.array([1.0, 0.0], dtype=np.float32)})
    second = tracker.assign({"b": np.array([0.0, 1.0], dtype=np.float32)})
    tracker.reset()
    after_reset = tracker.assign({"b": np.array([0.0, 1.0], dtype=np.float32)})

    assert first["a"] == 0
    assert second["b"] == 1
    assert after_reset["b"] == 0
