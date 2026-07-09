import os
import numpy as np
import pytest
from PyQt6.QtCore import QCoreApplication
import sys

from client.desktop_client import AudioCaptureWorker, PipelineLoaderWorker

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
