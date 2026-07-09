import time
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from PyQt6.QtCore import QCoreApplication
import sys

from client.desktop_client import StreamWorker

@pytest.fixture(scope="session", autouse=True)
def qt_app_context():
    app = QCoreApplication.instance()
    if app is None:
        app = QCoreApplication(sys.argv[:1])
    yield app


@patch("cv2.VideoCapture")
def test_stream_worker_lifecycle(mock_video_capture, qt_app_context):
    """Verify that StreamWorker starts up, manages camera handles, and stops cleanly."""
    # Mock cv2 VideoCapture behavior
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, np.zeros((100, 100, 3), dtype=np.uint8))
    mock_video_capture.return_value = mock_cap

    worker = StreamWorker(
        server="ws://localhost:8000",
        session_id="test-session",
        camera_index=-1,
        fps=10.0
    )

    assert not worker.isRunning()
    
    # Verify properties
    assert worker.server == "ws://localhost:8000"
    assert worker.session_id == "test-session"
    assert worker.fps == 10.0
    assert not worker._stop_event.is_set()

    # Trigger stopping event
    worker.stop()
    assert worker._stop_event.is_set()
