import time
import pytest
import numpy as np
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import uvicorn
import threading
from PyQt6.QtCore import QCoreApplication
import sys

from client.desktop_client import ResponseGeneratorWorker

# Setup local mock speech server
mock_app = FastAPI()

@mock_app.post("/chat_stream")
def chat_stream():
    # Return mock streaming audio chunks (1 second of float32 silence at 24000 Hz)
    def generate():
        silence = np.zeros(24000, dtype=np.float32)
        yield silence.tobytes()
    return StreamingResponse(generate(), media_type="application/octet-stream")

@mock_app.get("/last_turn")
def last_turn():
    return {
        "user": "Merhaba tercih danışmanı",
        "assistant": "Merhaba! İTÜ kariyer rehberine hoş geldiniz."
    }

class ServerThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.config = uvicorn.Config(mock_app, host="127.0.0.1", port=9009, log_level="warning")
        self.server = uvicorn.Server(self.config)

    def run(self):
        self.server.run()

    def shutdown(self):
        self.server.should_exit = True


@pytest.fixture(scope="module")
def mock_speech_server():
    server = ServerThread()
    server.start()
    time.sleep(1.0)  # Wait for server startup
    yield "http://127.0.0.1:9009"
    server.shutdown()
    server.join()


class DummyDiarisationTrack:
    def itertracks(self, yield_label=True):
        # Return mock speaker segment track
        class MockSegment:
            start = 0.0
            end = 1.5
        yield MockSegment(), None, "SPEAKER_00"


class MockDiarisationPipeline:
    def __call__(self, file_path):
        return DummyDiarisationTrack()


def test_response_generator_integration(mock_speech_server):
    """Verify that ResponseGeneratorWorker correctly communicates with mock cascaded server."""
    # 1. Synthesize mock 16kHz audio input
    t = np.linspace(0, 1.5, int(16000 * 1.5), endpoint=False)
    audio_data = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    # 2. Instantiate worker pointing to mock server and mock pipeline
    pipeline = MockDiarisationPipeline()
    worker = ResponseGeneratorWorker(
        audio_data=audio_data,
        pipeline=pipeline,
        cascaded_url=mock_speech_server
    )

    # Variables to collect signal responses
    diar_turns = []
    text_results = {"user": "", "bot": ""}
    status_updates = []

    def on_text_ready(u, b):
        text_results["user"] = u
        text_results["bot"] = b

    worker.diarisation_ready.connect(lambda turns: diar_turns.extend(turns))
    worker.text_ready.connect(on_text_ready)
    worker.status_changed.connect(lambda s: status_updates.append(s))

    # Run the worker synchronously in the test thread to verify completion
    worker.run()

    # 3. Assertions
    assert len(diar_turns) == 1
    assert diar_turns[0]["speaker_id"] == 0
    assert text_results["user"] == "Merhaba tercih danışmanı"
    assert text_results["bot"] == "Merhaba! İTÜ kariyer rehberine hoş geldiniz."
    assert "Diarizing speech..." in status_updates
    assert "Speaking..." in status_updates
    assert "Idle" in status_updates
