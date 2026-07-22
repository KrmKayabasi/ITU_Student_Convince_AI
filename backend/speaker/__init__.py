"""
Speaker verification module — shared across orchestrator and speech_backend.

Provides neural speaker embedding (TitaNet-Small via sherpa-onnx),
voice activity detection (Silero VAD), audio capture utilities,
and threshold calibration.
"""

from backend.speaker.speaker_engine import (
    SpeakerEmbeddingEngine,
    SpeakerDatabase,
    SpeakerProfile,
)
from backend.speaker.vad_engine import (
    VADEngine,
    VADConfig,
    SpeechSegment,
)
from backend.speaker.audio_capture import (
    AudioCapture,
    AudioConfig,
    SimulatedAudioCapture,
    RingBuffer,
)

__all__ = [
    "SpeakerEmbeddingEngine",
    "SpeakerDatabase",
    "SpeakerProfile",
    "VADEngine",
    "VADConfig",
    "SpeechSegment",
    "AudioCapture",
    "AudioConfig",
    "SimulatedAudioCapture",
    "RingBuffer",
]
