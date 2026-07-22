"""
Target-Speaker-Conditioned Voice Pipeline Library.

Production-grade speaker verification using:
- sherpa-onnx + TitaNet/CAM++ (speaker embeddings)
- Silero VAD (voice activity detection)
- sounddevice (real-time audio capture)
"""

from lib.speaker_engine import SpeakerEmbeddingEngine, SpeakerDatabase, SpeakerProfile
from lib.vad_engine import VADEngine, VADConfig, SpeechSegment
from lib.audio_capture import AudioCapture, AudioConfig, SimulatedAudioCapture, RingBuffer

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
