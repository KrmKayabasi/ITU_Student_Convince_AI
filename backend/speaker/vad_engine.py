"""
Voice Activity Detection engine using sherpa-onnx's Silero VAD.

Provides:
- Per-frame VAD classification (speech vs silence)
- Utterance segmentation with silence-tail detection
- Streaming VAD with internal state management

Integrated from target_speaker_pipeline/lib/vad_engine.py
"""

from __future__ import annotations

import os
import time
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sherpa_onnx

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model path resolution
# ---------------------------------------------------------------------------

def _resolve_vad_model(model_path: str) -> str:
    """Resolve the Silero VAD model path."""
    if model_path and os.path.exists(model_path):
        return model_path

    env_path = os.environ.get("VAD_MODEL_PATH", "")
    if env_path and os.path.exists(env_path):
        return env_path

    # Try relative to this file (backend/speaker/ -> ../../)
    candidate = Path(__file__).parent.parent.parent / "models" / "silero_vad.onnx"
    if candidate.exists():
        return str(candidate)

    # Try speech_backend location (existing)
    candidate2 = Path(__file__).parent.parent / "speech_backend" / "silero_vad.onnx"
    if candidate2.exists():
        return str(candidate2)

    # Try cwd
    cwd_candidate = Path("silero_vad.onnx")
    if cwd_candidate.exists():
        return str(cwd_candidate)

    return str(candidate)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class VADConfig:
    """VAD configuration."""

    model_path: str = "models/silero_vad.onnx"
    sample_rate: int = 16000
    window_size: int = 512          # samples per frame (32ms at 16kHz)
    num_threads: int = 2
    provider: str = "cpu"

    # Silero VAD thresholds (controls hangover behavior internally)
    speech_threshold: float = 0.5    # probability above this = speech
    min_silence_duration: float = 0.3  # seconds of silence to end segment
    min_speech_duration: float = 0.25  # seconds of speech to start segment
    max_speech_duration: float = 20.0  # force-flush after this many seconds

    # Additional gates
    energy_threshold: float = 0.002    # RMS below this = always silence
    segment_min_duration: float = 0.5  # discard segments shorter than this

    # How many CONSECUTIVE silence frames before we consider the utterance complete
    extra_silence_tail_frames: int = 10  # 10 * 32ms = 320ms


# ---------------------------------------------------------------------------
# Speech segment
# ---------------------------------------------------------------------------

@dataclass
class SpeechSegment:
    """A detected speech utterance."""

    audio: np.ndarray
    start_time: float
    end_time: float
    duration: float
    vad_confidence: float
    rms_energy: float
    frame_count: int
    sample_rate: int = 16000

    @property
    def is_valid(self) -> bool:
        return (
            self.duration >= 1.0
            and self.vad_confidence >= 0.5
            and self.rms_energy >= 0.002
        )

    def __repr__(self) -> str:
        return (
            f"SpeechSegment(dur={self.duration:.2f}s, "
            f"vad={self.vad_confidence:.2f}, "
            f"rms={self.rms_energy:.4f}, "
            f"frames={self.frame_count})"
        )


# ---------------------------------------------------------------------------
# VAD Engine
# ---------------------------------------------------------------------------

class VADEngine:
    """
    Streaming VAD using sherpa-onnx's Silero VAD model.

    The VAD model maintains INTERNAL state (speech/silence hangover counters).
    We wrap it with frame accumulation and segment extraction.

    Usage:
        vad = VADEngine()
        for chunk in audio_stream:
            is_speech = vad.process(chunk)
            if vad.has_segment():
                segment = vad.get_segment()
    """

    def __init__(self, config: VADConfig | None = None):
        self.config = config or VADConfig()

        resolved_model = _resolve_vad_model(self.config.model_path)

        silero_cfg = sherpa_onnx.SileroVadModelConfig(
            model=resolved_model,
            threshold=self.config.speech_threshold,
            min_silence_duration=self.config.min_silence_duration,
            min_speech_duration=self.config.min_speech_duration,
            window_size=self.config.window_size,
            max_speech_duration=self.config.max_speech_duration,
        )
        vad_cfg = sherpa_onnx.VadModelConfig(
            silero_vad=silero_cfg,
            sample_rate=self.config.sample_rate,
            num_threads=self.config.num_threads,
            provider=self.config.provider,
            debug=False,
        )
        self._vad = sherpa_onnx.VadModel.create(vad_cfg)

        # Internal accumulators
        self._speech_buffer: list[np.ndarray] = []
        self._speech_start_time: float | None = None
        self._silence_counter: int = 0
        self._frame_count: int = 0
        self._vad_confidences: list[float] = []
        self._rms_values: list[float] = []

        # State
        self.is_in_speech: bool = False
        self.current_is_speech: bool = False

        # Stats
        self.total_speech_frames: int = 0
        self.total_silence_frames: int = 0
        self.segment_count: int = 0

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def process(self, audio_chunk: np.ndarray) -> bool:
        """
        Process one audio frame. Returns True if VAD classifies as speech.

        Args:
            audio_chunk: 1-D float32, must be exactly window_size samples.

        Returns:
            True if this frame contains speech.
        """
        chunk = np.asarray(audio_chunk, dtype=np.float32).squeeze()

        # Ensure correct length
        ws = self._vad.window_size()
        if len(chunk) < ws:
            chunk = np.pad(chunk, (0, ws - len(chunk)), mode="constant")
        elif len(chunk) > ws:
            chunk = chunk[:ws]

        # Energy pre-filter
        rms = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))
        if rms < self.config.energy_threshold:
            self.current_is_speech = False
            self._on_silence_frame(chunk, rms)
            self.total_silence_frames += 1
            return False

        # Run Silero VAD (the model has internal state)
        try:
            self.current_is_speech = self._vad.is_speech(chunk.tolist())
        except Exception:
            self.current_is_speech = False

        if self.current_is_speech:
            self.total_speech_frames += 1
            self._on_speech_frame(chunk, rms)
        else:
            self.total_silence_frames += 1
            self._on_silence_frame(chunk, rms)

        return self.current_is_speech

    def _on_speech_frame(self, chunk: np.ndarray, rms: float) -> None:
        """Handle detected speech frame."""
        self._silence_counter = 0

        if not self.is_in_speech:
            self.is_in_speech = True
            self._speech_start_time = time.time()

        self._speech_buffer.append(chunk.copy())
        self._vad_confidences.append(0.85)  # model doesn't expose raw prob
        self._rms_values.append(rms)
        self._frame_count += 1

    def _on_silence_frame(self, chunk: np.ndarray, rms: float) -> None:
        """Handle detected silence frame."""
        if self.is_in_speech:
            self._silence_counter += 1

    def has_segment(self) -> bool:
        """Check if a complete segment is ready."""
        if not self.is_in_speech:
            return False
        silent_frames = self.config.extra_silence_tail_frames
        return self._silence_counter >= silent_frames

    def get_segment(self) -> SpeechSegment | None:
        """Extract the accumulated speech segment and reset."""
        if not self._speech_buffer:
            self._reset_state()
            return None

        audio = np.concatenate(self._speech_buffer).astype(np.float32)
        duration = len(audio) / self.config.sample_rate

        segment = SpeechSegment(
            audio=audio,
            start_time=self._speech_start_time or 0.0,
            end_time=time.time(),
            duration=duration,
            vad_confidence=(
                float(np.mean(self._vad_confidences))
                if self._vad_confidences
                else 0.0
            ),
            rms_energy=(
                float(np.mean(self._rms_values))
                if self._rms_values
                else 0.0
            ),
            frame_count=self._frame_count,
            sample_rate=self.config.sample_rate,
        )

        self._reset_state()

        if duration < self.config.segment_min_duration:
            logger.debug(f"Discarding short segment: {duration:.2f}s")
            return None

        self.segment_count += 1
        return segment

    def flush(self) -> SpeechSegment | None:
        """Force-flush any accumulated speech (e.g., on shutdown)."""
        if not self._speech_buffer:
            return None
        old_counter = self._silence_counter
        self._silence_counter = 999
        seg = self.get_segment()
        self._silence_counter = old_counter
        return seg

    def _reset_state(self) -> None:
        """Reset internal accumulators."""
        self._speech_buffer = []
        self._speech_start_time = None
        self._silence_counter = 0
        self._frame_count = 0
        self._vad_confidences = []
        self._rms_values = []
        self.is_in_speech = False

    def reset(self) -> None:
        """Full reset including VAD model state."""
        self._reset_state()
        self._vad.reset()
        self.current_is_speech = False

    @property
    def window_samples(self) -> int:
        return self._vad.window_size()

    @property
    def window_duration_ms(self) -> float:
        return self.window_samples / self.config.sample_rate * 1000
