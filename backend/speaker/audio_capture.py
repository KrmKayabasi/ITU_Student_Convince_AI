"""
Real-time audio capture via sounddevice.

Provides:
- Callback-based microphone input stream
- Lock-free ring buffer for thread-safe audio transfer
- Simulated audio for testing without a microphone

Integrated from target_speaker_pipeline/lib/audio_capture.py
"""

from __future__ import annotations

import time
import threading
import logging
from dataclasses import dataclass

import numpy as np

try:
    import sounddevice as sd

    SD_AVAILABLE = True
except ImportError:
    SD_AVAILABLE = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lock-free ring buffer (SPSC)
# ---------------------------------------------------------------------------

class RingBuffer:
    """
    Single-producer, single-consumer lock-free ring buffer.

    Producer (audio callback) writes frames.
    Consumer (VAD thread) reads frames.
    """

    def __init__(self, capacity_frames: int, frame_size: int):
        self._buffer = np.zeros((capacity_frames, frame_size), dtype=np.float32)
        self._capacity = capacity_frames
        self._frame_size = frame_size
        self._write_idx = 0
        self._read_idx = 0

    def write(self, frame: np.ndarray) -> bool:
        """
        Write a frame. Returns False if buffer is full (overflow).
        Non-blocking — safe to call from audio callback.
        """
        next_idx = (self._write_idx + 1) % self._capacity
        if next_idx == self._read_idx:
            return False  # buffer full, drop frame

        self._buffer[self._write_idx] = frame[: self._frame_size]
        self._write_idx = next_idx
        return True

    def read(self) -> np.ndarray | None:
        """
        Read a frame. Returns None if buffer is empty.
        Non-blocking.
        """
        if self._write_idx == self._read_idx:
            return None

        frame = self._buffer[self._read_idx].copy()
        self._read_idx = (self._read_idx + 1) % self._capacity
        return frame

    @property
    def available(self) -> int:
        """Number of frames available to read."""
        if self._write_idx >= self._read_idx:
            return self._write_idx - self._read_idx
        return self._capacity - self._read_idx + self._write_idx

    @property
    def overflow_count(self) -> int:
        """Number of overflows since last reset (approximate)."""
        return 0

    def reset(self) -> None:
        self._write_idx = 0
        self._read_idx = 0


# ---------------------------------------------------------------------------
# Audio capture configuration
# ---------------------------------------------------------------------------

@dataclass
class AudioConfig:
    sample_rate: int = 16000
    channels: int = 1
    blocksize: int = 512  # samples per callback (32ms at 16kHz)
    dtype: str = "float32"
    ring_buffer_capacity_s: float = 10.0  # seconds of audio buffered

    @property
    def ring_capacity_frames(self) -> int:
        return int(self.ring_buffer_capacity_s * self.sample_rate / self.blocksize)


# ---------------------------------------------------------------------------
# Audio capture
# ---------------------------------------------------------------------------

class AudioCapture:
    """
    Real-time microphone capture with ring buffer for thread-safe consumption.

    Usage:
        cap = AudioCapture()
        cap.start()
        while True:
            frame = cap.read()
            if frame is not None:
                process(frame)
    """

    def __init__(self, config: AudioConfig | None = None):
        self.config = config or AudioConfig()
        self.sd_available = SD_AVAILABLE

        self._stream: sd.InputStream | None = None
        self._ring = RingBuffer(
            capacity_frames=self.config.ring_capacity_frames,
            frame_size=self.config.blocksize,
        )
        self._lock = threading.Lock()
        self._running = False

        # Metrics
        self.overflow_count: int = 0
        self.callback_count: int = 0
        self.callback_jitter_max: float = 0.0
        self._last_callback_time: float | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> bool:
        """Start audio capture. Returns True on success."""
        if not self.sd_available:
            logger.warning("sounddevice not available, audio capture disabled")
            return False

        if self._running:
            return True

        try:
            self._stream = sd.InputStream(
                samplerate=self.config.sample_rate,
                channels=self.config.channels,
                blocksize=self.config.blocksize,
                callback=self._audio_callback,
                dtype=self.config.dtype,
            )
            self._stream.start()
            self._running = True
            logger.info(
                f"Audio capture started: {self.config.sample_rate}Hz, "
                f"blocksize={self.config.blocksize}"
            )
            return True
        except Exception as exc:
            logger.error(f"Failed to open microphone: {exc}")
            self.sd_available = False
            return False

    def stop(self) -> None:
        """Stop audio capture and release resources."""
        self._running = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        logger.info("Audio capture stopped")

    # ------------------------------------------------------------------
    # Audio callback (runs in sounddevice's internal thread)
    # ------------------------------------------------------------------

    def _audio_callback(self, indata, frames, time_info, status) -> None:
        """
        Called by sounddevice every blocksize samples.
        MUST NOT block — just writes to ring buffer.
        """
        self.callback_count += 1

        if status:
            if hasattr(status, "input_overflow") and status.input_overflow:
                self.overflow_count += 1
            if hasattr(status, "output_underflow") and status.output_underflow:
                pass

        # Track jitter
        now = time.perf_counter()
        if self._last_callback_time is not None:
            jitter = abs(
                (now - self._last_callback_time)
                - (self.config.blocksize / self.config.sample_rate)
            )
            if jitter > self.callback_jitter_max:
                self.callback_jitter_max = jitter
        self._last_callback_time = now

        if len(indata) == 0:
            return

        frame = indata.squeeze().astype(np.float32)
        if not self._ring.write(frame):
            self.overflow_count += 1

    # ------------------------------------------------------------------
    # Consumer API
    # ------------------------------------------------------------------

    def read(self) -> np.ndarray | None:
        """
        Read the next available audio frame.
        Returns None if no frame is available.
        Thread-safe — call from any thread.
        """
        return self._ring.read()

    def read_all(self) -> list[np.ndarray]:
        """Read all available frames at once."""
        frames = []
        while True:
            frame = self._ring.read()
            if frame is None:
                break
            frames.append(frame)
        return frames

    @property
    def available(self) -> int:
        return self._ring.available

    @property
    def is_running(self) -> bool:
        return self._running


# ---------------------------------------------------------------------------
# Simulated audio (for testing without microphone)
# ---------------------------------------------------------------------------

class SimulatedAudioCapture:
    """
    Generates synthetic audio with configurable speaker profiles.
    Used when no microphone is available or for testing.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        blocksize: int = 512,
    ):
        self.sample_rate = sample_rate
        self.blocksize = blocksize
        self._running = False
        self._rng = np.random.RandomState(42)

        # Speaker timbre profiles (L2-normalized spectral envelopes)
        self.target_timbre = self._make_voice(
            f0=180, f1=700, f2=2200, f3=3200, brightness=1.2
        )
        self.nontarget_timbre = self._make_voice(
            f0=120, f1=500, f2=1500, f3=2800, brightness=0.7
        )

        # Simulation state (set externally)
        self.target_speaking = False
        self.nontarget_speaking = False
        self.ambient_noise_level = 0.001

    @staticmethod
    def _make_voice(
        f0: float, f1: float, f2: float, f3: float, brightness: float
    ) -> dict:
        return {"f0": f0, "f1": f1, "f2": f2, "f3": f3, "brightness": brightness}

    def generate_frame(self) -> np.ndarray:
        """Generate one audio frame based on current simulation state."""
        t = (
            np.arange(self.blocksize, dtype=np.float32)
            / self.sample_rate
            + self._rng.uniform(0, 0.1)
        )

        frame = np.zeros(self.blocksize, dtype=np.float32)

        if self.target_speaking and self.nontarget_speaking:
            frame += self._voice_frame(t, self.target_timbre, amplitude=0.15)
            frame += self._voice_frame(
                t + self._rng.uniform(0, 0.02),
                self.nontarget_timbre,
                amplitude=0.12,
            )
        elif self.target_speaking:
            frame += self._voice_frame(t, self.target_timbre, amplitude=0.2)
        elif self.nontarget_speaking:
            frame += self._voice_frame(t, self.nontarget_timbre, amplitude=0.2)

        # Ambient noise
        frame += self._rng.normal(0, self.ambient_noise_level, self.blocksize)

        return frame.astype(np.float32)

    def _voice_frame(
        self, t: np.ndarray, voice: dict, amplitude: float = 0.2
    ) -> np.ndarray:
        """Synthesize a voiced frame with formant structure."""
        rng = self._rng

        # Glottal pulse (buzz at F0 + jitter)
        f0 = voice["f0"] * (1.0 + rng.uniform(-0.01, 0.01))
        phase = 2.0 * np.pi * f0 * t + rng.uniform(0, 2 * np.pi)
        glottal = np.sin(phase) + 0.3 * np.sin(2 * phase) + 0.1 * np.sin(3 * phase)

        # Formant filter
        f1_gain = voice["brightness"] * voice["f1"] / 1000.0
        f2_gain = voice["brightness"] * voice["f2"] / 2500.0
        f3_gain = voice["brightness"] * voice["f3"] / 4000.0

        formant_mod = (
            1.0
            + f1_gain * 0.3 * np.sin(2 * np.pi * voice["f1"] * t)
            + f2_gain * 0.2 * np.sin(2 * np.pi * voice["f2"] * t)
            + f3_gain * 0.1 * np.sin(2 * np.pi * voice["f3"] * t)
        )

        # Shimmer (amplitude variation)
        shimmer = 1.0 + rng.uniform(-0.03, 0.03)

        return (amplitude * shimmer * glottal * formant_mod).astype(np.float32)

    def read(self) -> np.ndarray | None:
        """Generate the next frame."""
        return self.generate_frame()

    @property
    def is_running(self) -> bool:
        return self._running

    start = lambda self: setattr(self, "_running", True) or True
    stop = lambda self: setattr(self, "_running", False)
    sd_available = False
