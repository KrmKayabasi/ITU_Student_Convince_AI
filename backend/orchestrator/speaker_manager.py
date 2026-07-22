"""
Async speaker verification manager for the orchestrator.

Wraps the speaker engine with non-blocking embedding extraction
(via ThreadPoolExecutor) and audio format conversion for the
orchestrator's PCM16 byte stream.

Usage:
    manager = SpeakerManager()
    await manager.initialize()

    # Enrollment from raw float32 audio
    quality = await manager.enroll("baydogan", float32_audio)

    # Per-chunk processing (non-blocking, accumulates internally)
    await manager.process_audio_chunk(pcm16_bytes)

    # Verification
    is_match, score = await manager.verify_accumulated("baydogan")
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

# Resolve the shared speaker module. In local dev it lives at ../speaker
# (relative to this file); in Docker it lives at /srv/speaker.  Add the
# resolved path so the import works in both environments.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SPEAKER_DIR = os.path.join(_HERE, "..", "speaker")
if os.path.isdir(_SPEAKER_DIR):
    sys.path.insert(0, os.path.abspath(_SPEAKER_DIR))

from speaker_engine import SpeakerEmbeddingEngine, SpeakerDatabase

logger = logging.getLogger("orchestrator.speaker")


class SpeakerManager:
    """
    Async speaker verification manager.

    Owns the embedding engine and profile database.
    Runs CPU-bound embedding extraction in a thread pool to avoid
    blocking the async event loop.
    """

    def __init__(
        self,
        *,
        model_path: str = "",
        profiles_dir: str = "profiles",
        verify_threshold: float = 0.45,
        update_threshold: float = 0.65,
        max_stored_embeddings: int = 50,
        accumulation_duration_s: float = 1.5,
        bargein_cooldown_s: float = 2.0,
        enabled: bool = True,
        bargein_enabled: bool = True,
    ):
        self.model_path = model_path
        self.profiles_dir = profiles_dir
        self.verify_threshold = verify_threshold
        self.update_threshold = update_threshold
        self.max_stored_embeddings = max_stored_embeddings
        self.accumulation_duration_s = accumulation_duration_s
        self.bargein_cooldown_s = bargein_cooldown_s
        self.enabled = enabled
        self.bargein_enabled = bargein_enabled

        # Initialized in initialize()
        self._engine: SpeakerEmbeddingEngine | None = None
        self._db: SpeakerDatabase | None = None
        self._executor: ThreadPoolExecutor | None = None

        # Target speaker tracking
        self.target_speaker_id: str | None = None

        # Audio accumulation buffer (for streaming chunks)
        self._audio_buffer: list[np.ndarray] = []
        self._sample_rate: int = 16000
        self._accumulation_samples: int = int(accumulation_duration_s * self._sample_rate)

        # Barge-in state
        self._last_bargein_time: float = 0.0
        self._last_speaker_status: str = "unknown"
        self._last_speaker_score: float = 0.0
        self._last_emit_time: float = 0.0
        self._emit_debounce_s: float = 1.0  # debounce speaker status emissions

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Load models and profile database (call once on startup)."""
        loop = asyncio.get_running_loop()
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="speaker")

        # Load engine (CPU-bound, run in thread)
        def _load():
            engine = SpeakerEmbeddingEngine(model_path=self.model_path, num_threads=2)
            db = SpeakerDatabase(storage_dir=self.profiles_dir)
            return engine, db

        self._engine, self._db = await loop.run_in_executor(self._executor, _load)
        logger.info(
            "SpeakerManager initialized: model=%s dim=%d profiles=%d",
            self._engine.model_path,
            self._engine.dim,
            self._db.speaker_count,
        )

    async def close(self) -> None:
        """Flush profiles and shut down thread pool."""
        if self._db is not None:
            self._db.flush()
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        logger.info("SpeakerManager closed")

    @property
    def is_ready(self) -> bool:
        return self._engine is not None and self._db is not None and self.enabled

    # ------------------------------------------------------------------
    # Enrollment
    # ------------------------------------------------------------------

    async def enroll(self, speaker_id: str, audio: np.ndarray) -> dict:
        """
        Enroll a speaker from float32 audio samples.

        Args:
            speaker_id: Unique speaker identifier.
            audio: 1-D float32 array, 16kHz, 1-10 seconds.

        Returns:
            {"status": "enrolled", "speaker_id": ..., "quality": ..., "embedding_dim": ...}
        """
        if not self.is_ready:
            return {"status": "error", "message": "SpeakerManager not initialized"}

        loop = asyncio.get_running_loop()
        embedding = await loop.run_in_executor(
            self._executor, self._engine.extract, audio
        )

        profile = self._db.add_or_update(
            speaker_id,
            embedding,
            max_stored=self.max_stored_embeddings,
        )

        self.target_speaker_id = speaker_id

        return {
            "status": "enrolled",
            "speaker_id": speaker_id,
            "quality": round(profile.quality, 3),
            "embedding_dim": self._engine.dim,
            "embedding_count": len(profile.embeddings),
        }

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    async def verify(
        self, audio: np.ndarray, speaker_id: str | None = None
    ) -> tuple[bool, float, str | None]:
        """
        Verify a float32 audio segment against an enrolled speaker.

        Args:
            audio: 1-D float32 array, 16kHz.
            speaker_id: Speaker to verify against (default: target_speaker_id).

        Returns:
            (is_match, score, matched_speaker_id)
        """
        if not self.is_ready:
            return False, 0.0, None

        sid = speaker_id or self.target_speaker_id
        if sid is None:
            return False, 0.0, None

        loop = asyncio.get_running_loop()
        embedding = await loop.run_in_executor(
            self._executor, self._engine.extract, audio
        )

        is_match, score = self._db.verify(sid, embedding, threshold=self.verify_threshold)

        # If match and score is high enough, update the profile
        if is_match and score >= self.update_threshold:
            self._db.add_or_update(sid, embedding, max_stored=self.max_stored_embeddings)

        return is_match, float(score), sid if is_match else None

    async def verify_embedding(
        self, embedding: np.ndarray, speaker_id: str | None = None
    ) -> tuple[bool, float, str | None]:
        """Verify a pre-extracted embedding directly."""
        if not self.is_ready:
            return False, 0.0, None

        sid = speaker_id or self.target_speaker_id
        if sid is None:
            return False, 0.0, None

        is_match, score = self._db.verify(sid, embedding, threshold=self.verify_threshold)
        return is_match, float(score), sid if is_match else None

    # ------------------------------------------------------------------
    # Streaming audio accumulation (for orchestrator PCM16 path)
    # ------------------------------------------------------------------

    def pcm16_to_float32(self, pcm16: bytes) -> np.ndarray:
        """Convert little-endian PCM16 bytes to float32 [-1, 1] array."""
        if not pcm16:
            return np.zeros(0, dtype=np.float32)
        pcm = np.frombuffer(pcm16, dtype="<i2")
        return (pcm.astype(np.float32) / 32768.0).astype(np.float32)

    def accumulate_chunk(self, float32_chunk: np.ndarray) -> None:
        """Add a float32 audio chunk to the accumulation buffer."""
        self._audio_buffer.append(float32_chunk.copy())

    def has_enough_audio(self) -> bool:
        """Check if enough audio has accumulated for a reliable embedding."""
        total = sum(len(c) for c in self._audio_buffer)
        return total >= self._accumulation_samples

    def get_accumulated_audio(self) -> np.ndarray:
        """Get the accumulated audio and clear the buffer."""
        if not self._audio_buffer:
            return np.zeros(0, dtype=np.float32)
        audio = np.concatenate(self._audio_buffer).astype(np.float32)
        self._audio_buffer.clear()
        return audio

    def get_accumulated_duration(self) -> float:
        """Get the duration of accumulated audio in seconds."""
        total = sum(len(c) for c in self._audio_buffer)
        return total / self._sample_rate

    # ------------------------------------------------------------------
    # Orchestrator integration
    # ------------------------------------------------------------------

    async def process_audio_chunk(
        self, pcm16: bytes
    ) -> dict | None:
        """
        Process one PCM16 audio chunk from the browser.

        Converts to float32, accumulates, and periodically runs
        speaker verification. Returns a dict if a speaker event
        should be emitted, None otherwise.

        Returns:
            None (no event to emit) or
            {"type": "speaker", "status": "target"|"nontarget"|"unknown", ...}
            {"type": "barge_in", "reason": "nontarget_speaker", ...}
        """
        if not self.is_ready:
            return None

        # Convert and accumulate
        float32_chunk = self.pcm16_to_float32(pcm16)
        if len(float32_chunk) == 0:
            return None
        self.accumulate_chunk(float32_chunk)

        # Only run verification when enough audio has accumulated
        if not self.has_enough_audio():
            return None

        # Extract accumulated audio and run verification
        audio = self.get_accumulated_audio()

        # Check energy — skip near-silence
        rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
        if rms < 0.002:
            return None

        # Run embedding extraction + verification in thread pool
        loop = asyncio.get_running_loop()

        def _extract_and_verify():
            embedding = self._engine.extract(audio)
            if self.target_speaker_id and self._db:
                is_match, score = self._db.verify(
                    self.target_speaker_id,
                    embedding,
                    threshold=self.verify_threshold,
                )
                # Search all profiles
                best_id, best_score = self._db.search(
                    embedding, threshold=self.verify_threshold
                )
                return embedding, is_match, float(score), best_id, float(best_score)
            return embedding, False, 0.0, None, 0.0

        try:
            embedding, is_match, score, best_id, best_score = (
                await loop.run_in_executor(self._executor, _extract_and_verify)
            )
        except Exception:
            logger.exception("Speaker verification failed")
            return None

        now = time.time()

        # Determine speaker status
        if is_match:
            status = "target"
            speaker_id = self.target_speaker_id
            display_score = score

            # Update profile with this embedding (adaptive)
            if score >= self.update_threshold:
                self._db.add_or_update(
                    self.target_speaker_id,
                    embedding,
                    max_stored=self.max_stored_embeddings,
                )
        elif best_id is not None:
            status = "nontarget"
            speaker_id = best_id
            display_score = best_score
        else:
            status = "unknown"
            speaker_id = None
            display_score = best_score

        self._last_speaker_status = status
        self._last_speaker_score = display_score

        # Debounce emission
        if now - self._last_emit_time < self._emit_debounce_s:
            return None
        self._last_emit_time = now

        return {
            "type": "speaker",
            "status": status,
            "speaker_id": speaker_id,
            "score": round(display_score, 3),
        }

    def check_bargein(self, is_ai_speaking: bool) -> dict | None:
        """
        Check if the last speaker status should trigger a barge-in.

        Only triggers when:
        - Speaker verification is enabled and barge-in is enabled
        - AI is currently speaking
        - A NON-TARGET speaker was detected
        - Cooldown has elapsed
        """
        if not self.bargein_enabled or not self.enabled:
            return None

        if not is_ai_speaking:
            return None

        if self._last_speaker_status != "nontarget":
            return None

        now = time.time()
        if now - self._last_bargein_time < self.bargein_cooldown_s:
            return None

        self._last_bargein_time = now
        return {
            "type": "barge_in",
            "reason": "nontarget_speaker",
            "score": round(self._last_speaker_score, 3),
        }

    # ------------------------------------------------------------------
    # Profile management
    # ------------------------------------------------------------------

    def get_profiles(self) -> list[dict]:
        """List all enrolled speaker profiles."""
        if self._db is None:
            return []
        profiles = []
        for sid, profile in self._db.get_all().items():
            profiles.append({
                "speaker_id": sid,
                "quality": round(profile.quality, 3),
                "embedding_count": len(profile.embeddings),
                "created_at": profile.created_at,
                "updated_at": profile.updated_at,
            })
        return profiles

    def remove_profile(self, speaker_id: str) -> bool:
        """Remove a speaker profile."""
        if self._db is None:
            return False
        if speaker_id == self.target_speaker_id:
            self.target_speaker_id = None
        return self._db.remove(speaker_id)

    def reset_all(self) -> int:
        """Remove all profiles. Returns count of deleted profiles."""
        if self._db is None:
            return 0
        count = self._db.speaker_count
        for sid in list(self._db.all_speaker_ids):
            self._db.remove(sid)
        self.target_speaker_id = None
        return count

    @property
    def is_enrolled(self) -> bool:
        return self.target_speaker_id is not None and self._db is not None
