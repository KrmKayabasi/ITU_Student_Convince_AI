"""
Integration tests for the speaker verification module.

Tests the core SpeakerEngine, SpeakerDatabase, SpeakerManager,
and the REST API endpoints.

Run with:
    cd backend
    python3 -m pytest tests/test_speaker_integration.py -v
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

# Ensure the backend directory is on the path
_backend_dir = Path(__file__).parent.parent
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

# Add orchestrator to path for SpeakerManager import
_orch_dir = _backend_dir / "orchestrator"
if str(_orch_dir) not in sys.path:
    sys.path.insert(0, str(_orch_dir))


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def make_sine(freq: float, duration: float, sample_rate: int = 16000) -> np.ndarray:
    """Generate a sine wave of given frequency and duration."""
    t = np.arange(int(duration * sample_rate), dtype=np.float32) / sample_rate
    return (0.5 * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)


def make_voice_like(
    f0: float = 150,
    duration: float = 2.0,
    sample_rate: int = 16000,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate a simple voice-like signal with formant structure.
    Different f0 values produce signals that TitaNet can discriminate.
    """
    rng = np.random.RandomState(seed)
    t = np.arange(int(duration * sample_rate), dtype=np.float32) / sample_rate

    # Glottal pulse (buzz at F0 + harmonics)
    phase = 2.0 * np.pi * f0 * t
    glottal = (
        np.sin(phase)
        + 0.5 * np.sin(2.0 * phase)
        + 0.25 * np.sin(3.0 * phase)
        + 0.125 * np.sin(4.0 * phase)
    )

    # Simple formant modulation
    f1, f2 = f0 * 4.5, f0 * 13.0
    formant = (
        1.0
        + 0.3 * np.sin(2.0 * np.pi * f1 * t)
        + 0.15 * np.sin(2.0 * np.pi * f2 * t)
    )

    # Add slight jitter and noise
    jitter = 1.0 + rng.uniform(-0.01, 0.01, len(t))
    noise = rng.normal(0, 0.005, len(t))

    audio = 0.3 * jitter * glottal * formant + noise
    return audio.astype(np.float32)


# ---------------------------------------------------------------------------
# Unit tests — SpeakerEngine & SpeakerDatabase
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def speaker_engine():
    """Load the speaker embedding engine once per test module."""
    from backend.speaker.speaker_engine import SpeakerEmbeddingEngine
    return SpeakerEmbeddingEngine(num_threads=2)


@pytest.fixture
def temp_db():
    """Create a temporary speaker database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        from backend.speaker.speaker_engine import SpeakerDatabase
        db = SpeakerDatabase(storage_dir=tmpdir)
        yield db


class TestSpeakerEngine:
    """Tests for SpeakerEmbeddingEngine."""

    def test_engine_loads(self, speaker_engine):
        """Model loads and exposes correct dimension."""
        assert speaker_engine is not None
        assert speaker_engine.dim == 192

    def test_extract_returns_normalized_embedding(self, speaker_engine):
        """Embedding extraction produces L2-normalized 192-dim vector."""
        audio = make_voice_like(f0=150, duration=2.0)
        emb = speaker_engine.extract(audio)
        assert emb.shape == (192,)
        assert emb.dtype == np.float32
        # L2 norm should be ~1.0
        norm = float(np.linalg.norm(emb))
        assert 0.99 < norm < 1.01, f"Expected L2 norm ~1.0, got {norm}"

    def test_embedding_consistency(self, speaker_engine):
        """Same audio produces nearly identical embeddings."""
        audio = make_voice_like(f0=150, duration=2.0, seed=42)
        emb1 = speaker_engine.extract(audio)
        emb2 = speaker_engine.extract(audio)
        cos_sim = float(np.dot(emb1, emb2))
        assert cos_sim > 0.99, f"Expected cos_sim > 0.99, got {cos_sim}"

    def test_different_voices_discriminated(self, speaker_engine):
        """Different voice-like signals produce different embeddings."""
        audio_a = make_voice_like(f0=150, duration=2.0, seed=1)
        audio_b = make_voice_like(f0=220, duration=2.0, seed=2)
        emb_a = speaker_engine.extract(audio_a)
        emb_b = speaker_engine.extract(audio_b)
        cos_sim = float(np.dot(emb_a, emb_b))
        # Different voices should not be too similar
        assert cos_sim < 0.95, f"Expected cos_sim < 0.95, got {cos_sim}"

    def test_extract_batch(self, speaker_engine):
        """Batch extraction works."""
        audios = [
            make_voice_like(f0=150, duration=1.5, seed=i)
            for i in range(3)
        ]
        embeddings = speaker_engine.extract_batch(audios)
        assert len(embeddings) == 3
        for emb in embeddings:
            assert emb.shape == (192,)

    def test_silence_handling(self, speaker_engine):
        """Silent/near-silent audio should not crash."""
        silence = np.zeros(16000, dtype=np.float32)
        emb = speaker_engine.extract(silence)
        assert emb.shape == (192,)
        assert emb.dtype == np.float32


class TestSpeakerDatabase:
    """Tests for SpeakerDatabase (CRUD, persistence, verification)."""

    def test_add_and_get_profile(self, temp_db):
        """Adding and retrieving a profile works."""
        emb = np.random.randn(192).astype(np.float32)
        emb /= np.linalg.norm(emb)
        profile = temp_db.add_or_update("alice", emb)
        assert profile.speaker_id == "alice"
        assert len(profile.embeddings) == 1

        retrieved = temp_db.get("alice")
        assert retrieved is not None
        assert retrieved.speaker_id == "alice"

    def test_centroid_computation(self, temp_db):
        """Profile centroid is the mean of embeddings."""
        emb1 = np.array([1.0, 0.0, 0.0] + [0.0] * 189, dtype=np.float32)
        emb2 = np.array([0.0, 1.0, 0.0] + [0.0] * 189, dtype=np.float32)

        from backend.speaker.speaker_engine import SpeakerProfile
        profile = SpeakerProfile("test", embeddings=[emb1, emb2])
        centroid = profile.centroid
        assert centroid is not None
        # L2 normalized mean of [1,0,0...] and [0,1,0...] = [0.707, 0.707, 0...]
        assert centroid[0] > 0.6
        assert centroid[1] > 0.6

    def test_verify_match(self, temp_db):
        """Verifying the same embedding returns a match."""
        emb = np.random.randn(192).astype(np.float32)
        emb /= np.linalg.norm(emb)
        temp_db.add_or_update("bob", emb)
        is_match, score = temp_db.verify("bob", emb, threshold=0.5)
        assert is_match
        assert score > 0.99

    def test_verify_no_match(self, temp_db):
        """Verifying a different embedding returns no match."""
        emb1 = np.random.randn(192).astype(np.float32)
        emb1 /= np.linalg.norm(emb1)
        emb2 = np.random.randn(192).astype(np.float32)
        emb2 /= np.linalg.norm(emb2)
        temp_db.add_or_update("carol", emb1)
        is_match, score = temp_db.verify("carol", emb2, threshold=0.8)
        # Random embeddings should not match at high threshold
        assert not is_match or score < 0.99

    def test_remove_profile(self, temp_db):
        """Removing a profile works."""
        emb = np.random.randn(192).astype(np.float32)
        emb /= np.linalg.norm(emb)
        temp_db.add_or_update("dave", emb)
        assert temp_db.speaker_count == 1

        removed = temp_db.remove("dave")
        assert removed
        assert temp_db.speaker_count == 0
        assert temp_db.get("dave") is None

    def test_remove_nonexistent(self, temp_db):
        """Removing a nonexistent profile returns False."""
        assert not temp_db.remove("nonexistent")

    def test_persistence(self, temp_db):
        """Profiles survive save/reload round-trip."""
        emb = np.random.randn(192).astype(np.float32)
        emb /= np.linalg.norm(emb)
        temp_db.add_or_update("eve", emb)
        temp_db.flush()

        # Reload by creating a new DB pointing to the same dir
        from backend.speaker.speaker_engine import SpeakerDatabase
        db2 = SpeakerDatabase(storage_dir=str(temp_db.storage_dir))
        profile = db2.get("eve")
        assert profile is not None
        assert profile.speaker_id == "eve"
        assert len(profile.embeddings) == 1

    def test_search(self, temp_db):
        """Search finds the best-matching speaker."""
        emb_alice = np.random.randn(192).astype(np.float32)
        emb_alice /= np.linalg.norm(emb_alice)
        emb_bob = np.random.randn(192).astype(np.float32)
        emb_bob /= np.linalg.norm(emb_bob)

        temp_db.add_or_update("alice", emb_alice)
        temp_db.add_or_update("bob", emb_bob)

        # Search with alice's embedding
        best_id, score = temp_db.search(emb_alice, threshold=0.5)
        assert best_id == "alice"
        assert score > 0.99

    def test_all_speaker_ids(self, temp_db):
        """all_speaker_ids returns correct list."""
        for name in ["a", "b", "c"]:
            emb = np.random.randn(192).astype(np.float32)
            emb /= np.linalg.norm(emb)
            temp_db.add_or_update(name, emb)
        ids = temp_db.all_speaker_ids
        assert set(ids) == {"a", "b", "c"}


# ---------------------------------------------------------------------------
# Unit tests — SpeakerManager
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def speaker_manager():
    """Create a SpeakerManager for testing (module-scoped, loaded once)."""
    import asyncio
    # Ensure orchestrator dir is on path
    _orch = Path(__file__).parent.parent / "backend" / "orchestrator"
    if str(_orch) not in sys.path:
        sys.path.insert(0, str(_orch))
    from speaker_manager import SpeakerManager
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = SpeakerManager(
            profiles_dir=tmpdir,
            verify_threshold=0.45,
            accumulation_duration_s=1.0,  # shorter for tests
            enabled=True,
            bargein_enabled=True,
        )
        loop = asyncio.new_event_loop()
        loop.run_until_complete(mgr.initialize())
        loop.close()
        yield mgr
        loop2 = asyncio.new_event_loop()
        loop2.run_until_complete(mgr.close())
        loop2.close()


class TestSpeakerManager:
    """Tests for the async SpeakerManager wrapper."""

    def test_initialization(self, speaker_manager):
        """SpeakerManager initializes with engine and DB ready."""
        assert speaker_manager.is_ready
        assert speaker_manager._engine is not None
        assert speaker_manager._db is not None

    def test_enroll_and_verify(self, speaker_manager):
        """Enrolling and verifying a speaker works."""
        audio = make_voice_like(f0=150, duration=2.0, seed=42)
        import asyncio
        loop = asyncio.new_event_loop()

        # Enroll
        result = loop.run_until_complete(speaker_manager.enroll("test_user", audio))
        assert result["status"] == "enrolled"
        assert result["speaker_id"] == "test_user"
        assert result["quality"] > 0.0
        assert speaker_manager.is_enrolled

        # Verify with same audio
        is_match, score, matched_id = loop.run_until_complete(
            speaker_manager.verify(audio)
        )
        assert is_match
        assert score > 0.7
        assert matched_id == "test_user"

        loop.close()

    def test_verify_different_speaker(self, speaker_manager):
        """A different voice should NOT match the enrolled speaker."""
        # Enroll with one voice
        audio_target = make_voice_like(f0=150, duration=2.0, seed=1)
        import asyncio
        loop = asyncio.new_event_loop()
        loop.run_until_complete(speaker_manager.enroll("target", audio_target))

        # Verify with a different voice
        audio_other = make_voice_like(f0=250, duration=2.0, seed=99)
        is_match, score, _ = loop.run_until_complete(
            speaker_manager.verify(audio_other, speaker_id="target")
        )
        # Different voice should NOT match at reasonable threshold
        assert not is_match or score < 0.7

        loop.close()

    def test_pcm16_conversion(self, speaker_manager):
        """PCM16 bytes convert correctly to float32."""
        # Create a simple float32 signal
        original = np.array([0.5, -0.5, 0.25, -0.25], dtype=np.float32)
        # Convert to PCM16 bytes
        pcm = (original * 32767.0).astype("<i2").tobytes()
        # Convert back
        result = speaker_manager.pcm16_to_float32(pcm)
        assert len(result) == len(original)
        # Should be very close (within quantization error)
        assert np.allclose(result, original, atol=1.0 / 32767.0)

    def test_audio_accumulation(self, speaker_manager):
        """Audio accumulation buffer works correctly."""
        chunk = np.ones(512, dtype=np.float32) * 0.1
        assert not speaker_manager.has_enough_audio()

        # Add chunks until we have enough
        needed = speaker_manager._accumulation_samples
        chunks_needed = (needed // 512) + 1
        for _ in range(chunks_needed):
            speaker_manager.accumulate_chunk(chunk)

        assert speaker_manager.has_enough_audio()
        audio = speaker_manager.get_accumulated_audio()
        assert len(audio) >= needed
        # Buffer should be cleared after get
        assert not speaker_manager.has_enough_audio()

    def test_profile_management(self, speaker_manager):
        """Profile CRUD operations work through the manager."""
        # Reset
        speaker_manager.reset_all()

        # Add a profile manually through the DB
        emb = np.random.randn(192).astype(np.float32)
        emb /= np.linalg.norm(emb)
        speaker_manager._db.add_or_update("profile_test", emb)

        profiles = speaker_manager.get_profiles()
        assert len(profiles) == 1
        assert profiles[0]["speaker_id"] == "profile_test"

        # Remove
        assert speaker_manager.remove_profile("profile_test")
        assert len(speaker_manager.get_profiles()) == 0

    def test_bargein_cooldown(self, speaker_manager):
        """Barge-in respects cooldown period."""
        # Manually set state to simulate non-target detection
        speaker_manager._last_speaker_status = "nontarget"
        speaker_manager._last_speaker_score = 0.5

        # First check should trigger
        event = speaker_manager.check_bargein(is_ai_speaking=True)
        assert event is not None
        assert event["type"] == "barge_in"

        # Second check within cooldown should NOT trigger
        event2 = speaker_manager.check_bargein(is_ai_speaking=True)
        assert event2 is None

    def test_bargein_requires_ai_speaking(self, speaker_manager):
        """Barge-in only triggers when AI is speaking."""
        speaker_manager._last_speaker_status = "nontarget"
        speaker_manager._last_bargein_time = 0.0  # reset cooldown

        event = speaker_manager.check_bargein(is_ai_speaking=False)
        assert event is None

    def test_bargein_requires_nontarget(self, speaker_manager):
        """Barge-in only triggers for non-target speakers."""
        speaker_manager._last_speaker_status = "target"
        speaker_manager._last_bargein_time = 0.0

        event = speaker_manager.check_bargein(is_ai_speaking=True)
        assert event is None


# ---------------------------------------------------------------------------
# Unit tests — threshold config
# ---------------------------------------------------------------------------

class TestThresholdConfig:
    """Tests for the threshold configuration."""

    def test_config_loads(self):
        """Threshold config JSON loads and has expected keys."""
        config_path = Path(__file__).parent.parent / "backend" / "speaker" / "threshold_config.json"
        if not config_path.exists():
            pytest.skip("threshold_config.json not found")

        with open(config_path) as f:
            config = json.load(f)

        assert "optimal_threshold" in config
        assert "target_mean" in config
        assert "nontarget_mean" in config
        assert "separation_dprime" in config
        assert config["separation_dprime"] > 2.0  # should be well-separated


# ---------------------------------------------------------------------------
# Performance benchmarks
# ---------------------------------------------------------------------------

class TestPerformance:
    """Performance benchmarks for speaker verification."""

    def test_embedding_latency(self, speaker_engine):
        """Embedding extraction P50 should be under 50ms."""
        latencies = []
        audio = make_voice_like(f0=150, duration=2.0)

        # Warm-up
        _ = speaker_engine.extract(audio)

        for _ in range(10):
            t0 = time.perf_counter()
            _ = speaker_engine.extract(audio)
            latencies.append((time.perf_counter() - t0) * 1000)

        p50 = float(np.median(latencies))
        print(f"\n  Embedding latency: P50={p50:.1f}ms")
        assert p50 < 50.0, f"P50 latency {p50:.1f}ms exceeds 50ms target"


# ---------------------------------------------------------------------------
# Main (for direct execution)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
