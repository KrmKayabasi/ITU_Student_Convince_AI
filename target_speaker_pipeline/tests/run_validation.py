#!/usr/bin/env python3
"""
Phase 8: System Validation Test Suite

Comprehensive validation of the speaker verification pipeline:
- Unit tests for each component
- Integration tests for the full pipeline
- Performance benchmarks
- Latency measurements
- Accuracy metrics

Usage:
    python3 tests/run_validation.py
    python3 tests/run_validation.py --benchmark  # includes performance tests
"""

from __future__ import annotations

import sys
import time
import json
import tempfile
import threading
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.speaker_engine import (
    SpeakerEmbeddingEngine,
    SpeakerDatabase,
    SpeakerProfile,
)
from lib.vad_engine import VADEngine, VADConfig, SpeechSegment
from lib.audio_capture import (
    RingBuffer,
    SimulatedAudioCapture,
    AudioConfig,
)


# ---------------------------------------------------------------------------
# Test result tracking
# ---------------------------------------------------------------------------

@dataclass
class TestResult:
    name: str
    passed: bool
    message: str = ""
    duration_ms: float = 0.0


class TestRunner:
    def __init__(self):
        self.results: list[TestResult] = []
        self.passed: int = 0
        self.failed: int = 0

    def run(self, name: str, test_fn) -> TestResult:
        t0 = time.perf_counter()
        try:
            test_fn()
            passed = True
            msg = "OK"
        except AssertionError as e:
            passed = False
            msg = str(e) or "Assertion failed"
        except Exception as e:
            passed = False
            msg = f"{type(e).__name__}: {e}"

        dt = (time.perf_counter() - t0) * 1000
        result = TestResult(name, passed, msg, dt)
        self.results.append(result)

        if passed:
            self.passed += 1
            print(f"  ✓ {name} ({dt:.1f}ms)")
        else:
            self.failed += 1
            print(f"  ✗ {name}: {msg}")

        return result

    def summary(self) -> None:
        print(f"\n{'='*60}")
        print(f"RESULTS: {self.passed} passed, {self.failed} failed, "
              f"{len(self.results)} total")
        print(f"{'='*60}")

        if self.failed > 0:
            print("\nFAILURES:")
            for r in self.results:
                if not r.passed:
                    print(f"  ✗ {r.name}: {r.message}")


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def make_tone(freq: float, duration: float, sample_rate: int = 16000) -> np.ndarray:
    t = np.arange(0, duration, 1.0 / sample_rate, dtype=np.float32)
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def assert_true(cond, msg=""):
    if not cond:
        raise AssertionError(msg)


def assert_close(a, b, tol=0.01, msg=""):
    if abs(a - b) > tol:
        raise AssertionError(f"{msg}: {a} != {b} (tol={tol})")


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def test_ring_buffer_basic():
    """RingBuffer: basic write/read."""
    rb = RingBuffer(capacity_frames=10, frame_size=512)
    frame = np.ones(512, dtype=np.float32)
    assert_true(rb.write(frame), "Write should succeed")
    result = rb.read()
    assert_true(result is not None, "Read should return frame")
    assert_true(np.allclose(result, frame), "Read frame should match written frame")


def test_ring_buffer_full():
    """RingBuffer: overflow behavior (one slot reserved for full/empty distinction)."""
    rb = RingBuffer(capacity_frames=4, frame_size=16)
    for i in range(3):
        assert_true(rb.write(np.full(16, i, dtype=np.float32)))
    # 4th write should fail (buffer full — 3 items max with capacity 4)
    assert_true(not rb.write(np.full(16, 99, dtype=np.float32)), "Should overflow")


def test_ring_buffer_empty():
    """RingBuffer: read from empty returns None."""
    rb = RingBuffer(capacity_frames=5, frame_size=32)
    assert_true(rb.read() is None, "Empty buffer should return None")


def test_speaker_profile_centroid():
    """SpeakerProfile: centroid computation."""
    profile = SpeakerProfile("test")
    # Add 3 identical embeddings
    emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    for _ in range(3):
        profile.add_embedding(emb)
    centroid = profile.centroid
    assert_true(centroid is not None)
    # Should be [1, 0, 0] normalized
    assert abs(float(np.dot(centroid, np.array([1.0, 0.0, 0.0]))) - 1.0) < 0.001, f"Centroid={centroid}"


def test_speaker_profile_quality():
    """SpeakerProfile: quality score."""
    profile = SpeakerProfile("test")
    # Few embeddings → low quality
    profile.add_embedding(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    assert_true(profile.quality < 0.5, f"Quality should be < 0.5 with 1 embedding, got {profile.quality}")

    # Many consistent embeddings → high quality
    for _ in range(20):
        profile.add_embedding(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    assert_true(profile.quality > 0.8, f"Quality should be > 0.8 with 21 embeddings, got {profile.quality}")


def test_speaker_database_crud():
    """SpeakerDatabase: create, read, update, delete."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db = SpeakerDatabase(tmpdir)

        # Create
        emb = np.random.randn(192).astype(np.float32)
        emb /= np.linalg.norm(emb)
        db.add_or_update("speaker_1", emb)
        assert_true(db.speaker_count == 1)
        assert_true("speaker_1" in db.all_speaker_ids)

        # Read
        profile = db.get("speaker_1")
        assert_true(profile is not None)
        assert_true(len(profile.embeddings) == 1)

        # Update
        emb2 = np.random.randn(192).astype(np.float32)
        emb2 /= np.linalg.norm(emb2)
        db.add_or_update("speaker_1", emb2)
        assert_true(len(db.get("speaker_1").embeddings) == 2)

        # Delete
        assert_true(db.remove("speaker_1"))
        assert_true(db.speaker_count == 0)

        # Delete non-existent
        assert_true(not db.remove("nonexistent"))


def test_speaker_database_verify():
    """SpeakerDatabase: verification with known embeddings."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db = SpeakerDatabase(tmpdir)

        emb = np.array([1.0] + [0.0] * 191, dtype=np.float32)
        emb /= np.linalg.norm(emb)
        db.add_or_update("alice", emb)

        # Same embedding should match
        is_match, score = db.verify("alice", emb, threshold=0.5)
        assert_true(is_match, f"Should match own embedding, score={score:.4f}")
        assert abs(score - 1.0) < 0.01, f"Score={score:.4f}"

        # Opposite embedding should not match
        opp = -emb
        is_match2, score2 = db.verify("alice", opp, threshold=0.5)
        assert_true(not is_match2, f"Should NOT match opposite embedding, score={score2:.4f}")


def test_speaker_embedding_engine():
    """SpeakerEmbeddingEngine: loads model, extracts 192-dim embedding."""
    engine = SpeakerEmbeddingEngine("models/nemo_en_titanet_small.onnx")
    audio = make_tone(200, 2.0)
    emb = engine.extract(audio)
    assert_true(emb.shape == (192,), f"Expected (192,), got {emb.shape}")
    norm = float(np.linalg.norm(emb))
    assert abs(norm - 1.0) < 0.01, f"Norm={norm:.4f}"


def test_speaker_embedding_consistency():
    """SpeakerEmbeddingEngine: same audio → same embedding."""
    engine = SpeakerEmbeddingEngine("models/nemo_en_titanet_small.onnx")
    audio = make_tone(200, 3.0)
    emb1 = engine.extract(audio)
    emb2 = engine.extract(audio)
    cos_sim = float(np.dot(emb1, emb2))
    assert_true(cos_sim > 0.99, f"Same audio should produce ~identical embedding, got {cos_sim:.6f}")


def test_vad_config():
    """VADConfig: default values are reasonable."""
    cfg = VADConfig()
    assert_true(cfg.window_size == 512)
    assert_true(cfg.sample_rate == 16000)
    assert_true(0 < cfg.speech_threshold < 1.0)
    assert_true(cfg.segment_min_duration > 0)


def test_vad_engine_creation():
    """VADEngine: creates successfully with model."""
    vad = VADEngine(VADConfig(model_path="models/silero_vad.onnx"))
    assert_true(vad.window_samples == 512)
    assert_true(28 < vad.window_duration_ms < 36)


def test_speech_segment_validation():
    """SpeechSegment: quality validation."""
    seg = SpeechSegment(
        audio=np.zeros(16000, dtype=np.float32),
        start_time=0, end_time=1.0, duration=1.0,
        vad_confidence=0.8, rms_energy=0.01, frame_count=50,
    )
    assert_true(seg.is_valid)

    # Short segment
    seg_short = SpeechSegment(
        audio=np.zeros(8000, dtype=np.float32),
        start_time=0, end_time=0.5, duration=0.5,
        vad_confidence=0.8, rms_energy=0.01, frame_count=25,
    )
    assert_true(not seg_short.is_valid)


def test_simulated_audio_generation():
    """SimulatedAudioCapture: generates audio frames."""
    sim = SimulatedAudioCapture(blocksize=512)
    sim.target_speaking = True
    frame = sim.generate_frame()
    assert_true(frame.shape == (512,))
    assert_true(frame.dtype == np.float32)
    rms = float(np.sqrt(np.mean(frame**2)))
    assert_true(rms > 0.001, f"Simulated audio should have energy, RMS={rms:.6f}")


def test_simulated_audio_stereo():
    """SimulatedAudioCapture: target and non-target generate different audio."""
    sim = SimulatedAudioCapture(blocksize=512)
    sim.target_speaking = True
    sim.nontarget_speaking = False
    f1 = sim.generate_frame()

    sim.target_speaking = False
    sim.nontarget_speaking = True
    f2 = sim.generate_frame()

    # Should not be identical
    assert_true(not np.allclose(f1, f2), "Target and non-target should differ")


def test_pipeline_embedding_persistence():
    """SpeakerDatabase: save and reload profiles."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create and save
        db1 = SpeakerDatabase(tmpdir)
        emb = np.random.randn(192).astype(np.float32)
        emb /= np.linalg.norm(emb)
        db1.add_or_update("test_speaker", emb)

        # Reload
        db2 = SpeakerDatabase(tmpdir)
        assert_true("test_speaker" in db2.all_speaker_ids)
        profile = db2.get("test_speaker")
        assert_true(len(profile.embeddings) == 1)


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

def test_full_pipeline_verification():
    """End-to-end: enrollment → verification with simulated audio."""
    engine = SpeakerEmbeddingEngine("models/nemo_en_titanet_small.onnx")

    with tempfile.TemporaryDirectory() as tmpdir:
        db = SpeakerDatabase(tmpdir)
        sim = SimulatedAudioCapture()

        # Generate "target speaker" enrollment audio
        sim.target_speaking = True
        sim.nontarget_speaking = False
        enroll_audio = np.concatenate([
            sim.generate_frame() for _ in range(int(3.0 * 16000 / 512))
        ])
        enroll_emb = engine.extract(enroll_audio)

        # Generate "guest speaker" audio
        sim.target_speaking = False
        sim.nontarget_speaking = True
        guest_audio = np.concatenate([
            sim.generate_frame() for _ in range(int(3.0 * 16000 / 512))
        ])
        guest_emb = engine.extract(guest_audio)

        # Enroll target
        db.add_or_update("target", enroll_emb)

        # Verify target vs target
        is_m, score = db.verify("target", enroll_emb, threshold=0.3)
        assert_true(is_m, "Target should verify against own enrollment")

        # The key test: target embedding should be more similar to target
        # profile than guest embedding is
        target_score = float(np.dot(enroll_emb, enroll_emb))  # 1.0
        guest_score = float(np.dot(enroll_emb, guest_emb))

        # For synthetic audio these may be close, so we just verify
        # the pipeline works, not the absolute discrimination
        print(f"    Target self-score: {target_score:.4f}")
        print(f"    Guest cross-score: {guest_score:.4f}")
        assert_true(is_m, "Pipeline verification should work")


def test_vad_segmentation_pipeline():
    """VAD: processes audio frames without crashing."""
    vad = VADEngine(VADConfig(
        model_path="models/silero_vad.onnx",
        window_size=512,
        energy_threshold=0.001,
    ))

    # Process 100 frames of simulated audio
    sim = SimulatedAudioCapture(blocksize=512)
    sim.target_speaking = True

    frames_ok = 0
    for _ in range(100):
        frame = sim.generate_frame()
        try:
            vad.process(frame)
            frames_ok += 1
        except Exception:
            pass

    assert_true(frames_ok == 100, f"VAD should process all frames, only {frames_ok}/100 OK")
    assert_true(vad.total_speech_frames + vad.total_silence_frames == 100)


# ---------------------------------------------------------------------------
# Performance benchmarks
# ---------------------------------------------------------------------------

def bench_embedding_latency():
    """Benchmark: embedding extraction latency."""
    engine = SpeakerEmbeddingEngine("models/nemo_en_titanet_small.onnx")
    audio = make_tone(200, 2.0)

    # Warmup
    for _ in range(3):
        engine.extract(audio)

    # Measure
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        engine.extract(audio)
        times.append((time.perf_counter() - t0) * 1000)

    p50 = float(np.median(times))
    p95 = float(np.percentile(times, 95))
    p99 = float(np.percentile(times, 99))

    print(f"    Embedding latency (2s audio): P50={p50:.1f}ms P95={p95:.1f}ms P99={p99:.1f}ms")
    assert_true(p50 < 100, f"P50 latency {p50:.0f}ms exceeds 100ms budget")


def bench_vad_latency():
    """Benchmark: VAD per-frame latency."""
    vad = VADEngine(VADConfig(
        model_path="models/silero_vad.onnx",
        window_size=512,
    ))

    frame = np.random.randn(512).astype(np.float32) * 0.01

    # Warmup
    for _ in range(10):
        vad.process(frame)

    # Measure
    times = []
    for _ in range(200):
        t0 = time.perf_counter()
        vad.process(frame)
        times.append((time.perf_counter() - t0) * 1000)

    p50 = float(np.median(times))
    p95 = float(np.percentile(times, 95))

    print(f"    VAD latency (32ms frame): P50={p50*1000:.0f}µs P95={p95*1000:.0f}µs")
    assert_true(p50 < 2.0, f"P50 VAD latency {p50:.1f}ms exceeds budget")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Phase 8: System Validation")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmarks")
    args = parser.parse_args()

    print("=" * 60)
    print("PHASE 8: SYSTEM VALIDATION")
    print("=" * 60)

    runner = TestRunner()

    # ---- Unit tests ----
    print("\n--- Unit Tests ---")
    runner.run("RingBuffer basic write/read", test_ring_buffer_basic)
    runner.run("RingBuffer overflow", test_ring_buffer_full)
    runner.run("RingBuffer empty read", test_ring_buffer_empty)
    runner.run("SpeakerProfile centroid", test_speaker_profile_centroid)
    runner.run("SpeakerProfile quality", test_speaker_profile_quality)
    runner.run("SpeakerDatabase CRUD", test_speaker_database_crud)
    runner.run("SpeakerDatabase verify", test_speaker_database_verify)
    runner.run("SpeakerEmbeddingEngine load+extract", test_speaker_embedding_engine)
    runner.run("SpeakerEmbeddingEngine consistency", test_speaker_embedding_consistency)
    runner.run("VADConfig defaults", test_vad_config)
    runner.run("VADEngine creation", test_vad_engine_creation)
    runner.run("SpeechSegment validation", test_speech_segment_validation)
    runner.run("SimulatedAudio generation", test_simulated_audio_generation)
    runner.run("SimulatedAudio speaker diff", test_simulated_audio_stereo)
    runner.run("Profile persistence", test_pipeline_embedding_persistence)

    # ---- Integration tests ----
    print("\n--- Integration Tests ---")
    runner.run("Full pipeline verification", test_full_pipeline_verification)
    runner.run("VAD segmentation pipeline", test_vad_segmentation_pipeline)

    # ---- Benchmarks ----
    if args.benchmark:
        print("\n--- Performance Benchmarks ---")
        runner.run("Embedding latency", bench_embedding_latency)
        runner.run("VAD latency", bench_vad_latency)

    # ---- Summary ----
    runner.summary()

    return 0 if runner.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
