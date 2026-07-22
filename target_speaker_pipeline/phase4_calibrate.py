#!/usr/bin/env python3
"""
Phase 4: Threshold Calibration

Calibrates speaker verification thresholds using enrollment and test
audio samples. Computes score distributions, DET curve metrics, and
recommends optimal threshold based on desired FAR/FRR trade-off.

Usage:
    # With audio files in a directory:
    python3 phase4_calibrate.py --enroll-dir audio/enrollment --test-dir audio/test

    # With synthetic data for pipeline validation:
    python3 phase4_calibrate.py --synthetic
"""

from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from lib.speaker_engine import SpeakerEmbeddingEngine, SpeakerDatabase


# ---------------------------------------------------------------------------
# Calibration data structures
# ---------------------------------------------------------------------------

@dataclass
class CalibrationResult:
    """Result of threshold calibration."""

    # Score distributions
    target_scores: list[float] = field(default_factory=list)
    nontarget_scores: list[float] = field(default_factory=list)

    # Computed metrics
    optimal_threshold: float = 0.5
    eer: float = 1.0
    eer_threshold: float = 0.5
    far_at_1percent_frr: float = 1.0
    frr_at_1percent_far: float = 1.0

    # Score statistics
    target_mean: float = 0.0
    target_std: float = 0.0
    nontarget_mean: float = 0.0
    nontarget_std: float = 0.0
    separation_dprime: float = 0.0  # d' = (μ₁ - μ₂) / √((σ₁² + σ₂²)/2)

    # Recommended operating point
    recommended_threshold: float = 0.5
    recommended_far: float = 0.0
    recommended_frr: float = 0.0

    @property
    def num_target_trials(self) -> int:
        return len(self.target_scores)

    @property
    def num_nontarget_trials(self) -> int:
        return len(self.nontarget_scores)

    @property
    def is_reliable(self) -> bool:
        """Calibration is reliable if separation d' > 1.0."""
        return self.separation_dprime >= 1.0

    def to_dict(self) -> dict:
        return {
            "optimal_threshold": self.optimal_threshold,
            "eer": self.eer,
            "eer_threshold": self.eer_threshold,
            "far_at_1percent_frr": self.far_at_1percent_frr,
            "frr_at_1percent_far": self.frr_at_1percent_far,
            "target_mean": self.target_mean,
            "target_std": self.target_std,
            "nontarget_mean": self.nontarget_std,
            "nontarget_std": self.nontarget_std,
            "separation_dprime": self.separation_dprime,
            "recommended_threshold": self.recommended_threshold,
            "recommended_far": self.recommended_far,
            "recommended_frr": self.recommended_frr,
            "num_target_trials": self.num_target_trials,
            "num_nontarget_trials": self.num_nontarget_trials,
        }

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


# ---------------------------------------------------------------------------
# Calibration engine
# ---------------------------------------------------------------------------

class ThresholdCalibrator:
    """
    Computes speaker verification thresholds from trial scores.

    Methodology:
    1. Collect target scores (same speaker, different utterances)
    2. Collect non-target scores (different speakers)
    3. Sweep thresholds, compute FAR and FRR at each point
    4. Find EER (where FAR ≈ FRR)
    5. Recommend operating threshold based on application requirements
    """

    def __init__(self, model_path: str = "models/nemo_en_titanet_small.onnx"):
        self.engine = SpeakerEmbeddingEngine(model_path)

    def calibrate_from_embeddings(
        self,
        target_pairs: list[tuple[np.ndarray, np.ndarray]],
        nontarget_pairs: list[tuple[np.ndarray, np.ndarray]],
    ) -> CalibrationResult:
        """
        Calibrate from pre-computed embedding pairs.

        Args:
            target_pairs: (enrollment_emb, test_emb) for SAME speaker.
            nontarget_pairs: (enrollment_emb, test_emb) for DIFFERENT speakers.
        """
        result = CalibrationResult()

        # Compute scores
        for enroll_emb, test_emb in target_pairs:
            score = self._cosine_similarity(enroll_emb, test_emb)
            result.target_scores.append(score)

        for enroll_emb, test_emb in nontarget_pairs:
            score = self._cosine_similarity(enroll_emb, test_emb)
            result.nontarget_scores.append(score)

        self._compute_metrics(result)
        return result

    def calibrate_from_audio(
        self,
        enrollment_audio: dict[str, list[np.ndarray]],  # speaker_id -> [audio_samples]
        test_same_speaker: dict[str, list[np.ndarray]],  # speaker_id -> [test samples]
        test_different_speakers: list[tuple[str, np.ndarray]],  # [(enrolled_speaker_id, other_audio)]
    ) -> CalibrationResult:
        """
        Calibrate from raw audio samples.

        Args:
            enrollment_audio: {speaker_id: [enrollment utterances]}
            test_same_speaker: {speaker_id: [test utterances from same speaker]}
            test_different_speakers: [(enrolled_id, other_speaker_audio)]
        """
        # Build enrollment centroids
        centroids: dict[str, np.ndarray] = {}
        for sid, utterances in enrollment_audio.items():
            embs = self.engine.extract_batch(utterances)
            centroid = np.mean(embs, axis=0).astype(np.float32)
            norm = float(np.linalg.norm(centroid))
            if norm > 1e-8:
                centroid /= norm
            centroids[sid] = centroid

        result = CalibrationResult()

        # Target scores
        for sid, test_utts in test_same_speaker.items():
            if sid not in centroids:
                continue
            centroid = centroids[sid]
            test_embs = self.engine.extract_batch(test_utts)
            for emb in test_embs:
                score = float(np.dot(centroid, emb))
                result.target_scores.append(score)

        # Non-target scores
        for enrolled_sid, other_audio in test_different_speakers:
            if enrolled_sid not in centroids:
                continue
            centroid = centroids[enrolled_sid]
            other_emb = self.engine.extract(other_audio)
            score = float(np.dot(centroid, other_emb))
            result.nontarget_scores.append(score)

        self._compute_metrics(result)
        return result

    def calibrate_synthetic(self) -> CalibrationResult:
        """
        Generate synthetic calibration data for pipeline validation.
        Uses the engine itself to create diverse test embeddings.
        """
        from lib.audio_capture import SimulatedAudioCapture

        sim = SimulatedAudioCapture()
        fs = 16000

        # Generate enrollment and test audio for multiple "speakers"
        speakers = {}
        sim_profiles = [
            ("alice", {"f0": 220, "f1": 800, "f2": 2400, "f3": 3500, "brightness": 1.3}),
            ("bob", {"f0": 110, "f1": 500, "f2": 1500, "f3": 2600, "brightness": 0.6}),
            ("carol", {"f0": 195, "f1": 720, "f2": 2100, "f3": 3100, "brightness": 1.0}),
            ("dave", {"f0": 90, "f1": 450, "f2": 1300, "f3": 2400, "brightness": 0.5}),
            ("eve", {"f0": 250, "f1": 880, "f2": 2600, "f3": 3700, "brightness": 1.4}),
        ]

        enrollment_audio: dict[str, list[np.ndarray]] = {}
        test_same: dict[str, list[np.ndarray]] = {}
        test_diff: list[tuple[str, np.ndarray]] = []

        for sid, profile in sim_profiles:
            sim.target_timbre = sim._make_voice(**profile)
            sim.target_speaking = True
            sim.nontarget_speaking = False

            # Generate 3 enrollment utterances (2s each)
            enroll_utts = []
            for _ in range(3):
                audio = np.concatenate([
                    sim.generate_frame()
                    for _ in range(int(2.0 * fs / sim.blocksize))
                ])
                enroll_utts.append(audio)
            enrollment_audio[sid] = enroll_utts

            # Generate 2 test utterances (different from enrollment)
            test_utts = []
            for _ in range(2):
                audio = np.concatenate([
                    sim.generate_frame()
                    for _ in range(int(2.0 * fs / sim.blocksize))
                ])
                test_utts.append(audio)
            test_same[sid] = test_utts

        # Non-target pairs (each speaker vs each other speaker's enrollment)
        for i, (sid_a, _) in enumerate(sim_profiles):
            for j, (sid_b, profile_b) in enumerate(sim_profiles):
                if i == j:
                    continue
                sim.target_timbre = sim._make_voice(**profile_b)
                audio = np.concatenate([
                    sim.generate_frame()
                    for _ in range(int(2.0 * fs / sim.blocksize))
                ])
                test_diff.append((sid_a, audio))

        return self.calibrate_from_audio(enrollment_audio, test_same, test_diff)

    # ------------------------------------------------------------------
    # Metrics computation
    # ------------------------------------------------------------------

    def _compute_metrics(self, result: CalibrationResult) -> None:
        """Compute all calibration metrics from score distributions."""
        target = np.array(result.target_scores)
        nontarget = np.array(result.nontarget_scores)

        if len(target) == 0 or len(nontarget) == 0:
            print("WARNING: Empty score distributions — cannot compute metrics")
            return

        # Statistics
        result.target_mean = float(np.mean(target))
        result.target_std = float(np.std(target))
        result.nontarget_mean = float(np.mean(nontarget))
        result.nontarget_std = float(np.std(nontarget))

        # d-prime (separation)
        pooled_std = np.sqrt((result.target_std ** 2 + result.nontarget_std ** 2) / 2)
        if pooled_std > 1e-8:
            result.separation_dprime = (
                result.target_mean - result.nontarget_mean
            ) / pooled_std
        else:
            result.separation_dprime = 0.0

        # Sweep thresholds
        thresholds = np.linspace(-1.0, 1.0, 1001)
        best_threshold = 0.5
        best_eer = 1.0

        for thresh in thresholds:
            far = np.mean(nontarget >= thresh)
            frr = np.mean(target < thresh)

            if abs(far - frr) < abs(
                np.mean(nontarget >= best_threshold)
                - np.mean(target < best_threshold)
            ):
                best_eer = max(far, frr)
                best_threshold = thresh

        result.eer = float(best_eer)
        result.eer_threshold = float(best_threshold)

        # FAR at various FRR operating points
        result.far_at_1percent_frr = self._far_at_frr(thresholds, target, nontarget, 0.01)
        result.frr_at_1percent_far = self._frr_at_far(thresholds, target, nontarget, 0.01)

        # Recommended operating point: minimize minDCF-like cost
        # Weigh false accepts 10x more than false rejects (security-sensitive)
        best_cost = float("inf")
        best_op_thresh = best_threshold
        best_far = 0.0
        best_frr = 0.0

        for thresh in thresholds:
            far = float(np.mean(nontarget >= thresh))
            frr = float(np.mean(target < thresh))
            cost = 10.0 * far + 1.0 * frr  # FA cost = 10x FR cost
            if cost < best_cost:
                best_cost = cost
                best_op_thresh = float(thresh)
                best_far = far
                best_frr = frr

        result.recommended_threshold = best_op_thresh
        result.recommended_far = best_far
        result.recommended_frr = best_frr
        result.optimal_threshold = result.recommended_threshold

    @staticmethod
    def _far_at_frr(
        thresholds: np.ndarray,
        target: np.ndarray,
        nontarget: np.ndarray,
        target_frr: float,
    ) -> float:
        """Find the FAR at a given FRR level."""
        for thresh in thresholds:
            frr = float(np.mean(target < thresh))
            if frr >= target_frr:
                return float(np.mean(nontarget >= thresh))
        return 1.0

    @staticmethod
    def _frr_at_far(
        thresholds: np.ndarray,
        target: np.ndarray,
        nontarget: np.ndarray,
        target_far: float,
    ) -> float:
        """Find the FRR at a given FAR level."""
        for thresh in thresholds[::-1]:  # high to low
            far = float(np.mean(nontarget >= thresh))
            if far <= target_far:
                return float(np.mean(target < thresh))
        return 1.0

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        a_norm = a / (float(np.linalg.norm(a)) + 1e-8)
        b_norm = b / (float(np.linalg.norm(b)) + 1e-8)
        return float(np.dot(a_norm, b_norm))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 4: Threshold Calibration")
    parser.add_argument(
        "--model",
        default="models/nemo_en_titanet_small.onnx",
        help="Speaker model path",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic data for calibration",
    )
    parser.add_argument(
        "--output",
        default="threshold_config.json",
        help="Output JSON file for calibration results",
    )
    args = parser.parse_args()

    calibrator = ThresholdCalibrator(args.model)

    if args.synthetic:
        print("Generating synthetic calibration data...")
        result = calibrator.calibrate_synthetic()
    else:
        print("ERROR: No calibration data source specified.")
        print("Use --synthetic for synthetic data, or provide --enroll-dir and --test-dir")
        print("for real audio files.")
        sys.exit(1)

    # Print results
    print("\n" + "=" * 60)
    print("PHASE 4: THRESHOLD CALIBRATION RESULTS")
    print("=" * 60)
    print(f"Trials: {result.num_target_trials} target, {result.num_nontarget_trials} nontarget")
    print()
    print(f"Score distributions:")
    print(f"  Target:     μ={result.target_mean:.4f}  σ={result.target_std:.4f}")
    print(f"  Non-target: μ={result.nontarget_mean:.4f}  σ={result.nontarget_std:.4f}")
    print(f"  Separation d'={result.separation_dprime:.2f}")
    print()
    print(f"EER: {result.eer:.3f} at threshold={result.eer_threshold:.3f}")
    print(f"FAR @ 1% FRR: {result.far_at_1percent_frr:.4f}")
    print(f"FRR @ 1% FAR: {result.frr_at_1percent_far:.4f}")
    print()
    print(f"Recommended operating point:")
    print(f"  Threshold: {result.recommended_threshold:.3f}")
    print(f"  FAR:       {result.recommended_far:.3f}")
    print(f"  FRR:       {result.recommended_frr:.3f}")
    print()

    if result.is_reliable:
        print("✓ Calibration RELIABLE — d' > 1.0, speakers are well-separated")
    else:
        print("⚠ Calibration MARGINAL — d' < 1.0, check data quality or model")

    # Save
    result.save(args.output)
    print(f"\nResults saved to: {args.output}")

    return result


if __name__ == "__main__":
    main()
