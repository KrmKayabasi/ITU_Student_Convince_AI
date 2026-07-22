#!/usr/bin/env python3
"""
Phase 1: Offline Speaker Verification

Validates that the speaker embedding model (TitaNet via sherpa-onnx) can
reliably differentiate between speakers using pre-recorded or synthetic audio.

Usage:
    python3 phase1_offline_verify.py [--model models/nemo_en_titanet_small.onnx]
"""

from __future__ import annotations

import sys
import os
import argparse
import time
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from lib.speaker_engine import SpeakerEmbeddingEngine, SpeakerDatabase


# ---------------------------------------------------------------------------
# Test case definition
# ---------------------------------------------------------------------------

@dataclass
class TestCase:
    """A single speaker verification test case."""

    name: str
    speaker_id: str        # who is actually speaking
    audio: np.ndarray      # the audio waveform
    sample_rate: int = 16000
    expected_match: bool | None = None  # None = enrollment sample (no check)
    expected_speaker: str | None = None  # for identification tests


# ---------------------------------------------------------------------------
# Test audio generation
# ---------------------------------------------------------------------------

def generate_test_audio(
    sample_rate: int = 16000,
    duration_s: float = 3.0,  # longer = more reliable embeddings
    seed: int = 42,
) -> dict[str, list[np.ndarray]]:
    """
    Generate synthetic test audio for multiple "speakers" with distinct
    vocal tract characteristics.

    KEY INSIGHT: A speaker embedding model needs CONSISTENT vocal tract
    characteristics across utterances, but the PHONETIC content should vary.
    Real speaker identity comes from:
    1. Vocal tract length (affects ALL formant positions proportionally)
    2. Glottal pulse shape (harmonic richness and spectral tilt)
    3. Prosodic patterns (F0 range, speaking rate)

    Each "speaker" here has FIXED vocal tract parameters. Different
    "utterances" vary the fundamental frequency trajectory and formant
    emphasis patterns (simulating different phonemes) while keeping
    the underlying vocal tract identity constant.
    """
    rng = np.random.RandomState(seed)
    t = np.arange(0, duration_s, 1.0 / sample_rate, dtype=np.float64)
    n_samples = len(t)

    speakers = {}

    # Each speaker profile: (base_f0, vocal_tract_scale, spectral_tilt, hnr, glottal_shape)
    # - base_f0: fundamental frequency in Hz
    # - vocal_tract_scale: >1 = shorter tract (higher formants), <1 = longer tract
    # - spectral_tilt: negative = steeper rolloff (darker), positive = flatter (brighter)
    # - hnr: harmonics-to-noise ratio dB (higher = clearer voice)
    # - glottal_shape: >0 = pressed/tense, <0 = breathy/lax
    profiles = {
        "speaker_alice": (240.0, 1.15, -1.5, 28.0, 0.3),   # very high, very bright, clear
        "speaker_bob": (105.0, 0.82, -8.0, 15.0, -0.4),      # very low, very dark, breathy
        "speaker_carol": (195.0, 1.00, -3.5, 22.0, 0.0),     # mid-range female
        "speaker_dave": (85.0, 0.75, -10.0, 12.0, -0.3),      # extremely low, gravelly
    }

    # Default formant frequencies for a neutral vowel (schwa) in an average male
    # These are then scaled by vocal_tract_scale to create individual voices
    base_formants = np.array([600.0, 1700.0, 2600.0, 3500.0, 4500.0])

    for name, (f0_base, vt_scale, spectral_tilt, hnr, glottal_shape) in profiles.items():
        # SCALE FORMANTS by vocal tract length (shorter tract = all formants higher)
        formant_centers = base_formants * vt_scale

        utterances = []
        for utt_idx in range(3):
            audio = np.zeros(n_samples, dtype=np.float64)

            # --- FUNDAMENTAL FREQUENCY TRAJECTORY (natural pitch contour) ---
            # Each utterance has a different pitch contour = different "prosody"
            # but the same speaker identity
            f0_traj = np.zeros(n_samples, dtype=np.float64)

            # Use several control points for natural-sounding F0
            n_controls = 5 + utt_idx  # different number per utterance
            control_times = np.sort(rng.uniform(0, duration_s, n_controls))
            control_f0s = f0_base * rng.uniform(0.85, 1.15, n_controls)

            # Interpolate between control points
            for i in range(n_samples):
                ti = t[i]
                # Find bracketing control points
                for j in range(n_controls - 1):
                    if control_times[j] <= ti < control_times[j + 1]:
                        frac = (ti - control_times[j]) / (
                            control_times[j + 1] - control_times[j]
                        )
                        f0_traj[i] = (
                            control_f0s[j] * (1 - frac) + control_f0s[j + 1] * frac
                        )
                        break
                else:
                    if ti < control_times[0]:
                        f0_traj[i] = control_f0s[0]
                    else:
                        f0_traj[i] = control_f0s[-1]

            # Add slight jitter (cycle-to-cycle F0 variation — natural)
            f0_traj *= 1.0 + rng.normal(0, 0.005, n_samples)

            # --- GLOTTAL SOURCE ---
            # Compute phase incrementally for accurate harmonics
            phase = np.cumsum(2.0 * np.pi * f0_traj) / sample_rate
            phase += rng.uniform(0, 2 * np.pi)  # random start phase

            # Generate glottal waveform (LF-model approximation with shape parameter)
            glottal = np.zeros(n_samples, dtype=np.float64)
            for h in range(1, 12):  # 11 harmonics — richer spectrum
                amp = np.exp(spectral_tilt * np.log(h + glottal_shape))
                glottal += amp * np.sin(h * phase)

            # Normalize glottal source
            glottal /= max(np.abs(glottal).max(), 0.01)

            # --- FORMANTS (vocal tract filter) ---
            # Different utterances emphasize different formants = different vowels
            # But the formant CENTERS stay the same (same vocal tract)
            formant_emphasis = np.ones(5, dtype=np.float64)
            if utt_idx == 0:
                # /a/-like: emphasis on F1, F2
                formant_emphasis = np.array([1.4, 1.2, 0.8, 0.5, 0.3])
            elif utt_idx == 1:
                # /i/-like: emphasis on F2, F3
                formant_emphasis = np.array([0.6, 0.8, 1.3, 1.1, 0.5])
            else:
                # /u/-like: emphasis on F1, low F2
                formant_emphasis = np.array([1.3, 0.9, 0.7, 0.5, 0.3])

            # Apply formant filtering (time-domain approximation via resonators)
            # We use a cascade of 2nd-order resonators at each formant
            filtered = glottal.copy()
            for fi in range(5):
                f_center = formant_centers[fi]
                bw = f_center * 0.1  # bandwidth ~10% of center (natural)
                emphasis = formant_emphasis[fi]

                # Simple 2nd-order resonator (difference equation)
                r = np.exp(-np.pi * bw / sample_rate)
                theta = 2.0 * np.pi * f_center / sample_rate
                a1 = -2.0 * r * np.cos(theta)
                a2 = r * r
                b0 = (1.0 - r * r) * emphasis  # scale by emphasis

                # Apply filter
                y = np.zeros(n_samples, dtype=np.float64)
                y_prev1, y_prev2 = 0.0, 0.0
                for n in range(n_samples):
                    y[n] = b0 * filtered[n] - a1 * y_prev1 - a2 * y_prev2
                    y_prev2 = y_prev1
                    y_prev1 = y[n]
                filtered = y

            # Normalize
            filtered /= max(np.abs(filtered).max(), 0.01)

            # --- NOISE COMPONENT (breathiness) ---
            noise_level = 10.0 ** (-hnr / 20.0)  # convert HNR dB to ratio
            breath = rng.normal(0, noise_level, n_samples)

            # --- AMPLITUDE ENVELOPE ---
            env = np.ones(n_samples)
            attack = int(0.03 * sample_rate)
            release = int(0.08 * sample_rate)
            env[:attack] = np.linspace(0, 1, attack)
            env[-release:] = np.linspace(1, 0, release)

            # Slight amplitude modulation (natural shimmer)
            am_mod = 1.0 + 0.02 * np.sin(2.0 * np.pi * 5.0 * t + rng.uniform(0, 2 * np.pi))

            # --- FINAL MIX ---
            signal = (filtered + breath) * env * am_mod
            signal *= 0.35 / max(np.abs(signal).max(), 0.01)
            signal *= 1.0 + rng.normal(0, 0.002, n_samples)  # tiny quantization-like noise

            utterances.append(signal.astype(np.float32))

        speakers[name] = utterances

    return speakers


# ---------------------------------------------------------------------------
# Offline verification
# ---------------------------------------------------------------------------

def run_offline_verification(
    model_path: str = "models/nemo_en_titanet_small.onnx",
    output_dir: str = "profiles",
) -> dict:
    """
    Run the full offline verification battery.

    Returns a results dictionary with all metrics.
    """
    print("=" * 70)
    print("PHASE 1: Offline Speaker Verification")
    print("=" * 70)
    print(f"Model: {model_path}")
    print()

    # -- Initialize --
    engine = SpeakerEmbeddingEngine(model_path)
    print(f"Model loaded. Embedding dimension: {engine.dim}")

    db = SpeakerDatabase(output_dir)
    print(f"Database: {output_dir}")

    # -- Generate test data --
    print("\n" + "-" * 50)
    print("Generating synthetic test audio...")
    test_audio = generate_test_audio()
    for name, utterances in test_audio.items():
        print(f"  {name}: {len(utterances)} utterances x {len(utterances[0])/16000:.1f}s")

    # -- Enrollment --
    print("\n" + "-" * 50)
    print("ENROLLMENT: Registering each speaker with 2 utterances...")

    # Clear any previous test profiles
    for sid in list(test_audio.keys()):
        db.remove(sid)

    for name, utterances in test_audio.items():
        # Enroll with first TWO utterances for a more robust centroid
        emb1 = engine.extract(utterances[0])
        emb2 = engine.extract(utterances[1])
        db.add_or_update(name, emb1, metadata={"source": "synthetic"})
        db.add_or_update(name, emb2, metadata={"source": "synthetic"})
        profile = db.get(name)
        print(f"  Enrolled: {name} ({len(profile.embeddings)} embeddings, quality={profile.quality:.3f})")

    print(f"  Total speakers: {db.speaker_count}")

    # -- Verification: same-speaker tests --
    print("\n" + "-" * 50)
    print("VERIFICATION: Same-speaker tests (should ALL match)...")
    same_speaker_scores = []
    for name, utterances in test_audio.items():
        for i, utt in enumerate(utterances):
            emb = engine.extract(utt)
            is_match, score = db.verify(name, emb, threshold=0.3)
            same_speaker_scores.append(score)
            status = "✓" if is_match else "✗ FAIL"
            print(f"  {name} utterance {i}: {status} (score={score:.4f})")

    # -- Verification: cross-speaker tests --
    print("\n" + "-" * 50)
    print("VERIFICATION: Cross-speaker tests (should ALL fail)...")
    cross_speaker_scores = []
    speaker_ids = list(test_audio.keys())
    for i, sid_a in enumerate(speaker_ids):
        for j, sid_b in enumerate(speaker_ids):
            if i == j:
                continue
            emb = engine.extract(test_audio[sid_b][0])
            is_match, score = db.verify(sid_a, emb, threshold=0.3)
            cross_speaker_scores.append(score)
            status = "✗ FAIL" if is_match else "✓"
            if is_match:
                print(f"  {sid_b} voice vs {sid_a} profile: {status} (score={score:.4f})")

    # Only print the problematic ones
    false_accepts = [s for s in cross_speaker_scores if s > 0.3]
    false_rejects = [s for s in same_speaker_scores if s < 0.3]
    if false_accepts:
        print(f"  FALSE ACCEPTS: {len(false_accepts)}/{len(cross_speaker_scores)}")
    if false_rejects:
        print(f"  FALSE REJECTS: {len(false_rejects)}/{len(same_speaker_scores)}")
    if not false_accepts and not false_rejects:
        print(f"  ALL {len(cross_speaker_scores)} cross-speaker tests PASSED ✓")

    # -- Identification --
    print("\n" + "-" * 50)
    print("IDENTIFICATION: Who is speaking?...")
    correct = 0
    total = 0
    for name, utterances in test_audio.items():
        for i, utt in enumerate(utterances):
            emb = engine.extract(utt)
            best_id, best_score = db.search(emb, threshold=0.3)
            total += 1
            if best_id == name:
                correct += 1
                print(f"  {name} utt {i}: → {best_id} ✓ ({best_score:.4f})")
            else:
                print(f"  {name} utt {i}: → {best_id} ✗ MISMATCH (should be {name})")

    identification_accuracy = correct / total if total > 0 else 0
    print(f"\n  Identification accuracy: {correct}/{total} = {identification_accuracy:.1%}")

    # -- Summary --
    same_mean = float(np.mean(same_speaker_scores)) if same_speaker_scores else 0
    same_min = float(np.min(same_speaker_scores)) if same_speaker_scores else 0
    cross_mean = float(np.mean(cross_speaker_scores)) if cross_speaker_scores else 0
    cross_max = float(np.max(cross_speaker_scores)) if cross_speaker_scores else 0

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Same-speaker scores:    mean={same_mean:.4f}  min={same_min:.4f}")
    print(f"  Cross-speaker scores:   mean={cross_mean:.4f}  max={cross_max:.4f}")
    print(f"  Separation margin:      {same_min - cross_max:.4f}")
    print(f"  Identification accuracy: {identification_accuracy:.1%}")
    print(f"  False accept rate:      {len(false_accepts)}/{len(cross_speaker_scores)}")
    print(f"  False reject rate:      {len(false_rejects)}/{len(same_speaker_scores)}")

    # -- Threshold analysis --
    print("\n" + "-" * 50)
    print("THRESHOLD SWEEP: EER estimation...")
    thresholds = np.linspace(0.0, 1.0, 101)
    for thresh in thresholds:
        far = sum(1 for s in cross_speaker_scores if s >= thresh) / len(cross_speaker_scores)
        frr = sum(1 for s in same_speaker_scores if s < thresh) / len(same_speaker_scores)
        if abs(far - frr) < 0.05:
            print(f"  threshold={thresh:.2f}: FAR={far:.3f}  FRR={frr:.3f}  EER≈{max(far,frr):.3f}")
            break

    # Find actual EER
    best_eer = 1.0
    best_thresh = 0.0
    for thresh in thresholds:
        far = sum(1 for s in cross_speaker_scores if s >= thresh) / max(1, len(cross_speaker_scores))
        frr = sum(1 for s in same_speaker_scores if s < thresh) / max(1, len(same_speaker_scores))
        eer = (far + frr) / 2
        if abs(far - frr) < abs(
            sum(1 for s in cross_speaker_scores if s >= best_thresh) / max(1, len(cross_speaker_scores))
            - sum(1 for s in same_speaker_scores if s < best_thresh) / max(1, len(same_speaker_scores))
        ):
            best_eer = max(far, frr)
            best_thresh = thresh

    print(f"  Estimated EER: {best_eer:.3f} at threshold={best_thresh:.2f}")

    # Final verdict
    print("\n" + "=" * 70)
    if identification_accuracy >= 0.9 and len(false_accepts) == 0 and len(false_rejects) <= 1:
        print("✓ PHASE 1 PASSED — Speaker differentiation is reliable.")
        print("  Ready for Phase 2: Streaming VAD Pipeline.")
    else:
        print("✗ PHASE 1 NEEDS ATTENTION — Check the failures above.")
    print("=" * 70)

    return {
        "same_speaker_mean": same_mean,
        "same_speaker_min": same_min,
        "cross_speaker_mean": cross_mean,
        "cross_speaker_max": cross_max,
        "separation_margin": same_min - cross_max,
        "identification_accuracy": identification_accuracy,
        "false_accepts": len(false_accepts),
        "false_rejects": len(false_rejects),
        "estimated_eer": best_eer,
        "optimal_threshold": best_thresh,
        "model_path": model_path,
        "embedding_dim": engine.dim,
    }


# ---------------------------------------------------------------------------
# Real audio test (optional — record from microphone)
# ---------------------------------------------------------------------------

def record_from_mic(duration_s: float = 3.0, sample_rate: int = 16000) -> np.ndarray | None:
    """Record audio from the default microphone."""
    try:
        import sounddevice as sd

        print(f"Recording {duration_s}s of audio... (speak now)")
        audio = sd.rec(
            int(duration_s * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
        )
        sd.wait()
        print("Recording complete.")
        return audio.squeeze().astype(np.float32)
    except ImportError:
        print("sounddevice not available, cannot record.")
        return None
    except Exception as exc:
        print(f"Recording failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Phase 1: Offline Speaker Verification"
    )
    parser.add_argument(
        "--model",
        default="models/nemo_en_titanet_small.onnx",
        help="Path to ONNX speaker embedding model",
    )
    parser.add_argument(
        "--output-dir",
        default="profiles",
        help="Speaker profile storage directory",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Record real audio from microphone for testing",
    )
    parser.add_argument(
        "--speaker-name",
        default="user",
        help="Name for recorded speaker enrollment",
    )
    args = parser.parse_args()

    # Run synthetic verification
    results = run_offline_verification(args.model, args.output_dir)

    # Optionally record real audio
    if args.record:
        print("\n" + "=" * 70)
        print("REAL AUDIO TEST")
        print("=" * 70)

        engine = SpeakerEmbeddingEngine(args.model)
        db = SpeakerDatabase(args.output_dir)

        # Enroll
        audio = record_from_mic(3.0)
        if audio is not None:
            emb = engine.extract(audio)
            db.add_or_update(args.speaker_name, emb, metadata={"source": "microphone"})
            print(f"Enrolled '{args.speaker_name}' from microphone recording.")

            # Verify
            input("Press Enter to record a verification sample...")
            audio2 = record_from_mic(3.0)
            if audio2 is not None:
                emb2 = engine.extract(audio2)
                is_match, score = db.verify(args.speaker_name, emb2, threshold=0.3)
                print(f"Verification: {'MATCH' if is_match else 'NO MATCH'} (score={score:.4f})")

    return results


if __name__ == "__main__":
    main()
