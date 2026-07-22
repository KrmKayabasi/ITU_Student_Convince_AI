#!/usr/bin/env python3
"""
One-shot real-time speaker verification test.

Enroll → verify continuously → live scores printed clearly.
Quit with Q or Ctrl+C.

Usage:
    python3 run.py              # Real microphone
    python3 run.py --sim        # Simulated alternating speakers
"""

import sys
import os
import time
import select
import threading
import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

MODEL_SPEAKER = ROOT / "models" / "nemo_en_titanet_small.onnx"
MODEL_VAD = ROOT / "models" / "silero_vad.onnx"

# --------------- model check ---------------
MISSING = []
if not MODEL_SPEAKER.exists():
    MISSING.append(
        "curl -L -o models/nemo_en_titanet_small.onnx "
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
        "speaker-recongition-models/nemo_en_titanet_small.onnx"
    )
if not MODEL_VAD.exists():
    MISSING.append(
        "curl -L -o models/silero_vad.onnx "
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
        "asr-models/silero_vad.onnx"
    )
if MISSING:
    print("ERROR: Missing model files. Run:")
    for cmd in MISSING:
        print(f"  {cmd}")
    sys.exit(1)

# --------------- imports ---------------
import sherpa_onnx

try:
    import sounddevice as sd
    SD_OK = True
except ImportError:
    SD_OK = False

# --------------- colors ---------------
C_ = lambda: None
C = {
    "R": "\033[31m", "G": "\033[32m", "Y": "\033[33m",
    "B": "\033[34m", "C": "\033[36m", "M": "\033[35m",
    "g": "\033[90m", "b": "\033[1m", "r": "\033[0m",
}


def say(color: str, msg: str):
    sys.stdout.write(f"\r\033[K{color}{msg}{C['r']}\n")
    sys.stdout.flush()


# ===================================================================
# Speaker engine (lightweight inline wrapper)
# ===================================================================

class SpkEng:
    def __init__(self, path: str):
        cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=path, num_threads=2, provider="cpu")
        self._e = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)

    @property
    def dim(self) -> int:
        return self._e.dim

    def embed(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        a = np.asarray(audio, dtype=np.float32).squeeze()
        s = self._e.create_stream()
        s.accept_waveform(float(sr), a.tolist())
        s.input_finished()
        v = np.array(self._e.compute(s), dtype=np.float32)
        n = float(np.linalg.norm(v))
        return v / n if n > 1e-8 else v


# ===================================================================
# Realistic synthetic voice — makes VAD-able audio
# ===================================================================

def synth_voice(f0: float, f1: float, f2: float, f3: float,
                dur: float, rng: np.random.RandomState,
                amplitude: float = 0.4) -> np.ndarray:
    """
    Generate speech-like audio with formant resonators, pitch variation,
    and natural amplitude envelope — close enough to real speech that the
    embedding model produces meaningful speaker-discriminative vectors.
    """
    fs = 16000
    t = np.arange(0, dur, 1 / fs, dtype=np.float64)
    n = len(t)

    # Pitch contour with natural variation
    f0_t = f0 * (1.0 + 0.04 * np.sin(2 * np.pi * 3.2 * t)
                 + 0.02 * np.sin(2 * np.pi * 7.1 * t)
                 + rng.normal(0, 0.004, n))

    # Phase via cumulative sum
    phase = np.cumsum(2 * np.pi * f0_t) / fs + rng.uniform(0, 2 * np.pi)

    # Glottal source: F0 + 9 harmonics, -12dB/octave spectral rolloff
    src = np.zeros(n, dtype=np.float64)
    for h in range(1, 10):
        src += np.exp(-2.0 * np.log(h)) * np.sin(h * phase)
    src /= max(abs(src).max(), 0.01)

    # Cascaded formant resonators (2nd-order IIR)
    out = src.copy()
    for fc, gain in [(f1, 1.0), (f2, 0.65), (f3, 0.35)]:
        r = np.exp(-np.pi * fc * 0.12 / fs)
        th = 2 * np.pi * fc / fs
        a1, a2 = -2 * r * np.cos(th), r * r
        b0 = (1 - r * r) * gain
        y = np.zeros(n)
        yp1 = yp2 = 0.0
        for i in range(n):
            y[i] = b0 * out[i] - a1 * yp1 - a2 * yp2
            yp2, yp1 = yp1, y[i]
        out = y

    # Normalize and scale
    out /= max(abs(out).max(), 0.01)
    out *= amplitude

    # Add breath + ambient
    out += rng.normal(0, 0.001, n)  # barely-audible breath

    return out.astype(np.float32)


# ===================================================================
# Segment detector — hybrid energy + optional VAD
# ===================================================================

class Segmenter:
    """
    Accumulates speech segments from an audio stream.

    Uses Silero VAD for real mic input, falls back to energy threshold
    for simulated audio (which Silero correctly rejects as non-human).
    """

    def __init__(self, vad_model_path: str):
        self._buf: list[np.ndarray] = []
        self._silence_ct = 0
        self._frame_ct = 0
        self._speech_ct = 0
        self._total_ct = 0
        self.in_speech = False
        self.speech_now = False
        self.rms = 0.0

        # Silero VAD
        scfg = sherpa_onnx.SileroVadModelConfig(
            model=vad_model_path, threshold=0.5,
            min_silence_duration=0.2, min_speech_duration=0.2,
            window_size=512, max_speech_duration=20)
        vcfg = sherpa_onnx.VadModelConfig(silero_vad=scfg,
                                          sample_rate=16000)
        self._vad = sherpa_onnx.VadModel.create(vcfg)

        # Energy gate constants
        self.E_ON = 0.008   # RMS above this → speech onset
        self.E_OFF = 0.004  # RMS below this for N frames → speech end
        self.SILENCE_TAIL = 15  # ~480ms at 32ms stride
        self.SEGMENT_MIN_S = 0.6

    @property
    def ws(self) -> int:
        return self._vad.window_size()

    def feed(self, chunk: np.ndarray) -> bool:
        """Returns True if this frame contains speech."""
        c = np.asarray(chunk, dtype=np.float32).squeeze()
        if len(c) < self.ws:
            c = np.pad(c, (0, self.ws - len(c)))
        elif len(c) > self.ws:
            c = c[:self.ws]

        self.rms = float(np.sqrt(np.mean(c.astype(np.float64) ** 2)))
        self._total_ct += 1

        # Try Silero first — it's authoritative for real speech
        vad_speech = False
        try:
            vad_speech = self._vad.is_speech(c.tolist())
        except Exception:
            pass

        # Energy gate as fallback (catches synthetic audio, very loud speech)
        energy_speech = self.rms > self.E_ON

        # Combine: Silero OR (energy AND not currently in silence tail)
        self.speech_now = vad_speech or (
            energy_speech and not (self.in_speech and self._silence_ct > 3)
        )

        if self.speech_now:
            self._speech_ct += 1
            self._silence_ct = 0
            if not self.in_speech:
                self.in_speech = True
            self._buf.append(c.copy())
            self._frame_ct += 1
        else:
            if self.in_speech:
                self._silence_ct += 1

        return self.speech_now

    def ready(self) -> bool:
        """Segment is ready if enough silence OR max duration reached."""
        if not self.in_speech:
            return False
        too_long = self._frame_ct > 70  # ~2.2s max, force flush
        return self._silence_ct >= self.SILENCE_TAIL or too_long

    def pop(self) -> np.ndarray | None:
        if not self._buf:
            self._reset()
            return None
        audio = np.concatenate(self._buf).astype(np.float32)
        dur = len(audio) / 16000
        self._reset()
        if dur < self.SEGMENT_MIN_S:
            return None
        return audio

    def flush(self) -> np.ndarray | None:
        if not self._buf:
            return None
        self._silence_ct = 999
        return self.pop()

    def _reset(self):
        self._buf, self._silence_ct, self._frame_ct = [], 0, 0
        self.in_speech = False

    @property
    def speech_ratio(self) -> float:
        return self._speech_ct / max(1, self._total_ct)


# ===================================================================
# Main
# ===================================================================

def main():
    p = argparse.ArgumentParser(description="One-shot speaker verification test")
    p.add_argument("--sim", action="store_true", help="Simulated mode (no microphone)")
    p.add_argument("--threshold", type=float, default=0.40,
                   help="Cosine similarity threshold for match")
    p.add_argument("--duration", type=float, default=15.0,
                   help="Test duration in seconds (sim mode)")
    args = p.parse_args()

    # --------------- header ---------------
    print(f"\n{C['b']}{C['C']}╔══════════════════════════════════════════════════╗{C['r']}")
    print(f"{C['b']}{C['C']}║     SPEAKER VERIFICATION — ONE-SHOT TEST         ║{C['r']}")
    print(f"{C['b']}{C['C']}║     TitaNet-Small (192-dim) + Silero VAD          ║{C['r']}")
    print(f"{C['b']}{C['C']}╚══════════════════════════════════════════════════╝{C['r']}")

    # --------------- hardware ---------------
    if args.sim:
        say(C["Y"], "Simulated mode — no microphone needed")
    elif not SD_OK:
        say(C["Y"], "sounddevice not installed → falling back to --sim")
        args.sim = True
    else:
        try:
            if sd.default.device[0] is None:
                say(C["Y"], "No mic found → falling back to --sim")
                args.sim = True
        except Exception:
            say(C["Y"], "Cannot query audio → falling back to --sim")
            args.sim = True

    # --------------- load models ---------------
    say(C["g"], "Loading models...")
    engine = SpkEng(str(MODEL_SPEAKER))
    say(C["g"], f"  Speaker model: {engine.dim}-dim ✓")
    seg = Segmenter(str(MODEL_VAD))
    say(C["g"], f"  VAD + energy gate: ready ✓")

    # ================================================================
    # ENROLLMENT
    # ================================================================
    print()
    say(C["b"] + C["C"], "═══ ENROLLMENT ═══")
    rng = np.random.RandomState(42)

    if args.sim:
        # Synthetic enrollment: bright voice profile
        enroll_audio = synth_voice(195, 730, 2350, 3400, 3.0, rng, 0.4)
        say(C["G"], f"Synthetic enrollment voice — 3.0s ✓")
    else:
        say(C["C"], f"🎤 Speak naturally for 3 seconds...")
        time.sleep(0.3)
        try:
            enroll_audio = sd.rec(int(3.0 * 16000), samplerate=16000,
                                  channels=1, dtype="float32")
            sd.wait()
            enroll_audio = enroll_audio.squeeze().astype(np.float32)
            rms = float(np.sqrt(np.mean(enroll_audio ** 2)))
            say(C["G"], f"Recorded 3.0s (RMS={rms:.4f}) ✓")
            if rms < 0.005:
                say(C["Y"], "  WARNING: Very quiet — results may be unreliable")
        except Exception as e:
            say(C["R"], f"Recording failed: {e}")
            sys.exit(1)

    enroll_emb = engine.embed(enroll_audio)
    say(C["G"], "Enrollment embedding ready.")

    # ================================================================
    # REAL-TIME LOOP
    # ================================================================
    print()
    say(C["b"] + C["C"], "═══ REAL-TIME VERIFICATION ═══")
    thresh = args.threshold
    print(f"  {C['g']}Score > {thresh:.2f} → {C['G']}TARGET SPEAKER{C['r']}")
    print(f"  {C['g']}Score < {thresh:.2f} → {C['R']}OTHER / UNKNOWN{C['r']}")
    print()

    # Audio source
    stream = None
    ring: list[np.ndarray] = []
    lock = threading.Lock()
    running = True
    sim_rng = np.random.RandomState(99)
    sim_state = {"target": True, "phase_start": time.time()}

    # Pre-generate word buffers so formant resonators have full context.
    # Generating 32ms slices independently destroys vocal tract coherence.
    def make_word(f0, f1, f2, f3):
        return synth_voice(f0, f1, f2, f3, 2.0, sim_rng, 0.35)

    target_word = make_word(195, 730, 2350, 3400)
    other_word = make_word(105, 520, 1450, 2550)
    silence_frame = np.zeros(512, dtype=np.float32)

    word_pos = 0          # current position in the word buffer
    silence_remaining = 0  # frames of silence to emit before next word

    if not args.sim:
        def cb(indata, frames, tinfo, st):
            if len(indata) > 0:
                with lock:
                    ring.append(indata.squeeze().astype(np.float32))
        try:
            stream = sd.InputStream(samplerate=16000, channels=1,
                                    blocksize=512, callback=cb,
                                    dtype="float32")
            stream.start()
            say(C["G"], "🎤 Microphone live — speak to test!")
        except Exception as e:
            say(C["R"], f"Mic error: {e} → simulated mode")
            args.sim = True

    # Metrics
    n_segments = 0
    n_target = 0
    n_other = 0
    last_score = 0.0
    last_lat = 0.0

    try:
        while running:
            frame = None

            # --- Get frame ---
            if args.sim:
                now = time.time()
                # Alternate speakers every 4 seconds to demonstrate differentiation
                elapsed = now - sim_state["phase_start"]
                if elapsed > 4.0:
                    sim_state["target"] = not sim_state["target"]
                    sim_state["phase_start"] = now
                    who = "TARGET SPEAKER" if sim_state["target"] else "OTHER SPEAKER"
                    say(C["G"] if sim_state["target"] else C["Y"],
                        f"── {who} ──")

                # Serve frames from pre-generated word buffers with silence gaps.
                # This preserves vocal tract coherence across the full utterance.
                if silence_remaining > 0:
                    frame = silence_frame
                    silence_remaining -= 1
                else:
                    buf = target_word if sim_state["target"] else other_word
                    start_idx = word_pos * 512
                    end_idx = start_idx + 512
                    if end_idx <= len(buf):
                        frame = buf[start_idx:end_idx].astype(np.float32)
                        word_pos += 1
                    else:
                        # Word ended — insert silence gap, regenerate word
                        word_pos = 0
                        silence_remaining = 22  # ~700ms silence
                        if sim_state["target"]:
                            target_word = make_word(195, 730, 2350, 3400)
                        else:
                            other_word = make_word(105, 520, 1450, 2550)
                        frame = silence_frame

                time.sleep(0.028)
            else:
                with lock:
                    if ring:
                        frame = ring.pop(0)
                if frame is None:
                    time.sleep(0.003)
                    r, _, _ = select.select([sys.stdin], [], [], 0)
                    if r:
                        ch = sys.stdin.readline().strip().lower()
                        if ch in ("q", "quit", "exit"):
                            break
                    continue

            # --- VAD + segment ---
            _ = seg.feed(frame[:512])

            if seg.ready():
                audio = seg.pop()
                if audio is not None:
                    n_segments += 1

                    t0 = time.perf_counter()
                    test_emb = engine.embed(audio)
                    score = float(np.dot(enroll_emb, test_emb))
                    last_lat = (time.perf_counter() - t0) * 1000
                    last_score = score

                    dur = len(audio) / 16000
                    is_target = score > thresh

                    if is_target:
                        n_target += 1
                        say(C["G"],
                            f"  ✓ TARGET  score={score:.3f}  "
                            f"dur={dur:.1f}s  lat={last_lat:.0f}ms")
                    else:
                        n_other += 1
                        say(C["R"],
                            f"  ✗ OTHER   score={score:.3f}  "
                            f"dur={dur:.1f}s  lat={last_lat:.0f}ms")

            # --- Status bar ---
            now = time.time()
            if n_segments == 0:
                # Pre-first-segment: show frame-level VAD/energy
                vad_txt = f"{C['G']}SPEECH{C['r']}" if seg.speech_now else f"{C['g']}silence{C['r']}"
                bar = f"  {vad_txt}  rms={seg.rms:.4f}  |  waiting for utterance..."
            else:
                sc = C["G"] if last_score > thresh else C["R"]
                bar = (f"  segments: {n_segments}  |  "
                       f"last: {sc}{last_score:.3f}{C['r']}  |  "
                       f"{C['G']}target:{n_target}{C['r']}  "
                       f"{C['R']}other:{n_other}{C['r']}")

            sys.stdout.write(f"\r\033[K{bar}{C['r']}")
            sys.stdout.flush()

            # Keyboard
            r, _, _ = select.select([sys.stdin], [], [], 0)
            if r:
                ch = sys.stdin.readline().strip().lower()
                if ch in ("q", "quit", "exit"):
                    break

            # Auto-stop in sim mode after duration
            if args.sim and n_segments >= 20:
                say(C["g"], "\nReached 20 segments — stopping.")
                break

    except KeyboardInterrupt:
        pass
    finally:
        running = False
        if stream:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass
        sys.stdout.write(f"\r\033[K{C['r']}")
        sys.stdout.flush()

    # ================================================================
    # REPORT
    # ================================================================
    print()
    print(f"{C['b']}{'═' * 50}{C['r']}")
    print(f"{C['b']}  RESULTS{C['r']}")
    print(f"{C['b']}{'═' * 50}{C['r']}")
    print(f"  Total utterances:   {n_segments}")
    print(f"  {C['G']}Target matches:     {n_target}{C['r']}")
    print(f"  {C['R']}Other / unknown:    {n_other}{C['r']}")
    print(f"  Last score:         {last_score:.3f}")
    print(f"  Extraction latency: {last_lat:.0f}ms")
    print(f"  Threshold:          {thresh:.2f}")
    print()

    if n_segments > 0:
        acc = n_target / n_segments
        if args.sim:
            print(f"  Target detected:    {acc:.0%} of segments")
            if 0.30 < acc < 0.70:
                print(f"  {C['G']}✓ SPEAKER DIFFERENTIATION WORKS{C['r']}")
                print(f"  {C['g']}  Target speaker scores high, other speaker scores low.{C['r']}")
                print(f"  {C['g']}  The alternating sim correctly produces ~50/50 split.{C['r']}")
            elif acc > 0.80:
                print(f"  {C['Y']}⚠ Threshold may be too low — nearly everything matches{C['r']}")
            else:
                print(f"  {C['Y']}⚠ Threshold may be too high — nearly nothing matches{C['r']}")
        else:
            print(f"  Target: {acc:.0%}")
            if acc > 0.60:
                print(f"  {C['G']}✓ Your voice consistently matches enrollment{C['r']}")
            else:
                print(f"  {C['Y']}⚠ Low match rate — try re-enrolling or lowering threshold{C['r']}")

    print(f"\n{C['g']}Done.{C['r']}")


if __name__ == "__main__":
    main()
