#!/usr/bin/env python3
"""
Phase 2 + 3: Real-Time Speaker Verification Pipeline

Streaming audio → VAD → speaker embedding → verification → decision.

Supports:
- Live microphone via sounddevice
- Simulated audio for testing
- Interactive keyboard controls
- Real-time status display

Usage:
    python3 phase3_realtime_pipeline.py           # Live microphone
    python3 phase3_realtime_pipeline.py --sim     # Simulated mode
"""

from __future__ import annotations

import sys
import os
import time
import select
import signal
import threading
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from lib.speaker_engine import SpeakerEmbeddingEngine, SpeakerDatabase
from lib.vad_engine import VADEngine, VADConfig, SpeechSegment
from lib.audio_capture import (
    AudioCapture,
    AudioConfig,
    SimulatedAudioCapture,
    SD_AVAILABLE,
)

# ---------------------------------------------------------------------------
# Terminal colors
# ---------------------------------------------------------------------------

C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_RED = "\033[31m"
C_GREEN = "\033[32m"
C_YELLOW = "\033[33m"
C_BLUE = "\033[34m"
C_CYAN = "\033[36m"
C_MAGENTA = "\033[35m"
C_GRAY = "\033[90m"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """Full pipeline configuration."""

    # Speaker model
    speaker_model: str = "models/nemo_en_titanet_small.onnx"
    embedding_dim: int = 192

    # VAD
    vad_model: str = "models/silero_vad.onnx"
    sample_rate: int = 16000
    blocksize: int = 512  # 32ms

    # Verification
    verify_threshold: float = 0.45   # cosine similarity threshold
    update_threshold: float = 0.65    # threshold for updating profile (stricter)

    # Enrollment
    min_enrollment_utterances: int = 3
    max_stored_embeddings: int = 30

    # Barge-in
    bargein_cooldown_s: float = 2.0

    # Auto-reset
    auto_reset_silence_s: float = 30.0

    # Storage
    profiles_dir: str = "profiles"


# ---------------------------------------------------------------------------
# Decision types
# ---------------------------------------------------------------------------

@dataclass
class VerificationDecision:
    """Result of a verification decision."""

    type: str  # "target", "nontarget", "unknown", "silence"
    speaker_id: str | None
    score: float
    segment: SpeechSegment | None
    timestamp: float

    def __repr__(self) -> str:
        if self.type == "target":
            return f"TARGET({self.speaker_id}, {self.score:.3f})"
        elif self.type == "nontarget":
            return f"NONTARGET({self.score:.3f})"
        elif self.type == "silence":
            return "SILENCE"
        return f"UNKNOWN({self.score:.3f})"


# ---------------------------------------------------------------------------
# Real-Time Speaker Verifier
# ---------------------------------------------------------------------------

class RealTimeSpeakerVerifier:
    """
    Main pipeline orchestrator.

    Threads:
    - Audio callback (sounddevice) → ring buffer
    - Main loop (this thread) → VAD → embedding → verification → decision
    """

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()

        # ---- Components ----
        self.engine = SpeakerEmbeddingEngine(
            self.config.speaker_model,
            num_threads=2,
        )
        self.db = SpeakerDatabase(self.config.profiles_dir)

        vad_cfg = VADConfig(
            model_path=self.config.vad_model,
            sample_rate=self.config.sample_rate,
            window_size=self.config.blocksize,
        )
        self.vad = VADEngine(vad_cfg)

        audio_cfg = AudioConfig(
            sample_rate=self.config.sample_rate,
            blocksize=self.config.blocksize,
        )
        self.audio = AudioCapture(audio_cfg)
        self.sim_audio = SimulatedAudioCapture(
            sample_rate=self.config.sample_rate,
            blocksize=self.config.blocksize,
        )

        # ---- State ----
        self.target_speaker_id: str | None = None
        self.is_enrolled: bool = False
        self.is_ai_speaking: bool = False
        self.running: bool = False
        self.sim_mode: bool = False

        self.silence_duration: float = 0.0
        self.last_bargein_time: float = 0.0
        self.last_status_line: str = ""
        self.last_vad_state: str = "Silence"

        # ---- Metrics ----
        self.total_segments: int = 0
        self.target_segments: int = 0
        self.nontarget_segments: int = 0
        self.bargein_count: int = 0
        self.embedding_latency_ms: float = 0.0

        # ---- Callbacks ----
        self.on_decision: callable | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> bool:
        """Start the pipeline. Returns True on success."""
        if self.sim_mode:
            self.sim_audio.start()
            print(f"{C_YELLOW}[Pipeline] Simulated audio mode{C_RESET}")
        elif not self.audio.start():
            print(f"{C_YELLOW}[Pipeline] Falling back to simulated mode{C_RESET}")
            self.sim_mode = True
            self.sim_audio.start()
        else:
            print(f"{C_GREEN}[Pipeline] Live microphone active{C_RESET}")

        self.running = True
        return True

    def stop(self) -> None:
        """Stop the pipeline."""
        self.running = False
        self.audio.stop()
        self.sim_audio.stop()
        self.db.flush()

    # ------------------------------------------------------------------
    # Enrollment
    # ------------------------------------------------------------------

    def enroll(self, speaker_id: str, audio: np.ndarray) -> float:
        """
        Enroll a speaker from an audio sample.

        Returns quality score (0-1).
        """
        emb = self.engine.extract(audio)
        self.db.add_or_update(
            speaker_id,
            emb,
            max_stored=self.config.max_stored_embeddings,
        )
        self.target_speaker_id = speaker_id
        self.is_enrolled = True
        return self.db.get(speaker_id).quality

    def enroll_current_segment(self, speaker_id: str) -> float:
        """Enroll from the most recent segment. Call after process()."""
        # This is used by wake-word enrollment
        # For now, we'd need to track the last segment
        return 0.0

    def clear_enrollment(self) -> None:
        """Clear the target speaker profile."""
        if self.target_speaker_id:
            self.db.remove(self.target_speaker_id)
        self.target_speaker_id = None
        self.is_enrolled = False

    # ------------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------------

    def step(self) -> VerificationDecision | None:
        """
        Process one audio frame. Returns a decision if a segment completed.
        """
        # 1. Get audio frame
        if self.sim_mode:
            frame = self.sim_audio.read()
        else:
            frame = self.audio.read()

        if frame is None:
            return None

        # 2. VAD
        is_speech = self.vad.process(frame)

        # 3. Check for complete segment
        if not self.vad.has_segment():
            return None

        segment = self.vad.get_segment()
        if segment is None or not segment.is_valid:
            return None

        self.total_segments += 1

        # 4. Extract embedding (timed)
        t0 = time.perf_counter()
        embedding = self.engine.extract(segment.audio)
        self.embedding_latency_ms = (time.perf_counter() - t0) * 1000

        # 5. Verify
        decision = self._verify(embedding, segment)
        return decision

    def _verify(
        self,
        embedding: np.ndarray,
        segment: SpeechSegment,
    ) -> VerificationDecision:
        """Run verification and return a decision."""
        now = time.time()

        if not self.is_enrolled or self.target_speaker_id is None:
            return VerificationDecision(
                type="unknown",
                speaker_id=None,
                score=0.0,
                segment=segment,
                timestamp=now,
            )

        # Check if this is the target speaker
        is_match, score = self.db.verify(
            self.target_speaker_id,
            embedding,
            threshold=self.config.verify_threshold,
        )

        if is_match:
            # Update profile if score is high enough (prevents drift corruption)
            if score >= self.config.update_threshold:
                self.db.add_or_update(
                    self.target_speaker_id,
                    embedding,
                    max_stored=self.config.max_stored_embeddings,
                )

            self.target_segments += 1
            return VerificationDecision(
                type="target",
                speaker_id=self.target_speaker_id,
                score=score,
                segment=segment,
                timestamp=now,
            )

        # Search for any other enrolled speaker
        best_id, best_score = self.db.search(
            embedding,
            threshold=self.config.verify_threshold,
        )

        if best_id is not None and best_id != self.target_speaker_id:
            self.nontarget_segments += 1
            return VerificationDecision(
                type="nontarget",
                speaker_id=best_id,
                score=best_score,
                segment=segment,
                timestamp=now,
            )

        # No match — unknown speaker
        return VerificationDecision(
            type="unknown",
            speaker_id=None,
            score=score,
            segment=segment,
            timestamp=now,
        )

    # ------------------------------------------------------------------
    # Barge-in check
    # ------------------------------------------------------------------

    def check_bargein(self, decision: VerificationDecision) -> bool:
        """
        Check if this decision should trigger a barge-in.

        Returns True if the AI assistant should be interrupted.
        """
        if not self.is_ai_speaking:
            return False

        if decision.type != "nontarget":
            return False

        now = time.time()
        if now - self.last_bargein_time < self.config.bargein_cooldown_s:
            return False

        self.last_bargein_time = now
        self.bargein_count += 1
        return True


# ---------------------------------------------------------------------------
# Interactive CLI
# ---------------------------------------------------------------------------

class InteractiveCLI:
    """Interactive terminal UI for the speaker verification pipeline."""

    def __init__(self, sim_mode: bool = False):
        self.config = PipelineConfig()
        self.pipeline = RealTimeSpeakerVerifier(self.config)

        if sim_mode:
            self.pipeline.sim_mode = True

        self.target_speaking = False
        self.nontarget_speaking = False

    def run(self) -> None:
        self._print_header()

        if not self.pipeline.start():
            print("Failed to start pipeline. Exiting.")
            return

        self._print_help()

        last_step = time.perf_counter()
        step_interval = self.config.blocksize / self.config.sample_rate
        last_status = time.time()

        # Silence tracking
        last_speech_time = time.time()

        try:
            while self.pipeline.running:
                # Process audio
                now = time.perf_counter()
                if now - last_step >= step_interval:
                    decision = self.pipeline.step()
                    last_step = now

                    if decision is not None:
                        last_speech_time = time.time()
                        self._handle_decision(decision)

                # Auto-reset check
                silence_dur = time.time() - last_speech_time
                if (
                    silence_dur > self.config.auto_reset_silence_s
                    and self.pipeline.is_enrolled
                ):
                    self._log(C_YELLOW, f"[Auto-Reset] {silence_dur:.0f}s silence — profile cleared")
                    self.pipeline.clear_enrollment()

                # Status display (10Hz)
                if time.time() - last_status > 0.1:
                    self._render_status()
                    last_status = time.time()

                # Keyboard
                rlist, _, _ = select.select([sys.stdin], [], [], 0.01)
                if rlist:
                    line = sys.stdin.readline().strip()
                    if line:
                        if not self._handle_key(line[0].lower()):
                            break

        except KeyboardInterrupt:
            pass
        finally:
            self.pipeline.stop()
            sys.stdout.write(f"\r\033[K{C_RESET}")
            sys.stdout.flush()

    # ------------------------------------------------------------------
    # Decision handling
    # ------------------------------------------------------------------

    def _handle_decision(self, decision: VerificationDecision) -> None:
        """React to a verification decision."""

        # Barge-in check
        if self.pipeline.check_bargein(decision):
            self._log(
                C_BOLD + C_RED,
                f"[Barge-in #{self.pipeline.bargein_count}] "
                f"Non-target speaker detected — interrupting AI!",
            )
            self.pipeline.is_ai_speaking = False

        # Log noteworthy events
        if decision.type == "target":
            if self.pipeline.target_segments % 5 == 0:
                self._log(
                    C_GREEN,
                    f"[Target] {decision.speaker_id} verified "
                    f"(score={decision.score:.3f}, "
                    f"lat={self.pipeline.embedding_latency_ms:.0f}ms)",
                )
        elif decision.type == "nontarget":
            self._log(
                C_YELLOW,
                f"[Non-Target] {decision.speaker_id} "
                f"(score={decision.score:.3f})",
            )

    # ------------------------------------------------------------------
    # Keyboard
    # ------------------------------------------------------------------

    def _handle_key(self, char: str) -> bool:
        if char == "q":
            self._log(C_RED, "Shutting down...")
            self.pipeline.running = False
            return False
        elif char == "w":
            self._enroll()
        elif char == "c":
            self.pipeline.clear_enrollment()
            self._log(C_YELLOW, "Enrollment cleared")
        elif char == "t":
            self.target_speaking = not self.target_speaking
            self.pipeline.sim_audio.target_speaking = self.target_speaking
            self._log(C_BLUE, f"Target speaker: {'ON' if self.target_speaking else 'OFF'}")
        elif char == "n":
            self.nontarget_speaking = not self.nontarget_speaking
            self.pipeline.sim_audio.nontarget_speaking = self.nontarget_speaking
            self._log(C_BLUE, f"Non-target speaker: {'ON' if self.nontarget_speaking else 'OFF'}")
        elif char == "a":
            self.pipeline.is_ai_speaking = not self.pipeline.is_ai_speaking
            self._log(C_BLUE, f"AI speaking: {'ON' if self.pipeline.is_ai_speaking else 'OFF'}")
        elif char == "h":
            self._print_help()
        return True

    def _enroll(self) -> None:
        """Enroll the current target speaker."""
        if self.pipeline.sim_mode:
            # Simulated enrollment: generate a 3-second "voice sample"
            fs = self.config.sample_rate
            duration = 3.0
            audio = np.zeros(int(fs * duration), dtype=np.float32)
            for i in range(0, len(audio), self.config.blocksize):
                end = min(i + self.config.blocksize, len(audio))
                frame = self.pipeline.sim_audio.generate_frame()
                audio[i:end] = frame[: end - i]

            quality = self.pipeline.enroll("target_user", audio)
            self._log(
                C_GREEN,
                f"Enrolled 'target_user' (simulated, quality={quality:.2f})",
            )
        else:
            # Live enrollment: record 3 seconds
            self._log(C_BLUE, "Recording 3 seconds for enrollment... speak now!")
            try:
                import sounddevice as sd
                audio = sd.rec(
                    int(3.0 * self.config.sample_rate),
                    samplerate=self.config.sample_rate,
                    channels=1,
                    dtype="float32",
                )
                sd.wait()
                audio = audio.squeeze().astype(np.float32)
                quality = self.pipeline.enroll("target_user", audio)
                self._log(
                    C_GREEN,
                    f"Enrolled 'target_user' (live, quality={quality:.2f})",
                )
            except Exception as e:
                self._log(C_RED, f"Enrollment failed: {e}")

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def _print_header(self) -> None:
        print(f"\n{C_BOLD}{C_CYAN}╔══════════════════════════════════════════════════════╗{C_RESET}")
        print(f"{C_BOLD}{C_CYAN}║   TARGET-SPEAKER VERIFICATION PIPELINE v2.0         ║{C_RESET}")
        print(f"{C_BOLD}{C_CYAN}║   Model: TitaNet-Small (192-dim) + Silero VAD       ║{C_RESET}")
        print(f"{C_BOLD}{C_CYAN}╚══════════════════════════════════════════════════════╝{C_RESET}")

    def _print_help(self) -> None:
        print(f"\n{C_BOLD}Controls:{C_RESET}")
        print(f"  {C_BOLD}w{C_RESET} - Enroll target speaker")
        print(f"  {C_BOLD}c{C_RESET} - Clear enrollment")
        print(f"  {C_BOLD}t{C_RESET} - Toggle target speaker (sim mode)")
        print(f"  {C_BOLD}n{C_RESET} - Toggle non-target speaker (sim mode)")
        print(f"  {C_BOLD}a{C_RESET} - Toggle AI speaking")
        print(f"  {C_BOLD}h{C_RESET} - This help")
        print(f"  {C_BOLD}q{C_RESET} - Quit\n")

    def _render_status(self) -> None:
        """Render the single-line status bar."""
        p = self.pipeline
        c = self.config

        # Enrollment
        enroll = f"{C_GREEN}Enrolled{C_RESET}" if p.is_enrolled else f"{C_RED}No Profile{C_RESET}"

        # VAD state
        vad_state = "Speech" if p.vad.current_is_speech else "Silence"
        vad_color = C_GREEN if p.vad.current_is_speech else C_GRAY
        vad = f"VAD:{vad_color}{vad_state}{C_RESET}"

        # AI state
        ai = f"{C_CYAN}AI:ON{C_RESET}" if p.is_ai_speaking else f"{C_GRAY}AI:OFF{C_RESET}"

        # Sim state
        if p.sim_mode:
            tgt = f"{C_GREEN}T:ON{C_RESET}" if self.target_speaking else f"{C_GRAY}T:OFF{C_RESET}"
            nt = f"{C_YELLOW}N:ON{C_RESET}" if self.nontarget_speaking else f"{C_GRAY}N:OFF{C_RESET}"
            sim = f"| {tgt} {nt}"
        else:
            sim = ""

        # Stats
        stats = (
            f"| Seg:{p.total_segments} "
            f"T:{p.target_segments} "
            f"NT:{p.nontarget_segments} "
            f"B:{p.bargein_count}"
        )

        # Latency
        lat = f"| Emb:{p.embedding_latency_ms:.0f}ms"

        line = f"\r{enroll} | {vad} | {ai} {sim} {stats} {lat}"
        sys.stdout.write(f"\r\033[K{line}{C_RESET}")
        sys.stdout.flush()

    def _log(self, color: str, msg: str) -> None:
        """Print a log line, preserving the status bar."""
        sys.stdout.write(f"\r\033[K{color}{msg}{C_RESET}\n")
        sys.stdout.flush()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Phase 2+3: Real-Time Speaker Verification Pipeline"
    )
    parser.add_argument("--sim", action="store_true", help="Use simulated audio")
    parser.add_argument(
        "--model",
        default="models/nemo_en_titanet_small.onnx",
        help="Speaker model path",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.45,
        help="Verification cosine similarity threshold",
    )
    args = parser.parse_args()

    cli = InteractiveCLI(sim_mode=args.sim)
    cli.config.speaker_model = args.model
    cli.config.verify_threshold = args.threshold
    cli.run()


if __name__ == "__main__":
    main()
