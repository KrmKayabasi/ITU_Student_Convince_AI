#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Background workers for the İTÜ Student Convince AI desktop client.

Extracted from desktop_client.py to keep the GUI module focused on UI concerns.
Each worker runs in its own QThread or threading context and communicates with
the main thread exclusively through Qt signals.
"""

from __future__ import annotations

import os
import queue
import tempfile
import threading
import json
import time
import importlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wav
import httpx
from collections import deque

from PyQt6.QtCore import QThread, pyqtSignal

# ── Constants ─────────────────────────────────────────────────────────────────

SAMPLE_RATE = 16000


def _load_sounddevice():
    try:
        import sounddevice as sd
    except OSError as e:
        raise RuntimeError(
            "PortAudio system library is missing. Install it first, e.g. "
            "on Ubuntu/Debian: sudo apt install libportaudio2 portaudio19-dev"
        ) from e
    return sd


def list_audio_devices() -> tuple[list[tuple[int, str]], list[tuple[int, str]]]:
    sd = _load_sounddevice()
    inputs = []
    outputs = []
    for index, device in enumerate(sd.query_devices()):
        name = str(device.get("name", f"Device {index}"))
        if device.get("max_input_channels", 0) > 0:
            inputs.append((index, name))
        if device.get("max_output_channels", 0) > 0:
            outputs.append((index, name))
    return inputs, outputs


def _ensure_sherpa_onnxruntime_link() -> None:
    onnx_spec = importlib.util.find_spec("onnxruntime")
    sherpa_spec = importlib.util.find_spec("sherpa_onnx")
    if onnx_spec is None or sherpa_spec is None or onnx_spec.origin is None or sherpa_spec.origin is None:
        return

    onnx_capi_dir = Path(onnx_spec.origin).parent / "capi"
    candidates = sorted(onnx_capi_dir.glob("libonnxruntime.so*"))
    if not candidates:
        return

    site_packages = Path(sherpa_spec.origin).parent.parent
    sherpa_libs = site_packages / "sherpa_onnx.libs"
    link_path = sherpa_libs / "libonnxruntime.so"
    if link_path.exists():
        return

    sherpa_libs.mkdir(exist_ok=True)
    target = candidates[0]
    try:
        link_path.symlink_to(os.path.relpath(target, sherpa_libs))
    except OSError:
        pass


def _load_sherpa_onnx():
    try:
        return importlib.import_module("sherpa_onnx")
    except ImportError as e:
        if "libonnxruntime.so" not in str(e):
            raise
        sys.modules.pop("sherpa_onnx", None)
        _ensure_sherpa_onnxruntime_link()
        return importlib.import_module("sherpa_onnx")


class SpeakerTracker:
    """Assign stable anonymous speaker IDs across utterances in one session."""

    def __init__(self, threshold: float = 0.35):
        self.threshold = threshold
        self._centroids: dict[int, np.ndarray] = {}
        self._counts: dict[int, int] = {}
        self._next_id = 0

    @staticmethod
    def _normalize(embedding: np.ndarray) -> np.ndarray | None:
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm <= 1e-8:
            return None
        return vector / norm

    def reset(self) -> None:
        self._centroids.clear()
        self._counts.clear()
        self._next_id = 0

    def assign(self, local_embeddings: dict) -> dict:
        normalized = {
            label: vector
            for label, embedding in local_embeddings.items()
            if (vector := self._normalize(embedding)) is not None
        }
        assignments = {}
        available_global_ids = set(self._centroids)

        candidates = []
        for label, vector in normalized.items():
            for global_id in available_global_ids:
                distance = 1.0 - float(np.dot(vector, self._centroids[global_id]))
                candidates.append((distance, label, global_id))

        used_labels = set()
        used_global_ids = set()
        for distance, label, global_id in sorted(candidates):
            if distance > self.threshold:
                break
            if label in used_labels or global_id in used_global_ids:
                continue
            assignments[label] = global_id
            used_labels.add(label)
            used_global_ids.add(global_id)

        for label in normalized:
            if label not in assignments:
                assignments[label] = self._next_id
                self._next_id += 1

        for label, global_id in assignments.items():
            vector = normalized[label]
            if global_id not in self._centroids:
                self._centroids[global_id] = vector
                self._counts[global_id] = 1
                continue
            count = self._counts[global_id]
            updated = self._centroids[global_id] * count + vector
            normalized_update = self._normalize(updated)
            if normalized_update is not None:
                self._centroids[global_id] = normalized_update
                self._counts[global_id] = count + 1

        return assignments


# ── Pipeline Loader ───────────────────────────────────────────────────────────

class PipelineLoaderWorker(QThread):
    """Loads DiariZen Pipeline in background to prevent GUI freeze on startup."""
    loaded = pyqtSignal(object)

    def run(self):
        try:
            print("[Model] DiariZen pipeline model is loading in background...", flush=True)
            from diarizen.pipelines.inference import DiariZenPipeline
            import torch

            # Detect hardware acceleration device
            if torch.cuda.is_available():
                device = torch.device("cuda")
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")

            pipeline = DiariZenPipeline.from_pretrained(
                "BUT-FIT/diarizen-wavlm-large-s80-md-v2"
            ).to(device)
            print(f"[Model] DiariZen pipeline loaded successfully on {device}!", flush=True)
            self.loaded.emit(pipeline)
        except Exception as e:
            print(f"[Model Error] Background pipeline loading failed: {e}", flush=True)
            self.loaded.emit(None)


# ── Audio Capture ─────────────────────────────────────────────────────────────

class AudioCaptureWorker(QThread):
    """Microphone audio capturer supporting Manual and Silero VAD modes.

    Thread safety: the audio callback runs in a real-time sounddevice thread
    while the main Qt thread reads/writes is_recording, use_vad, and buffer.
    A threading.Lock guards all mutable state shared between the two.
    """
    speech_started = pyqtSignal()
    speech_stopped = pyqtSignal()
    audio_recorded = pyqtSignal(np.ndarray)
    level_changed = pyqtSignal(float, float)
    error_occurred = pyqtSignal(str)

    def __init__(self, sample_rate=SAMPLE_RATE, input_device=None,
                 noise_suppression=True, ns_level=2, auto_gain_control=True,
                 use_vad=True, capture_enabled=False):
        super().__init__()
        self.sd = _load_sounddevice()
        self.sample_rate = sample_rate
        self.input_device = input_device
        self._lock = threading.Lock()
        self._is_recording = False
        self.is_active = True
        self._use_vad = use_vad
        self._capture_enabled = capture_enabled
        self._noise_suppression_enabled = noise_suppression
        self._buffer: list = []
        self._audio_processor = None

        try:
            from pywebrtc_audio import AudioProcessor

            self._audio_processor = AudioProcessor(
                sample_rate=sample_rate,
                num_channels=1,
                noise_suppression=True,
                high_pass_filter=True,
                auto_gain_control=auto_gain_control,
                echo_cancellation=False,
                ns_level=max(0, min(3, ns_level)),
                agc_max_gain_db=12.0,
            )
        except Exception as e:
            print(f"[Audio Warning] Noise suppression unavailable: {e}", flush=True)

        # Load Silero VAD for voice activity boundary detection
        sherpa_onnx = _load_sherpa_onnx()
        vad_model_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "backend", "speech_backend", "silero_vad.onnx"
        ))
        if not os.path.exists(vad_model_path):
            for p in ["silero_vad.onnx", "../silero_vad.onnx",
                       "../../silero_vad.onnx", "../../../silero_vad.onnx"]:
                if os.path.exists(p):
                    vad_model_path = p
                    break

        config = sherpa_onnx.VadModelConfig()
        config.silero_vad.model = vad_model_path
        config.sample_rate = self.sample_rate
        config.silero_vad.threshold = 0.5
        config.silero_vad.min_silence_duration = 0.8
        config.silero_vad.min_speech_duration = 0.3
        self.vad = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=60)

    # ── Thread-safe property accessors for cross-thread state ─────────────────

    @property
    def is_recording(self) -> bool:
        with self._lock:
            return self._is_recording

    @is_recording.setter
    def is_recording(self, value: bool) -> None:
        with self._lock:
            self._is_recording = value

    @property
    def use_vad(self) -> bool:
        with self._lock:
            return self._use_vad

    @use_vad.setter
    def use_vad(self, value: bool) -> None:
        with self._lock:
            self._use_vad = value

    @property
    def capture_enabled(self) -> bool:
        with self._lock:
            return self._capture_enabled

    @capture_enabled.setter
    def capture_enabled(self, value: bool) -> None:
        with self._lock:
            self._capture_enabled = value

    @property
    def noise_suppression_enabled(self) -> bool:
        with self._lock:
            return self._noise_suppression_enabled

    @noise_suppression_enabled.setter
    def noise_suppression_enabled(self, value: bool) -> None:
        with self._lock:
            self._noise_suppression_enabled = value

    @property
    def noise_suppression_available(self) -> bool:
        return self._audio_processor is not None

    @property
    def buffer(self) -> list:
        with self._lock:
            return list(self._buffer)

    @buffer.setter
    def buffer(self, value: list) -> None:
        with self._lock:
            self._buffer = list(value)

    def _buffer_append(self, samples: np.ndarray) -> None:
        with self._lock:
            self._buffer.append(samples)

    def _buffer_len(self) -> int:
        with self._lock:
            return len(self._buffer)

    def _buffer_drain(self) -> list:
        with self._lock:
            buf = self._buffer
            self._buffer = []
            return buf

    def stop(self):
        self.is_active = False

    def run(self):
        sd = self.sd
        pre_roll = deque(maxlen=20)  # 400ms pre-speech frames
        self.vad.reset()
        capture_was_enabled = False
        meter_frame = 0
        processor_failed = False

        def callback(indata, frames, time_info, status):
            nonlocal capture_was_enabled, meter_frame, processor_failed
            if not self.is_active:
                raise sd.CallbackAbort()

            if status:
                self.error_occurred.emit(f"Audio input warning: {status}")

            if not self.capture_enabled:
                if capture_was_enabled:
                    self.is_recording = False
                    self._buffer_drain()
                    pre_roll.clear()
                    self.vad.reset()
                    if self._audio_processor is not None:
                        self._audio_processor.reset()
                capture_was_enabled = False
                return

            if not capture_was_enabled:
                self.vad.reset()
                pre_roll.clear()
                if self._audio_processor is not None:
                    self._audio_processor.reset()
                capture_was_enabled = True

            raw_samples = np.ascontiguousarray(indata[:, 0], dtype=np.float32)
            samples = raw_samples
            if (
                self.noise_suppression_enabled
                and self._audio_processor is not None
                and not processor_failed
            ):
                try:
                    samples = self._audio_processor.process(raw_samples)
                except Exception as e:
                    processor_failed = True
                    self.error_occurred.emit(f"Noise suppression disabled after error: {e}")
                    samples = raw_samples

            samples = np.nan_to_num(samples, copy=False).astype(np.float32, copy=False)
            meter_frame += 1
            if meter_frame >= 5:
                meter_frame = 0
                raw_rms = float(np.sqrt(np.mean(raw_samples**2))) if len(raw_samples) else 0.0
                clean_rms = float(np.sqrt(np.mean(samples**2))) if len(samples) else 0.0
                self.level_changed.emit(raw_rms, clean_rms)

            # Re-read VAD mode each callback (it can change mid-stream)
            use_vad_snapshot = self.use_vad

            # Prevent memory overflow: if speech exceeds 45s, stop and emit
            if self.is_recording and self._buffer_len() * 320 >= 16000 * 45:
                self.is_recording = False
                self.speech_stopped.emit()
                buf = self._buffer_drain()
                if buf:
                    if pre_roll:
                        pr_samples = np.concatenate(list(pre_roll))
                        full_audio = np.concatenate([pr_samples, np.concatenate(buf)])
                    else:
                        full_audio = np.concatenate(buf)
                    self.audio_recorded.emit(full_audio)
                self.vad.reset()
                return

            if use_vad_snapshot:
                self.vad.accept_waveform(samples)
                if self.vad.is_speech_detected():
                    if not self.is_recording:
                        self.is_recording = True
                        self.speech_started.emit()
                    self._buffer_append(samples)
                else:
                    if self.is_recording:
                        if not self.vad.empty():
                            self.vad.pop()
                            self.is_recording = False
                            self.speech_stopped.emit()
                            buf = self._buffer_drain()
                            if buf:
                                if pre_roll:
                                    pr_samples = np.concatenate(list(pre_roll))
                                    full_audio = np.concatenate([pr_samples, np.concatenate(buf)])
                                else:
                                    full_audio = np.concatenate(buf)
                                self.audio_recorded.emit(full_audio)
                            self.vad.reset()
                    else:
                        pre_roll.append(samples)
            else:
                if self.is_recording:
                    self._buffer_append(samples)

        try:
            stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                blocksize=320,  # 20ms frames
                callback=callback,
                device=self.input_device,
            )

            with stream:
                while self.is_active:
                    self.msleep(100)
        except Exception as e:
            self.error_occurred.emit(f"Microphone unavailable: {e}")


# ── Response Generator ────────────────────────────────────────────────────────

class ResponseGeneratorWorker(QThread):
    """Executes Speaker Diarisation, Gemma Inference, and streams audio playout."""
    status_changed = pyqtSignal(str)
    diarisation_ready = pyqtSignal(list)
    text_ready = pyqtSignal(str, str)

    def __init__(self, audio_data, pipeline, cascaded_url="http://localhost:8002",
                 auth_token=None, session_id="default", diarization_enabled=True,
                 output_device=None, speaker_tracker=None):
        super().__init__()
        self.audio_data = audio_data
        self.pipeline = pipeline
        self.cascaded_url = cascaded_url
        self.auth_token = auth_token
        self.session_id = session_id
        self.diarization_enabled = diarization_enabled
        self.output_device = output_device
        self.speaker_tracker = speaker_tracker
        self.speaker_id = None
        self.is_interrupted = False

    def interrupt(self):
        self.is_interrupted = True

    def _make_headers(self) -> dict:
        headers = {"X-Session-ID": self.session_id}
        if self.speaker_id is not None:
            headers["X-Speaker-ID"] = str(self.speaker_id)
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        return headers

    def run(self):
        if len(self.audio_data) == 0:
            return

        turn_started = time.perf_counter()

        # 1. Run DiariZen Speaker Diarisation
        turns = []
        tmp_path = None
        diarization_started = time.perf_counter()
        if self.diarization_enabled and self.pipeline is not None:
            self.status_changed.emit("Diarizing speech...")
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                    tmp_path = tmp_file.name
                pcm_audio = (np.clip(self.audio_data, -1.0, 1.0) * 32767).astype(np.int16)
                wav.write(tmp_path, SAMPLE_RATE, pcm_audio)
                try:
                    diar_results, local_embeddings = self.pipeline(
                        tmp_path,
                        return_embeddings=True,
                    )
                except TypeError:
                    diar_results = self.pipeline(tmp_path)
                    local_embeddings = {}

                if self.speaker_tracker is not None and local_embeddings:
                    speaker_map = self.speaker_tracker.assign(local_embeddings)
                else:
                    speaker_map = {}
                for turn, _, speaker in diar_results.itertracks(yield_label=True):
                    if speaker not in speaker_map:
                        speaker_map[speaker] = len(speaker_map)
                    turns.append({
                        "start": float(turn.start),
                        "end": float(turn.end),
                        "speaker_id": speaker_map[speaker],
                    })
            except Exception as e:
                print(f"[Diarisation Error] Failed to run diarisation: {e}", flush=True)
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)
        self.diarisation_ready.emit(turns)
        if turns:
            durations = {}
            for turn in turns:
                speaker_id = turn["speaker_id"]
                durations[speaker_id] = durations.get(speaker_id, 0.0) + turn["end"] - turn["start"]
            self.speaker_id = max(durations, key=durations.get)
        diarization_ms = (time.perf_counter() - diarization_started) * 1000
        print(
            f"[Perf] diarization={self.diarization_enabled and self.pipeline is not None} "
            f"diarization_ms={diarization_ms:.0f}",
            flush=True,
        )

        # 2. Call Cascaded Speech Server
        self.status_changed.emit("Generating response...")
        playback_queue = queue.Queue()

        def play_callback(outdata, frames, time_info, status):
            try:
                data = playback_queue.get_nowait()
                if len(data) < frames:
                    outdata[:len(data), 0] = data
                    # ── Micro-fade to prevent DC click from zero-padding ──────
                    # An abrupt jump from the last sample to 0.0 creates a
                    # broadband transient (audible pop).  A 16-sample linear
                    # fade from the last value to zero eliminates it.
                    n_pad = frames - len(data)
                    if n_pad > 0 and len(data) > 0:
                        last_val = float(data[-1])
                        fade_len = min(n_pad, 16)
                        for j in range(fade_len):
                            outdata[len(data) + j, 0] = last_val * (1.0 - (j + 1) / fade_len)
                        # Remaining samples (if any) stay at 0.0
                        outdata[len(data) + fade_len:, 0] = 0.0
                    else:
                        outdata[len(data):, 0] = 0.0
                else:
                    outdata[:, 0] = data[:frames]
            except queue.Empty:
                outdata[:, 0] = 0.0

        # Create output playback stream.  The sample rate is read from the
        # server's X-Sample-Rate response header so it always matches the
        # actual TTS output (Piper VITS = 22050 Hz, XTTS = 24000 Hz, etc.).
        # Falls back to 24000 for backward compatibility with servers that
        # don't send the header.
        play_stream = None
        try:
            sd = _load_sounddevice()
            # Stream playout chunk-by-chunk from speech server POST request
            headers = self._make_headers()
            with httpx.Client(timeout=60.0) as client:
                with client.stream(
                    "POST",
                    f"{self.cascaded_url}/chat_stream",
                    content=self.audio_data.tobytes(),
                    headers=headers,
                ) as response:
                    if response.status_code == 401:
                        self.status_changed.emit("Auth Error (Idle)")
                        return
                    if response.status_code != 200:
                        self.status_changed.emit("Server Error (Idle)")
                        return

                    # ── Read actual sample rate from server header ──────────
                    sample_rate = int(response.headers.get("X-Sample-Rate", "24000"))
                    print(f"[Client] Server reports sample rate: {sample_rate} Hz", flush=True)

                    # Build the output stream NOW so it matches the actual rate
                    play_stream = sd.OutputStream(
                        samplerate=sample_rate,
                        channels=1,
                        callback=play_callback,
                        blocksize=512,
                        device=self.output_device,
                    )
                    play_stream.start()

                    self.status_changed.emit("Speaking...")
                    # ── Byte-aligned reassembly ───────────────────────────
                    # iter_bytes(chunk_size=4096) does NOT guarantee alignment
                    # with the server's 1024-float32 (4096-byte) yield chunks.
                    # TCP fragmentation or HTTP chunk aggregation can deliver
                    # partial reads.  We buffer partial bytes and reassemble
                    # at float32 (4-byte) boundaries to prevent garbled audio.
                    _byte_buf = b""
                    first_audio_received = False
                    for chunk in response.iter_bytes(chunk_size=1024 * 4):
                        if self.is_interrupted:
                            break
                        if chunk and not first_audio_received:
                            first_audio_received = True
                            print(
                                f"[Perf] first_audio_ms="
                                f"{(time.perf_counter() - turn_started) * 1000:.0f}",
                                flush=True,
                            )
                        _byte_buf += chunk
                        # Process complete float32 samples (4 bytes each)
                        n_complete = (len(_byte_buf) // 4) * 4
                        if n_complete > 0:
                            audio_chunk = np.frombuffer(_byte_buf[:n_complete], dtype=np.float32)
                            _byte_buf = _byte_buf[n_complete:]
                            if len(audio_chunk) > 0:
                                for i in range(0, len(audio_chunk), 512):
                                    sub_chunk = audio_chunk[i:i + 512]
                                    playback_queue.put(sub_chunk)
                    # Flush any remaining bytes (should be 0 after final chunk)

            # Wait for playout queue to finish before closing
            while not playback_queue.empty() and not self.is_interrupted:
                self.msleep(50)

        except Exception as e:
            print(f"[Speech Playout Error] Streaming failed: {e}", flush=True)
            self.status_changed.emit("Idle (Connection Error)")
        finally:
            if play_stream:
                try:
                    play_stream.stop()
                    play_stream.close()
                except Exception:
                    pass

        # 3. Retrieve response texts from `/last_turn`
        if not self.is_interrupted:
            self.status_changed.emit("Finalizing turn...")
            try:
                with httpx.Client(timeout=5.0) as client:
                    txt_res = client.get(
                        f"{self.cascaded_url}/last_turn",
                        headers=self._make_headers(),
                    )
                    if txt_res.status_code == 200:
                        data = txt_res.json()
                        user_text = data.get("user", "")
                        bot_text = data.get("assistant", "")
                        self.text_ready.emit(user_text, bot_text)
            except Exception as e:
                print(f"[Metadata Error] Failed to retrieve conversation turn: {e}", flush=True)

        self.status_changed.emit("Idle")


class SpeechResetWorker(QThread):
    completed = pyqtSignal(bool, str)

    def __init__(self, server_url: str, session_id: str, auth_token=None):
        super().__init__()
        self.server_url = server_url
        self.session_id = session_id
        self.auth_token = auth_token

    def run(self):
        headers = {"X-Session-ID": self.session_id}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        try:
            response = httpx.post(
                f"{self.server_url}/reset",
                headers=headers,
                timeout=5.0,
            )
            response.raise_for_status()
            self.completed.emit(True, "Session reset")
        except Exception as e:
            self.completed.emit(False, f"Session reset failed: {e}")


# ── Camera Stream ─────────────────────────────────────────────────────────────

class StreamWorker(QThread):
    """Processes webcam capture and handles connection to CV Scoring Ingestion pipeline."""
    frame_ready = pyqtSignal(object)
    debug_ready = pyqtSignal(dict)
    profile_ready = pyqtSignal(dict)
    focus_ready = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, server: str, session_id: str, camera_index: int, fps: float,
                 auth_token=None, cap=None):
        super().__init__()
        self.server = server
        self.session_id = session_id
        self.camera_index = camera_index
        self.fps = fps
        self.auth_token = auth_token
        self.cap = cap
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        import asyncio
        import websockets

        async def main_loop():
            cap = self.cap
            if cap is None:
                cap = cv2.VideoCapture(self.camera_index)
            if not cap.isOpened():
                message = f"Camera index {self.camera_index} cannot be opened; voice remains available"
                print(f"[CV Error] {message}")
                self.error_occurred.emit(message)
                return

            # Build URIs with optional auth token as query parameter
            token_suffix = f"?token={self.auth_token}" if self.auth_token else ""
            uri_stream = f"{self.server}/stream/{self.session_id}{token_suffix}"
            uri_debug = f"{self.server}/debug/{self.session_id}{token_suffix}"
            uri_profile = f"{self.server}/profile/{self.session_id}{token_suffix}"
            uri_focus = f"{self.server}/focus/{self.session_id}{token_suffix}"

            async def send_frames(ws):
                interval = 1.0 / self.fps
                while not self._stop_event.is_set():
                    ret, frame = cap.read()
                    if not ret:
                        await asyncio.sleep(0.01)
                        continue
                    self.frame_ready.emit(frame)

                    # Encode JPEG and stream to CV server
                    _, jpeg = cv2.imencode(".jpg", frame)
                    try:
                        await ws.send(jpeg.tobytes())
                    except Exception:
                        break
                    await asyncio.sleep(interval)

            async def receive(uri, signal):
                while not self._stop_event.is_set():
                    try:
                        async with websockets.connect(uri, open_timeout=2) as ws:
                            async for msg in ws:
                                if self._stop_event.is_set():
                                    return
                                signal.emit(json.loads(msg))
                    except Exception:
                        if not self._stop_event.is_set():
                            await asyncio.sleep(1.0)

            while not self._stop_event.is_set():
                tasks = []
                try:
                    async with websockets.connect(uri_stream, open_timeout=2) as ws:
                        tasks = [
                            asyncio.create_task(send_frames(ws)),
                            asyncio.create_task(receive(uri_debug, self.debug_ready)),
                            asyncio.create_task(receive(uri_profile, self.profile_ready)),
                            asyncio.create_task(receive(uri_focus, self.focus_ready)),
                        ]
                        _, pending = await asyncio.wait(
                            tasks,
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                        for task in pending:
                            task.cancel()
                        await asyncio.gather(*pending, return_exceptions=True)
                except Exception as e:
                    if not self._stop_event.is_set():
                        self.error_occurred.emit(f"CV service unavailable; retrying: {e}")
                        await asyncio.sleep(1.0)
                finally:
                    for task in tasks:
                        if not task.done():
                            task.cancel()

            cap.release()

        # Run event loop inside thread
        import cv2
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(main_loop())
