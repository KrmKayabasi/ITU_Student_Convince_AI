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

import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
import httpx
from collections import deque

from PyQt6.QtCore import QThread, pyqtSignal

# ── Constants ─────────────────────────────────────────────────────────────────

SAMPLE_RATE = 16000


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

    def __init__(self, sample_rate=SAMPLE_RATE):
        super().__init__()
        self.sample_rate = sample_rate
        self._lock = threading.Lock()
        self._is_recording = False
        self.is_active = True
        self._use_vad = True
        self._buffer: list = []

        # Load Silero VAD for voice activity boundary detection
        import sherpa_onnx
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
        pre_roll = deque(maxlen=20)  # 400ms pre-speech frames
        self.vad.reset()

        def callback(indata, frames, time_info, status):
            if not self.is_active:
                raise sd.CallbackAbort()
            samples = indata[:, 0].copy()

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

        stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            blocksize=320,  # 20ms frames
            callback=callback,
        )

        with stream:
            while self.is_active:
                self.msleep(100)


# ── Response Generator ────────────────────────────────────────────────────────

class ResponseGeneratorWorker(QThread):
    """Executes Speaker Diarisation, Gemma Inference, and streams audio playout."""
    status_changed = pyqtSignal(str)
    diarisation_ready = pyqtSignal(list)
    text_ready = pyqtSignal(str, str)

    def __init__(self, audio_data, pipeline, cascaded_url="http://localhost:8002",
                 auth_token=None):
        super().__init__()
        self.audio_data = audio_data
        self.pipeline = pipeline
        self.cascaded_url = cascaded_url
        self.auth_token = auth_token
        self.is_interrupted = False

    def interrupt(self):
        self.is_interrupted = True

    def _make_headers(self) -> dict:
        headers = {}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        return headers

    def run(self):
        if len(self.audio_data) == 0:
            return

        # 1. Run DiariZen Speaker Diarisation
        self.status_changed.emit("Diarizing speech...")
        turns = []
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_path = tmp_file.name
            # Convert float32 to int16 for WAV output
            pcm_audio = (self.audio_data * 32767).astype(np.int16)
            wav.write(tmp_path, 16000, pcm_audio)

            if self.pipeline:
                diar_results = self.pipeline(tmp_path)
                speaker_map = {}
                for turn, _, speaker in diar_results.itertracks(yield_label=True):
                    if speaker not in speaker_map:
                        speaker_map[speaker] = len(speaker_map)
                    turns.append({
                        "start": float(turn.start),
                        "end": float(turn.end),
                        "speaker_id": speaker_map[speaker],
                    })
            self.diarisation_ready.emit(turns)
        except Exception as e:
            print(f"[Diarisation Error] Failed to run diarisation: {e}", flush=True)
            self.diarisation_ready.emit([])
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

        # 2. Call Cascaded Speech Server
        self.status_changed.emit("Generating response...")
        playback_queue = queue.Queue()

        def play_callback(outdata, frames, time_info, status):
            try:
                data = playback_queue.get_nowait()
                if len(data) < frames:
                    outdata[:len(data), 0] = data
                    outdata[len(data):, 0] = 0
                else:
                    outdata[:, 0] = data[:frames]
            except queue.Empty:
                outdata[:, 0] = 0

        # Create output playback stream.  The sample rate is read from the
        # server's X-Sample-Rate response header so it always matches the
        # actual TTS output (Piper VITS = 22050 Hz, XTTS = 24000 Hz, etc.).
        # Falls back to 24000 for backward compatibility with servers that
        # don't send the header.
        play_stream = None
        try:
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
                    )
                    play_stream.start()

                    self.status_changed.emit("Speaking...")
                    for chunk in response.iter_bytes(chunk_size=1024 * 4):
                        if self.is_interrupted:
                            break
                        audio_chunk = np.frombuffer(chunk, dtype=np.float32)
                        if len(audio_chunk) > 0:
                            for i in range(0, len(audio_chunk), 512):
                                sub_chunk = audio_chunk[i:i + 512]
                                playback_queue.put(sub_chunk)

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


# ── Camera Stream ─────────────────────────────────────────────────────────────

class StreamWorker(QThread):
    """Processes webcam capture and handles connection to CV Scoring Ingestion pipeline."""
    frame_ready = pyqtSignal(object)
    debug_ready = pyqtSignal(dict)
    profile_ready = pyqtSignal(dict)
    focus_ready = pyqtSignal(dict)

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
                print(f"[CV Error] Camera index {self.camera_index} cannot be opened.")
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

            async def recv_debug():
                async for ws in websockets.connect(uri_debug):
                    try:
                        async for msg in ws:
                            self.debug_ready.emit(json.loads(msg))
                    except websockets.ConnectionClosed:
                        continue

            async def recv_profile():
                async for ws in websockets.connect(uri_profile):
                    try:
                        async for msg in ws:
                            self.profile_ready.emit(json.loads(msg))
                    except websockets.ConnectionClosed:
                        continue

            async def recv_focus():
                async for ws in websockets.connect(uri_focus):
                    try:
                        async for msg in ws:
                            self.focus_ready.emit(json.loads(msg))
                    except websockets.ConnectionClosed:
                        continue

            try:
                async with websockets.connect(uri_stream) as ws:
                    await asyncio.gather(
                        send_frames(ws),
                        recv_debug(),
                        recv_profile(),
                        recv_focus(),
                        return_exceptions=True,
                    )
            except Exception as e:
                print(f"[CV WS Error] WS connection lost: {e}")
            finally:
                cap.release()

        # Run event loop inside thread
        import cv2
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(main_loop())
