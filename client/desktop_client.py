#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
İTÜ Student Convince AI - PyQt6 Desktop Client.
Unified, self-contained desktop interface featuring:
  - Real-time webcam video stream capturing
  - Live CV scoring metrics (Attention, posture energy, openness, emotion)
  - Automatic Voice Activity Detection (Silero VAD) & Manual Push-to-Talk recording
  - Offline DiariZen Speaker Diarisation (color-coded speaker-specific turns)
  - Direct HTTP connection to host-native Gemma 12B Speech Server
  - Low-latency real-time response audio streaming and playout
"""

import sys
import os
import time
import queue
import argparse
import threading
import tempfile
import numpy as np
import cv2
import httpx
import sounddevice as sd
import scipy.io.wavfile as wav

from PyQt6.QtCore import QUrl, Qt, QThread, pyqtSignal, QProcess, QTimer
from PyQt6.QtGui import QImage, QPixmap, QFont
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QPlainTextEdit,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QGridLayout,
    QFrame,
    QGroupBox,
    QScrollArea,
    QCheckBox,
)

# Unified constant settings
SAMPLE_RATE = 16000  # VAD/Inference sample rate

# ─────────────────────────────────────────────────────────────────────────────
# 1. Background Workers (Threads)
# ─────────────────────────────────────────────────────────────────────────────

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


class AudioCaptureWorker(QThread):
    """Microphone audio capturer supporting Manual and Silero VAD modes."""
    speech_started = pyqtSignal()
    speech_stopped = pyqtSignal()
    audio_recorded = pyqtSignal(np.ndarray)

    def __init__(self, sample_rate=16000):
        super().__init__()
        self.sample_rate = sample_rate
        self.is_recording = False
        self.is_active = True
        self.use_vad = True
        self.buffer = []

        # Load Silero VAD for voice activity boundary detection
        import sherpa_onnx
        vad_model_path = "/Users/baydogan/Documents/ComputerScience/Projects/Turkish_Speech_to_Speech/cascaded_architecture/silero_vad.onnx"
        if not os.path.exists(vad_model_path):
            for p in ["silero_vad.onnx", "../silero_vad.onnx", "../../silero_vad.onnx", "../../../silero_vad.onnx"]:
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

    def stop(self):
        self.is_active = False

    def run(self):
        from collections import deque
        pre_roll = deque(maxlen=20)  # 400ms pre-speech frames
        self.vad.reset()

        def callback(indata, frames, time_info, status):
            if not self.is_active:
                raise sd.CallbackAbort()
            samples = indata[:, 0].copy()

            if self.use_vad:
                self.vad.accept_waveform(samples)
                if self.vad.is_speech_detected():
                    if not self.is_recording:
                        self.is_recording = True
                        self.speech_started.emit()
                    self.buffer.append(samples)
                else:
                    if self.is_recording:
                        if not self.vad.empty():
                            self.vad.pop()
                            self.is_recording = False
                            self.speech_stopped.emit()
                            if self.buffer:
                                if pre_roll:
                                    pr_samples = np.concatenate(list(pre_roll))
                                    full_audio = np.concatenate([pr_samples, np.concatenate(self.buffer)])
                                else:
                                    full_audio = np.concatenate(self.buffer)
                                self.audio_recorded.emit(full_audio)
                            self.buffer = []
                            self.vad.reset()
                    else:
                        pre_roll.append(samples)
            else:
                if self.is_recording:
                    self.buffer.append(samples)

        stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            blocksize=320,  # 20ms frames
            callback=callback
        )

        with stream:
            while self.is_active:
                self.msleep(100)


class ResponseGeneratorWorker(QThread):
    """Executes Speaker Diarisation, Gemma Inference, and streams audio playout."""
    status_changed = pyqtSignal(str)
    diarisation_ready = pyqtSignal(list)
    text_ready = pyqtSignal(str, str)

    def __init__(self, audio_data, pipeline, cascaded_url="http://localhost:8002"):
        super().__init__()
        self.audio_data = audio_data
        self.pipeline = pipeline
        self.cascaded_url = cascaded_url
        self.is_interrupted = False

    def interrupt(self):
        self.is_interrupted = True

    def run(self):
        if len(self.audio_data) == 0:
            return

        # 1. Run DiariZen Speaker Diarisation
        self.status_changed.emit("Diarizing speech...")
        turns = []
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_path = tmp_file.name
        try:
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
                        "speaker_id": speaker_map[speaker]
                    })
            self.diarisation_ready.emit(turns)
        except Exception as e:
            print(f"[Diarisation Error] Failed to run diarisation: {e}", flush=True)
            self.diarisation_ready.emit([])
        finally:
            if os.path.exists(tmp_path):
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

        # Create output playback stream matching server output (24000 Hz)
        play_stream = sd.OutputStream(
            samplerate=24000,
            channels=1,
            callback=play_callback,
            blocksize=512
        )

        try:
            play_stream.start()

            # Stream playout chunk-by-chunk from speech server POST request
            with httpx.Client(timeout=60.0) as client:
                with client.stream(
                    "POST",
                    f"{self.cascaded_url}/chat_stream",
                    content=self.audio_data.tobytes()
                ) as response:
                    if response.status_code != 200:
                        self.status_changed.emit("Server Error!")
                        return
                    
                    self.status_changed.emit("Speaking...")
                    for chunk in response.iter_bytes(chunk_size=1024 * 4):
                        if self.is_interrupted:
                            break
                        audio_chunk = np.frombuffer(chunk, dtype=np.float32)
                        if len(audio_chunk) > 0:
                            for i in range(0, len(audio_chunk), 512):
                                sub_chunk = audio_chunk[i:i+512]
                                playback_queue.put(sub_chunk)

            # Wait for playout queue to finish before closing
            while not playback_queue.empty() and not self.is_interrupted:
                self.msleep(50)

        except Exception as e:
            print(f"[Speech Playout Error] Streaming failed: {e}", flush=True)
        finally:
            play_stream.stop()
            play_stream.close()

        # 3. Retrieve response texts from `/last_turn`
        if not self.is_interrupted:
            self.status_changed.emit("Finalizing turn...")
            try:
                with httpx.Client(timeout=5.0) as client:
                    txt_res = client.get(f"{self.cascaded_url}/last_turn")
                    if txt_res.status_code == 200:
                        data = txt_res.json()
                        user_text = data.get("user", "")
                        bot_text = data.get("assistant", "")
                        self.text_ready.emit(user_text, bot_text)
            except Exception as e:
                print(f"[Metadata Error] Failed to retrieve conversation turn: {e}", flush=True)

        self.status_changed.emit("Idle")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Camera Stream Thread
# ─────────────────────────────────────────────────────────────────────────────

class StreamWorker(QThread):
    """Processes webcam capture and handles connection to CV Scoring Ingestion pipeline."""
    frame_ready = pyqtSignal(object)
    debug_ready = pyqtSignal(dict)
    profile_ready = pyqtSignal(dict)
    focus_ready = pyqtSignal(dict)

    def __init__(self, server: str, session_id: str, camera_index: int, fps: float):
        super().__init__()
        self.server = server
        self.session_id = session_id
        self.camera_index = camera_index
        self.fps = fps
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        import asyncio
        import websockets
        async def main_loop():
            cap = cv2.VideoCapture(self.camera_index)
            if not cap.isOpened():
                print(f"[CV Error] Camera index {self.camera_index} cannot be opened.")
                return

            # Connect websocket streams asynchronously
            uri_stream = f"{self.server}/stream/{self.session_id}"
            uri_debug = f"{self.server}/debug/{self.session_id}"
            uri_profile = f"{self.server}/profile/{self.session_id}"
            uri_focus = f"{self.server}/focus/{self.session_id}"

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
                        return_exceptions=True
                    )
            except Exception as e:
                print(f"[CV WS Error] WS connection lost: {e}")
            finally:
                cap.release()

        # Run event loop inside thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(main_loop())


# ─────────────────────────────────────────────────────────────────────────────
# 3. Main GUI Interface
# ─────────────────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.setWindowTitle("İTÜ AI Tercih Danışmanı - Desktop Voice Client")
        self.resize(1100, 700)

        # Main layouts setup
        self.qprocesses = {}
        self.status_labels = {}
        self.diarisation_timeline = []

        # Start background model loader
        self.pipeline = None
        self.loader = PipelineLoaderWorker()
        self.loader.loaded.connect(self._on_pipeline_loaded)
        self.loader.start()

        # Start VAD audio capture worker
        self.audio_worker = AudioCaptureWorker(sample_rate=SAMPLE_RATE)
        self.audio_worker.speech_started.connect(self._on_speech_started)
        self.audio_worker.speech_stopped.connect(self._on_speech_stopped)
        self.audio_worker.audio_recorded.connect(self._on_audio_recorded)
        self.audio_worker.start()

        # UI elements
        self.active_worker = None

        # Build Main Panels
        main_layout = QHBoxLayout()
        left_panel = self._build_left_panel()
        right_panel = self._build_right_panel()
        main_layout.addWidget(left_panel, stretch=1)
        main_layout.addWidget(right_panel, stretch=1)

        central = QWidget()
        central.setLayout(main_layout)
        self.setCentralWidget(central)

        # Style sheet (dark theme look)
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #121212;
                color: #e0e0e0;
            }
            QLabel {
                font-family: 'Segoe UI', Arial;
            }
        """)

        # CV stream workers
        self._latest_debug = None
        self._latest_profile = None
        self._latest_focus = None
        self.worker = StreamWorker(args.server, args.session_id, args.camera_index, args.fps)
        self.worker.frame_ready.connect(self._on_frame)
        self.worker.debug_ready.connect(self._on_debug)
        self.worker.profile_ready.connect(self._on_profile)
        self.worker.focus_ready.connect(self._on_focus)
        self.worker.start()

        # Status polling timer
        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self._check_services_status)
        self.status_timer.start(3000)

    def _on_pipeline_loaded(self, pipeline):
        self.pipeline = pipeline
        if self.pipeline:
            self.lbl_status.setText("Status: Idle")
            self.btn_hold.setEnabled(True)
            self.chk_vad.setEnabled(True)
        else:
            self.lbl_status.setText("Status: Model load failed")

    def _build_left_panel(self) -> QWidget:
        panel = QFrame()
        panel.setFrameShape(QFrame.Shape.StyledPanel)
        panel.setStyleSheet("border: 1px solid #2d2d2d; background-color: #1a1a1a;")
        vbox = QVBoxLayout(panel)

        # Webcam preview
        self.video_label = QLabel("kamera bekleniyor...")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(320, 240)
        self.video_label.setStyleSheet("background-color: #111; border-radius: 6px;")
        vbox.addWidget(self.video_label, stretch=3)

        # Metrics text console
        self.metrics_view = QPlainTextEdit()
        self.metrics_view.setReadOnly(True)
        self.metrics_view.setStyleSheet("font-family: 'Menlo', monospace; font-size: 11px; background-color: #0f0f0f; border: none; padding: 6px;")
        vbox.addWidget(self.metrics_view, stretch=2)

        # CV Pipeline Docker Control Box
        self.service_box = QGroupBox("CV Pipeline Service Manager")
        self.service_box.setStyleSheet("QGroupBox { font-weight: bold; color: #eceff1; border: 1px solid #444; border-radius: 6px; margin-top: 12px; padding-top: 12px; }")
        s_layout = QVBoxLayout(self.service_box)

        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start CV Pipeline")
        self.start_btn.setStyleSheet("background-color: #1b5e20; color: white; font-weight: bold; padding: 6px; border-radius: 4px;")
        self.start_btn.clicked.connect(self._start_cv_docker)
        
        self.stop_btn = QPushButton("Stop CV Pipeline")
        self.stop_btn.setStyleSheet("background-color: #b71c1c; color: white; font-weight: bold; padding: 6px; border-radius: 4px;")
        self.stop_btn.clicked.connect(self._stop_cv_docker)
        
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        s_layout.addLayout(btn_layout)

        # Health grid status indicator
        grid = QGridLayout()
        services_list = ["speech_server", "cv_pipeline"]
        for i, name in enumerate(services_list):
            lbl_name = QLabel(f"{name.upper()}:")
            lbl_name.setStyleSheet("font-weight: bold; color: #b0bec5;")
            lbl_status = QLabel("STOPPED")
            lbl_status.setStyleSheet("color: #ff5252; font-weight: bold;")
            grid.addWidget(lbl_name, 0, i * 2)
            grid.addWidget(lbl_status, 0, i * 2 + 1)
            self.status_labels[name] = lbl_status
        s_layout.addLayout(grid)

        self.log_viewer = QPlainTextEdit()
        self.log_viewer.setReadOnly(True)
        self.log_viewer.setStyleSheet("background-color: #0f0f0f; color: #b0bec5; font-family: monospace; font-size: 10px; border: none;")
        s_layout.addWidget(self.log_viewer, stretch=1)
        vbox.addWidget(self.service_box, stretch=2)

        return panel

    def _build_right_panel(self) -> QWidget:
        panel = QFrame()
        panel.setFrameShape(QFrame.Shape.StyledPanel)
        panel.setStyleSheet("border: 1px solid #2d2d2d; background-color: #1e1e1e;")
        vbox = QVBoxLayout(panel)

        header = QLabel("İTÜ Tercih Danışmanı (Gemma 12B Voice Assistant)")
        header.setStyleSheet("font-size: 16px; font-weight: bold; color: #eceff1; padding: 6px;")
        vbox.addWidget(header)

        # Scrollable Chat History bubbles
        self.chat_scroll = QScrollArea()
        self.chat_scroll.setWidgetResizable(True)
        self.chat_scroll.setStyleSheet("background-color: #121212; border: none; border-radius: 8px;")
        self.chat_history_widget = QWidget()
        self.chat_layout = QVBoxLayout(self.chat_history_widget)
        self.chat_layout.addStretch()
        self.chat_scroll.setWidget(self.chat_history_widget)
        vbox.addWidget(self.chat_scroll, stretch=8)

        # Controls & status panel
        ctrl_box = QGroupBox("Voice Controls")
        ctrl_box.setStyleSheet("QGroupBox { font-weight: bold; color: #eceff1; border: 1px solid #333; border-radius: 6px; padding: 12px; }")
        ctrl_layout = QVBoxLayout(ctrl_box)

        self.lbl_status = QLabel("Status: Loading AI model...")
        self.lbl_status.setStyleSheet("font-size: 12px; font-weight: bold; color: #ffd740;")
        ctrl_layout.addWidget(self.lbl_status)

        hbox_toggles = QHBoxLayout()
        self.chk_vad = QCheckBox("Auto-Talk (VAD)")
        self.chk_vad.setChecked(True)
        self.chk_vad.setEnabled(False)
        self.chk_vad.stateChanged.connect(self._on_vad_toggle)
        
        self.btn_hold = QPushButton("Hold to Talk")
        self.btn_hold.setEnabled(False)
        self.btn_hold.setStyleSheet("background-color: #0288d1; color: white; padding: 8px; font-weight: bold; border-radius: 4px;")
        self.btn_hold.pressed.connect(self._on_hold_pressed)
        self.btn_hold.released.connect(self._on_hold_released)

        self.btn_interrupt = QPushButton("Interrupt Playout")
        self.btn_interrupt.setStyleSheet("background-color: #ff8f00; color: white; padding: 8px; font-weight: bold; border-radius: 4px;")
        self.btn_interrupt.clicked.connect(self._on_interrupt)

        hbox_toggles.addWidget(self.chk_vad)
        hbox_toggles.addWidget(self.btn_hold)
        hbox_toggles.addWidget(self.btn_interrupt)
        ctrl_layout.addLayout(hbox_toggles)
        
        vbox.addWidget(ctrl_box, stretch=2)
        return panel

    def _add_chat_bubble(self, sender: str, text: str, speaker_id: int | None = None) -> None:
        bubble_widget = QWidget()
        bubble_layout = QHBoxLayout(bubble_widget)
        bubble_layout.setContentsMargins(5, 5, 5, 5)

        bubble = QLabel(text)
        bubble.setWordWrap(True)
        bubble.setMaximumWidth(int(self.chat_scroll.width() * 0.75))

        if sender == "user":
            bubble_layout.addStretch()
            # Dynamic speaker Turn color accent based on DiariZen outputs!
            if speaker_id == 1:
                bg_color = "#004d40"  # Speaker 1: Teal
                text_meta = "User (Speaker 1):"
            elif speaker_id == 2:
                bg_color = "#4a148c"  # Speaker 2: Purple
                text_meta = "User (Speaker 2):"
            else:
                bg_color = "#0d47a1"  # Speaker 0: Blue
                text_meta = "User (Speaker 0):"

            bubble.setStyleSheet(f"background-color: {bg_color}; color: white; border-radius: 12px; padding: 10px; font-size: 13px;")
            bubble_layout.addWidget(bubble)
        else:
            text_meta = "İTÜ Danışmanı:"
            bubble.setStyleSheet("background-color: #263238; color: #eceff1; border-radius: 12px; padding: 10px; font-size: 13px;")
            bubble_layout.addWidget(bubble)
            bubble_layout.addStretch()

        wrapper_widget = QWidget()
        wrapper_layout = QVBoxLayout(wrapper_widget)
        wrapper_layout.setContentsMargins(2, 2, 2, 2)
        
        lbl_meta = QLabel(text_meta)
        lbl_meta.setStyleSheet("color: #78909c; font-size: 10px; font-weight: bold;")
        if sender == "user":
            lbl_meta.setAlignment(Qt.AlignmentFlag.AlignRight)
        else:
            lbl_meta.setAlignment(Qt.AlignmentFlag.AlignLeft)

        wrapper_layout.addWidget(lbl_meta)
        wrapper_layout.addWidget(bubble_widget)
        self.chat_layout.addWidget(wrapper_widget)

        # Autoscroll
        QTimer.singleShot(100, lambda: self.chat_scroll.verticalScrollBar().setValue(
            self.chat_scroll.verticalScrollBar().maximum()
        ))

    # Recording Trigger Handlers
    def _on_vad_toggle(self, state):
        self.audio_worker.use_vad = self.chk_vad.isChecked()
        if self.chk_vad.isChecked():
            self.btn_hold.setEnabled(False)
        else:
            self.btn_hold.setEnabled(True)

    def _on_hold_pressed(self):
        if not self.chk_vad.isChecked():
            self.lbl_status.setText("Status: Recording...")
            self.audio_worker.is_recording = True
            self.audio_worker.buffer = []

    def _on_hold_released(self):
        if not self.chk_vad.isChecked() and self.audio_worker.is_recording:
            self.audio_worker.is_recording = False
            self.lbl_status.setText("Status: Processing voice...")
            if self.audio_worker.buffer:
                recorded = np.concatenate(self.audio_worker.buffer)
                self.audio_worker.buffer = []
                self._on_audio_recorded(recorded)

    def _on_speech_started(self):
        self.lbl_status.setText("Status: Listening...")

    def _on_speech_stopped(self):
        self.lbl_status.setText("Status: Processing voice...")

    def _on_audio_recorded(self, audio_data):
        # Stop any running response worker first
        self._on_interrupt()

        # Start response worker (diarisation + POST streaming)
        self.active_worker = ResponseGeneratorWorker(
            audio_data, 
            self.pipeline, 
            f"http://localhost:8002"
        )
        self.active_worker.status_changed.connect(lambda s: self.lbl_status.setText(f"Status: {s}"))
        self.active_worker.diarisation_ready.connect(self._on_diarisation_ready)
        self.active_worker.text_ready.connect(self._on_text_ready)
        self.active_worker.start()

    def _on_diarisation_ready(self, turns):
        self.diarisation_timeline = turns

    def _on_text_ready(self, user_text, bot_text):
        # Determine dominant speaker from diarisation list
        speaker_id = 0
        if self.diarisation_timeline:
            # For simplicity, assign the speaker tag of the longest turn
            longest_duration = 0
            for turn in self.diarisation_timeline:
                duration = turn["end"] - turn["start"]
                if duration > longest_duration:
                    longest_duration = duration
                    speaker_id = turn["speaker_id"]

        if user_text:
            self._add_chat_bubble("user", user_text, speaker_id)
        if bot_text:
            self._add_chat_bubble("advisor", bot_text)

    def _on_interrupt(self):
        if self.active_worker and self.active_worker.isRunning():
            self.active_worker.interrupt()
            self.active_worker.wait(1000)
            self.lbl_status.setText("Status: Idle")

    # Webcam and CV data callbacks
    def _on_frame(self, frame) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(
            self.video_label.width(),
            self.video_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.video_label.setPixmap(pixmap)

    def _on_debug(self, data: dict) -> None:
        self._latest_debug = data
        self._refresh_metrics()

    def _on_profile(self, data: dict) -> None:
        self._latest_profile = data
        self._refresh_metrics()

    def _on_focus(self, data: dict) -> None:
        self._latest_focus = data
        self._refresh_metrics()

    def _refresh_metrics(self) -> None:
        text = _format_metrics(self._latest_debug, self._latest_profile, self._latest_focus)
        self.metrics_view.setPlainText(text)

    # Docker pipeline controls
    def _start_cv_docker(self) -> None:
        self.log_viewer.appendPlainText("Starting CV Pipeline Container (docker compose up -d)...")
        proc = QProcess(self)
        proc.readyReadStandardOutput.connect(self._on_docker_stdout)
        proc.readyReadStandardError.connect(self._on_docker_stderr)
        proc.start("docker", ["compose", "up", "-d"])
        self.qprocesses["docker_compose"] = proc

    def _stop_cv_docker(self) -> None:
        self.log_viewer.appendPlainText("Stopping CV Pipeline Container (docker compose down)...")
        proc = QProcess(self)
        proc.readyReadStandardOutput.connect(self._on_docker_stdout)
        proc.readyReadStandardError.connect(self._on_docker_stderr)
        proc.start("docker", ["compose", "down"])
        self.qprocesses["docker_compose"] = proc

    def _on_docker_stdout(self) -> None:
        sender = self.sender()
        if isinstance(sender, QProcess):
            data = sender.readAllStandardOutput().data().decode("utf-8", errors="replace")
            self.log_viewer.appendPlainText(f"[DOCKER] {data.strip()}")

    def _on_docker_stderr(self) -> None:
        sender = self.sender()
        if isinstance(sender, QProcess):
            data = sender.readAllStandardError().data().decode("utf-8", errors="replace")
            self.log_viewer.appendPlainText(f"[DOCKER ERR] {data.strip()}")

    def _check_services_status(self) -> None:
        # Check CV pipeline container status
        proc = QProcess(self)
        proc.finished.connect(lambda exit_code, exit_status, p=proc: self._on_cv_status_finished(p))
        proc.start("docker", ["compose", "ps", "--format", "{{.Service}} {{.State}}"])

        # Check port 8002 netcat connection
        nc_proc = QProcess(self)
        nc_proc.finished.connect(lambda exit_code, exit_status: self._on_nc_finished(exit_code))
        nc_proc.start("nc", ["-z", "-G", "1", "127.0.0.1", "8002"])

    def _on_cv_status_finished(self, proc: QProcess) -> None:
        data = proc.readAllStandardOutput().data().decode("utf-8", errors="replace")
        state = "STOPPED"
        for line in data.strip().split("\n"):
            parts = line.strip().split()
            if len(parts) >= 2 and parts[0] == "cv-pipeline":
                if "running" in parts[1].lower():
                    state = "RUNNING"
                elif "starting" in parts[1].lower():
                    state = "STARTING"
        lbl = self.status_labels.get("cv_pipeline")
        if lbl:
            lbl.setText(state)
            if state == "RUNNING":
                lbl.setStyleSheet("color: #69f0ae; font-weight: bold;")
            elif state == "STARTING":
                lbl.setStyleSheet("color: #ffd740; font-weight: bold;")
            else:
                lbl.setStyleSheet("color: #ff5252; font-weight: bold;")

    def _on_nc_finished(self, exit_code: int) -> None:
        state = "RUNNING" if exit_code == 0 else "STOPPED"
        lbl = self.status_labels.get("speech_server")
        if lbl:
            lbl.setText(state)
            if state == "RUNNING":
                lbl.setStyleSheet("color: #69f0ae; font-weight: bold;")
            else:
                lbl.setStyleSheet("color: #ff5252; font-weight: bold;")

    def closeEvent(self, event) -> None:
        self._on_interrupt()
        self.audio_worker.stop()
        self.audio_worker.wait(2000)
        self.worker.stop()
        self.worker.wait(2000)
        event.accept()


# ─────────────────────────────────────────────────────────────────────────────
# 4. Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(x, nd=3):
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)

def _format_metrics(debug: dict | None, profile: dict | None, focus: dict | None) -> str:
    lines: list[str] = []
    if debug is not None:
        raw = debug.get("raw", {})
        smoothed = debug.get("smoothed", {})
        live = debug.get("live_score_preview", {})
        emotion_scores = raw.get("emotion_scores") or {}
        top_emotion = ", ".join(
            f"{k}={_fmt(v, 2)}" for k, v in sorted(emotion_scores.items(), key=lambda kv: -kv[1])[:3]
        )
        lines += [
            f"state={debug.get('state')}  face={raw.get('face_present')}",
            f"focused={debug.get('is_focused')}  focus_time={_fmt(debug.get('focus_time'))}",
            f"lean(raw)={_fmt(raw.get('lean'))}  delta_lean={_fmt(smoothed.get('delta_lean'))}  baseline={_fmt(smoothed.get('baseline_lean'))}",
            f"eye_contact(raw)={_fmt(raw.get('eye_contact'))}  avg={_fmt(smoothed.get('avg_eye_contact'))}  head_yaw={_fmt(raw.get('head_yaw_deg'), 1)}",
            f"spine_ratio={_fmt(raw.get('spine_ratio'))}  shoulder_tilt={_fmt(raw.get('shoulder_tilt'))}  arms_crossed={raw.get('arms_crossed')}",
            f"emotion={raw.get('emotion_label')}  ({top_emotion})",
            f"live attn={_fmt(live.get('attention'))}  open={_fmt(live.get('openness'))}  energy={_fmt(live.get('energy'))}",
        ]
    else:
        lines.append("/debug baglantisi bekleniyor...")

    if focus is not None:
        lines.append(f"[/focus] is_focused={focus.get('is_focused')}  focus_time={_fmt(focus.get('focus_time'))}")

    if profile is not None:
        scores = profile.get("scores", {})
        lines.append(
            f"[/profile TEK SEFERLIK] attn={scores.get('attention')} open={scores.get('openness')} energy={scores.get('energy')}"
        )

    return "\n".join(lines)

def parse_args():
    p = argparse.ArgumentParser(description="İTÜ AI Tercih Danışmanı - Desktop Client")
    p.add_argument("session_id", nargs="?", default="camera-test-1")
    p.add_argument("--camera-index", type=int, default=0)
    p.add_argument("--server", default="ws://localhost:8000")
    p.add_argument("--fps", type=float, default=15.0)
    return p.parse_args()

def main() -> None:
    args = parse_args()
    app = QApplication(sys.argv[:1])
    window = MainWindow(args)
    window.show()
    app.exec()

if __name__ == "__main__":
    main()
