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

Architecture:
  client/workers.py   — background QThread workers (audio capture, CV streaming,
                         response generation, pipeline loading)
  client/metrics.py   — CV metrics text formatting
  client/desktop_client.py — PyQt6 MainWindow + entry point (this file)
"""

from __future__ import annotations

import sys
import os
import argparse
import numpy as np
import cv2
import httpx
from urllib.parse import urlparse

from PyQt6.QtCore import Qt, QTimer, QProcess

# Monkey-patch torch.load to default weights_only=False to support pyannote/speechbrain models in PyTorch 2.6
import torch
_orig_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

try:
    import torch.torch_version
    torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])
except Exception:
    pass
from PyQt6.QtGui import QImage, QPixmap
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

from client.workers import (
    PipelineLoaderWorker,
    AudioCaptureWorker,
    ResponseGeneratorWorker,
    StreamWorker,
    SAMPLE_RATE,
)
from client.metrics import format_metrics


# ── Dark theme stylesheet ─────────────────────────────────────────────────────

DARK_THEME = """
    QMainWindow, QWidget {
        background-color: #121212;
        color: #e0e0e0;
    }
    QLabel {
        font-family: 'Segoe UI', Arial;
    }
"""


# ── Panel builders ────────────────────────────────────────────────────────────

def _build_left_panel(main_window: MainWindow) -> QWidget:
    """Build the left panel: webcam preview, metrics, and Docker controls."""
    panel = QFrame()
    panel.setFrameShape(QFrame.Shape.StyledPanel)
    panel.setStyleSheet("border: 1px solid #2d2d2d; background-color: #1a1a1a;")
    vbox = QVBoxLayout(panel)

    # Webcam preview
    main_window.video_label = QLabel("kamera bekleniyor...")
    main_window.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    main_window.video_label.setMinimumSize(320, 240)
    main_window.video_label.setStyleSheet("background-color: #111; border-radius: 6px;")
    vbox.addWidget(main_window.video_label, stretch=3)

    # Metrics text console
    main_window.metrics_view = QPlainTextEdit()
    main_window.metrics_view.setReadOnly(True)
    main_window.metrics_view.setStyleSheet(
        "font-family: 'Menlo', monospace; font-size: 11px; "
        "background-color: #0f0f0f; border: none; padding: 6px;"
    )
    vbox.addWidget(main_window.metrics_view, stretch=2)

    # CV Pipeline Docker Control Box
    service_box = QGroupBox("CV Pipeline Service Manager")
    service_box.setStyleSheet(
        "QGroupBox { font-weight: bold; color: #eceff1; border: 1px solid #444; "
        "border-radius: 6px; margin-top: 12px; padding-top: 12px; }"
    )
    s_layout = QVBoxLayout(service_box)

    btn_layout = QHBoxLayout()
    main_window.start_btn = QPushButton("Start CV Pipeline")
    main_window.start_btn.setStyleSheet(
        "background-color: #1b5e20; color: white; font-weight: bold; "
        "padding: 6px; border-radius: 4px;"
    )
    main_window.start_btn.clicked.connect(main_window._start_cv_docker)

    main_window.stop_btn = QPushButton("Stop CV Pipeline")
    main_window.stop_btn.setStyleSheet(
        "background-color: #b71c1c; color: white; font-weight: bold; "
        "padding: 6px; border-radius: 4px;"
    )
    main_window.stop_btn.clicked.connect(main_window._stop_cv_docker)

    btn_layout.addWidget(main_window.start_btn)
    btn_layout.addWidget(main_window.stop_btn)
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
        main_window.status_labels[name] = lbl_status
    s_layout.addLayout(grid)

    main_window.log_viewer = QPlainTextEdit()
    main_window.log_viewer.setReadOnly(True)
    main_window.log_viewer.setStyleSheet(
        "background-color: #0f0f0f; color: #b0bec5; "
        "font-family: monospace; font-size: 10px; border: none;"
    )
    s_layout.addWidget(main_window.log_viewer, stretch=1)
    vbox.addWidget(service_box, stretch=2)

    return panel


def _build_right_panel(main_window: MainWindow) -> QWidget:
    """Build the right panel: chat history, voice controls, status."""
    panel = QFrame()
    panel.setFrameShape(QFrame.Shape.StyledPanel)
    panel.setStyleSheet("border: 1px solid #2d2d2d; background-color: #1e1e1e;")
    vbox = QVBoxLayout(panel)

    header = QLabel("İTÜ Tercih Danışmanı (Gemma 12B Voice Assistant)")
    header.setStyleSheet("font-size: 16px; font-weight: bold; color: #eceff1; padding: 6px;")
    vbox.addWidget(header)

    # Scrollable Chat History bubbles
    main_window.chat_scroll = QScrollArea()
    main_window.chat_scroll.setWidgetResizable(True)
    main_window.chat_scroll.setStyleSheet(
        "background-color: #121212; border: none; border-radius: 8px;"
    )
    main_window.chat_history_widget = QWidget()
    main_window.chat_layout = QVBoxLayout(main_window.chat_history_widget)
    main_window.chat_layout.addStretch()
    main_window.chat_scroll.setWidget(main_window.chat_history_widget)
    vbox.addWidget(main_window.chat_scroll, stretch=8)

    # Controls & status panel
    ctrl_box = QGroupBox("Voice Controls")
    ctrl_box.setStyleSheet(
        "QGroupBox { font-weight: bold; color: #eceff1; border: 1px solid #333; "
        "border-radius: 6px; padding: 12px; }"
    )
    ctrl_layout = QVBoxLayout(ctrl_box)

    main_window.lbl_status = QLabel("Status: Loading AI model...")
    main_window.lbl_status.setStyleSheet("font-size: 12px; font-weight: bold; color: #ffd740;")
    ctrl_layout.addWidget(main_window.lbl_status)

    hbox_toggles = QHBoxLayout()
    main_window.chk_vad = QCheckBox("Auto-Talk (VAD)")
    main_window.chk_vad.setChecked(True)
    main_window.chk_vad.setEnabled(False)
    main_window.chk_vad.stateChanged.connect(main_window._on_vad_toggle)

    main_window.btn_hold = QPushButton("Hold to Talk")
    main_window.btn_hold.setEnabled(False)
    main_window.btn_hold.setStyleSheet(
        "background-color: #0288d1; color: white; padding: 8px; "
        "font-weight: bold; border-radius: 4px;"
    )
    main_window.btn_hold.pressed.connect(main_window._on_hold_pressed)
    main_window.btn_hold.released.connect(main_window._on_hold_released)

    main_window.btn_interrupt = QPushButton("Interrupt Playout")
    main_window.btn_interrupt.setStyleSheet(
        "background-color: #ff8f00; color: white; padding: 8px; "
        "font-weight: bold; border-radius: 4px;"
    )
    main_window.btn_interrupt.clicked.connect(main_window._on_interrupt)

    hbox_toggles.addWidget(main_window.chk_vad)
    hbox_toggles.addWidget(main_window.btn_hold)
    hbox_toggles.addWidget(main_window.btn_interrupt)
    ctrl_layout.addLayout(hbox_toggles)

    vbox.addWidget(ctrl_box, stretch=2)
    return panel


# ── Main Window ───────────────────────────────────────────────────────────────

_SPEAKER_COLORS = {
    1: "#004d40",  # Teal
    2: "#4a148c",  # Purple
}


class MainWindow(QMainWindow):
    """Main desktop window orchestrating CV pipeline, audio, and chat UI."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.speech_server_url = args.speech_server
        self.setWindowTitle("İTÜ AI Tercih Danışmanı - Desktop Voice Client")
        self.resize(1100, 700)

        # State
        self.qprocesses: dict = {}
        self.status_labels: dict = {}
        self.diarisation_timeline: list = []
        self.pipeline = None
        self.active_worker = None
        self._latest_debug = None
        self._latest_profile = None
        self._latest_focus = None
        self._poll_counter = 0

        # Start background model loader
        self.loader = PipelineLoaderWorker()
        self.loader.loaded.connect(self._on_pipeline_loaded)
        self.loader.start()

        # Start VAD audio capture worker
        self.audio_worker = AudioCaptureWorker(sample_rate=SAMPLE_RATE)
        self.audio_worker.speech_started.connect(self._on_speech_started)
        self.audio_worker.speech_stopped.connect(self._on_speech_stopped)
        self.audio_worker.audio_recorded.connect(self._on_audio_recorded)
        self.audio_worker.start()

        # Build Main Panels
        main_layout = QHBoxLayout()
        main_layout.addWidget(_build_left_panel(self), stretch=1)
        main_layout.addWidget(_build_right_panel(self), stretch=1)

        central = QWidget()
        central.setLayout(main_layout)
        self.setCentralWidget(central)
        self.setStyleSheet(DARK_THEME)

        # Initialize video capture on the main GUI thread to comply with macOS AVFoundation restrictions
        self.cap = cv2.VideoCapture(args.camera_index)

        # CV stream worker
        self.worker = StreamWorker(
            args.server, args.session_id, args.camera_index, args.fps,
            auth_token=args.auth_token, cap=self.cap
        )
        self.worker.frame_ready.connect(self._on_frame)
        self.worker.debug_ready.connect(self._on_debug)
        self.worker.profile_ready.connect(self._on_profile)
        self.worker.focus_ready.connect(self._on_focus)
        self.worker.start()

        # Status polling timer (every 3 seconds)
        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self._check_services_status)
        self.status_timer.start(3000)

    # ── Pipeline / model callbacks ────────────────────────────────────────────

    def _on_pipeline_loaded(self, pipeline):
        self.pipeline = pipeline
        if self.pipeline:
            self.lbl_status.setText("Status: Idle")
            self.btn_hold.setEnabled(True)
            self.chk_vad.setEnabled(True)
        else:
            self.lbl_status.setText("Status: Model load failed")

    # ── Chat bubble rendering ─────────────────────────────────────────────────

    def _add_chat_bubble(self, sender: str, text: str,
                         speaker_id: int | None = None) -> None:
        bubble = QLabel(text)
        bubble.setWordWrap(True)
        bubble.setMaximumWidth(int(self.chat_scroll.width() * 0.75))

        if sender == "user":
            bg_color = _SPEAKER_COLORS.get(speaker_id or 0, "#0d47a1")
            sid = speaker_id or 0
            text_meta = f"User (Speaker {sid}):"
            bubble.setStyleSheet(
                f"background-color: {bg_color}; color: white; "
                f"border-radius: 12px; padding: 10px; font-size: 13px;"
            )
        else:
            text_meta = "İTÜ Danışmanı:"
            bubble.setStyleSheet(
                "background-color: #263238; color: #eceff1; "
                "border-radius: 12px; padding: 10px; font-size: 13px;"
            )

        # Bubble layout
        bubble_widget = QWidget()
        bubble_layout = QHBoxLayout(bubble_widget)
        bubble_layout.setContentsMargins(5, 5, 5, 5)
        if sender == "user":
            bubble_layout.addStretch()
            bubble_layout.addWidget(bubble)
        else:
            bubble_layout.addWidget(bubble)
            bubble_layout.addStretch()

        # Metadata label
        wrapper = QWidget()
        wrapper_layout = QVBoxLayout(wrapper)
        wrapper_layout.setContentsMargins(2, 2, 2, 2)

        lbl_meta = QLabel(text_meta)
        lbl_meta.setStyleSheet("color: #78909c; font-size: 10px; font-weight: bold;")
        lbl_meta.setAlignment(
            Qt.AlignmentFlag.AlignRight if sender == "user"
            else Qt.AlignmentFlag.AlignLeft
        )
        wrapper_layout.addWidget(lbl_meta)
        wrapper_layout.addWidget(bubble_widget)
        self.chat_layout.addWidget(wrapper)

        # Autoscroll
        QTimer.singleShot(100, lambda: self.chat_scroll.verticalScrollBar().setValue(
            self.chat_scroll.verticalScrollBar().maximum()
        ))

    # ── Recording trigger handlers ─────────────────────────────────────────────

    def _on_vad_toggle(self, state):
        self.audio_worker.use_vad = self.chk_vad.isChecked()
        self.btn_hold.setEnabled(not self.chk_vad.isChecked())

    def _on_hold_pressed(self):
        if not self.chk_vad.isChecked():
            self.lbl_status.setText("Status: Recording...")
            self.audio_worker.is_recording = True
            self.audio_worker.buffer = []

    def _on_hold_released(self):
        if not self.chk_vad.isChecked() and self.audio_worker.is_recording:
            self.audio_worker.is_recording = False
            self.lbl_status.setText("Status: Processing voice...")
            buf = self.audio_worker._buffer_drain()
            if buf:
                self._on_audio_recorded(np.concatenate(buf))

    def _on_speech_started(self):
        self.lbl_status.setText("Status: Listening...")

    def _on_speech_stopped(self):
        self.lbl_status.setText("Status: Processing voice...")

    def _on_audio_recorded(self, audio_data):
        self._on_interrupt()

        self.active_worker = ResponseGeneratorWorker(
            audio_data,
            self.pipeline,
            self.speech_server_url,
            auth_token=self.args.auth_token,
        )
        self.active_worker.status_changed.connect(
            lambda s: self.lbl_status.setText(f"Status: {s}")
        )
        self.active_worker.diarisation_ready.connect(self._on_diarisation_ready)
        self.active_worker.text_ready.connect(self._on_text_ready)
        self.active_worker.start()

    def _on_diarisation_ready(self, turns):
        self.diarisation_timeline = turns

    def _on_text_ready(self, user_text, bot_text):
        # Determine dominant speaker from diarisation list
        speaker_id = 0
        if self.diarisation_timeline:
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
            self.active_worker.wait(50)
            self.lbl_status.setText("Status: Idle")

    # ── Webcam and CV data callbacks ──────────────────────────────────────────

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
        self.metrics_view.setPlainText(
            format_metrics(self._latest_debug, self._latest_profile, self._latest_focus)
        )

    # ── Docker pipeline controls ──────────────────────────────────────────────

    def _start_cv_docker(self) -> None:
        active_proc = self.qprocesses.get("docker_compose")
        if active_proc and active_proc.state() != QProcess.ProcessState.NotRunning:
            self.log_viewer.appendPlainText(
                "[Docker Alert] A docker compose action is already in progress."
            )
            return

        self.log_viewer.appendPlainText("Starting CV Pipeline Container (docker compose up -d)...")
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)

        proc = QProcess(self)
        proc.readyReadStandardOutput.connect(self._on_docker_stdout)
        proc.readyReadStandardError.connect(self._on_docker_stderr)
        proc.finished.connect(
            lambda ec, es, p=proc: self._on_docker_finished(p, "Start", ec)
        )
        proc.start("docker", ["compose", "up", "-d"])
        self.qprocesses["docker_compose"] = proc

    def _stop_cv_docker(self) -> None:
        active_proc = self.qprocesses.get("docker_compose")
        if active_proc and active_proc.state() != QProcess.ProcessState.NotRunning:
            self.log_viewer.appendPlainText(
                "[Docker Alert] A docker compose action is already in progress."
            )
            return

        self.log_viewer.appendPlainText("Stopping CV Pipeline Container (docker compose down)...")
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)

        proc = QProcess(self)
        proc.readyReadStandardOutput.connect(self._on_docker_stdout)
        proc.readyReadStandardError.connect(self._on_docker_stderr)
        proc.finished.connect(
            lambda ec, es, p=proc: self._on_docker_finished(p, "Stop", ec)
        )
        proc.start("docker", ["compose", "down"])
        self.qprocesses["docker_compose"] = proc

    def _on_docker_finished(self, proc: QProcess, action: str, exit_code: int) -> None:
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
        if exit_code == 0:
            self.log_viewer.appendPlainText(
                f"[Docker Success] {action} command completed successfully."
            )
        else:
            self.log_viewer.appendPlainText(
                f"[Docker Error] {action} command failed with exit code {exit_code}."
            )

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

    # ── Service health polling ────────────────────────────────────────────────

    def _check_services_status(self) -> None:
        """Poll service health via lightweight HTTP requests instead of spawning
        subprocesses every cycle.  Docker compose ps is only called every 3rd
        cycle (every ~9 seconds) to reduce overhead."""
        self._poll_counter += 1
        if self._poll_counter % 3 == 0:
            proc = QProcess(self)
            proc.finished.connect(
                lambda ec, es, p=proc: self._on_cv_status_finished(p)
            )
            proc.start("docker", ["compose", "ps", "--format", "{{.Service}} {{.State}}"])

        # Check speech server via HTTP /health endpoint
        parsed = urlparse(self.speech_server_url)
        health_url = (
            f"{parsed.scheme}://{parsed.hostname or '127.0.0.1'}"
            f":{parsed.port or 8002}/health"
        )
        try:
            with httpx.Client(timeout=1.5) as client:
                resp = client.get(health_url)
                self._on_nc_finished(0 if resp.status_code == 200 else 1)
        except Exception:
            self._on_nc_finished(1)

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
            color = {"RUNNING": "#69f0ae", "STARTING": "#ffd740"}.get(state, "#ff5252")
            lbl.setStyleSheet(f"color: {color}; font-weight: bold;")

    def _on_nc_finished(self, exit_code: int) -> None:
        state = "RUNNING" if exit_code == 0 else "STOPPED"
        lbl = self.status_labels.get("speech_server")
        if lbl:
            lbl.setText(state)
            color = "#69f0ae" if state == "RUNNING" else "#ff5252"
            lbl.setStyleSheet(f"color: {color}; font-weight: bold;")

    def closeEvent(self, event) -> None:
        self._on_interrupt()
        self.audio_worker.stop()
        self.audio_worker.wait(2000)
        self.worker.stop()
        self.worker.wait(2000)
        if hasattr(self, "cap") and self.cap:
            self.cap.release()
        event.accept()


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="İTÜ AI Tercih Danışmanı - Desktop Client")
    p.add_argument("session_id", nargs="?", default="camera-test-1")
    p.add_argument("--camera-index", type=int, default=0)
    p.add_argument("--server", default="ws://localhost:8000")
    p.add_argument("--fps", type=float, default=15.0)
    p.add_argument("--speech-server", default="http://localhost:8002")
    p.add_argument("--auth-token", default=os.environ.get("ITU_AUTH_TOKEN"),
                   help="Bearer token for server authentication")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    app = QApplication(sys.argv[:1])
    window = MainWindow(args)
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
