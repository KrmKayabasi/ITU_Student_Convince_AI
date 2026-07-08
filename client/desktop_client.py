"""
Cihazin gercek kamerasini kullanan test istemcisi (PyQt6 panel).

Mimari kararlar geregi (bkz. docs/SPRINT.md Bolum 2), cv-pipeline modulunun
kendisi KAMERAYA DOKUNMAZ — yalnizca network uzerinden JPEG kare alir. Bu
script, o "robot/kiosk" istemcisinin yerini tutar: yerel webcam'i acar,
kareleri encode edip /stream'e gonderir; /profile (tek seferlik), /focus
(periyodik) ve /debug (SADECE test icin, ~0.3sn'de tum ham+turetilmis
degerler) kanallarindan gelenleri canli olarak panelde gosterir.

Pencere HBox olarak 2 esit panele bolunur:
  - Sol : kamera goruntusu + canli metrikler
  - Sag : AIme_Girl (Unmute) frontend'i, QWebEngineView icinde gomulu

AIme_Girl backend'i (STT/TTS/LLM) GPU/CUDA gerektirdigi icin bu makinede
calismayabilir; sag panel yalnizca frontend'e (varsayilan http://localhost:3000)
baglanir. Backend baska bir yerde (ör. GPU'lu sunucu) calisiyorsa --aime-url
ile o adresi verin.

Kullanim:
  python camera_client.py [session_id] [--camera-index 0] [--server ws://localhost:8000] [--aime-url http://localhost:3000]

Cikmak icin pencereyi kapatin (veya Ctrl+C).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import threading

import cv2
import websockets
from pathlib import Path
from PyQt6.QtCore import QUrl, Qt, QThread, pyqtSignal, QProcess, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
    QGroupBox,
    QPushButton,
    QGridLayout,
)


def parse_args():
    p = argparse.ArgumentParser(description="Gercek kamera ile cv-pipeline test istemcisi (PyQt6)")
    p.add_argument("session_id", nargs="?", default="camera-test-1")
    p.add_argument("--camera-index", type=int, default=0, help="cv2.VideoCapture cihaz indeksi")
    p.add_argument("--server", default="ws://localhost:8000", help="cv-pipeline sunucu adresi")
    p.add_argument("--fps", type=float, default=15.0)
    p.add_argument(
        "--aime-url",
        default="http://localhost:3000",
        help="AIme_Girl (Unmute) frontend adresi (sag panelde gomulu olarak gosterilir)",
    )
    return p.parse_args()


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


class StreamWorker(QThread):
    """Kamerayi acar, /stream'e kare basar; /profile, /focus, /debug kanallarini
    dinler. Kendi asyncio event loop'unu bu thread icinde calistirir, sonuclari
    Qt sinyalleriyle (otomatik olarak queued/thread-safe) GUI thread'e iletir."""

    frame_ready = pyqtSignal(object)  # np.ndarray (BGR)
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
        try:
            asyncio.run(self._main())
        except Exception as exc:  # noqa: BLE001 - arka plan thread'de son durak
            print(f"[{self.session_id}] worker hata: {exc}")

    async def _main(self) -> None:
        await asyncio.gather(
            self._push_camera_frames(),
            self._listen_profile(),
            self._listen_focus(),
            self._listen_debug(),
        )

    async def _push_camera_frames(self) -> None:
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"kamera acilamadi (index={self.camera_index})")

        uri = f"{self.server}/stream/{self.session_id}"
        interval = 1.0 / self.fps
        try:
            async with websockets.connect(uri) as ws:
                print(f"[{self.session_id}] kameradan {uri} adresine akis basladi")
                while not self._stop_event.is_set():
                    ok, frame = cap.read()
                    if not ok:
                        await asyncio.sleep(0.05)
                        continue

                    self.frame_ready.emit(frame.copy())

                    ok, buf = cv2.imencode(".jpg", frame)
                    if ok:
                        await ws.send(buf.tobytes())
                    await asyncio.sleep(interval)
        finally:
            cap.release()

    async def _listen_profile(self) -> None:
        uri = f"{self.server}/profile/{self.session_id}"
        async with websockets.connect(uri) as ws:
            print(f"[{self.session_id}] /profile dinleniyor")
            while not self._stop_event.is_set():
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                data = json.loads(msg)
                self.profile_ready.emit(data)
                print(f"[{self.session_id}] PROFILE (tek seferlik): {json.dumps(data, ensure_ascii=False)}")

    async def _listen_focus(self) -> None:
        uri = f"{self.server}/focus/{self.session_id}"
        async with websockets.connect(uri) as ws:
            print(f"[{self.session_id}] /focus dinleniyor")
            while not self._stop_event.is_set():
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                data = json.loads(msg)
                self.focus_ready.emit(data)
                print(f"[{self.session_id}] FOCUS: {data}")

    async def _listen_debug(self) -> None:
        """SADECE test/dogrulama icin: /debug'dan gelen anlik ham+turetilmis
        degerleri hem panele hem (spam olmasin diye seyreltilmis) terminale basar."""
        uri = f"{self.server}/debug/{self.session_id}"
        tick = 0
        async with websockets.connect(uri) as ws:
            print(f"[{self.session_id}] /debug dinleniyor (yalnizca test amacli kanal)")
            while not self._stop_event.is_set():
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                data = json.loads(msg)
                self.debug_ready.emit(data)
                tick += 1
                if tick % 3 == 0:  # ~1sn'de bir terminale bas, panel her karede zaten guncel
                    print(f"[{self.session_id}] DEBUG: {json.dumps(data, ensure_ascii=False)}")


class MainWindow(QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.setWindowTitle(f"cv-pipeline canli panel — {args.session_id}")
        self.resize(1400, 800)

        self._latest_debug: dict | None = None
        self._latest_profile: dict | None = None
        self._latest_focus: dict | None = None
        self.qprocesses: dict[str, QProcess] = {}

        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        left = self._build_left_panel()
        right = self._build_aime_panel(args.aime_url)

        layout.addWidget(left, 1)
        layout.addWidget(right, 1)

        self.worker = StreamWorker(args.server, args.session_id, args.camera_index, args.fps)
        self.worker.frame_ready.connect(self._on_frame)
        self.worker.debug_ready.connect(self._on_debug)
        self.worker.profile_ready.connect(self._on_profile)
        self.worker.focus_ready.connect(self._on_focus)
        self.worker.start()

        # Start QTimer to poll docker compose container statuses
        self.docker_timer = QTimer(self)
        self.docker_timer.timeout.connect(self._check_docker_status)
        self.docker_timer.start(3000)

    def _build_left_panel(self) -> QWidget:
        panel = QFrame()
        panel.setFrameShape(QFrame.Shape.StyledPanel)
        vbox = QVBoxLayout(panel)

        self.video_label = QLabel("kamera bekleniyor...")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(320, 240)
        self.video_label.setStyleSheet("background-color: #111; color: #ccc;")
        vbox.addWidget(self.video_label, stretch=3)

        self.metrics_view = QPlainTextEdit()
        self.metrics_view.setReadOnly(True)
        self.metrics_view.setStyleSheet("font-family: Menlo, monospace; font-size: 11px;")
        vbox.addWidget(self.metrics_view, stretch=2)

        # Service Manager Group Box
        self.service_box = QGroupBox("AIme Girl (Unmute) Service Manager")
        self.service_box.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #444;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px 0 3px;
            }
        """)
        s_layout = QVBoxLayout(self.service_box)

        # Control buttons
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start All Services")
        self.start_btn.setStyleSheet("background-color: #1b5e20; color: white; font-weight: bold; padding: 6px; border-radius: 4px;")
        self.start_btn.clicked.connect(self._start_all_services)
        
        self.stop_btn = QPushButton("Stop All Services")
        self.stop_btn.setStyleSheet("background-color: #b71c1c; color: white; font-weight: bold; padding: 6px; border-radius: 4px;")
        self.stop_btn.clicked.connect(self._stop_all_services)
        
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        s_layout.addLayout(btn_layout)

        # Grid for status indicators
        grid = QGridLayout()
        self.status_labels = {}
        services_list = ["llm", "stt", "tts", "backend", "frontend"]
        for i, name in enumerate(services_list):
            lbl_name = QLabel(f"{name.upper()}:")
            lbl_name.setStyleSheet("font-weight: bold;")
            lbl_status = QLabel("STOPPED")
            lbl_status.setStyleSheet("color: #ff5252; font-weight: bold;")
            grid.addWidget(lbl_name, i // 3, (i % 3) * 2)
            grid.addWidget(lbl_status, i // 3, (i % 3) * 2 + 1)
            self.status_labels[name] = lbl_status
            
        s_layout.addLayout(grid)

        # Terminal viewer for active logs
        self.log_viewer = QPlainTextEdit()
        self.log_viewer.setReadOnly(True)
        self.log_viewer.setPlaceholderText("Console logs from backend services...")
        self.log_viewer.setStyleSheet("""
            background-color: #1e1e1e;
            color: #b0bec5;
            font-family: 'Courier New', Courier, monospace;
            font-size: 10px;
            border: 1px solid #333;
        """)
        s_layout.addWidget(self.log_viewer, stretch=1)
        
        vbox.addWidget(self.service_box, stretch=2)

        return panel

    def _build_aime_panel(self, url: str) -> QWidget:
        """AIme_Girl (Unmute) frontend'ini gomen panel. Backend (STT/TTS/LLM)
        GPU/CUDA gerektirdigi icin burada calistirilmiyor; yalnizca frontend'e
        (varsayilan http://localhost:3000, --aime-url ile degistirilebilir)
        baglanan bir web view gosterilir."""
        panel = QFrame()
        panel.setFrameShape(QFrame.Shape.StyledPanel)
        vbox = QVBoxLayout(panel)
        vbox.setContentsMargins(0, 0, 0, 0)

        self.aime_view = QWebEngineView()
        self.aime_view.setUrl(QUrl(url))
        vbox.addWidget(self.aime_view)

        return panel

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

    def _start_all_services(self) -> None:
        self.log_viewer.appendPlainText("Starting all dockerized services (docker compose up -d)...")
        if "docker_compose" in self.qprocesses and self.qprocesses["docker_compose"].state() != QProcess.ProcessState.NotRunning:
            self.log_viewer.appendPlainText("Docker compose build/run is already in progress.")
            return
            
        proc = QProcess(self)
        env = QProcess.systemEnvironment()
        hf_token_path = Path.home() / ".cache" / "huggingface" / "token"
        if hf_token_path.exists():
            cached_token = hf_token_path.read_text().strip()
            env.append(f"HUGGING_FACE_HUB_TOKEN={cached_token}")
            env.append(f"HF_TOKEN={cached_token}")
        proc.setProcessEnvironment(env)
        
        proc.readyReadStandardOutput.connect(self._on_docker_stdout)
        proc.readyReadStandardError.connect(self._on_docker_stderr)
        
        proc.start("docker", ["compose", "up", "-d"])
        self.qprocesses["docker_compose"] = proc

    def _stop_all_services(self) -> None:
        self.log_viewer.appendPlainText("Stopping all dockerized services (docker compose down)...")
        proc = QProcess(self)
        proc.readyReadStandardOutput.connect(self._on_docker_stdout_direct)
        proc.readyReadStandardError.connect(self._on_docker_stderr_direct)
        proc.start("docker", ["compose", "down"])
        proc.waitForFinished(15000)
        self.log_viewer.appendPlainText("Docker compose down completed.")

    def _on_docker_stdout(self) -> None:
        proc = self.qprocesses.get("docker_compose")
        if proc:
            data = proc.readAllStandardOutput().data().decode("utf-8", errors="replace")
            self.log_viewer.appendPlainText(f"[DOCKER] {data.strip()}")

    def _on_docker_stderr(self) -> None:
        proc = self.qprocesses.get("docker_compose")
        if proc:
            data = proc.readAllStandardError().data().decode("utf-8", errors="replace")
            self.log_viewer.appendPlainText(f"[DOCKER ERR] {data.strip()}")

    def _on_docker_stdout_direct(self) -> None:
        sender = self.sender()
        if isinstance(sender, QProcess):
            data = sender.readAllStandardOutput().data().decode("utf-8", errors="replace")
            self.log_viewer.appendPlainText(f"[DOCKER] {data.strip()}")

    def _on_docker_stderr_direct(self) -> None:
        sender = self.sender()
        if isinstance(sender, QProcess):
            data = sender.readAllStandardError().data().decode("utf-8", errors="replace")
            self.log_viewer.appendPlainText(f"[DOCKER ERR] {data.strip()}")

    def _check_docker_status(self) -> None:
        proc = QProcess(self)
        proc.readyReadStandardOutput.connect(lambda p=proc: self._on_status_check_finished(p))
        proc.start("/bin/bash", ["-c", "docker compose ps --format '{{.Service}} {{.State}}'"])

    def _on_status_check_finished(self, proc: QProcess) -> None:
        if proc.state() != QProcess.ProcessState.NotRunning:
            proc.waitForFinished(1000)
        data = proc.readAllStandardOutput().data().decode("utf-8", errors="replace")
        
        states = {
            "llm": "STOPPED",
            "stt": "STOPPED",
            "tts": "STOPPED",
            "backend": "STOPPED",
            "frontend": "STOPPED"
        }
        
        mapping = {
            "llm": "llm",
            "stt": "stt",
            "tts": "tts",
            "aime-backend": "backend",
            "aime-frontend": "frontend"
        }
        
        for line in data.strip().split("\n"):
            parts = line.strip().split()
            if len(parts) >= 2:
                svc, state = parts[0], parts[1]
                key = mapping.get(svc)
                if key:
                    if "running" in state.lower():
                        states[key] = "RUNNING"
                    elif "starting" in state.lower() or "restarting" in state.lower():
                        states[key] = "STARTING"
                        
        for key, state in states.items():
            lbl = self.status_labels.get(key)
            if lbl:
                lbl.setText(state)
                if state == "RUNNING":
                    lbl.setStyleSheet("color: #69f0ae; font-weight: bold;")
                elif state == "STARTING":
                    lbl.setStyleSheet("color: #ffd740; font-weight: bold;")
                else:
                    lbl.setStyleSheet("color: #ff5252; font-weight: bold;")

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override ismi
        self._stop_all_services()
        self.worker.stop()
        self.worker.wait(2000)
        event.accept()


def main() -> None:
    args = parse_args()
    app = QApplication(sys.argv[:1])
    window = MainWindow(args)
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
