# Setup & Installation Guide - ITU Student Convince AI

Follow these instructions to set up your virtual environments, install speech dependencies, and launch all components natively or using Docker.

---

## 🛠️ Prerequisites

*   **Python**: Version 3.12 (highly recommended).
*   **Docker Desktop**: Required to run the CV scoring pipeline container.
*   **uv**: The fast Python package installer (`curl -LsSf https://astral.sh/uv | sh`).

---

## 🚀 Step-by-Step Installation

### 1. Unified Dependency Setup
We use a single unified virtual environment (`.venv`) at the root of the project to run both the CV pipeline and the PyQt6 client. To set it up and resolve all package constraints automatically:

```bash
# Clone the repository (if not already done)
cd ITU_Student_Convince_AI

# Run the unified setup script
./scripts/setup_all.sh
```
*This will create a fresh `.venv` in the root directory, install all required camera, media processing, and GUI dependencies, and install the local `diarizen` and `pyannote-audio` packages.*

---

## 🏃 Running the Pipeline

Follow this sequence to launch the entire system:

### Step 1: Deploy the Speech Server on the NVIDIA H200 Server
On the remote NVIDIA H200 GPU server, run the Docker Compose command to build and launch the unquantized Gemma 12B, Whisper Large v3 Turbo, and Piper VITS stack with full multi-GPU hardware acceleration:
```bash
cd backend/speech_backend
docker compose -f docker-compose.server.yml up -d --build
```
*The server will start listening at `http://<H200-SERVER-IP>:8002`.*

### Step 2: Start the CV Ingestion Backend on the Client Device
Run the Docker container on your local client machine to process webcam frames:
```bash
# Start the Docker container on the client device in the background
docker compose up -d
```
*The scoring service will run at `http://localhost:8000`.*

### Step 3: Launch the PyQt6 Client Device App
Run the PyQt6 interface on your client device, pointing it to the remote H200 server:
```bash
uv run python client/desktop_client.py --speech-server http://<H200-SERVER-IP>:8002
```
*What this does:*
1. Runs the camera display, gaze/posture tracking (MediaPipe/ONNX), and local speaker diarisation (DiariZen) natively on the client device.
2. Captures user voice boundaries using native-compiled **Rust-bound Silero VAD** (`sherpa-onnx`) to eliminate Python interpreter loop overhead.
3. Automatically posts audio turns to the remote H200 server and plays back the returned 24kHz audio stream.
