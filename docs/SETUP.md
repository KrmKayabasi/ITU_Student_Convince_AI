# Setup & Installation Guide - ITU Student Convince AI

Follow these instructions to set up virtual environments, build submodules with hardware optimizations, and run the services natively or using Docker.

---

## 🛠️ Prerequisites

*   **Python**: Version 3.12 (highly recommended).
*   **Rust Toolchain**: Required to compile the `moshi-server` native library.
*   **CMake**: Required to build native submodules.
*   **Node.js & pnpm**: Required for the Next.js frontend web UI.
*   **uv**: The fast python package installer (`curl -LsSf https://astral.sh/uv | sh`).

---

## 🚀 Step-by-Step Installation

### 1. Setup the Root (CV Pipeline & Client)
```bash
# Clone the repository (if not already done)
cd ITU_Student_Convince_AI

# Create virtual environment in root
uv venv
source .venv/bin/activate

# Install main requirements
uv pip install -r requirements.txt

# Install camera client requirements (PyQt6)
uv pip install -r requirements-camera.txt
```

### 2. Setup the Voice Agent Backend (`backend/voice_agent/`)
The voice agent backend contains native Rust submodules for real-time audio and diarisation.

```bash
cd backend/voice_agent/

# Clean old virtual environments if any
rm -rf .venv

# Create virtual environment
uv venv
source .venv/bin/activate

# Compile & install requirements (which will compile pyannote-audio and diarizen)
uv pip install -e .
uv pip install pytest
```

#### Hardware Accelerations:
*   **Apple Silicon (macOS)**:
    Start scripts compile the backend with Metal acceleration (`--features metal`).
*   **NVIDIA GPU (Linux)**:
    Start scripts compile the backend with CUDA acceleration (`--features cuda`).

---

## 🏃 Running the Pipeline

### 1. Start the CV Backend
Run the FastAPI scoring server:
```bash
source .venv/bin/activate
uvicorn backend.cv_pipeline.main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Start the Desktop Client (with Services Manager)
Run the PyQt6 interface:
```bash
source .venv/bin/activate
python client/desktop_client.py
```
Inside the PyQt6 window, simply click **"Start All Services"** to start the entire Unmute stack in the background.

---

## 🐳 Docker Stack (Unified Deployment)

You can launch the integrated multi-container services with:
```bash
docker compose up --build
```
*Note: The entire stack is strictly dockerized. You can run all 6 microservices (STT, TTS, LLM, Backend, Frontend, and CV pipeline) inside Docker using this single unified command, or let the PyQt6 desktop client manage them for you.*
