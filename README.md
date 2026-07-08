# ITU Student Convince AI

ITU Student Convince AI is an enterprise-grade, context-aware conversational AI system. It integrates a real-time **Computer Vision (CV) Description Pipeline** (to track user focus, posture, and emotional state) with a natural **Voice Agent** (powered by Moshi speech-to-speech and Gemma 4 LLM) to simulate interactive, real-time mock convince scenarios.

---

## 📂 Repository Structure

The repository is organized following professional software engineering standards to separate backend, frontend, desktop client, and test paths:

```
ITU_Student_Convince_AI/
├── backend/
│   ├── cv_pipeline/         # FastAPI computer vision description pipeline
│   └── voice_agent/         # FastAPI voice agent backend (STT/TTS/LLM)
├── frontend/                # Next.js web avatar interface
├── client/                  # PyQt6 desktop client and webcam ingestion
├── tests/                   # Unified test suite
│   ├── cv_pipeline/         # CV scoring and signal extraction tests
│   └── voice_agent/         # Voice agent and speaker diarisation tests
├── docs/                    # Technical documentation and sprint specifications
├── Dockerfile               # Main container configuration for CV pipeline
└── docker-compose.yml       # Unified orchestrator local stack
```

---

## ⚡ End-to-End Startup Instructions

Follow this step-by-step guide to launch all systems:

### Step 1: Pre-requisites & Setup
Ensure you have **Docker Desktop** installed and running on your machine.
Then, install the unified python dependencies for the scoring backend and desktop client:
```bash
# 1. Create a virtual environment at root
uv venv
source .venv/bin/activate

# 2. Install all required dependencies (webcam, CV, models, PyQt6 GUI)
uv pip install fastapi uvicorn "websockets>=13.1,<17.0" opencv-python opencv-python-headless numpy mediapipe onnxruntime PyQt6 PyQt6-WebEngine pytest
```

### Step 2: Start the CV Ingestion Backend
The FastAPI server processes video streams sent by the client. Run it inside the root directory:
```bash
# Run the FastAPI server (make sure you activated the virtual environment)
uv run uvicorn backend.cv_pipeline.main:app --reload --host 0.0.0.0 --port 8000
```
*The server will start listening at `http://localhost:8000`. It will search for models automatically in the local `./models/` folder.*

### Step 3: Run the PyQt6 Desktop Client
Launch the PyQt6 webcam panel and control board:
```bash
uv run python client/desktop_client.py
```
*A window will open displaying:
- **Left Panel**: Your webcam feed with live scoring overlay (Attention, Posture, and active Emotions).
- **Right Panel**: The AI companion avatar interface.*

### Step 4: Bootstrap the Conversational AI Stack
Inside the PyQt6 Desktop Client UI, click the **"Start All Services"** button at the bottom of the left panel.
This will:
1. Automatically load your Hugging Face credentials (`~/.cache/huggingface/token`).
2. Run `docker compose up -d` in the background to build and start the STT, TTS, LLM (Gemma 4 E2B), Voice Backend, and Frontend containers.
3. Show the live container health indicators (`RUNNING` / `STARTING` / `STOPPED`) on the dashboard using dynamic status checks.

*To stop everything when you're done, simply click **"Stop All Services"** or close the PyQt6 window (which automatically runs `docker compose down` inside the background).*

---

## 📖 Documentation

For detailed guides, please refer to the following documents in the `docs/` folder:

*   **[Setup Guide (docs/SETUP.md)](file:///Users/baydogan/Documents/ComputerScience/Projects/ITU_Student_Convince_AI/docs/SETUP.md)**: Detailed step-by-step instructions on setting up virtual environments, building submodules, compiling Apple Metal/CUDA hardware accelerations, and downloading weights.
*   **[System Architecture (docs/ARCHITECTURE.md)](file:///Users/baydogan/Documents/ComputerScience/Projects/ITU_Student_Convince_AI/docs/ARCHITECTURE.md)**: Detailed specifications of the decoupled CV Description Pipeline, PyQt6 client panel, and Unmute Voice Agent stack.
*   **[Sprint Specification (docs/SPRINT.md)](file:///Users/baydogan/Documents/ComputerScience/Projects/ITU_Student_Convince_AI/docs/SPRINT.md)**: Original sprint design decisions, communication protocols, and timing bounds.
