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

## ⚡ Quick Start

### 1. Run the CV Scoring Backend
The CV Backend processes camera frames to calculate engagement metrics (attention, posture energy, openness):
```bash
# Create and activate virtual environment in the root
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt

# Run the FastAPI server
uvicorn backend.cv_pipeline.main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Run the PyQt6 Desktop Client (with Integrated Service Manager)
The desktop client opens your webcam, streams video to the CV backend, and embeds the interactive Voice Agent UI. It features a built-in manager to start/stop the entire voice backend container stack (LLM, STT, TTS, Backend, and Web interface) in the background via Docker Compose:
```bash
# Install desktop client requirements
uv pip install -r requirements-camera.txt

# Run the PyQt6 panel
python client/desktop_client.py
```
*Click **"Start All Services"** inside the desktop client UI to automatically bootstrap the entire dockerized voice agent stack using `docker compose up -d` in the background.*

---

## 📖 Documentation

For detailed guides, please refer to the following documents in the `docs/` folder:

*   **[Setup Guide (docs/SETUP.md)](file:///Users/baydogan/Documents/ComputerScience/Projects/ITU_Student_Convince_AI/docs/SETUP.md)**: Detailed step-by-step instructions on setting up virtual environments, building submodules, compiling Apple Metal/CUDA hardware accelerations, and downloading weights.
*   **[System Architecture (docs/ARCHITECTURE.md)](file:///Users/baydogan/Documents/ComputerScience/Projects/ITU_Student_Convince_AI/docs/ARCHITECTURE.md)**: Detailed specifications of the decoupled CV Description Pipeline, PyQt6 client panel, and Unmute Voice Agent stack.
*   **[Sprint Specification (docs/SPRINT.md)](file:///Users/baydogan/Documents/ComputerScience/Projects/ITU_Student_Convince_AI/docs/SPRINT.md)**: Original sprint design decisions, communication protocols, and timing bounds.
