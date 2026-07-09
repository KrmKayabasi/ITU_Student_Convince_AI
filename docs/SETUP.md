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

### Step 1: Start the host-native Gemma 12B Speech Server
The Speech-to-Speech server runs natively on the host to leverage Apple Silicon GPU (MPS) / Metal acceleration for Gemma 12B and Piper VITS:
```bash
./scripts/start_cascaded_speech_server.sh
```
*The server will start listening at `http://localhost:8002`.*

### Step 2: Start the CV Ingestion Backend (Docker)
The CV Pipeline processes webcam frames sent by the desktop client to calculate attention and engagement metrics:
```bash
# Start the Docker container in the background
docker compose up -d
```
*The scoring service will run at `http://localhost:8000`.*

### Step 3: Start the PyQt6 Desktop Client
Launch the webcam and voice interface:
```bash
uv run python client/desktop_client.py
```
*Click **"Start CV Pipeline"** inside the desktop client GUI to automatically trigger the Docker container if it isn't already running.*
