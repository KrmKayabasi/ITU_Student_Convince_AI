#!/bin/bash
# Unified setup script for İTÜ Student Convince AI.
# Handles environment creation and resolves all package dependencies.

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
echo "=== Starting unified dependency setup inside $PROJECT_ROOT ==="

# --- 1. Set up Root Environment (FastAPI Backend + PyQt6 GUI + Voice Assistant) ---
echo "--- Step 1: Setting up root virtual environment (.venv) ---"
cd "$PROJECT_ROOT"

# Clean old virtualenv
if [ -d ".venv" ]; then
    echo "Cleaning up existing root .venv..."
    rm -rf .venv
fi

echo "Creating fresh .venv using uv..."
uv venv

echo "Activating root .venv..."
source .venv/bin/activate

echo "Installing non-conflicting core and UI requirements..."
# Install requirements with relaxed bounds to prevent websocket conflicts
uv pip install fastapi uvicorn "websockets>=13.1,<17.0" opencv-python opencv-python-headless numpy==1.26.4 mediapipe onnxruntime PyQt6 PyQt6-WebEngine pytest

echo "Installing editable voice agent and diarisen speaker diarisation packages..."
uv pip install -e backend/voice_agent

echo "Root virtual environment setup successfully completed!"
deactivate

# --- 2. Ensure permissions ---
echo "--- Step 2: Granting executable permissions to all setup/run scripts ---"
chmod +x "$PROJECT_ROOT"/scripts/*.sh

echo "=== Setup completed successfully! ==="
echo "You can now run: "
echo "  1) uv run uvicorn backend.cv_pipeline.main:app --reload --host 0.0.0.0 --port 8000"
echo "  2) uv run python client/desktop_client.py"
