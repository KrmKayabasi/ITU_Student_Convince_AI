#!/bin/bash
# Unified setup script for İTÜ Student Convince AI.
# Handles environment creation and resolves all package dependencies.

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
echo "=== Starting unified dependency setup inside $PROJECT_ROOT ==="

# --- 1. Set up Root Environment (FastAPI Backend + PyQt6 GUI) ---
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
# Install requirements with relaxed bounds to prevent websocket 13.1 vs 16.0 conflicts
uv pip install fastapi uvicorn "websockets>=13.1,<17.0" opencv-python opencv-python-headless numpy==1.26.4 mediapipe onnxruntime PyQt6 PyQt6-WebEngine pytest

echo "Root virtual environment setup successfully completed!"
deactivate

# --- 2. Set up Voice Agent Backend Environment (backend/voice_agent/.venv) ---
echo "--- Step 2: Setting up Voice Agent virtual environment (backend/voice_agent/.venv) ---"
cd "$PROJECT_ROOT/backend/voice_agent"

# Clean old virtualenv
if [ -d ".venv" ]; then
    echo "Cleaning up existing voice_agent .venv..."
    rm -rf .venv
fi

echo "Creating fresh voice_agent .venv using uv..."
uv venv

echo "Activating voice_agent .venv..."
source .venv/bin/activate

echo "Installing voice agent and native local packages..."
uv pip install -e .
uv pip install pytest

echo "Voice Agent virtual environment setup successfully completed!"
deactivate

# --- 3. Ensure permissions ---
echo "--- Step 3: Granting executable permissions to all setup/run scripts ---"
chmod +x "$PROJECT_ROOT"/backend/voice_agent/dockerless/*.sh
chmod +x "$PROJECT_ROOT"/scripts/*.sh

echo "=== Setup completed successfully! ==="
echo "You can now run: "
echo "  1) uv run uvicorn backend.cv_pipeline.main:app --reload --host 0.0.0.0 --port 8000"
echo "  2) uv run python client/desktop_client.py"
