#!/bin/bash
# Unified setup script for İTÜ Student Convince AI.
# Handles environment creation and resolves all package dependencies.

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
echo "=== Starting unified dependency setup inside $PROJECT_ROOT ==="

if ! ldconfig -p 2>/dev/null | grep -q "libportaudio"; then
    echo "WARNING: PortAudio was not found. Microphone support requires:" >&2
    echo "  sudo apt install libportaudio2 portaudio19-dev" >&2
fi

# --- 1. Set up Root Environment (FastAPI Backend + PyQt6 GUI + Voice Assistant) ---
echo "--- Step 1: Setting up root virtual environment (.venv) ---"
cd "$PROJECT_ROOT"

# Clean old virtualenv
if [ -d ".venv" ]; then
    echo "Cleaning up existing root .venv..."
    rm -rf .venv
fi

echo "Creating and syncing fresh .venv using uv with Python 3.12..."
uv sync --python 3.12

echo "Root virtual environment setup successfully completed!"

# --- 2. Ensure permissions ---
echo "--- Step 2: Granting executable permissions to all setup/run scripts ---"
chmod +x "$PROJECT_ROOT"/scripts/*.sh

echo "=== Setup completed successfully! ==="
echo "You can now run: "
echo "  1) uv run uvicorn backend.cv_pipeline.main:app --reload --host 0.0.0.0 --port 8000"
echo "  2) uv run python client/desktop_client.py"
