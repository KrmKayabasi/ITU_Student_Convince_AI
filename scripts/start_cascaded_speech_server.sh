#!/bin/bash
# Script to launch the integrated host-native cascaded speech-to-speech server (Gemma 4 12B + Piper VITS).

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
echo "=== Bootstrapping Cascaded Speech-to-Speech Server ==="
echo "Project Root: $PROJECT_ROOT"

cd "$PROJECT_ROOT/backend/speech_backend"

# Start the FastAPI server using the root unified virtual environment
echo "Starting server using root unified .venv Python..."
exec "../../.venv/bin/python" server.py
