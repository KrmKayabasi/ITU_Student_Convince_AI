#!/bin/bash
# Script to launch the host-native cascaded speech-to-speech server (Gemma 4 12B + Piper VITS).

set -e

SPEECH_DIR="/Users/baydogan/Documents/ComputerScience/Projects/Turkish_Speech_to_Speech"
echo "=== Bootstrapping Cascaded Speech-to-Speech Server ==="
echo "Project Path: $SPEECH_DIR"

if [ ! -d "$SPEECH_DIR" ]; then
    echo "[Error] Speech-to-speech project directory not found at $SPEECH_DIR!"
    exit 1
fi

cd "$SPEECH_DIR/cascaded_architecture"

# Start the FastAPI server using the dedicated MLX/MPS-enabled virtual environment
echo "Starting server using venv Python..."
exec ../venv/bin/python server.py
