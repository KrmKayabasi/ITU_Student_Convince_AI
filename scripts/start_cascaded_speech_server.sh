#!/bin/bash
# Script to launch the integrated cascaded speech-to-speech server.
# Uses Coqui XTTS v2 (character-based, Turkish-native, zero espeak-ng).
# Requires the Python 3.11 venv from Turkish_Speech_to_Speech reference repo.

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REFERENCE_VENV="/Users/baydogan/Documents/ComputerScience/Projects/Turkish_Speech_to_Speech/venv"

echo "=== Bootstrapping Cascaded Speech-to-Speech Server ==="
echo "Project Root: $PROJECT_ROOT"
echo "Python venv:  $REFERENCE_VENV"
echo "TTS Backend:  Coqui XTTS v2 (24000 Hz, Turkish-native)"
echo ""

cd "$PROJECT_ROOT/backend/speech_backend"

# Ensure backend.* imports resolve
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Force XTTS v2 backend
export TTS_MODEL_ID="xtts"

exec "$REFERENCE_VENV/bin/python3" server.py
