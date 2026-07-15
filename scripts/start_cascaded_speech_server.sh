#!/bin/bash
# Script to launch the integrated cascaded speech-to-speech server.
# Uses Coqui XTTS v2 (character-based, Turkish-native, zero espeak-ng).
# Requires a Python 3.11 venv (XTTS v2 does not support Python 3.12+).

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# Override with your own path if you keep the 3.11 venv elsewhere:
#   REFERENCE_VENV=/path/to/venv ./scripts/start_cascaded_speech_server.sh
REFERENCE_VENV="${REFERENCE_VENV:-$PROJECT_ROOT/.venv311}"

if [ ! -x "$REFERENCE_VENV/bin/python3" ]; then
    echo "ERROR: No Python 3.11 venv found at $REFERENCE_VENV" >&2
    echo "" >&2
    echo "Create it once with:" >&2
    echo "  python3.11 -m venv $REFERENCE_VENV" >&2
    echo "  $REFERENCE_VENV/bin/pip install -r backend/speech_backend/requirements_server.txt" >&2
    echo "" >&2
    echo "Or point at an existing 3.11 venv:" >&2
    echo "  REFERENCE_VENV=/path/to/venv ./scripts/start_cascaded_speech_server.sh" >&2
    exit 1
fi

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
