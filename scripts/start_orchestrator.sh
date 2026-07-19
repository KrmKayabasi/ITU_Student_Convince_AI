#!/bin/bash
# Launch the İTÜ Convince AI realtime orchestrator (Gemini Live + CV injection).
#
# Requires GOOGLE_API_KEY (Google AI Studio). The CV pipeline (:8000) should be
# running so the orchestrator can subscribe to /profile and /focus.
#
# Usage:
#   GOOGLE_API_KEY=... ./scripts/start_orchestrator.sh

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [ -z "$GOOGLE_API_KEY" ]; then
  echo "ERROR: GOOGLE_API_KEY is not set." >&2
  echo "Get a key from Google AI Studio and export it, e.g.:" >&2
  echo "  export GOOGLE_API_KEY=your_key_here" >&2
  exit 1
fi

cd "$PROJECT_ROOT/backend/orchestrator"

# Optional: pick a venv via ORCH_PYTHON, else fall back to python3.
PYTHON="${ORCH_PYTHON:-python3}"

echo "=== İTÜ Convince AI — Realtime Orchestrator ==="
echo "Model:       ${GEMINI_LIVE_MODEL:-gemini-2.5-flash-native-audio-preview-12-2025}"
echo "API version: ${GEMINI_API_VERSION:-v1alpha}"
echo "CV pipeline: ${CV_PIPELINE_WS_URL:-ws://localhost:8000}"
echo "Port:        ${ORCH_PORT:-8001}"
echo ""

exec "$PYTHON" server.py
