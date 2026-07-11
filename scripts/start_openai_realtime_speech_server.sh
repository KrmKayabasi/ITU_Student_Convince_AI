#!/bin/bash
# Launch the speech server in OpenAI Realtime mode.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [ -x "$PROJECT_ROOT/.venv/bin/python" ]; then
  PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
fi

if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "OPENAI_API_KEY must be set." >&2
  exit 1
fi

export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export SPEECH_PROVIDER="${SPEECH_PROVIDER:-openai_realtime}"
export OPENAI_REALTIME_MODEL="${OPENAI_REALTIME_MODEL:-gpt-realtime-2.1}"
export OPENAI_REALTIME_VOICE="${OPENAI_REALTIME_VOICE:-marin}"
export OPENAI_REALTIME_LANGUAGE="${OPENAI_REALTIME_LANGUAGE:-tr}"

cd "$PROJECT_ROOT/backend/speech_backend"
exec "$PYTHON_BIN" server.py
