#!/bin/bash
set -ex
cd "$(dirname "$0")/.."

# Apple Silicon (Metal/MPS) icin Turkce TTS servisi.
#
# services/turkish/tts_server.py zaten SHERPA_TTS_PROVIDER env var'i ile
# provider seciyor; onnxruntime'in macOS wheel'leri "coreml" provider'ini
# destekledigi icin burada Apple GPU/Neural Engine kullanilir. Kod degisikligi
# gerekmiyor, CPU/Linux yolu (bkz. start_tts.sh, docker-compose.turkish.yml)
# degismeden kalir.
#
# coreml provider'i yuklenemezse SHERPA_TTS_PROVIDER=cpu ile tekrar deneyin.
#
# Ayarlanabilir env var'lar:
#   SHERPA_TTS_MODEL_URL, SHERPA_TTS_SPEED, TTS_CHUNK_WORDS, ...
#   (bkz. services/turkish/tts_server.py)

export SHERPA_TTS_PROVIDER="${SHERPA_TTS_PROVIDER:-coreml}"
# SHERPA_TTS_MODEL_DIR varsayilani (/models/sherpa-tts) docker-compose.turkish.yml'in
# volume mount'una gore; dockerless/host'ta /models yazilabilir olmadigi icin
# repo-local, gitignore'lu volumes/ altina yonlendiriyoruz.
export SHERPA_TTS_MODEL_DIR="${SHERPA_TTS_MODEL_DIR:-$(pwd)/volumes/turkish-tts-models}"
mkdir -p "$SHERPA_TTS_MODEL_DIR"

uv run --with-requirements services/turkish/requirements-tts.txt -- \
  fastapi run services/turkish/tts_server.py --host 0.0.0.0 --port 8089
