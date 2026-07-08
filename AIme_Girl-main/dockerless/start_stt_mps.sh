#!/bin/bash
set -ex
cd "$(dirname "$0")/.."

# Apple Silicon (Metal/MPS) icin Turkce STT servisi.
#
# faster-whisper'in CTranslate2 motoru MPS desteklemedigi (yalnizca cuda/cpu)
# icin burada Apple'in kendi MLX framework'unu kullanan mlx-whisper backend'i
# secilir (STT_BACKEND=mlx). CUDA/CPU yolu (services/turkish/whisper_stt_server.py
# icindeki FasterWhisperTranscriber, bkz. start_stt.sh) degismeden kalir.
#
# Ayarlanabilir env var'lar:
#   WHISPER_MLX_MODEL   - varsayilan: mlx-community/whisper-large-v3-turbo
#   WHISPER_LANGUAGE    - varsayilan: tr
#   WHISPER_WARMUP_SEC  - ilk istekte gecikme olmasin diye baslangicta isitma suresi

export STT_BACKEND=mlx
export WHISPER_LANGUAGE="${WHISPER_LANGUAGE:-tr}"

uv run --with-requirements services/turkish/requirements-stt.txt --with mlx-whisper -- \
  fastapi run services/turkish/whisper_stt_server.py --host 0.0.0.0 --port 8090
