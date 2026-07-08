#!/bin/bash
set -ex
cd "$(dirname "$0")/.."

# Apple Silicon (Metal/MPS) icin LLM servisi.
#
# vLLM (bkz. start_llm.sh) yalnizca CUDA/ROCm/CPU/TPU destekler, Apple
# Silicon'da calismaz. unmute backend'i (unmute/llm/llm_utils.py) LLM'e
# standart OpenAI client'i ile KYUTAI_LLM_URL uzerinden konusuyor; hangi
# sunucu oldugu onemli degil, OpenAI-uyumlu /v1/chat/completions + /v1/models
# saglayan her sey calisir. mlx-lm paketinin mlx_lm.server modulu bunu Metal
# uzerinden sagliyor.
#
# Backend'i baslatirken modeli acikca verin (autoselect_model() tam olarak
# 1 model bekliyor):
#   KYUTAI_LLM_MODEL=${MLX_LLM_MODEL:-mlx-community/gemma-3-1b-it-4bit} \
#     UNMUTE_DEFAULT_LANGUAGE=tr ./dockerless/start_backend.sh
#
# NOT: mlx-lm 0.31.3, transformers>=5.0.0 istiyor ama transformers 5.13.0 ile
# tokenizer kaydinda kirilan bir uyumsuzluk var (AutoTokenizer.register:
# "'str' object has no attribute '__module__'"). transformers<5'i tek bir
# "uv run --with mlx-lm --with transformers<5" cagrisinda pinlemek uv'nin
# resolver'ini mlx-lm'i cok daha eski (concurrency bayraklari olmayan,
# es zamanli isteklerde "There is no Stream(gpu, 0) in current thread"
# hatasiyla kilitlenen) bir surume dusurmeye zorluyor. Bunun yerine mlx-lm'i
# guncel surumde tutup transformers'i AYRI bir adimda zorla 4.57.6'ya
# indiriyoruz (mlx-lm'in calisma zamani kodu aslinda 5.x'e ihtiyac duymuyor,
# sadece pyproject'teki alt sinir asiri yuksek).
VENV_DIR="$(dirname "$0")/.venv-llm-mps"
uv venv "$VENV_DIR" --python 3.12
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
uv pip install mlx-lm
uv pip install "transformers==4.57.6" --reinstall

mlx_lm.server \
  --model "${MLX_LLM_MODEL:-mlx-community/gemma-3-1b-it-4bit}" \
  --host 0.0.0.0 \
  --port 8091
