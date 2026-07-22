#!/bin/bash
set -ex
cd "$(dirname "$0")/.."
uv tool run vllm@v0.11.0 serve \
  --model=google/gemma-4-E2B-it \
  --max-model-len=2048 \
  --dtype=bfloat16 \
  --gpu-memory-utilization=0.3 \
  --port=8091
