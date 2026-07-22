#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NETWORK_NAME="${PODMAN_NETWORK_NAME:-unmute-turkish}"

if [[ -x /usr/bin/nvidia-container-runtime ]]; then
  PODMAN_GPU_ARGS_DEFAULT="--runtime /usr/bin/nvidia-container-runtime -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all"
elif [[ -d /usr/share/containers/oci/hooks.d ]]; then
  PODMAN_GPU_ARGS_DEFAULT="--security-opt=label=disable --hooks-dir=/usr/share/containers/oci/hooks.d"
else
  PODMAN_GPU_ARGS_DEFAULT="--device nvidia.com/gpu=0"
fi

container_names=(
  unmute-frontend
  unmute-backend
  unmute-stt
  unmute-tts
  unmute-llm
)

load_env() {
  if [[ -f "$ROOT_DIR/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$ROOT_DIR/.env"
    set +a
  fi
}

podman_rm_existing() {
  for name in "${container_names[@]}"; do
    podman rm -f "$name" >/dev/null 2>&1 || true
  done
}

ensure_dirs() {
  mkdir -p \
    "$ROOT_DIR/volumes/hf-cache" \
    "$ROOT_DIR/volumes/vllm-cache" \
    "$ROOT_DIR/volumes/turkish-tts-models"
}

ensure_network() {
  podman network exists "$NETWORK_NAME" || podman network create "$NETWORK_NAME" >/dev/null
}

ensure_nvidia_cdi() {
  if [[ -z "${PODMAN_GPU_ARGS:-}" ]]; then
    return
  fi
  if [[ "${PODMAN_GPU_ARGS:-}" == *"--hooks-dir"* || "${PODMAN_GPU_ARGS:-}" == *"nvidia-container-runtime"* ]]; then
    return
  fi
  if [[ "${PODMAN_GPU_ARGS:-}" != *"nvidia.com/gpu"* ]]; then
    return
  fi
  if ! command -v nvidia-ctk >/dev/null 2>&1; then
    cat <<'EOF' >&2
nvidia-ctk was not found. Install/configure NVIDIA Container Toolkit for Podman,
or run CPU-only with PODMAN_GPU_ARGS="" WHISPER_DEVICE=cpu.
EOF
    return
  fi
  if [[ -f /etc/cdi/nvidia.yaml ]]; then
    return
  fi

  if [[ -w /etc ]]; then
    echo "Installing NVIDIA CDI spec into /etc/cdi/nvidia.yaml..." >&2
    mkdir -p /etc/cdi
    nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
  else
    cat <<'EOF' >&2
Podman cannot resolve NVIDIA CDI devices until the CDI spec is installed in /etc/cdi.
Run this once, then rerun this script:

  sudo mkdir -p /etc/cdi
  sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml

For CPU-only testing, run:

  PODMAN_GPU_ARGS="" WHISPER_DEVICE=cpu services/turkish/podman_pipeline.sh up
EOF
    exit 1
  fi
}

build_images() {
  podman build \
    -t unmute-frontend:podman \
    -f "$ROOT_DIR/frontend/hot-reloading.Dockerfile" \
    "$ROOT_DIR/frontend"

  podman build \
    -t unmute-backend:podman \
    --target hot-reloading \
    -f "$ROOT_DIR/Dockerfile" \
    "$ROOT_DIR"

  podman build \
    -t unmute-turkish-stt:podman \
    --build-arg "INSTALL_CUDA_WHEELS=${INSTALL_CUDA_WHEELS:-true}" \
    -f "$ROOT_DIR/services/turkish/stt.Dockerfile" \
    "$ROOT_DIR"

  podman build \
    -t unmute-turkish-tts:podman \
    --build-arg "INSTALL_SUPERTONIC=${INSTALL_SUPERTONIC:-false}" \
    -f "$ROOT_DIR/services/turkish/tts.Dockerfile" \
    "$ROOT_DIR"
}

up() {
  load_env
  LLM_MODEL="${KYUTAI_LLM_MODEL:-google/gemma-4-E2B-it}"
  PODMAN_GPU_ARGS="${PODMAN_GPU_ARGS-$PODMAN_GPU_ARGS_DEFAULT}"
  ensure_dirs
  ensure_network
  ensure_nvidia_cdi
  build_images
  podman_rm_existing

  local gpu_args=()
  if [[ -n "$PODMAN_GPU_ARGS" ]]; then
    # Intentionally uses shell splitting so PODMAN_GPU_ARGS can contain multiple args.
    # Set PODMAN_GPU_ARGS="" to disable GPU device flags.
    # shellcheck disable=SC2206
    gpu_args=($PODMAN_GPU_ARGS)
  fi

  podman run -d \
    --name unmute-llm \
    --network "$NETWORK_NAME" \
    --network-alias llm \
    "${gpu_args[@]}" \
    -e "HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN:-}" \
    -v "$ROOT_DIR/volumes/hf-cache:/root/.cache/huggingface:Z" \
    -v "$ROOT_DIR/volumes/vllm-cache:/root/.cache/vllm:Z" \
    docker.io/vllm/vllm-openai:latest \
    --model="$LLM_MODEL" \
    --max-model-len=2048 \
    --dtype=auto \
    --gpu-memory-utilization=0.70 \
    --max-num-seqs=64 \
    --enable-prefix-caching

  podman run -d \
    --name unmute-stt \
    --network "$NETWORK_NAME" \
    --network-alias stt \
    "${gpu_args[@]}" \
    -p 8090:8080 \
    -e "HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN:-}" \
    -e "WHISPER_MODEL=${WHISPER_MODEL:-Systran/faster-whisper-large-v3}" \
    -e "WHISPER_DEVICE=${WHISPER_DEVICE:-cuda}" \
    -e "WHISPER_COMPUTE_TYPE=${WHISPER_COMPUTE_TYPE:-int8_float16}" \
    -e "WHISPER_CPU_COMPUTE_TYPE=${WHISPER_CPU_COMPUTE_TYPE:-int8}" \
    -e "WHISPER_FALLBACK_TO_CPU=${WHISPER_FALLBACK_TO_CPU:-true}" \
    -e "WHISPER_WARMUP_SEC=${WHISPER_WARMUP_SEC:-1.0}" \
    -e "WHISPER_LANGUAGE=${WHISPER_LANGUAGE:-tr}" \
    -e "WHISPER_BEAM_SIZE=${WHISPER_BEAM_SIZE:-1}" \
    -e "STT_SPEECH_RMS_THRESHOLD=${STT_SPEECH_RMS_THRESHOLD:-0.008}" \
    -e "STT_END_SILENCE_SEC=${STT_END_SILENCE_SEC:-0.35}" \
    -e "STT_NOISE_FLOOR_ALPHA=${STT_NOISE_FLOOR_ALPHA:-0.02}" \
    -e "STT_SPEECH_FACTOR=${STT_SPEECH_FACTOR:-3.0}" \
    -e "STT_SPEECH_OFF_FACTOR=${STT_SPEECH_OFF_FACTOR:-1.5}" \
    -e "STT_PREROLL_SEC=${STT_PREROLL_SEC:-0.2}" \
    -v "$ROOT_DIR/volumes/hf-cache:/root/.cache/huggingface:Z" \
    unmute-turkish-stt:podman

  podman run -d \
    --name unmute-tts \
    --network "$NETWORK_NAME" \
    --network-alias tts \
    -p 8089:8080 \
    -e "HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN:-}" \
    -e "TURKISH_TTS_BACKEND=${TURKISH_TTS_BACKEND:-sherpa}" \
    -e "SHERPA_TTS_MODEL_DIR=/models/sherpa-tts" \
    -e "SHERPA_TTS_MODEL_URL=${SHERPA_TTS_MODEL_URL:-https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-tr_TR-dfki-medium.tar.bz2}" \
    -e "SHERPA_TTS_PROVIDER=${SHERPA_TTS_PROVIDER:-cpu}" \
    -e "SHERPA_TTS_NUM_THREADS=${SHERPA_TTS_NUM_THREADS:-2}" \
    -e "SHERPA_TTS_SPEED=${SHERPA_TTS_SPEED:-1.05}" \
    -e "TTS_CHUNK_WORDS=${TTS_CHUNK_WORDS:-8}" \
    -e "SUPERTONIC_LANG=${SUPERTONIC_LANG:-tr}" \
    -e "SUPERTONIC_VOICE=${SUPERTONIC_VOICE:-M1}" \
    -e "SUPERTONIC_STEPS=${SUPERTONIC_STEPS:-5}" \
    -e "SUPERTONIC_SPEED=${SUPERTONIC_SPEED:-1.1}" \
    -v "$ROOT_DIR/volumes/turkish-tts-models:/models:Z" \
    -v "$ROOT_DIR/volumes/hf-cache:/root/.cache/huggingface:Z" \
    unmute-turkish-tts:podman

  podman run -d \
    --name unmute-backend \
    --network "$NETWORK_NAME" \
    --network-alias backend \
    -p 8000:80 \
    -e KYUTAI_STT_URL=ws://stt:8080 \
    -e KYUTAI_TTS_URL=ws://tts:8080 \
    -e KYUTAI_LLM_URL=http://llm:8000 \
    -e "KYUTAI_LLM_MODEL=$LLM_MODEL" \
    -e "KYUTAI_LLM_MAX_TOKENS=${KYUTAI_LLM_MAX_TOKENS:-160}" \
    -e "UNMUTE_DEFAULT_LANGUAGE=${UNMUTE_DEFAULT_LANGUAGE:-tr}" \
    -e "KYUTAI_LLM_TOP_P=${KYUTAI_LLM_TOP_P:-0.9}" \
    -e "KYUTAI_LLM_REPETITION_PENALTY=${KYUTAI_LLM_REPETITION_PENALTY:-1.1}" \
    -e "NEWSAPI_API_KEY=${NEWSAPI_API_KEY:-}" \
    -v "$ROOT_DIR/unmute:/app/unmute:Z" \
    unmute-backend:podman

  podman run -d \
    --name unmute-frontend \
    --network "$NETWORK_NAME" \
    --network-alias frontend \
    -p 3000:3000 \
    -e NEXT_PUBLIC_IN_DOCKER=false \
    -v "$ROOT_DIR/frontend/src:/app/src:Z" \
    unmute-frontend:podman

  cat <<'EOF'
Started Turkish Unmute pipeline with Podman.

Open:
  http://localhost:3000

Useful logs:
  services/turkish/podman_pipeline.sh logs backend
  services/turkish/podman_pipeline.sh logs stt
  services/turkish/podman_pipeline.sh logs tts
  services/turkish/podman_pipeline.sh logs llm

Stop:
  services/turkish/podman_pipeline.sh down
EOF
}

down() {
  podman_rm_existing
}

logs() {
  local service="${1:-}"
  if [[ -z "$service" ]]; then
    podman logs -f unmute-frontend &
    podman logs -f unmute-backend &
    podman logs -f unmute-stt &
    podman logs -f unmute-tts &
    podman logs -f unmute-llm &
    wait
  else
    podman logs -f "unmute-$service"
  fi
}

wait_ready() {
  local timeout_sec="${1:-600}"
  local start
  start="$(date +%s)"
  while true; do
    if podman exec unmute-llm python3 -c 'import urllib.request; urllib.request.urlopen("http://127.0.0.1:8000/v1/models", timeout=3)' >/dev/null 2>&1; then
      podman exec unmute-llm python3 -c 'import json, urllib.request; models=json.load(urllib.request.urlopen("http://127.0.0.1:8000/v1/models", timeout=5)); model=models["data"][0]["id"]; payload=json.dumps({"model": model, "messages": [{"role": "system", "content": "Türkçe kısa cevap ver."}, {"role": "user", "content": "Merhaba"}], "max_tokens": 8, "temperature": 0.3}).encode(); urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions", data=payload, headers={"Content-Type": "application/json"}), timeout=60).read()' >/dev/null 2>&1 || true
      echo "LLM is ready."
      return 0
    fi
    if (( $(date +%s) - start >= timeout_sec )); then
      echo "Timed out waiting for LLM readiness after ${timeout_sec}s." >&2
      return 1
    fi
    sleep 5
  done
}

case "${1:-up}" in
  up)
    up
    ;;
  down)
    down
    ;;
  logs)
    logs "${2:-}"
    ;;
  wait)
    wait_ready "${2:-600}"
    ;;
  *)
    echo "Usage: $0 [up|down|wait [seconds]|logs [frontend|backend|stt|tts|llm]]" >&2
    exit 2
    ;;
esac
