#!/usr/bin/env bash
# Build-time model asset fetcher. Downloads MediaPipe Tasks models and the
# EmotiEffLib ONNX emotion weight into ./models/ so the Docker image can
# COPY them in at build-time (no runtime download, see Dockerfile).
#
# URLs point at MediaPipe's official model storage and the upstream
# EmotiEffLib (ex-HSEmotion) GitHub repo. Re-verify against
# https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker
# and https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
# if a download 404s in the future — Google occasionally reshuffles paths.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="${MODELS_DIR:-$SCRIPT_DIR/../models}"
mkdir -p "$MODELS_DIR"

FACE_LANDMARKER_URL="https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
POSE_LANDMARKER_URL="https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
EMOTION_ONNX_URL="https://raw.githubusercontent.com/sb-ai-lab/EmotiEffLib/main/models/affectnet_emotions/onnx/enet_b0_8_best_vgaf.onnx"

fetch() {
  local url="$1" dest="$2"
  if [ -s "$dest" ]; then
    echo "skip (exists): $dest"
    return
  fi
  echo "fetching: $url -> $dest"
  curl -fL --retry 3 --retry-delay 2 -o "$dest.part" "$url"
  mv "$dest.part" "$dest"
}

fetch "$FACE_LANDMARKER_URL" "$MODELS_DIR/face_landmarker.task"
fetch "$POSE_LANDMARKER_URL" "$MODELS_DIR/pose_landmarker_full.task"
fetch "$EMOTION_ONNX_URL" "$MODELS_DIR/emotion_enet_b0_8_best_vgaf.onnx"

echo "models ready in $MODELS_DIR:"
ls -la "$MODELS_DIR"
