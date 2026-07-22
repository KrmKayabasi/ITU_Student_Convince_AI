# İTÜ Convince AI — realtime orchestrator (Gemini Live + CV injection)
FROM python:3.11-slim

WORKDIR /srv/orchestrator

# System deps for scipy/numpy wheels are already provided by manylinux wheels;
# keep the image slim.
COPY backend/orchestrator/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# App code + the shared system persona (config.py looks for ./SYSTEM_PROMPT.md).
COPY backend/orchestrator/ ./

# Shared speaker verification module (used by speaker_manager.py).
COPY backend/speaker/ /srv/speaker/

# Download model files at build time (avoids copy-cache issues).
COPY scripts/download_models.py /tmp/download_models.py
RUN python3 /tmp/download_models.py --project-root /srv && \
    rm /tmp/download_models.py && \
    mkdir -p /srv/profiles && \
    ls -lh /srv/models/

COPY SYSTEM_PROMPT.md ./SYSTEM_PROMPT.md

ENV ORCH_HOST=0.0.0.0 \
    ORCH_PORT=8001 \
    SPEAKER_MODEL_PATH=/srv/models/nemo_en_titanet_small.onnx \
    VAD_MODEL_PATH=/srv/models/silero_vad.onnx \
    SPEAKER_PROFILES_DIR=/srv/profiles

EXPOSE 8001

CMD ["python", "server.py"]