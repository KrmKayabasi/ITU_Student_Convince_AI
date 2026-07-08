FROM python:3.12-slim

ARG INSTALL_CUDA_WHEELS=true

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    LD_LIBRARY_PATH=/usr/local/lib/python3.12/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.12/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH}

RUN apt-get update && apt-get install -y \
    curl \
    libgomp1 \
    --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

COPY services/turkish/requirements-stt.txt services/turkish/requirements-stt-cuda.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements-stt.txt && \
    if [ "$INSTALL_CUDA_WHEELS" = "true" ]; then pip install --no-cache-dir -r /tmp/requirements-stt-cuda.txt; fi

WORKDIR /app
COPY unmute ./unmute
COPY services ./services

EXPOSE 8080
CMD ["fastapi", "run", "services/turkish/whisper_stt_server.py", "--host", "0.0.0.0", "--port", "8080"]
