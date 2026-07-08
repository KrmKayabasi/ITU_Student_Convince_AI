FROM python:3.12-slim

ARG INSTALL_SUPERTONIC=false
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

RUN apt-get update && apt-get install -y \
    curl \
    libgomp1 \
    libsndfile1 \
    --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

COPY services/turkish/requirements-tts.txt /tmp/requirements-tts.txt
RUN pip install --no-cache-dir -r /tmp/requirements-tts.txt && \
    if [ "$INSTALL_SUPERTONIC" = "true" ]; then pip install --no-cache-dir supertonic; fi

WORKDIR /app
COPY unmute ./unmute
COPY services ./services

EXPOSE 8080
CMD ["fastapi", "run", "services/turkish/tts_server.py", "--host", "0.0.0.0", "--port", "8080"]
