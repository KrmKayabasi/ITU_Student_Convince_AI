# İTÜ Convince AI — realtime orchestrator (Gemini Live + CV injection)
FROM python:3.11-slim

WORKDIR /srv/orchestrator

# System deps for scipy/numpy wheels are already provided by manylinux wheels;
# keep the image slim.
COPY backend/orchestrator/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# App code + the shared system persona (config.py looks for ./SYSTEM_PROMPT.md).
COPY backend/orchestrator/ ./
COPY SYSTEM_PROMPT.md ./SYSTEM_PROMPT.md

ENV ORCH_HOST=0.0.0.0 \
    ORCH_PORT=8001

EXPOSE 8001

CMD ["python", "server.py"]
