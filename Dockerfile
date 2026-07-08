FROM python:3.11-slim

WORKDIR /srv

# opencv-python-headless'in ihtiyac duydugu sistem kutuphaneleri
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# MediaPipe .task model dosyalari ve EmotiEffLib .onnx agirligi build-time'da
# image'a gomulur (runtime indirme YOK, bkz. scripts/fetch_models.sh).
COPY models/ /srv/models/
ENV MODELS_DIR=/srv/models

COPY backend/cv_pipeline/ /srv/backend/cv_pipeline/
ENV PYTHONPATH=/srv

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "backend.cv_pipeline.main:app", "--host", "0.0.0.0", "--port", "8000"]
