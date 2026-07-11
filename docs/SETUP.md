# Setup & Installation Guide — ITU Student Convince AI

Follow these instructions to set up your virtual environments, install dependencies, and launch all components.

---

## 🛠️ Prerequisites

- **Python 3.12** — for CV pipeline and desktop client (project `.venv`, managed by `uv`)
- **Python 3.11+** — for the lightweight OpenAI Realtime speech server if running without a container
- **Docker or Podman**: Required to run the containerized services
- **uv**: The fast Python package installer (`curl -LsSf https://astral.sh/uv | sh`)
- **OpenAI API key** with access to `gpt-realtime-2.1`
- **Linux audio library** for microphone I/O:
  `sudo apt install libportaudio2 portaudio19-dev`

---

## 🚀 Step-by-Step Installation

### 1. Unified Dependency Setup

We use a single unified virtual environment (`.venv`, Python 3.12) at the root of the project to run the CV pipeline and the PyQt6 client.

```bash
# Clone the repository
git clone <repo-url>
cd ITU_Student_Convince_AI

# Sync the unified environment with Python 3.12
uv sync --python 3.12

# Or run the setup helper, which does the same sync and fixes script permissions
./scripts/setup_all.sh
```

*This creates `.venv` in the root, installs all camera, media processing, PyQt6 GUI, and audio dependencies, and installs the local `diarizen` and `pyannote-audio` packages.*

---

## 🏃 Running the Pipeline

Follow this sequence to launch the entire system:

### Step 1: Deploy the Speech Server

The default speech server uses **OpenAI Realtime** (`gpt-realtime-2.1`) and keeps the same local desktop VAD/diarisation flow. It does not need local Whisper, Gemma, XTTS, CUDA, or a Hugging Face cache.

**Start with plain Podman:**

```bash
cd ITU_Student_Convince_AI
export OPENAI_API_KEY=sk-...
export SPEECH_SERVER_TOKEN=$(openssl rand -hex 32)  # optional but recommended

podman build -f backend/speech_backend/Dockerfile.server -t itu-speech-realtime backend/speech_backend
podman run --rm -p 8002:8002 \
    -e OPENAI_API_KEY \
    -e SPEECH_SERVER_TOKEN \
    --name itu-speech-server \
    itu-speech-realtime
```

**Start with Podman Compose if you have a compose provider:**

```bash
cd ITU_Student_Convince_AI
export OPENAI_API_KEY=sk-...
export SPEECH_SERVER_TOKEN=$(openssl rand -hex 32)  # optional but recommended

podman compose -f backend/speech_backend/docker-compose.server.yml up --build
```

If your Podman install uses the Docker-compatible plugin, `docker compose -f backend/speech_backend/docker-compose.server.yml up --build` also works.

**Start without a container:**

```bash
cd ITU_Student_Convince_AI
source .venv/bin/activate
pip install -r backend/speech_backend/requirements_server.txt
export OPENAI_API_KEY=sk-...
./scripts/start_openai_realtime_speech_server.sh
```

The server listens at `http://localhost:8002`. `/health` reports `provider=openai_realtime` and `model=gpt-realtime-2.1`.

### Step 2: Start the CV Ingestion Backend

Run the Docker container on your local client machine to process webcam frames:

```bash
# Start the Docker container on the client device
docker compose up -d
```

The scoring service runs at `http://localhost:8000`.

Alternatively, you can click the **"Start CV Pipeline"** button inside the Desktop Client GUI.

### Step 3: Launch the PyQt6 Desktop Client

```bash
# Local speech server:
uv run python client/desktop_client.py

# Local speech server with SPEECH_SERVER_TOKEN enabled:
uv run python client/desktop_client.py --auth-token "$SPEECH_SERVER_TOKEN"

# Remote speech server with auth:
uv run python client/desktop_client.py \
    --speech-server http://<SERVER-IP>:8002 \
    --auth-token YOUR_TOKEN
```

*What this does:*
1. Opens the camera display with live gaze/posture/emotion tracking
2. Captures user voice boundaries using Silero VAD (sherpa-onnx)
3. Runs DiariZen speaker diarisation locally
4. Posts audio turns to the speech server
5. Reads `X-Sample-Rate` header and creates playback stream at the correct rate
6. Plays back the streamed audio response with byte-aligned reassembly

### Step 4: Standalone Speech Client (Terminal-Only)

If you don't need the GUI and just want to talk to the AI from the terminal:

```bash
cd ITU_Student_Convince_AI
source .venv/bin/activate
export SPEECH_SERVER_TOKEN=YOUR_TOKEN   # if auth is enabled on the server
python backend/speech_backend/client.py
```

---

## 🔒 Configuring Authentication

Authentication is **off by default** for development convenience. To enable it:

### Speech Server

```bash
# Generate a random token
export SPEECH_SERVER_TOKEN=$(openssl rand -hex 32)
export OPENAI_API_KEY=sk-...

# Start the server (it reads SPEECH_SERVER_TOKEN from env)
./scripts/start_openai_realtime_speech_server.sh
```

### CV Pipeline

```bash
export CV_PIPELINE_TOKEN=$(openssl rand -hex 32)
docker compose up -d
```

### Desktop Client

```bash
# Option 1: CLI flag
uv run python client/desktop_client.py --auth-token YOUR_TOKEN

# Option 2: Environment variable
export ITU_AUTH_TOKEN=YOUR_TOKEN
uv run python client/desktop_client.py
```

### Standalone Client

```bash
export SPEECH_SERVER_TOKEN=YOUR_TOKEN
python backend/speech_backend/client.py
```

---

## 🔧 Speech Provider Configuration

### OpenAI Realtime (default)

```bash
export SPEECH_PROVIDER=openai_realtime
export OPENAI_API_KEY=sk-...
export OPENAI_REALTIME_MODEL=gpt-realtime-2.1
export OPENAI_REALTIME_VOICE=marin
./scripts/start_openai_realtime_speech_server.sh
```

The server sends manually committed audio turns to Realtime, so local Silero VAD remains the authority for speech boundaries.

### Legacy Cascaded Mode

The old Whisper → Gemma → TTS path is still available for experiments, but it requires the old local model dependencies and GPU setup.

```bash
export SPEECH_PROVIDER=cascaded
./scripts/start_cascaded_speech_server.sh
```

---

## 🔧 Configuration Reference

### Speech Backend (`backend/speech_backend/config.py`)

| Variable | Default | Description |
|----------|---------|-------------|
| `SPEECH_PROVIDER` | `openai_realtime` | `openai_realtime` or legacy `cascaded` |
| `OPENAI_API_KEY` | required | API key for OpenAI Realtime |
| `OPENAI_REALTIME_MODEL` | `gpt-realtime-2.1` | Realtime model |
| `OPENAI_REALTIME_VOICE` | `marin` | Realtime voice |
| `OPENAI_REALTIME_TRANSCRIPTION_MODEL` | `gpt-realtime-whisper` | Input transcript model for `/last_turn` |
| `OPENAI_REALTIME_LANGUAGE` | `tr` | Input transcription language |
| `TTS_MODEL_ID` | `xtts` | Legacy cascaded TTS backend only |
| `TURKISH_TTS_BACKEND` | (empty) | Legacy cascaded TTS alt backend only |
| `GEMMA_MODEL_ID` | `google/gemma-4-e4b-it` | Legacy cascaded model ID |
| `GEMMA_QUANTIZATION` | `none` | Legacy cascaded quantization level |
| `DEVICE` | auto | Legacy cascaded device: `mps`, `cuda`, or `cpu` |
| `PLAYBACK_INTERRUPTION_MODE` | `both` | `both`, `key_only`, `vad_only`, `none` |
| `SERVER_HOST` | `0.0.0.0` | Bind address |
| `SERVER_PORT` | `8002` | Listen port |
| `SUPERTONIC_VOICE` | `M1` | Legacy cascaded Supertonic voice name |
| `SUPERTONIC_SPEED` | `1.1` | Legacy cascaded Supertonic speed factor |
| `SHERPA_TTS_SPEED` | `1.0` | Legacy cascaded Piper VITS speed factor |
| `SHERPA_TTS_SILENCE_SCALE` | `0.2` | Legacy cascaded Piper silence scale |

### CV Pipeline (`backend/cv_pipeline/config.py`)

| Variable | Default | Description |
|----------|---------|-------------|
| `CALIBRATION_SECONDS` | `3.0` | Baseline calibration duration |
| `NO_FACE_TIMEOUT_SECONDS` | `2.0` | Time before IDLE when no face |
| `SESSION_GC_TIMEOUT_SECONDS` | `120.0` | Stale session cleanup |
| `PROFILE_MIN_SAMPLES` | `30` | Eye contact samples before profile generation |
| `FOCUS_EMIT_INTERVAL_SECONDS` | `2.5` | Focus push interval |
| `FOCUS_EYE_CONTACT_THRESHOLD` | `0.5` | Minimum eye contact for "focused" |
| `GAZE_YAW_GATE_DEG` | `25.0` | Yaw threshold for gaze gating |
| `GAZE_PITCH_GATE_DEG` | `20.0` | Pitch threshold for gaze gating |
| `GAZE_GATE_PENALTY` | `0.4` | Penalty multiplier when gated |
| `MAX_NUM_FACES` | `3` | Maximum faces to detect |
| `MAX_NUM_POSES` | `3` | Maximum poses to detect |
| `EMOTION_INFER_HZ` | `1.0` | Emotion inference frequency |

---

## 🧪 Running Tests

```bash
source .venv/bin/activate
pytest tests/ -v
```

Tests cover gaze, posture, scoring, session state, processing, emotion, config, CV pipeline integration, and the Realtime bridge audio conversion layer. See `docs/ARCHITECTURE.md` for the full breakdown.

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: No module named 'backend'` | Run from project root with `PYTHONPATH=.` or use `uv run` |
| Camera not opening | Check `--camera-index` (try 0 or 1) |
| `Connection refused` to speech server | Verify server is running: `curl http://localhost:8002/health` |
| Docker container won't start | Run `docker compose ps` to check status; check logs with `docker compose logs cv-pipeline` |
| Auth errors (401) | Set `SPEECH_SERVER_TOKEN` or `ITU_AUTH_TOKEN` matching the server, or unset both for dev mode |
| OpenAI Realtime connection fails | Check `OPENAI_API_KEY`, outbound WebSocket access, and `OPENAI_REALTIME_MODEL=gpt-realtime-2.1` |
| No user/assistant text in chat bubble | Keep `OPENAI_REALTIME_TRANSCRIPTION_MODEL=gpt-realtime-whisper`; audio still streams even if transcript metadata is missing |
| Garbled audio | Check that `/chat_stream` returns `X-Sample-Rate: 24000` and the client is using the latest byte-aligned playback code |
