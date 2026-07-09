# Setup & Installation Guide — ITU Student Convince AI

Follow these instructions to set up your virtual environments, install dependencies, and launch all components.

---

## 🛠️ Prerequisites

- **Python 3.12** — for CV pipeline and desktop client (project `.venv`, managed by `uv`)
- **Python 3.11** — for the speech server (XTTS v2 requires Python < 3.12). Uses the venv at `Turkish_Speech_to_Speech/venv/`
- **Docker Desktop**: Required to run the CV scoring pipeline container
- **uv**: The fast Python package installer (`curl -LsSf https://astral.sh/uv | sh`)

---

## 🚀 Step-by-Step Installation

### 1. Unified Dependency Setup

We use a single unified virtual environment (`.venv`, Python 3.12) at the root of the project to run the CV pipeline and the PyQt6 client.

```bash
# Clone the repository
git clone <repo-url>
cd ITU_Student_Convince_AI

# Run the unified setup script
./scripts/setup_all.sh
```

*This creates `.venv` in the root, installs all camera, media processing, PyQt6 GUI, and audio dependencies, and installs the local `diarizen` and `pyannote-audio` packages.*

---

## 🏃 Running the Pipeline

Follow this sequence to launch the entire system:

### Step 1: Deploy the Speech Server

The speech server uses the **Python 3.11** venv from the Turkish_Speech_to_Speech reference repo, which has Coqui XTTS v2 installed (XTTS requires Python < 3.12).

**Start the server:**

```bash
cd ITU_Student_Convince_AI

# Start with XTTS v2 (default) — Python 3.11 venv
./scripts/start_cascaded_speech_server.sh
```

The server listens at `http://localhost:8002`. You'll see messages for Whisper, Gemma 4, and XTTS v2 loading.

**Alternative TTS backends (set before starting):**
```bash
# Piper VITS (espeak-based, 22050 Hz):
export TTS_MODEL_ID="vits-piper-tr_TR-fahrettin-medium"
./scripts/start_cascaded_speech_server.sh

# Supertonic (Turkish-native, 44100 Hz):
export TURKISH_TTS_BACKEND="supertonic"
./scripts/start_cascaded_speech_server.sh

# Dummy (sine wave, for testing audio pipeline):
export TURKISH_TTS_BACKEND="dummy"
./scripts/start_cascaded_speech_server.sh
```

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

# Start the server (it reads SPEECH_SERVER_TOKEN from env)
./scripts/start_cascaded_speech_server.sh
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

## 🔧 TTS Configuration

### Coqui XTTS v2 (default)

```bash
# No extra config needed — this is the default
# Uses fahrettin speaker voice (xtts_test.wav in speech_backend/)
# Sample rate: 24000 Hz
./scripts/start_cascaded_speech_server.sh
```

### Piper VITS (sherpa-onnx)

Requires the model directory to be present:

```bash
# Make sure the model is downloaded:
ls backend/speech_backend/vits-piper-tr_TR-fahrettin-medium/

# Switch to Piper:
export TTS_MODEL_ID="vits-piper-tr_TR-fahrettin-medium"
./scripts/start_cascaded_speech_server.sh
```

Piper-specific tuning (optional):
```bash
export SHERPA_TTS_SPEED="1.0"
export SHERPA_TTS_SILENCE_SCALE="0.2"
```

### Supertonic

```bash
export TURKISH_TTS_BACKEND="supertonic"
export SUPERTONIC_VOICE="M1"
export SUPERTONIC_SPEED="1.1"
./scripts/start_cascaded_speech_server.sh
```

---

## 🔧 Configuration Reference

### Speech Backend (`backend/speech_backend/config.py`)

| Variable | Default | Description |
|----------|---------|-------------|
| `TTS_MODEL_ID` | `xtts` | TTS backend: `xtts` (default), Piper path, `supertonic`, `dummy` |
| `TURKISH_TTS_BACKEND` | (empty) | Alt: `supertonic`, `dummy` |
| `GEMMA_MODEL_ID` | `google/gemma-4-e4b-it` | Model ID (server mode: `google/gemma-4-12B-it`) |
| `GEMMA_QUANTIZATION` | `none` | Quantization level: `none`, `int4`, `int8` |
| `DEVICE` | auto | `mps` (Mac), `cuda` (Linux), or `cpu` |
| `SPEECH_SERVER_MODE` | (empty) | When set, defaults to 12B model |
| `ENABLE_THINKING` | `0` | Set to `1` for chain-of-thought |
| `PLAYBACK_INTERRUPTION_MODE` | `both` | `both`, `key_only`, `vad_only`, `none` |
| `SERVER_HOST` | `0.0.0.0` | Bind address |
| `SERVER_PORT` | `8002` | Listen port |
| `SUPERTONIC_VOICE` | `M1` | Supertonic voice name |
| `SUPERTONIC_SPEED` | `1.1` | Supertonic speed factor |
| `SHERPA_TTS_SPEED` | `1.0` | Piper VITS speed factor |
| `SHERPA_TTS_SILENCE_SCALE` | `0.2` | Piper silence scale |

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

137 tests covering gaze, posture, scoring, session state, processing, emotion, config, and CV pipeline integration. See `docs/ARCHITECTURE.md` for the full test breakdown.

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: No module named 'backend'` | Run from project root with `PYTHONPATH=.` or use `uv run` |
| Camera not opening | Check `--camera-index` (try 0 or 1) |
| `Connection refused` to speech server | Verify server is running: `curl http://localhost:8002/health` |
| Docker container won't start | Run `docker compose ps` to check status; check logs with `docker compose logs cv-pipeline` |
| Auth errors (401) | Set `SPEECH_SERVER_TOKEN` or `ITU_AUTH_TOKEN` matching the server, or unset both for dev mode |
| XTTS fails with "No module named 'transformers'" | Must use Python 3.11 venv: `./scripts/start_cascaded_speech_server.sh` |
| XTTS "Python >= 3.9 and < 3.12" error | Python 3.12 not supported by Coqui TTS — use the reference venv (Python 3.11) |
| Hecelenme / spelled letters | With XTTS v2 this should not happen. If it persists, check audio pipeline (sample rate, byte alignment) |
| XTTS slow on first load | Normal — XTTS v2 is ~1.8 GB. Subsequent loads are cached |
