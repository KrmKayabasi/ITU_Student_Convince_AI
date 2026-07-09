# İTÜ AI Tercih Danışmanı — Student Convince AI

Welcome to the **İTÜ AI Preference Advisor**. This application is an interactive desktop assistant designed to help you prepare for advisor interviews. It combines real-time **Computer Vision** (tracking your camera feed for focus, eye contact, and posture) with a **Gemma 12B Voice Assistant** that hears you, understands who is speaking using speaker diarisation, and speaks back in Turkish.

---

## 🏛️ Architecture

```
┌──────────────────────────────────────────────────────────┐
│             PyQt6 Desktop Client (Mac/PC)                │
│  Webcam → CV frames        Mic → Silero VAD → DiariZen  │
│  client/desktop_client.py   client/workers.py            │
│  client/metrics.py                                       │
└───────────┬──────────────────────────┬───────────────────┘
            │ JPEG frames (WebSocket)  │ Audio (HTTP POST)
            ▼                          ▼
┌──────────────────────────┐  ┌────────────────────────────┐
│  CV Pipeline (FastAPI)   │  │  Speech Server (FastAPI)   │
│  backend/cv_pipeline/    │  │  backend/speech_backend/   │
│  Port 8000               │  │  Port 8002                 │
│                          │  │  Whisper → Gemma 12B → TTS │
└──────────────────────────┘  └────────────────────────────┘
```

### Key Components

| Component | Directory | Description |
|-----------|-----------|-------------|
| CV Pipeline | `backend/cv_pipeline/` | MediaPipe Face/Pose, ONNX emotion, session state machine |
| Speech Server | `backend/speech_backend/` | Whisper ASR → Gemma 12B → VITS TTS pipeline |
| Desktop Client | `client/` | PyQt6 GUI, Silero VAD, DiariZen diarisation, WebRTC audio |
| Tests | `tests/` | 137 pytest tests covering all modules |

### Client Module Layout

```
client/
├── desktop_client.py   — PyQt6 MainWindow + entry point (~430 lines)
├── workers.py          — Background QThread workers (audio, CV, response gen)
├── metrics.py          — CV metrics text formatting
└── mock_client.py      — Test/demo camera feed generator
```

---

## 🔒 Security

Both servers support **optional token-based authentication**:

| Server | Env Variable | Mechanism |
|--------|-------------|-----------|
| Speech Server (8002) | `SPEECH_SERVER_TOKEN` | `Authorization: Bearer <token>` header |
| CV Pipeline (8000) | `CV_PIPELINE_TOKEN` | `?token=<token>` query parameter on WebSocket |

When the env variable is **not set**, auth is disabled (development mode). When set, all non-health endpoints require the matching token. The `/health` endpoints are always open for monitoring tools.

For the desktop client:
```bash
# Via CLI flag:
uv run python client/desktop_client.py --auth-token YOUR_TOKEN

# Or via environment variable:
export ITU_AUTH_TOKEN=YOUR_TOKEN
```

---

## 🛠️ Step-by-Step Launch Instructions

Make sure **Docker Desktop** is open and running on your computer before starting.

### Prerequisites

- **Python 3.12** (recommended)
- **Docker Desktop**
- **uv** — fast Python package installer: `curl -LsSf https://astral.sh/uv | sh`

### Step 1: Set Up Python Dependencies

```bash
cd ITU_Student_Convince_AI
./scripts/setup_all.sh
```

*What this does:* Creates a `.venv` in the project root and installs all camera, media processing, PyQt6 GUI, and audio dependencies. Also installs the local `diarizen` and `pyannote-audio` packages.

---

### Step 2: Start the Gemma 12B Speech Server

Open a **new terminal window** and run:

```bash
cd ITU_Student_Convince_AI

# Start the speech-to-speech server
./scripts/start_cascaded_speech_server.sh
```

*What to expect:* You will see messages showing **Whisper Large v3 Turbo**, **Gemma 12B**, and the Turkish Voice Synthesizer are loading. Once finished, it will say `All models loaded and ready!` on port `8002`. Keep this terminal open.

**For production:** Set `SPEECH_SERVER_TOKEN` to enable authentication:
```bash
export SPEECH_SERVER_TOKEN=$(openssl rand -hex 32)
```

---

### Step 3: Start the Desktop Application

Open a **third terminal window**:

```bash
cd ITU_Student_Convince_AI
uv run python client/desktop_client.py
```

*What to expect:* A dark-themed application window opens:
- **Left Panel**: Webcam feed, live CV scoring metrics, Docker service controls
- **Right Panel**: Chat history with speaker-color-coded bubbles, voice controls
- **Status Label**: Shows `Status: Idle` once the DiariZen model loads

**For remote speech server:**
```bash
uv run python client/desktop_client.py --speech-server http://<H200-IP>:8002
```

---

### Step 4: Turn on the Camera Scoring Pipeline (Docker)

Inside the left panel of the Desktop Application GUI, click the green **"Start CV Pipeline"** button.
- *What this does:* Starts a Docker container running the CV pipeline (MediaPipe + ONNX) for posture, attention, and facial expression analysis.
- *Verification:* The status indicator turns green and reads `CV_PIPELINE: RUNNING`.

---

## 🎙️ How to Talk with the AI

### Auto-Talk (VAD) Mode (Default)
Simply start speaking into your microphone. The app detects speech boundaries automatically using Silero VAD, runs speaker diarisation, and sends audio to the Gemma server. The advisor's Turkish voice response plays through your speakers.

### Speaker Diarisation Colors
If multiple people speak in the room, each speaker gets a unique color:
- **Speaker 0** → Blue
- **Speaker 1** → Teal
- **Speaker 2** → Purple

### Manual Control
Uncheck **Auto-Talk (VAD)** and use **Hold to Talk** to manually control recording.

### Interrupting
Click the orange **"Interrupt Playout"** button or speak over the assistant in Auto-Talk mode.

---

## 🧪 Running Tests

```bash
source .venv/bin/activate
pytest tests/ -v
```

**137 tests** covering:
- Gaze computation (eye contact, gating, blink softening, head pose)
- Posture calculations (lean, spine ratio, arms crossed)
- Primary person selection (largest face, continuity hysteresis)
- Session state machine (IDLE → CALIBRATING → ACTIVE → IDLE)
- Scoring formulas (update_session, build_profile, build_focus)
- Dynamic confidence computation
- FrameSlot thread safety
- Emotion worker lifecycle and error handling
- Configuration defaults and env-var overrides
- Multi-kiosk session isolation
- Profile-once-per-person guarantee

---

## 📖 Further Reading

- **[System Architecture](docs/ARCHITECTURE.md)** — Component design, communication flows, auth
- **[Setup & Installation Guide](docs/SETUP.md)** — Detailed environment setup, Docker deployment, auth configuration
- **[Sprint Document](docs/SPRINT.md)** — CV pipeline implementation sprint tasks and methodology
- **[System Prompt](SYSTEM_PROMPT.md)** — The LLM prompt driving the İTÜ advisor persona

---

## 🔑 Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `SPEECH_SERVER_TOKEN` | (empty) | Enables Bearer token auth on speech server |
| `CV_PIPELINE_TOKEN` | (empty) | Enables query-param auth on CV WebSocket endpoints |
| `ITU_AUTH_TOKEN` | (empty) | Auth token used by the desktop client |
| `GEMMA_MODEL_ID` | `google/gemma-4-e4b-it` | Model ID for Gemma |
| `GEMMA_QUANTIZATION` | `none` | Quantization: `none`, `int4`, `int8` |
| `DEVICE` | `mps` (Mac) / `cuda` (Linux) | Compute device |
| `SPEECH_SERVER_MODE` | (empty) | When set, default model becomes 12B (server deployment) |
| `CALIBRATION_SECONDS` | `3.0` | CV calibration duration |
| `NO_FACE_TIMEOUT_SECONDS` | `2.0` | Time before IDLE when no face detected |
| `FOCUS_EMIT_INTERVAL_SECONDS` | `2.5` | Focus push interval |
| `PROFILE_MIN_SAMPLES` | `30` | Samples needed before profile is generated |
