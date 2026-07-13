# İTÜ AI Tercih Danışmanı — Student Convince AI

## What is this?

A real-time desktop assistant that helps İTÜ students prepare for advisor interviews. Point a webcam and microphone at them, and the app:

- 🎥 **Tracks their posture, eye contact, and facial expressions** using computer vision
- 🎙️ **Listens to their speech** with local Silero VAD boundaries
- 🧠 **Responds naturally in Turkish** using OpenAI Realtime `gpt-realtime-2.1`
- 👥 **Identifies who is speaking** (even in multi-person rooms) via DiariZen speaker diarisation

Built for İTÜ's Computer Engineering department. Runs on a Mac or Linux PC with a webcam and microphone.

---

## Quick Start

```bash
# 1. Install dependencies (Python 3.12, uv required)
sudo apt install libportaudio2 portaudio19-dev
uv sync --python 3.12

# 2. Start the speech server (terminal 1)
export OPENAI_API_KEY=sk-...
podman build -f backend/speech_backend/Dockerfile.server -t itu-speech-realtime backend/speech_backend
podman run --rm -p 8002:8002 --env-file .env --name itu-speech-server itu-speech-realtime

# 3. Start the desktop app (terminal 2)
uv run python client/desktop_client.py
```

A dark-themed window opens. Click **"Oturumu Başlat"**, then hold **"Hold to Talk"** while speaking. Festival Mode pauses microphone turn detection while GPT is responding, preventing the assistant from hearing itself.

The speech server is lightweight in the default OpenAI Realtime mode and does not need local CUDA, Whisper, Gemma, or TTS models. If you do not use Podman, see [Setup Guide](docs/SETUP.md) for the local Python command.

---

## Architecture

```
┌─────────────────────────────────────────────┐
│          PyQt6 Desktop Client                │
│  Webcam → CV frames    Mic → Silero VAD      │
│  client/desktop_client.py                    │
└──────────┬─────────────────────┬─────────────┘
           │ WebSocket           │ HTTP POST
           ▼                     ▼
┌──────────────────┐  ┌────────────────────────┐
│ CV Pipeline      │  │ Speech Server (:8002)   │
│ (Docker :8000)   │  │                        │
│ MediaPipe + ONNX │  │ OpenAI Realtime bridge │
│ emotion @ 1 Hz   │  │ gpt-realtime-2.1       │
└──────────────────┘  └────────────────────────┘
```

### Speech Pipeline Flow

```
User speaks → WebRTC noise suppression/AGC → Silero VAD → Audio sent via HTTP POST
→ OpenAI Realtime (gpt-realtime-2.1) handles speech understanding, reasoning, and voice
→ PCM audio deltas stream back to the desktop client
```

The desktop-side Silero VAD and DiariZen diarisation remain local. Only the old server-side Whisper → Gemma → TTS cascade is replaced.

### Why OpenAI Realtime?

`gpt-realtime-2.1` removes the separate STT, LLM, and TTS hops. That reduces turn latency, simplifies local deployment, and avoids running large speech/LLM models on the speech server.

The server keeps the same `/chat_stream`, `/last_turn`, `/reset`, and `/health` endpoints, so the PyQt client does not need a protocol rewrite.

---

## Speaking to the AI

**Default Festival Mode:** Click **Oturumu Başlat**, then hold **Hold to Talk**. Capture pauses while the assistant responds so speaker echo cannot create another turn.

**Hands-free mode:** Start with `--no-festival-mode`, or enable **Auto-Talk (VAD)** after starting the session.

**Interrupting the AI:** Click **Interrupt Playout**. Full-duplex voice interruption remains disabled until acoustic echo cancellation is added.

**New visitor:** Click **Yeni Ziyaretçi** to clear the visible conversation and reset that visitor's OpenAI session.

**Group conversation:** Do not press **Yeni Ziyaretçi** when several people are discussing the same topic. Pass push-to-talk between them. DiariZen matches anonymous voice embeddings across turns, keeps stable labels such as Konuşmacı 1 and Konuşmacı 2, and tells GPT which visitor produced each turn while preserving the shared conversation context.

Useful comparison flags:

```bash
# Skip DiariZen model loading and per-turn inference
uv run python client/desktop_client.py --no-diarization

# Disable WebRTC preprocessing for an A/B comparison
uv run python client/desktop_client.py --no-noise-suppression

# Aggressive festival noise suppression (0-3)
AUDIO_NS_LEVEL=3 uv run python client/desktop_client.py
```

**Multiple speakers:** Each person gets a color-coded bubble:
- Speaker 0 → Blue
- Speaker 1 → Teal
- Speaker 2 → Purple

---

## Security

Auth is **off by default** (development mode). To enable:

```bash
# Speech server
export SPEECH_SERVER_TOKEN=$(openssl rand -hex 32)
export OPENAI_API_KEY=sk-...

# Desktop client
uv run python client/desktop_client.py --auth-token YOUR_TOKEN
```

When enabled, all non-health endpoints require the token (constant-time comparison via `secrets.compare_digest`). `/health` remains open for monitoring.

---

## Tests

```bash
source .venv/bin/activate
pytest tests/ -v
```

Tests cover gaze, posture, face selection, scoring, session state, emotion worker, thread safety, config, and the Realtime bridge audio conversion layer.

---

## Project Structure

```
backend/
├── cv_pipeline/          FastAPI CV server (MediaPipe, ONNX, session management)
│   └── detectors/        Face, pose, gaze, posture, emotion, person selection
├── speech_backend/       FastAPI speech server (OpenAI Realtime by default)
│   └── training/         Legacy LoRA fine-tuning experiments for Gemma speech
└── voice_agent/          DiariZen speaker diarisation

client/
├── desktop_client.py     PyQt6 MainWindow (~430 lines)
├── workers.py            Background threads (audio, CV stream, response generation)
└── metrics.py            Live CV metrics display formatting

tests/                    pytest tests across all modules
docs/                     Architecture, setup, sprint documentation
scripts/                  Startup and setup scripts
```

---

## Configuration

All configurable via environment variables. Key ones:

| Variable | Default | What it does |
|----------|---------|-------------|
| `SPEECH_PROVIDER` | `openai_realtime` | Speech backend: OpenAI Realtime by default; `cascaded` keeps the old local path |
| `OPENAI_API_KEY` | required | API key for `gpt-realtime-2.1` |
| `OPENAI_REALTIME_MODEL` | `gpt-realtime-2.1` | Realtime speech model |
| `OPENAI_REALTIME_VOICE` | `marin` | Realtime output voice |
| `SPEECH_SERVER_TOKEN` | (off) | Enables auth on speech server |
| `GEMMA_MODEL_ID` | `google/gemma-4-e4b-it` | Used only when `SPEECH_PROVIDER=cascaded` |
| `DEVICE` | auto | Used only when `SPEECH_PROVIDER=cascaded` |

Full reference in [Setup Guide → Configuration](docs/SETUP.md#-configuration-reference).

---

## Docs

- **[Setup & Installation](docs/SETUP.md)** — Python environments, Docker, auth, all config options, troubleshooting
- **[Architecture](docs/ARCHITECTURE.md)** — Component details, Realtime bridge, thread safety, scoring model
- **[Sprint Doc](docs/SPRINT.md)** — CV pipeline implementation history and methodology
- **[System Prompt](SYSTEM_PROMPT.md)** — The LLM prompt that defines the İTÜ advisor persona
