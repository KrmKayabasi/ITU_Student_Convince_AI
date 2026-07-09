# İTÜ AI Tercih Danışmanı — Student Convince AI

## What is this?

A real-time desktop assistant that helps İTÜ students prepare for advisor interviews. Point a webcam and microphone at them, and the app:

- 🎥 **Tracks their posture, eye contact, and facial expressions** using computer vision
- 🎙️ **Listens to their speech** and transcribes it with Whisper
- 🧠 **Responds naturally in Turkish** using Gemma 4 12B and a Turkish-native voice
- 👥 **Identifies who is speaking** (even in multi-person rooms) via DiariZen speaker diarisation

Built for İTÜ's Computer Engineering department. Runs on a Mac or Linux PC with a webcam and microphone.

---

## Quick Start

```bash
# 1. Install dependencies (Python 3.12, uv required)
./scripts/setup_all.sh

# 2. Start the speech server (terminal 1)
./scripts/start_cascaded_speech_server.sh

# 3. Start the desktop app (terminal 2)
uv run python client/desktop_client.py
```

A dark-themed window opens. Click **"Start CV Pipeline"**, then start talking. The AI responds in Turkish with a natural voice.

Speech server needs **Python 3.11** (XTTS v2 requirement). The start script automatically uses the correct Python environment. If you don't have it, see [Setup Guide](docs/SETUP.md).

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
│ MediaPipe + ONNX │  │ Whisper → Gemma 4 12B  │
│ emotion @ 1 Hz   │  │ → Coqui XTTS v2 (TTS)  │
└──────────────────┘  └────────────────────────┘
```

### Speech Pipeline Flow

```
User speaks → Silero VAD detects boundaries → Audio sent via HTTP POST
→ Whisper transcribes (Turkish) → Gemma 4 12B generates response
→ Full response synthesized in ONE PASS by XTTS v2 → Audio streamed back
```

**Why one pass?** The full LLM response is accumulated, then sent to TTS as a single text. This gives the voice model complete sentence context — natural Turkish prosody, no unnatural pauses between fragments.

### Why XTTS v2?

The default TTS engine is **Coqui XTTS v2**, a character-based model that processes Turkish text directly. No intermediate phonemization step (unlike Piper VITS + espeak-ng, which maps text → phonetic symbols → token IDs → audio and can mispronounce individual characters).

| | Piper VITS + espeak-ng | XTTS v2 |
|---|---|---|
| Turkish characters (ğ,ş,ç,ö,ü,ı) | Needs correctly configured espeak data | **Native** |
| Uppercase (İTÜ) | Spells letter-by-letter without preprocessing | **Auto-normalizes** |
| Markdown (`*bold*`) | Reads `*` as "yıldız" | **Auto-strips** |
| Syllable artifacts | Possible (3 conversion layers) | **None** (1 conversion layer) |

To switch back to Piper or try other backends, set `TTS_MODEL_ID` before starting the server. See [Setup Guide](docs/SETUP.md) for details.

---

## Speaking to the AI

**Default mode (hands-free):** Just start talking. The app detects when you start and stop speaking using Silero VAD. After you finish, the AI responds automatically.

**Manual mode:** Uncheck "Auto-Talk (VAD)" and use the **Hold to Talk** button.

**Interrupting the AI:** Click **"Interrupt Playout"** or just speak over it.

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

# Desktop client
uv run python client/desktop_client.py --auth-token YOUR_TOKEN
```

When enabled, all non-health endpoints require the token (constant-time comparison via `secrets.compare_digest`). `/health` remains open for monitoring.

---

## Tests

```bash
source .venv/bin/activate
pytest tests/ -v    # 137 tests, ~5 seconds
```

Tests cover gaze, posture, face selection, scoring, session state, emotion worker, thread safety, and config.

---

## Project Structure

```
backend/
├── cv_pipeline/          FastAPI CV server (MediaPipe, ONNX, session management)
│   └── detectors/        Face, pose, gaze, posture, emotion, person selection
├── speech_backend/       FastAPI speech server (Whisper, Gemma 4, XTTS)
│   └── training/         LoRA fine-tuning for Gemma 4 on Turkish speech
└── voice_agent/          DiariZen speaker diarisation

client/
├── desktop_client.py     PyQt6 MainWindow (~430 lines)
├── workers.py            Background threads (audio, CV stream, response generation)
└── metrics.py            Live CV metrics display formatting

tests/                    137 pytest tests across all modules
docs/                     Architecture, setup, sprint documentation
scripts/                  Startup and setup scripts
```

---

## Configuration

All configurable via environment variables. Key ones:

| Variable | Default | What it does |
|----------|---------|-------------|
| `TTS_MODEL_ID` | `xtts` | TTS engine: `xtts`, Piper path, `supertonic`, `dummy` |
| `SPEECH_SERVER_TOKEN` | (off) | Enables auth on speech server |
| `GEMMA_MODEL_ID` | `google/gemma-4-e4b-it` | Which Gemma model to load |
| `DEVICE` | auto | `mps` (Mac), `cuda`, or `cpu` |

Full reference in [Setup Guide → Configuration](docs/SETUP.md#-configuration-reference).

---

## Docs

- **[Setup & Installation](docs/SETUP.md)** — Python environments, Docker, auth, all config options, troubleshooting
- **[Architecture](docs/ARCHITECTURE.md)** — Component details, TTS comparison, thread safety, scoring model
- **[Sprint Doc](docs/SPRINT.md)** — CV pipeline implementation history and methodology
- **[System Prompt](SYSTEM_PROMPT.md)** — The LLM prompt that defines the İTÜ advisor persona
