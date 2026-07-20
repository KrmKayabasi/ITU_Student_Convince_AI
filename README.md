# İTÜ AI Tercih Danışmanı — Student Convince AI

## What is this?

A real-time **browser kiosk** that talks prospective students into choosing İTÜ. A student stands in front of a screen, and **"Elif"** — an animated AI advisor with a hand-crafted SVG face — holds a natural Turkish voice conversation with them:

- 🗣️ **Real-time speech-to-speech** via Google **Gemini Live** (native audio, barge-in, affective dialog)
- 🎥 **Reads body language** — posture, eye contact, emotion — via the CV pipeline (MediaPipe + ONNX)
- 🧠 **CV feeds the conversation**: a one-time behavioral profile shapes Elif's opening line, and continuous focus tracking makes her *re-engage a distracted student* (face + voice together)
- 👩 **Lip-synced talking face** — blinks, breathes, tilts her head while listening, and leans in to grab attention

Built for İTÜ's Computer Engineering department promotion stands.

---

## Quick Start (version1 — Docker)

**Prerequisites:** Docker + Docker Compose, a webcam + mic, Chrome, and a **Google AI Studio API key** ([aistudio.google.com/apikey](https://aistudio.google.com/apikey)).

```bash
# 1. One-time: download CV model files (MediaPipe + emotion ONNX)
bash scripts/fetch_models.sh

# 2. Put your Gemini key in .env (compose reads it automatically)
echo "GOOGLE_API_KEY=your_key_here" > .env

# 3. Bring up the whole stack
docker compose up --build
```

Then open **http://localhost:8080/kiosk** in Chrome, allow camera + mic, and press **"Konuşmaya Başla"**.

**Design preview without any backend:** http://localhost:8080/kiosk?demo=1 — switch Elif's expression states, trigger the attention-grab, and test lip-sync with fake speech audio. (Also works with just `cd frontend && corepack pnpm dev` → http://localhost:3000/kiosk?demo=1.)

### Local development (no Docker for orchestrator/frontend)

```bash
# Terminal 1 — CV pipeline (Docker; needs ./models from step 1)
docker compose up --build cv-pipeline                # :8000

# Terminal 2 — orchestrator (Python 3.11)
python3.11 -m venv .venv-orch && source .venv-orch/bin/activate
pip install -r backend/orchestrator/requirements.txt
export GOOGLE_API_KEY=your_key_here
./scripts/start_orchestrator.sh                      # :8001

# Terminal 3 — kiosk UI
cd frontend && corepack pnpm install && corepack pnpm dev   # :3000/kiosk
```

**Verify Gemini in isolation** (writes a Turkish greeting to `smoke_out.wav`):

```bash
GOOGLE_API_KEY=... python backend/orchestrator/smoke_gemini.py
```

---

## Architecture (version1)

```
BROWSER KIOSK (Next.js, frontend/src/app/kiosk)     sessionId = crypto.randomUUID()
  mic  ─AudioWorklet→ 16k PCM16 ──────────────┐
  webcam ─JPEG ~10fps─────────────────────────┼──► CV PIPELINE (:8000)
  Elif (SVG face + rAF rig) ◄─ lip-sync RMS   │      MediaPipe + ONNX emotion
  playback ◄─ 24k PCM16 ──────────────────────┤      /profile (one-shot)
  CV /focus,/profile ─► avatar reactions      │      /focus   (~2.5s)
        │ WS /v1/realtime (binary PCM + JSON) │
        ▼                                     │
ORCHESTRATOR (:8001, backend/orchestrator)    │
  GeminiLiveBridge ──► Gemini Live (native audio, v1alpha)
  CvInjector ◄────────────────────────────────┘
    • /profile → Turkish opening hint (send_client_content, once)
    • /focus   → debounced re-engage steer + avatar seekAttention
GATEWAY (nginx :8080)  /→frontend  /api→cv-pipeline  /orch→orchestrator
```

- **`GOOGLE_API_KEY` never reaches the browser** — only the orchestrator talks to Gemini.
- Gemini's native VAD handles turn-taking and barge-in; sessions survive connection rotation via **session resumption** + context-window compression.
- Full details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) · [backend/orchestrator/README.md](backend/orchestrator/README.md) · UI spec: [docs/UI_DESIGN.md](docs/UI_DESIGN.md)

---

## Tests

```bash
# CV pipeline + legacy suites (137 tests)
pytest tests/ -v

# Orchestrator (22 tests: audio helpers, opener hints, focus debounce, bridge events)
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/orchestrator -q
```

---

## Configuration (key env vars)

| Variable | Default | What it does |
|----------|---------|-------------|
| `GOOGLE_API_KEY` | — | **Required.** Google AI Studio key for Gemini Live |
| `GEMINI_LIVE_MODEL` | `gemini-3.1-flash-live-preview` | Native-audio realtime model |
| `GEMINI_VOICE` | `Aoede` | Prebuilt voice (female; override if Turkish quality disappoints) |
| `FOCUS_LOSS_SECONDS` | `5` | Sustained focus loss before a re-engage nudge |
| `NUDGE_COOLDOWN_SECONDS` | `20` | Minimum gap between nudges |
| `PROFILE_WAIT_SECONDS` | `3` | Wait for the CV profile before a generic opener |
| `CV_PIPELINE_TOKEN` / `ORCH_TOKEN` | (off) | Optional auth tokens |

Full list: [backend/orchestrator/README.md](backend/orchestrator/README.md) and [backend/cv_pipeline/config.py](backend/cv_pipeline/config.py).

---

## Project Structure

```
backend/
├── cv_pipeline/          FastAPI CV server (MediaPipe, ONNX, session state machine)
│   └── detectors/        Face, pose, gaze, posture, emotion, person selection
├── orchestrator/         Gemini Live bridge + CV→LLM injection (version1 voice stack)
├── speech_backend/       LEGACY cascaded stack (Whisper→Gemma→XTTS) — offline fallback
└── voice_agent/          DiariZen speaker diarisation (used by the legacy desktop client)

frontend/src/app/kiosk/   Kiosk UI: Elif SVG face + rAF rig, realtime/webcam/CV hooks, demo mode
client/                   LEGACY PyQt6 desktop client — now an optional CV debug tool
deploy/nginx.conf         Single-entry gateway (WS-aware)
tests/                    137 CV/voice tests + 22 orchestrator tests
docs/                     Architecture, UI design spec, setup, sprint history
```

---

## Legacy stack (pre-version1)

The original desktop pipeline (PyQt6 client + cascaded Whisper→Gemma 4→Coqui XTTS server on :8002) still exists and works as an **offline fallback** — no cloud API needed:

```bash
./scripts/setup_all.sh                       # Python envs
./scripts/start_cascaded_speech_server.sh    # speech server (:8002, Python 3.11)
uv run python client/desktop_client.py       # PyQt6 desktop app
```

See [docs/SETUP.md](docs/SETUP.md) for its full configuration (TTS backends, `SPEECH_SERVER_TOKEN` auth, DiariZen speaker colors, etc.).

---

## Docs

- **[UI Design Spec](docs/UI_DESIGN.md)** — the "Elif" concept: art direction, layout, motion & face states
- **[Architecture](docs/ARCHITECTURE.md)** — component details, contracts, thread safety, scoring model
- **[Orchestrator](backend/orchestrator/README.md)** — Gemini Live bridge, CV injection, protocol, env vars
- **[Setup & Installation](docs/SETUP.md)** — legacy-stack environments, Docker, troubleshooting
- **[System Prompt](SYSTEM_PROMPT.md)** — the LLM persona: İTÜ advisor decision tree & knowledge base
