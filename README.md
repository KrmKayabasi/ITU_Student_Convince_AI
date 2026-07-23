# İTÜ AI Tercih Danışmanı — Student Convince AI

A real-time **browser kiosk** that talks prospective students into choosing İTÜ. A student stands in front of a screen and talks with **Haru** — an animated Live2D AI advisor — who holds a natural Turkish voice conversation, reads body language, and adapts.

> Built for İTÜ Computer Engineering promotion stands. Runs via Docker or a lightweight Python venv — pick what works for you.

---

## What it does

- 🗣️ **Real-time speech-to-speech** via Google **Gemini Live** (native audio, barge-in, Turkish voice)
- 👩 **Live2D avatar (Haru)** — a real animated character with lip-sync from the assistant's playback audio, blinks, breathing, and eye contact while you speak
- 🎥 **Reads body language** — posture, eye contact, attention — via the CV pipeline (MediaPipe + ONNX)
- 🧠 **CV feeds the conversation** — a one-time behavioral profile shapes Haru's opening line, and continuous focus tracking makes her *re-engage a distracted student* (voice + avatar lean-in together)
- 😊 **Emotion-driven expressions** *(optional)* — the assistant's tone is classified (go_emotions) and mapped to avatar expressions

The avatar is a **swappable renderer**: Live2D (default) or the hand-crafted SVG "Elif" face (`?avatar=svg`), both lip-synced from the same playback-audio RMS.

---

## Quick Start (Docker — recommended)

This is the supported path. Works the same on Linux, macOS, and Windows.

### Prerequisites

1. **Docker + Docker Compose** — [install Docker Desktop](https://www.docker.com/get-started/) (includes Compose). Verify:
   ```bash
   docker --version && docker compose version
   ```
2. **A webcam and microphone** (built-in laptop ones are fine).
3. **Google Chrome** (required — Firefox/Zen block WebGL on localhost).
4. **A Google AI Studio API key** — get one free at [aistudio.google.com/apikey](https://aistudio.google.com/apikey).

### Setup (3 commands)

```bash
# 1. Clone and enter the repo
git clone <repo-url> ITU_Student_Convince_AI
cd ITU_Student_Convince_AI

# 2. One-time: download CV model files (~50 MB; MediaPipe + emotion ONNX)
bash scripts/fetch_models.sh

# 3. Put your Gemini key in .env (Docker Compose reads this automatically)
echo "GOOGLE_API_KEY=your_key_here" > .env
```

### Run

```bash
docker compose up --build
```

**Skip MediaPipe (face detection):** The CV pipeline's face tracker isn't production-ready yet. Run with a mock signal so the avatar still reacts and the orchestrator works normally:

```bash
MOCK_FOCUS=true docker compose up --build
```

This skips MediaPipe model loading entirely and reports a synthetic "person present, focused" signal on `/tracking` and `/focus`.

The first build takes **5–10 minutes** (CV installs OpenCV/MediaPipe, frontend does a full Next.js build). Subsequent starts are fast due to caching.

Watch the logs for these ready signals:
- `orchestrator`: `[Orchestrator] key present` + listening on `:8001`
- `cv-pipeline`: `Uvicorn running on http://0.0.0.0:8000`
- `frontend`: `▲ Next.js 15.2.6`

### Open the kiosk

In **Chrome**:
```
http://localhost:8080/kiosk
```

- Chrome asks for **camera + microphone** permission → **Allow**
- Click **"Konuşmaya Başla"**
- Talk to Haru — she listens, looks at you, and replies in Turkish with lip-synced speech

That's it. The Live2D avatar is the default — no query params needed.

---

## Quick Start (Python venv — no Docker)

Prefer this path when you want lighter/faster iteration, can't install Docker, or need to debug Python services directly. Works on **macOS, Linux, and Windows** (PowerShell or Git Bash).

### Prerequisites

1. **Python 3.11+** — check with `python3 --version` (macOS/Linux) or `python --version` (Windows).
2. **Node.js 18+** — [install from nodejs.org](https://nodejs.org/). The installer includes `corepack` for pnpm.
3. **A webcam and microphone** (built-in laptop ones are fine).
4. **Google Chrome** (required — Firefox/Zen block WebGL on `localhost`).
5. **A Google AI Studio API key** — get one free at [aistudio.google.com/apikey](https://aistudio.google.com/apikey).
6. **Linux only** — system libraries for OpenCV:
   ```bash
   sudo apt-get install -y libgl1 libglib2.0-0
   ```
   macOS ships these via system frameworks (no extra packages). Windows wheels bundle everything.

### 1. Clone and enter the repo

```bash
git clone <repo-url> ITU_Student_Convince_AI
cd ITU_Student_Convince_AI
```

### 2. Download CV model files (~50 MB)

**macOS / Linux:**
```bash
bash scripts/fetch_models.sh
```

**Windows (PowerShell):**
```powershell
mkdir -p models
curl -L -o models/face_landmarker.task "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
curl -L -o models/pose_landmarker_full.task "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
curl -L -o models/emotion_enet_b0_8_best_vgaf.onnx "https://raw.githubusercontent.com/sb-ai-lab/EmotiEffLib/main/models/affectnet_emotions/onnx/enet_b0_8_best_vgaf.onnx"
```
Or install [Git Bash](https://gitforwindows.org/) and run `bash scripts/fetch_models.sh`.

### 3. Create the Python virtual environment

**macOS / Linux:**
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r backend/orchestrator/requirements.txt
deactivate
```

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r backend/orchestrator/requirements.txt
deactivate
```

> **Note for Windows users:** If `Activate.ps1` is blocked, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` first, or use Command Prompt with `.venv\Scripts\activate`.

**(Optional)** To enable emotion-driven avatar expressions (~200 MB extra for PyTorch + transformers):
```bash
source .venv/bin/activate         # macOS/Linux
# or: .venv\Scripts\activate       (Windows cmd)
pip install -r backend/orchestrator/requirements-emotion.txt
deactivate
```

### 4. Install frontend dependencies

```bash
cd frontend
corepack enable
corepack pnpm install
cd ..
```

### 5. Create the environment file

**macOS / Linux:**
```bash
echo 'GOOGLE_API_KEY=your_key_here' > .env
```

**Windows (PowerShell):**
```powershell
"GOOGLE_API_KEY=your_key_here" | Out-File -FilePath .env -Encoding utf8
```

> Use your actual key from [aistudio.google.com/apikey](https://aistudio.google.com/apikey).

### Run (4 terminals)

Open four terminal tabs, all from the repo root. The order matters — start CV pipeline first, wait for it to be ready, then start the rest.

---

**Terminal 1 — CV Pipeline** (MediaPipe computer vision, port `:8000`)

**macOS / Linux:**
```bash
source .venv/bin/activate
export MODELS_DIR="$(pwd)/models"

# Production (real MediaPipe face/pose/gaze):
uvicorn backend.cv_pipeline.main:app --host 0.0.0.0 --port 8000

# OR: mock mode — skips MediaPipe, reports "person present, focused"
MOCK_FOCUS=true uvicorn backend.cv_pipeline.main:app --host 0.0.0.0 --port 8000
```

**Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
$env:MODELS_DIR = "$(Get-Location)\models"

# Production:
uvicorn backend.cv_pipeline.main:app --host 0.0.0.0 --port 8000

# OR: mock mode
$env:MOCK_FOCUS = "true"
uvicorn backend.cv_pipeline.main:app --host 0.0.0.0 --port 8000
```

Ready signal: `Uvicorn running on http://0.0.0.0:8000`

---

**Terminal 2 — Orchestrator** (Gemini Live bridge, port `:8001`)

**macOS / Linux:**
```bash
source .venv/bin/activate
export GOOGLE_API_KEY=your_key_here
export CV_PIPELINE_WS_URL=ws://localhost:8000
cd backend/orchestrator
python server.py
```

**Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
$env:GOOGLE_API_KEY = "your_key_here"
$env:CV_PIPELINE_WS_URL = "ws://localhost:8000"
cd backend/orchestrator
python server.py
```

Ready signal: `[Orchestrator] key present` + listening on `:8001`

> If you installed the emotion classifier in step 3, add `export ENABLE_EMOTION=true` (macOS/Linux) or `$env:ENABLE_EMOTION = "true"` (PowerShell).

> **Typing `your_key_here` every time?** Read it from the `.env` file instead:
> ```bash
> # macOS/Linux
> export $(grep -v '^#' .env | xargs)
> # PowerShell
> Get-Content .env | ForEach-Object { if ($_ -match '^([^#].*?)=(.*)') { Set-Item "env:$($matches[1])" $matches[2] } }
> ```

---

**Terminal 3 — Kiosk UI** (Next.js dev server, port `:3000`)

```bash
cd frontend
corepack pnpm dev
```

Ready signal: `▲ Next.js 15.2.6`

---

**Terminal 4 — Camera feed** (optional; only if you aren't using the browser webcam)

If your setup streams frames from a separate camera client (e.g., the PyQt6 desktop client) instead of the browser's `getUserMedia`, you can skip this terminal. The browser-based kiosk sends camera frames through the `useBackendServerUrl` hook automatically — no extra terminal needed.

### Open the kiosk

In **Chrome**, navigate to:
```
http://localhost:3000/kiosk
```

- Chrome asks for **camera + microphone** permission → **Allow**
- Click **"Konuşmaya Başla"**
- Talk to Haru — she listens, looks at you, and replies in Turkish with lip-synced speech

> **Why no nginx gateway?** In development mode the frontend connects directly to the orchestrator (port `8001`) and CV pipeline (port `8000`) — no reverse proxy needed. The nginx gateway is a Docker convenience that collapses everything onto one port; you can still use it locally by installing nginx and pointing `deploy/nginx.conf`'s upstreams to `localhost` instead of Docker service names.

### Stopping

Press `Ctrl+C` in each terminal. The venv stays intact — next time just activate and run.

---

## Try it without any backend (avatar preview — Docker or venv)

To preview the avatar and tune lip-sync/expressions with **no Gemini key, no CV models, no Docker build**:

**Docker:**
```bash
docker compose up --build frontend
```

**venv (no Docker):**
```bash
cd frontend
corepack enable && corepack pnpm install && corepack pnpm dev
```
Then open in Chrome:
```
http://localhost:3000/kiosk?demo=1
```

The left **DEMO MODU** panel lets you: switch face states, pick an emotion, choose a fake audio source (`speech`/`sine`/`silent`) to drive lip-sync, trigger the attention grab, and toggle the avatar renderer (Live2D ↔ SVG).

---

## Avatar: Live2D vs SVG

| Mode | URL | When to use |
|------|-----|-------------|
| **Live2D (default)** | `/kiosk` or `/kiosk?avatar=live2d` | Normal use — real animated character |
| **SVG "Elif"** | `/kiosk?avatar=svg` | Fallback for machines without WebGL, or the original hand-crafted face |

Both renderers consume the **same** audio amplitude source, so lip-sync is identical. Override the default at build time with `NEXT_PUBLIC_AVATAR=svg` if your deployment needs SVG.

Full details, asset layout, model swapping, and license notes: **[docs/LIVE2D.md](docs/LIVE2D.md)**.

---

## Optional: emotion-driven expressions

By default the avatar's expression follows the conversation state (listening / speaking / thinking / concerned). To make it also react to the **emotional tone** of the assistant's words (joy / sadness / anger / surprise / …), enable the emotion classifier — a port of `jaison-core`'s `emotion_roberta` (go_emotions, 28 labels).

This adds **~200 MB** (PyTorch + transformers), so it's off by default.

**Docker:**
```bash
# Bring up the emotion-enabled orchestrator instead of the default one
docker compose up --build orchestrator-emotion cv-pipeline frontend gateway
```

**venv (no Docker):**
```bash
# Install emotion deps into the existing venv
source .venv/bin/activate          # or .venv\Scripts\activate on Windows
pip install -r backend/orchestrator/requirements-emotion.txt
# Then start the orchestrator with the flag set:
export ENABLE_EMOTION=true          # or $env:ENABLE_EMOTION = "true" on PowerShell
cd backend/orchestrator && python server.py
```

Then watch the orchestrator logs for:
```
emotion classifier loaded: model=SamLowe/roberta-base-go_emotions device=cpu
```

> **Gateway routing note:** the nginx gateway routes `/orch/` to the default `orchestrator:8001`. To reach the emotion-enabled instance (`orchestrator-emotion`) through `:8080`, point `deploy/nginx.conf`'s `/orch` upstream at `orchestrator-emotion:8001`. Or test the emotion instance directly on host port `:8002`. Details in [docs/LIVE2D.md](docs/LIVE2D.md).

---

## Architecture (version1)

```
BROWSER KIOSK (Next.js, frontend/src/app/kiosk)     sessionId = crypto.randomUUID()
  mic ── PCM16@16k ──► WS /orch/v1/realtime ──► orchestrator ──► Gemini Live
  Haru (Live2D) ◄─ lip-sync RMS                       (Gemini Live bridge)
  CV /focus,/profile ─► avatar reactions             CV /focus ──► re-engage steer
                                                       + avatar seekAttention
       │                                                  ▲
       └──────────── nginx gateway :8080 ────────────────┘
                (/ → frontend, /api → cv-pipeline, /orch → orchestrator)
```

**Four services** (`docker compose up`):
- **`cv-pipeline`** (:8000) — FastAPI. MediaPipe face/pose/gaze + ONNX emotion. Streams `/focus` + `/profile` over WebSocket.
- **`orchestrator`** (:8001) — FastAPI WebSocket bridge between browser and Gemini Live. Mic audio in → model audio + transcripts out. Subscribes to CV server-side for LLM context injection.
- **`frontend`** (:3000) — Next.js 15 kiosk UI. Live2D avatar, realtime session, webcam, subtitles.
- **`gateway`** (:8080) — nginx. Single entry point; WebSocket-aware.

Data flow for lip-sync: **Gemini audio (24 kHz PCM16) → playback AudioWorklet → AnalyserNode (RMS) → `AmplitudeSource` → Live2D `ParamMouthOpenY`**. The avatar's mouth moves with the assistant's actual voice.

Visitor departure handling: after a face has been observed, the kiosk starts a 5-second timer when tracking reports `absent`. Returning to frame or an `unknown` tracking state cancels the timer. If absence remains continuous and the CV session has reached `IDLE`, the frontend closes the realtime, webcam, and CV connections, then hard-reloads the page to clear conversation context for the next visitor. Starting with an empty frame does not trigger a reload.

Full details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) · [backend/orchestrator/README.md](backend/orchestrator/README.md) · UI spec: [docs/UI_DESIGN.md](docs/UI_DESIGN.md).

---

## Configuration (key env vars)

Docker Compose auto-loads `.env` from the repo root. When using venvs, `export` each variable in your shell or use a `.env` loader. All optional — defaults shown.

| Variable | Default | Purpose |
|----------|---------|---------|
| `GOOGLE_API_KEY` | *(required)* | Gemini Live API key |
| `GEMINI_LIVE_MODEL` | `gemini-3.1-flash-live-preview` | Live model |
| `GEMINI_VOICE` | `Aoede` | Prebuilt voice |
| `ENABLE_EMOTION` | `false` | Avatar emotion classification (adds ~200 MB) |
| `NEXT_PUBLIC_AVATAR` | `live2d` | Default avatar renderer (`live2d` or `svg`) |
| `ORCH_TOKEN` | *(empty)* | Optional bearer token to protect the orchestrator WS |
| `CV_PIPELINE_WS_URL` | `ws://localhost:8000` | CV pipeline WebSocket address |
| `MODELS_DIR` | `models` | Path to MediaPipe/ONNX model files |
| `MOCK_FOCUS` | `false` | Skip MediaPipe inference, return synthetic "focused" signal |
| `SPEAKER_ENABLED` | `true` | TitaNet-Small speaker verification & enrollment |
| `SPEAKER_BARGEIN_ENABLED` | `true` | Speaker-aware barge-in (target speaker ignored during AI playback) |

---

## Local development (no Docker for orchestrator/frontend)

For hacking on the orchestrator or frontend without rebuilding Docker on every change, see the full **[Quick Start (venv — no Docker)](#quick-start-python-venv--no-docker)** section above. Quick reference for the three terminals:

```bash
# Terminal 1 — CV pipeline (:8000)
source .venv/bin/activate && export MODELS_DIR="$(pwd)/models"

# Production (real MediaPipe):
uvicorn backend.cv_pipeline.main:app --host 0.0.0.0 --port 8000

# OR mock mode (no MediaPipe, synthetic "focused" signal):
MOCK_FOCUS=true uvicorn backend.cv_pipeline.main:app --host 0.0.0.0 --port 8000

# Terminal 2 — orchestrator (:8001)
source .venv/bin/activate && export GOOGLE_API_KEY=your_key_here
cd backend/orchestrator && python server.py

# Terminal 3 — kiosk UI (:3000)
cd frontend && corepack pnpm dev
```

**Verify Gemini in isolation** (writes a Turkish greeting to `smoke_out.wav`):
```bash
GOOGLE_API_KEY=... python backend/orchestrator/smoke_gemini.py
```

---

## Tests

```bash
# CV pipeline + legacy suites (137 tests)
pytest tests/

# Orchestrator (22 tests: audio helpers, opener hints, focus debounce, bridge events)
cd backend/orchestrator && pytest
```

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| **Old UI / no Live2D** | You're seeing the SVG face. Live2D is the default — make sure you rebuilt (`docker compose build --no-cache frontend`) and opened `/kiosk` (not a cached tab). |
| **"Live2D yüklenemedi"** | Open in **Chrome**, not Firefox/Zen (they block WebGL on localhost). Check DevTools Console. |
| **Avatar loads but mouth doesn't move** | Confirm mic permission; check orchestrator logs show `key present`. Try `?avatar=svg` to isolate whether it's avatar-specific or an audio pipeline issue. |
| **`WebGL context was lost`** | Use Chrome. Disable tracking protection on localhost. |
| **cv-pipeline build fails on `COPY models/`** | You skipped `bash scripts/fetch_models.sh`. Run it, then rebuild. |
| **Frontend build fails on `frozen-lockfile`** | You edited `package.json` without updating the lockfile. Run `corepack pnpm install` in `frontend/` and commit `pnpm-lock.yaml`. |
| **No audio at all** | Chrome: `chrome://settings/content/microphone` — ensure localhost allowed. Check `GOOGLE_API_KEY` in `.env`. |
| **venv: `ModuleNotFoundError: No module named 'backend'`** | Set `PYTHONPATH` to the repo root: `export PYTHONPATH="$(pwd)"` (macOS/Linux) or `$env:PYTHONPATH = "$(Get-Location)"` (PowerShell). The orchestrator does this automatically; the CV pipeline may need it if launched from outside the repo root. |
| **venv: `MODELS_DIR` is empty / MediaPipe can't find models** | Make sure you ran step 2 (`bash scripts/fetch_models.sh`). Verify `ls models/` shows three files. The CV pipeline expects `MODELS_DIR` to point at the `models/` directory. |
| **venv (Windows): `Activate.ps1` is blocked** | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` in an admin PowerShell, then retry. |
| **venv: `opencv-python-headless` install fails (Linux)** | Install system libraries: `sudo apt-get install -y libgl1 libglib2.0-0`. |
| **venv: orchestrator can't reach CV pipeline** | Make sure CV pipeline is running on `:8000` before starting orchestrator. Check `CV_PIPELINE_WS_URL` is set to `ws://localhost:8000`. |

---

## Project Structure

```
backend/
├── cv_pipeline/          FastAPI CV server (MediaPipe, ONNX, session state machine)
│   └── detectors/        Face, pose, gaze, posture, emotion, person selection
├── orchestrator/         Gemini Live bridge + CV→LLM injection (version1 voice stack)
│   ├── emotion.py        Optional go_emotions classifier → avatar expressions (ENABLE_EMOTION)
│   └── Dockerfile.emotion  Emotion-enabled image (installs torch/transformers)
├── speech_backend/       LEGACY cascaded stack (Whisper→Gemma→XTTS) — offline fallback
└── voice_agent/          DiariZen speaker diarisation (used by the legacy desktop client)

frontend/
  public/live2d/          Cubism Core + bundled Haru model for the Live2D avatar
  src/app/kiosk/          Kiosk UI: Live2D avatar (default) or SVG Elif, realtime/webcam/CV
                          hooks, emotion picker, demo mode

client/                   LEGACY PyQt6 desktop client — now an optional CV debug tool
deploy/nginx.conf         Single-entry gateway (WS-aware)
scripts/fetch_models.sh   Downloads MediaPipe + ONNX models into ./models/
docker-compose.yml        Base stack (cv-pipeline + orchestrator + frontend + gateway)
docker-compose.override.yml  Local overrides: orchestrator-emotion service + frontend :3000 port
docs/                     Architecture, UI design, Live2D, setup, sprint history
```

---

## Legacy stack (pre-version1)

The original desktop pipeline (PyQt6 client + cascaded Whisper→Gemma 4→Coqui XTTS server on :8002) still exists and works as an **offline fallback** — no cloud API needed:

```bash
./scripts/setup_all.sh                       # Python envs
./scripts/start_cascaded_speech_server.sh    # speech server (:8002, Python 3.11)
cd client && python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && python main.py
```

See [docs/SETUP.md](docs/SETUP.md) for the legacy stack. The version1 web kiosk (above) is the recommended path.

---

## Docs

- **[Live2D Avatar](docs/LIVE2D.md)** — Live2D default, swapping to SVG, emotion channel, assets & licenses
- **[UI Design Spec](docs/UI_DESIGN.md)** — art direction, layout, motion & face states
- **[Architecture](docs/ARCHITECTURE.md)** — component details, contracts, thread safety, scoring model
- **[Orchestrator](backend/orchestrator/README.md)** — Gemini Live bridge, CV injection, protocol, env vars
- **[Setup & Installation](docs/SETUP.md)** — legacy-stack environments, Docker, troubleshooting
- **[System Prompt](SYSTEM_PROMPT.md)** — the LLM persona: İTÜ advisor decision tree & knowledge base

---

## License

- **Code**: see repository license.
- **Live2D Cubism Core + Haru model**: © Live2D Inc. Free for small-scale / individual / educational use under the [Live2D license](https://www.live2d.com/en/sdk/license/). Review for commercial use.
- **pixi.js** / **pixi-live2d-display**: MIT.
