# System Architecture — ITU Student Convince AI

This document provides a technical overview of the components, communication flows, and design decisions of the ITU Student Convince AI system.

---

## 🆕 version1 — Web kiosk (Gemini Live + CV↔LLM + talking face)

> The `version1` branch finalizes the product as a **browser kiosk**. It replaces the cascaded Whisper→Gemma→XTTS speech server with **Google Gemini Live** (native-audio, realtime), wires the CV pipeline into the conversation, and adds an animated talking face. The cascaded/`openai_realtime` speech backends remain behind `SPEECH_PROVIDER` as fallbacks; the PyQt client is demoted to an optional CV-debug tool.

```
BROWSER (Next.js, frontend/src/app/kiosk)         sessionId = crypto.randomUUID()
  mic ─AudioWorklet(pcm16-capture)─16k PCM16─┐
  webcam ─<canvas>.jpeg ~10fps──────────────┼─► CV /stream/{id}   (:8000, unchanged)
  TalkingFace ◄ outputAnalyser RMS (lip-sync)│
  playback worklet ◄ 24k PCM16 ──────────────┤
  CV /focus,/profile ─► avatar reactions      │
        │  WS /v1/realtime?session_id=id  (binary PCM + control JSON)
        ▼
ORCHESTRATOR (backend/orchestrator, FastAPI :8001)   per session_id
  GeminiLiveBridge ─► Gemini Live (gemini-3.1-flash-live-preview, v1alpha:
                       affective_dialog + proactive_audio)
  CvInjector: /profile → inject_context(opener)   /focus → debounced steer + seekAttention
CV PIPELINE (backend/cv_pipeline, :8000) — UNCHANGED (/profile one-shot, /focus ~2.5s)
GATEWAY (nginx :8080) → /=frontend, /api=cv-pipeline, /orch=orchestrator (WS upgrade)
```

**One command:** `GOOGLE_API_KEY=... docker compose up --build` → kiosk at `http://localhost:8080/kiosk`.

| Concern | version1 answer |
|---------|-----------------|
| LLM/voice | Gemini Live native audio (16k PCM16 in / 24k PCM16 out), native VAD + barge-in |
| One-time profile | `CvInjector` formats a Turkish opener hint → `send_client_content` (once, informs greeting) |
| Continuous focus | `CvInjector` debounces `is_focused==False` (≥`FOCUS_LOSS_SECONDS`, `NUDGE_COOLDOWN_SECONDS`) → `send_realtime_input(text=…)` + `{"type":"seekAttention"}` to the avatar |
| Talking face | `TalkingFace` (Canvas-2D, amplitude-driven mouth + expressions) now; `RiveFace` (rigged `.riv`) as a drop-in upgrade |
| API key | server-side only (orchestrator); the browser never sees it |

Details: [`backend/orchestrator/README.md`](../backend/orchestrator/README.md).

---

## 🏛️ Component Overview

The system uses a **client-server architecture** separating lightweight native processing on the client device from heavy LLM/ASR compute on the remote server:

```
┌──────────────────────────────────────────────────────────────┐
│               PyQt6 Desktop Client (Mac/PC)                  │
│                                                              │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐  │
│  │ StreamWorker │  │ AudioCapture │  │ ResponseGenerator  │  │
│  │ (webcam→CV)  │  │ Worker (VAD) │  │ Worker (diar+play) │  │
│  └─────────────┘  └──────────────┘  └────────────────────┘  │
│                                                              │
│  client/desktop_client.py  — GUI orchestrator               │
│  client/workers.py         — Background QThread workers     │
│  client/metrics.py         — CV metrics formatting          │
└────────────┬──────────────────────────┬──────────────────────┘
             │ 1. JPEG frames           │ 3. PCM audio
             │    (WebSocket)           │    (HTTP POST)
             ▼                          ▼
┌──────────────────────────┐  ┌────────────────────────────────┐
│  CV Pipeline Container   │  │  Speech Server (FastAPI :8002) │
│  (FastAPI :8000)         │  │                                │
│                          │  │  Whisper Large v3 Turbo →      │
│  MediaPipe Face/Pose     │  │  Gemma 4 12B →                │
│  ONNX Emotion (1Hz)      │  │  Coqui XTTS v2 (24000 Hz)     │
│  Session State Machine   │  │                                │
│                          │  │  backend/speech_backend/       │
│  backend/cv_pipeline/    │  │  server.py                     │
└──────────────────────────┘  └────────────────────────────────┘
```

---

## 📁 Module Layout

### 1. Computer Vision Pipeline (`backend/cv_pipeline/`)

| File | Role |
|------|------|
| `main.py` | FastAPI app: WebSocket ingest (`/stream`, `/profile`, `/focus`, `/debug`), worker orchestration, session GC, WS token auth |
| `manager.py` | `SessionManager`: thread-safe `session_id → SessionData` registry |
| `session.py` | `SessionData` state machine (IDLE → CALIBRATING → ACTIVE), `RawSignals` dataclass, ring buffers |
| `processing.py` | `FrameSlot` (thread-safe drop-stale frame container), `SignalExtractor` (orchestrates all detectors) |
| `scoring.py` | `update_session()` state transitions, `build_initial_profile()`, `build_focus_payload()`, dynamic confidence |
| `config.py` | Centralized thresholds and constants, all overridable via environment variables |
| `detectors/face.py` | MediaPipe Face Landmarker wrapper (landmarks, blendshapes, head pose from transformation matrix) |
| `detectors/pose.py` | MediaPipe Pose Landmarker wrapper (world + image landmarks, bbox) |
| `detectors/gaze.py` | Eye contact from blendshapes + head pose, blink-aware, gating penalty |
| `detectors/posture.py` | Lean (shoulder_z − hip_z), spine ratio (yaw-invariant), arms crossed detection |
| `detectors/emotion.py` | ONNX emotion classifier (`enet_b0_8_best_vgaf`), background worker thread at ~1Hz, error-resilient |
| `detectors/select.py` | Primary person selection (largest bbox + continuity hysteresis), face-to-pose matching |

**Key design decisions:**
- **Drop-stale frame ingestion**: `FrameSlot` holds exactly one frame; new frames overwrite unprocessed ones. No unbounded queue.
- **VIDEO running mode**: MediaPipe requires strictly increasing timestamps; a monotonic counter is used (not wall-clock `time.monotonic()`).
- **World landmarks for posture**: Image-space z is noisy under monocular perspective; pose signals use metric-scale world coordinates.
- **Dynamic confidence**: All profile confidence values are computed from actual data characteristics, not hardcoded constants.

### 2. Speech Server (`backend/speech_backend/`)

| File | Role |
|------|------|
| `server.py` | FastAPI app: `/chat_stream`, `/reset`, `/last_turn`, `/health` with optional Bearer token auth. **One-pass synthesis**: accumulates full LLM response, then synthesizes entire text at once |
| `config.py` | Unified configuration (shared between server and standalone pipeline), Markdown-stripped system prompt, all env-overridable |
| `model_handler.py` | `GemmaAudioProcessor`: MLX (Mac) and CUDA (Linux) inference paths, streaming generation. Clause splitter: splits only on `. ! ?`, commas/newlines stay inside for prosody, digit-guard prevents number-with-period splits |
| `tts_handler.py` | `OfflineTTSHandler`: **Coqui XTTS v2** (default, 24000 Hz, character-based), Sherpa-ONNX/Piper VITS (22050 Hz, espeak-based), Supertonic, MMS-TTS, dummy (sine wave). Scoped monkey-patches restored after model load |
| `audio_handler.py` | `AudioHandler`: WebRTC noise suppression, Silero VAD, playout with barge-in interruption, SHA-256 verified model download |
| `pipeline.py` | Standalone speech-to-speech pipeline (alternative to client-server mode) |
| `client.py` | Thin client for remote H200 server (audio capture → POST → stream playout) |
| `client_config.py` | Client-side configuration (server URL, auth token, VAD params) |
| `training/train_multimodal.py` | LoRA fine-tuning script for Gemma 4 on Turkish speech datasets (uses `--audio-root` instead of hardcoded paths) |

### 3. Desktop Client (`client/`)

| File | Role |
|------|------|
| `desktop_client.py` | `MainWindow` (PyQt6): panel builders, chat bubbles, Docker controls, HTTP health polling (~430 lines) |
| `workers.py` | `PipelineLoaderWorker`, `AudioCaptureWorker` (thread-safe), `ResponseGeneratorWorker` (reads `X-Sample-Rate` header, byte-aligned reassembly), `StreamWorker` (auth token support) |
| `metrics.py` | `format_metrics()`: renders live CV metrics as a human-readable text block |

### 4. Test Suite (`tests/`)

```
tests/
├── conftest.py              — Shared fixtures (mock objects, sample data)
├── test_config.py           — 22 tests: defaults, env overrides, range validation
├── test_gaze.py             — 15 tests: eye contact, gating, blink, head pose matrix
├── test_posture.py          — 12 tests: lean, spine ratio, arms crossed
├── test_select.py           — 10 tests: face selection, continuity, pose matching
├── test_scoring.py          — 20 tests: update_session, profile, focus, dynamic confidence
├── test_session.py          — 12 tests: state machine, ring buffers, RawSignals
├── test_processing.py       — 10 tests: FrameSlot thread safety, crop_from_bbox
├── test_emotion.py          —  8 tests: worker lifecycle, labels, error handling
├── cv_pipeline/             — 16 integration tests (focus, gaze, posture, multi-kiosk, profile-once)
└── voice_agent/             —  4 tests (pipeline loader, audio capture, speech integration, e2e)
```

**137 tests total. Run with:** `pytest tests/ -v`

---

## 🔧 TTS Architecture — Coqui XTTS v2

The speech server uses **Coqui XTTS v2** as the default TTS engine (overridable via `TTS_MODEL_ID` env var).

### Why XTTS v2?

| Aspect | Piper VITS (espeak-ng) | Coqui XTTS v2 |
|--------|------------------------|---------------|
| Phonemization | espeak-ng → IPA → token IDs → VITS | **Character-based** — text directly |
| Turkish chars | Needs proper espeak-ng data + voice file | **Native** — ğ,ş,ç,ö,ü,ı handled directly |
| Uppercase | Spells letter-by-letter without lowercase | **Auto-normalizes** |
| Markdown chars | Reads `*` as "yıldız" | **Auto-strips** internally |
| Sample rate | 22050 Hz (Piper) | 24000 Hz |
| Hecelenme risk | **HIGH** (3-layer pipeline: text→phoneme→token→audio) | **ZERO** (single-layer: text→audio) |

### Pipeline flow:

```
Gemma 4 12B streaming response
       │
       ▼ clause splitter (. ! ?) — commas/newlines stay inside
accumulated full_response_parts[]
       │
       ▼ " ".join() — single text string
tts.synthesize(full_text)  ← ONE CALL, full context
       │
       ▼ XTTS v2 internally: lowercase → text split → encodec → hifi-gan
24000 Hz float32 waveform
       │
       ▼ stream in 1024-sample chunks → client
```

### Supported TTS backends:

| Backend | `TTS_MODEL_ID` / `TURKISH_TTS_BACKEND` | Sample rate | Requires |
|---------|----------------------------------------|-------------|----------|
| **Coqui XTTS v2** | `xtts` (default) | 24000 Hz | `TTS` package, Python < 3.12 |
| Piper VITS | local path to model dir | 22050 Hz | `sherpa-onnx` |
| Supertonic | `supertonic` | 44100 Hz | `supertonic` package |
| MMS-TTS | HF model ID | 16000 Hz | `transformers` |
| Dummy (sine) | `dummy` | 24000 Hz | None (dev/test) |

---

## 📡 Communication Protocols

### CV Pipeline (Port 8000)

| Endpoint | Direction | Protocol | Description |
|----------|-----------|----------|-------------|
| `/stream/{session_id}` | Client → Server | WebSocket (binary JPEG) | Ingests webcam frames |
| `/profile/{session_id}` | Server → Client | WebSocket (JSON) | One-shot rich behavioral profile |
| `/focus/{session_id}` | Server → Client | WebSocket (JSON) | Periodic focus metrics (~2.5s) |
| `/debug/{session_id}` | Server → Client | WebSocket (JSON) | Raw debug data (~0.3s, dev only) |
| `/health` | — | HTTP GET | Liveness/readiness probe |

### Speech Server (Port 8002)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/chat_stream` | POST | Audio in (float32 PCM), streamed audio out (header reports actual sample rate) |
| `/reset` | POST | Clear conversation history |
| `/last_turn` | GET | Retrieve last user/assistant text pair |
| `/health` | GET | Liveness probe (no auth required) |

---

## 🔒 Authentication

Both servers support **optional token-based auth** (off by default in development).

### Speech Server — Bearer Token

Set `SPEECH_SERVER_TOKEN` in the environment. All endpoints except `/health` require:
```
Authorization: Bearer <token>
```

### CV Pipeline — Query Parameter

Set `CV_PIPELINE_TOKEN` in the environment. All WebSocket endpoints require:
```
ws://host:8000/stream/session-1?token=<token>
```

### Token Validation

Tokens are compared using `secrets.compare_digest()` (constant-time, resistant to timing attacks). Invalid tokens receive HTTP 401 or WebSocket close code 4001.

---

## 🧵 Thread Safety

### FrameSlot
Uses `threading.Lock` for concurrent put (ingest thread) and get (worker thread) on the drop-stale frame container.

### AudioCaptureWorker
All shared mutable state (`is_recording`, `use_vad`, `buffer`) is protected by a `threading.Lock`. Property-based accessors enforce the lock from both the audio callback thread (real-time) and the Qt main thread.

### EmotionWorker
Background thread with its own lock for `_latest_crop`, `_label`, `_scores`. Five consecutive inference failures trigger an automatic reset to neutral to avoid serving stale predictions.

### SessionManager
Uses `threading.Lock` for `get_or_create`, `get`, `remove`, and `gc_stale_sessions`.

---

## 📊 CV Scoring Model

### Score Components
Three behavioral scores (attention, openness, energy) are computed as weighted sums of normalized signals:

```
attention = eye_contact×0.45 + lean_score×0.30 + emotion_energy×0.15 + spine_score×0.10
openness  = eye_contact×0.25 + lean_score×0.20 + emotion_energy×0.20 + spine_score×0.25 + arms_open×0.10
energy    = emotion_energy×0.50 + lean_score×0.20 + spine_score×0.20 + eye_contact×0.10
```

### Dynamic Confidence
All confidence scores are computed from actual measurements:

| Signal | Confidence Source |
|--------|------------------|
| Lean | Sigmoid ramp over sample count (0→1) |
| Eye contact | 0.9 with head pose data, 0.5 without |
| Spine | Physical plausibility check on ratio + tilt |
| Emotion | Top-1 vs top-2 probability margin |

### State Machine
```
IDLE ──(face detected)──→ CALIBRATING ──(3s elapsed)──→ ACTIVE
  ↑                             ↑                          │
  └──(no face > 2s)────────────┴──(new face)──────────────┘
```
- **IDLE**: No person in frame
- **CALIBRATING**: New person detected, collecting lean baseline samples
- **ACTIVE**: Normal analysis — ring buffers active, focus tracking, profile trigger armed

---

## ⚡ Performance Considerations

- **One-pass TTS synthesis**: Full LLM response synthesized at once — no clause-boundary artifacts, no inter-clause silence artifacts, no micro-fade events
- **Byte-aligned client**: `iter_bytes` chunks reassembled at float32 (4-byte) boundaries — immune to TCP fragmentation misalignment
- **Emotion inference at 1 Hz** in a background thread — does not block the 15fps main pipeline
- **Health checks via HTTP** instead of spawning `nc` and `docker compose ps` subprocesses (was 120/min, now ~7/min via httpx + throttled Docker polling)
- **MLX tempfile uses `delete=True`** — auto-cleanup by the OS, no manual `os.unlink` needed
- **Face detection limited to `MAX_NUM_FACES` (default 3)** to bound MediaPipe processing time
- **Pose detection limited to `MAX_NUM_POSES` (default 3)** similarly bounded
- **Client reads `X-Sample-Rate` header** — playback stream created at the exact rate reported by the server (24000 Hz for XTTS, 22050 for Piper)
