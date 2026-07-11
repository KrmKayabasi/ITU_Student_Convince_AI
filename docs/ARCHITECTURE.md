# System Architecture — ITU Student Convince AI

This document provides a technical overview of the components, communication flows, and design decisions of the ITU Student Convince AI system.

---

## 🏛️ Component Overview

The system uses a **client-server architecture** separating native client processing from the speech and CV backend services:

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
│                          │  │  OpenAI Realtime bridge        │
│  MediaPipe Face/Pose     │  │  gpt-realtime-2.1             │
│  ONNX Emotion (1Hz)      │  │  24kHz PCM stream              │
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
| `server.py` | FastAPI app: `/chat_stream`, `/reset`, `/last_turn`, `/health` with optional Bearer token auth. Defaults to OpenAI Realtime provider |
| `openai_realtime_bridge.py` | Persistent Realtime WebSocket bridge; resamples 16kHz float32 input to 24kHz PCM16 and streams PCM output back as float32 |
| `config.py` | Unified configuration, Markdown-stripped system prompt, provider switch, all env-overridable |
| `model_handler.py` | Legacy cascaded `GemmaAudioProcessor`: used only with `SPEECH_PROVIDER=cascaded` |
| `tts_handler.py` | Legacy cascaded `OfflineTTSHandler`: used only with `SPEECH_PROVIDER=cascaded` |
| `audio_handler.py` | `AudioHandler`: WebRTC noise suppression, Silero VAD, playout with barge-in interruption, SHA-256 verified model download |
| `pipeline.py` | Standalone speech-to-speech pipeline (alternative to client-server mode) |
| `client.py` | Thin terminal client for the speech server (audio capture → POST → stream playout) |
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

Run with: `pytest tests/ -v`

---

## 🔧 Speech Architecture — OpenAI Realtime

The speech server uses **OpenAI Realtime** (`gpt-realtime-2.1`) by default. This replaces the old server-side Whisper → Gemma → TTS cascade while preserving the desktop client's local Silero VAD and DiariZen diarisation.

### Why Realtime?

| Aspect | Cascaded local path | OpenAI Realtime path |
|--------|---------------------|----------------------|
| Components | Whisper + Gemma + XTTS/Piper | `gpt-realtime-2.1` |
| Local GPU need | High | None on speech server |
| Latency | Sequential STT, LLM, TTS hops | Audio deltas stream from one session |
| Client protocol | `/chat_stream` float32 PCM | Same `/chat_stream` contract |
| VAD/diarisation | Local client | Local client, unchanged |

### Pipeline Flow:

```
Desktop mic → Silero VAD turn boundary
       │
       ▼ DiariZen speaker diarisation for UI speaker labels
float32 PCM 16 kHz POST /chat_stream
       │
       ▼ resample to 24 kHz + PCM16 + input_audio_buffer.commit
OpenAI Realtime gpt-realtime-2.1
       │
       ▼ response.audio.delta PCM chunks
float32 PCM 24 kHz stream → desktop playout
```

### Speech Providers:

| Provider | Env | Requires |
|----------|-----|----------|
| OpenAI Realtime | `SPEECH_PROVIDER=openai_realtime` | `OPENAI_API_KEY`, outbound WebSocket access |
| Legacy cascaded | `SPEECH_PROVIDER=cascaded` | Whisper/Gemma/TTS dependencies and suitable GPU |

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

- **Realtime speech path**: OpenAI Realtime removes the separate STT, LLM, and TTS hops from the default server path
- **Byte-aligned client**: `iter_bytes` chunks reassembled at float32 (4-byte) boundaries — immune to TCP fragmentation misalignment
- **Emotion inference at 1 Hz** in a background thread — does not block the 15fps main pipeline
- **Health checks via HTTP** instead of spawning `nc` and `docker compose ps` subprocesses (was 120/min, now ~7/min via httpx + throttled Docker polling)
- **MLX tempfile uses `delete=True`** — auto-cleanup by the OS, no manual `os.unlink` needed
- **Face detection limited to `MAX_NUM_FACES` (default 3)** to bound MediaPipe processing time
- **Pose detection limited to `MAX_NUM_POSES` (default 3)** similarly bounded
- **Client reads `X-Sample-Rate` header** — playback stream created at the exact rate reported by the server (24000 Hz for Realtime)
