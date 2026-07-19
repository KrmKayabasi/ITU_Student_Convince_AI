# Realtime Orchestrator (version1)

Owns a **Google Gemini Live** native-audio session per kiosk visitor, bridges
audio between the browser and Gemini, and injects the CV pipeline's signals into
the conversation:

- **one-shot `/profile`** → a Turkish opening hint so the assistant greets in a
  way that fits the student's first impression;
- **continuous `/focus`** → a debounced re-engage nudge (voice + avatar
  attention-grab) when the student looks away.

It replaces the cascaded Whisper→Gemma→XTTS speech server. The CV pipeline
(`backend/cv_pipeline`, :8000) is unchanged.

## Layout

| File | Role |
|------|------|
| `server.py` | FastAPI WS `/v1/realtime` (binary PCM + control JSON), per-session runner, single outbound queue, `/v1/health` |
| `gemini_live_bridge.py` | Persistent Gemini Live session; `send_audio`/`steer`/`inject_context`/`receive`; normalizes events |
| `cv_injector.py` | Server-side subscriber to CV `/profile` (opener) + `/focus` (debounced nudge) |
| `cv_hints.py` | Pure profile→Turkish opening-hint formatting (unit tested) |
| `audio_helpers.py` | PCM16/float32 + resample helpers (reused from the OpenAI realtime bridge) |
| `config.py` | Env config + `SYSTEM_PROMPT.md` loader + markdown stripper |
| `smoke_gemini.py` | P1 live smoke test (writes `smoke_out.wav`) |

## Run (local)

```bash
python -m venv .venv && source .venv/bin/activate   # Python 3.11
pip install -r backend/orchestrator/requirements.txt
export GOOGLE_API_KEY=...            # Google AI Studio key
./scripts/start_orchestrator.sh      # serves :8001, subscribes to CV :8000
```

Verify the model/voice/flags end-to-end (needs the key):

```bash
GOOGLE_API_KEY=... python backend/orchestrator/smoke_gemini.py
# -> writes backend/orchestrator/smoke_out.wav (play it to confirm Turkish)
```

## Key env vars

| Var | Default | Notes |
|-----|---------|-------|
| `GOOGLE_API_KEY` | — | **required** (AI Studio) |
| `GEMINI_LIVE_MODEL` | `gemini-2.5-flash-native-audio-preview-12-2025` | native-audio model |
| `GEMINI_API_VERSION` | `v1alpha` | needed for affective + proactive |
| `GEMINI_VOICE` | `Aoede` | verify Turkish quality; override if needed |
| `GEMINI_AFFECTIVE_DIALOG` / `GEMINI_PROACTIVE_AUDIO` | `true` | v1alpha-only |
| `CV_PIPELINE_WS_URL` | `ws://localhost:8000` | CV pipeline base |
| `FOCUS_LOSS_SECONDS` | `5` | sustained `!is_focused` before a nudge |
| `NUDGE_COOLDOWN_SECONDS` | `20` | min gap between nudges |
| `PROFILE_WAIT_SECONDS` | `3` | wait for `/profile` before a generic opener |
| `ORCH_PORT` | `8001` | server port |

## Browser ↔ orchestrator protocol

- **Uplink**: binary = 16 kHz PCM16 mono (mic); text = `{"type":"interrupt"|"session.stop"}`.
- **Downlink**: binary = 24 kHz PCM16 mono (model); text = `ready` / `transcript` /
  `interrupt` (barge-in) / `seekAttention` / `turn_complete` / `error`.

## Tests

```bash
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/orchestrator -q
```

Covers audio helpers, opener-hint formatting, focus debounce/cooldown, and
Gemini event normalization. Live Gemini calls are not exercised (needs a key).
