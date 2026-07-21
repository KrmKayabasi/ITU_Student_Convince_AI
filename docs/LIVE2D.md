# Live2D Avatar

The kiosk ships with two interchangeable avatar renderers behind the same
`FaceStage` component:

| Mode  | Renderer                          | Lip-sync source          | Expressions          |
|-------|-----------------------------------|--------------------------|----------------------|
| `live2d` | `Live2DAvatar.tsx` — Cubism 4 model via `pixi-live2d-display` (**default**) | playback-audio RMS (`AmplitudeSource`) | FaceState **+** emotion overlay |
| `svg` | `AdvisorFace.tsx` — hand-crafted "Elif" SVG portrait (fallback) | same `AmplitudeSource` | FaceState only |

Both renderers consume the **same** `AmplitudeSource` (the `AnalyserNode` RMS
of the assistant playback audio), so lip-sync stays identical and in sync with
Gemini Live audio regardless of which face is shown.

## Selecting the avatar

- **Default**: bare `/kiosk` uses Live2D.
- Query param override: `/kiosk?avatar=svg` (force the SVG fallback) or
  `/kiosk?avatar=live2d` (force Live2D).
- Build-time default: set `NEXT_PUBLIC_AVATAR=svg` on the frontend if your
  deployment needs the SVG face as the default (e.g. machines without WebGL).
- Demo mode (no backend): `/kiosk?demo=1` — the `DemoPanel` also
  has an Avatar radio to switch live, plus an emotion dropdown to drive the
  Live2D expression map without a server.

## Emotion channel (optional, off by default)

Emotion-driven expressions are ported from `jaison-core`'s `emotion_roberta`
filter (`SamLowe/roberta-base-go_emotions`, 28 labels). The orchestrator
classifies the assistant's streamed transcript and pushes a new downlink
control message:

```json
{ "type": "emotion", "emotion": "joy" }
```

The frontend (`useRealtimeSession`) stores it and passes it to `FaceStage` →
`Live2DAvatar`, where `live2dExpressions.ts` maps the 28 go_emotions labels to
~6 expression families (joy / sadness / anger / fear / surprise / curiosity /
neutral) blended on top of the FaceState base pose.

**Enable on the orchestrator** (heavy — adds `torch` + `transformers`):

The emotion deps are split into a separate file so the default image stays
slim. When you want emotion, install both:

```bash
pip install -r backend/orchestrator/requirements.txt \
            -r backend/orchestrator/requirements-emotion.txt
# then run the orchestrator with ENABLE_EMOTION=true
```

For Docker, a `Dockerfile.emotion` + `docker-compose.override.yml` are included
(see "Testing in Docker" below).

Config knobs (`backend/orchestrator/config.py`, all env-overridable):
- `ENABLE_EMOTION` (default `false`) — master switch. When off, the
  orchestrator never imports torch and emits no `emotion` messages; the avatar
  uses FaceState-only expressions.
- `EMOTION_MODEL` (default `SamLowe/roberta-base-go_emotions`)
- `EMOTION_DEBOUNCE_S` (default `0.6`) — min interval between classifications
- `EMOTION_MIN_CHARS` (default `12`) — min accumulated assistant text

The classifier is lazy-loaded on the first qualifying transcript (so boot is
fast) and runs off the event loop via `asyncio.to_thread` — it can never block
the realtime audio loop. All failures are logged and swallowed; emotion is
cosmetic.

## Live2D assets

Bundled under `frontend/public/live2d/`:

```
frontend/public/live2d/
  live2dcubismcore.min.js        # Cubism 4 Core (see License below)
  models/
    Haru/                         # free Live2D sample model
      haru_greeter_t03.model3.json
      haru_greeter_t03.moc3
      haru_greeter_t03.physics3.json
      haru_greeter_t03.pose3.json
      haru_greeter_t03.2048/texture_00.png, texture_01.png
      expressions/F01..F08.exp3.json
      motion/haru_g_*.motion3.json
```

### Using a different model

Drop a new model folder under `frontend/public/live2d/models/<Name>/` and pass
its `.model3.json` URL via the `modelUrl` prop on `Live2DAvatar` (or change
`DEFAULT_MODEL_URL` in `useLive2DModel.ts`). The model must expose the Cubism 4
standard parameters the rig writes (`ParamMouthOpenY`, `ParamMouthForm`,
`ParamAngleX/Y/Z`, `ParamBodyAngleX`, `ParamEyeBallX/Y`, `ParamEyeLOpen/R`,
`ParamEyeSmile`, `ParamBrowLY/RY`, `ParamBreath`, `ParamCheek`) — the official
Hiyori / Haru / Shizuku samples all do. Missing params are silently skipped.

## Dependencies & compatibility

`frontend/package.json`:
- `pixi.js@^6.5.10` — **must stay on v6.** `pixi-live2d-display@0.4.0` is not
  compatible with pixi v8.
- `pixi-live2d-display@^0.4.0` — imported via the `pixi-live2d-display/cubism4`
  subpath (Cubism 4 models only; Cubism 2 is unused here).

Both are dynamically imported inside `useLive2DModel` and the component is
loaded via `next/dynamic({ ssr: false })`, so pixi never enters the SSR bundle
and the kiosk route only pays the cost when `?avatar=live2d` is active.

## License notes

- **Cubism Core** (`live2dcubismcore.min.js`): © Live2D Inc. Proprietary, but
  freely redistributable as "Redistributable Code" per the Live2D Cubism SDK
  license agreement, and free for small-scale / educational / individual use.
  İTÜ student project use is within the free tier. Review the header of the
  file itself for the exact terms.
- **Haru model**: official Live2D sample material, free for
  small-business/individual/educational use under the Live2D Free Material
  License. Same caveat as above.
- **pixi-live2d-display**: MIT.
- **pixi.js**: MIT.

If you ship this commercially, re-evaluate the Live2D licenses and consider a
proprietary Live2D distribution license.

## Testing in Docker

All testing stays inside Docker (no host Node/Python needed). The
`docker-compose.override.yml` (auto-merged by `docker compose up`) adds:

- `orchestrator-emotion` service — builds `Dockerfile.emotion` (torch +
  transformers) and runs with `ENABLE_EMOTION=true`. Host port `8002` to avoid
  clashing with the default orchestrator on `8001`.
- `frontend` `ports: ["3000:3000"]` — exposes the Next.js server directly so
  the avatar can be tested without the gateway.

### Path 1 — Avatar only, no backend (fastest)

Proves the Live2D model renders and lip-syncs. Needs no CV models and no
`GOOGLE_API_KEY`.

```bash
docker compose up --build frontend
# then in Chrome:
#   http://localhost:3000/kiosk?demo=1&avatar=live2d
```

In the left DEMO panel: pick `speech` under "Ses (lip-sync)" to drive the
mouth, cycle the "Duygu (emotion)" dropdown to test expressions, and toggle
the "Avatar" radio to swap Live2D ↔ SVG live.

### Path 2 — Full stack with Live2D (needs CV models + GOOGLE_API_KEY)

```bash
ls ./models                       # must be non-empty (see docs/SETUP.md)
export GOOGLE_API_KEY="..."
docker compose up --build
# then:
#   http://localhost:8080/kiosk?avatar=live2d   (through the gateway)
```

Mic → orchestrator → Gemini Live → 24 kHz PCM → `AnalyserNode` RMS →
`AmplitudeSource` → Live2D `ParamMouthOpenY`. Lip-sync is identical to the SVG
face because both renderers consume the same amplitude source.

### Path 3 — Full stack WITH emotion expressions

```bash
export GOOGLE_API_KEY="..."
docker compose up --build orchestrator-emotion cv-pipeline frontend gateway
```

Note: the gateway (`deploy/nginx.conf`) routes `/orch/` to the default
`orchestrator:8001`. To route through the gateway to the emotion-enabled
instance on `:8002`, either:

- hit the emotion orchestrator directly by pointing the browser at
  `orchestrator-emotion` (for isolated testing), or
- temporarily edit `deploy/nginx.conf`'s `/orch` upstream to
  `orchestrator-emotion:8001`, or
- rename the `orchestrator-emotion` service to `orchestrator` (commenting out
  the base one in `docker-compose.yml`).

Watch the orchestrator logs for
`emotion classifier loaded: model=SamLowe/roberta-base-go_emotions device=cpu`
on the first assistant transcript; the avatar's expression then tracks tone.

### Sanity checks

- **Lip-sync**: mouth opens/closes with audio; closes within ~150 ms on
  barge-in (`interrupt`).
- **seekAttention**: fires when the CV focus-loss nudge triggers (Path 2/3) —
  the model leans in for ~2.6 s. In demo mode use the "seekAttention ▶" button.
- **No regression**: `/kiosk` (no `?avatar=`) still shows the SVG Elif face.

