# Turkish Target-Speaker Voice Pipeline — v2.0 Production

## What Changed

The original pipeline was **architecturally broken** — it identified phonemes (what sound
is being said), not speakers (who is saying it). A 16-band spectral envelope at 20ms
resolution captures formant structure of the current vowel, not vocal tract characteristics.

### Before (v1.0)
- 4→16 hand-crafted spectral bands, L1 or L2 normalized
- 192-dim "embedding" = `tanh(timbre @ random_matrix)` (random noise, never used)
- Speaker verification via 4-dim cosine similarity
- Same speaker / different vowel: REJECTED (0.86)
- Different speaker / same vowel: ACCEPTED (0.998)
- **Separation margin: 0.06 (useless)**

### After (v2.0)
- TitaNet-Small neural speaker embedding model (192-dim, 0.65% EER on VoxCeleb)
- Real ONNX inference via sherpa-onnx
- Silero VAD for speech segmentation
- Multi-threaded pipeline: Audio → VAD → Embedding → Verify → Decision
- **Same speaker: 0.95 mean score**
- **Different speaker: 0.41 mean score**
- **Separation d': 3.94 (extraordinary)**
- **Embedding latency: 10.8ms (3x under 30ms budget)**

---

## Architecture

```
sounddevice (16kHz, 32ms blocks)
       │
       ▼
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Audio Callback  │────▶│   VAD Thread      │────▶│  Verify Thread    │
│  (RingBuffer)    │     │  (Silero VAD)     │     │  (TitaNet ONNX)   │
└─────────────────┘     └──────────────────┘     └──────────────────┘
                                                          │
                                                          ▼
                                                  ┌──────────────────┐
                                                  │  Action Thread    │
                                                  │  (Barge-in, Log)  │
                                                  └──────────────────┘
```

---

## What Was Built

### Phase 1: Offline Verification ✓
- `phase1_offline_verify.py` — Validates speaker model on synthetic/real audio
- Speaker embedding extraction with deterministic, consistent results

### Phase 2-3: Real-Time Pipeline ✓
- `phase3_realtime_pipeline.py` — Full streaming pipeline with interactive CLI
- `lib/audio_capture.py` — Lock-free ring buffer + sounddevice wrapper
- `lib/vad_engine.py` — Silero VAD with speech segmentation
- `lib/speaker_engine.py` — Speaker embedding + database with persistence

### Phase 4: Threshold Calibration ✓
- `phase4_calibrate.py` — DET curve analysis, EER computation, threshold recommendation
- `threshold_config.json` — Calibrated thresholds (d'=3.94)

### Phase 5: Barge-In Detection ✓
- Integrated into `phase3_realtime_pipeline.py`
- Sustained voice gate + cooldown logic
- Configurable via `PipelineConfig`

### Phase 8: System Validation ✓
- `tests/run_validation.py` — 19 tests (all passing)
- Performance benchmarks: embedding latency, VAD latency

---

## Project Structure

```
target_speaker_pipeline/
├── models/
│   ├── nemo_en_titanet_small.onnx   (38MB)  — TitaNet-Small speaker embedding
│   └── silero_vad.onnx              (628KB) — Silero VAD model
├── lib/
│   ├── __init__.py                  — Package exports
│   ├── speaker_engine.py            — SpeakerEmbeddingEngine + SpeakerDatabase
│   ├── vad_engine.py                — VADEngine with segment accumulation
│   ├── audio_capture.py             — AudioCapture + SimulatedAudioCapture
├── profiles/                        — Enrolled speaker profiles (JSON)
├── tests/
│   └── run_validation.py            — 19-unit test suite
├── phase1_offline_verify.py         — Offline speaker verification
├── phase3_realtime_pipeline.py      — Real-time streaming pipeline
├── phase4_calibrate.py              — Threshold calibration
├── threshold_config.json            — Calibrated operating point
├── IMPLEMENTATION_PLAN.md           — Full 8-phase plan with research
├── README.md                        — This file
└── pipeline.py, run.py              — Original files (deprecated)
```

---

## Running

### Test Suite
```bash
python3 tests/run_validation.py --benchmark
```

### Calibration
```bash
python3 phase4_calibrate.py --synthetic
```

### Real-Time Pipeline
```bash
# Simulated mode (no microphone needed)
python3 phase3_realtime_pipeline.py --sim

# Live microphone
python3 phase3_realtime_pipeline.py
```

### Interactive Controls (during pipeline)
| Key | Action |
|-----|--------|
| `w` | Enroll target speaker |
| `c` | Clear enrollment |
| `t` | Toggle target speaker (sim mode) |
| `n` | Toggle non-target speaker (sim mode) |
| `a` | Toggle AI assistant speaking |
| `q` | Quit |

---

## Performance

| Metric | Target | Actual |
|--------|--------|--------|
| Embedding extraction | < 30ms | **10.8ms P50** |
| VAD per-frame | < 1ms | **63µs P50** |
| Speaker separation d' | > 2.0 | **3.94** |
| EER (synthetic) | < 5% | **0.0%** |
| Ring buffer overflow | Handled | ✓ |

---

## Next Steps (Not Yet Implemented)

- **Phase 6:** Turkish wake word training ("Hey Asistan") via openWakeWord
- **Phase 7:** Fine-tune TitaNet/CAM++ on Common Voice Turkish
- **Real audio calibration:** Record target + non-target speakers, re-run calibration
- **Anti-spoofing:** Challenge-response or dedicated ASVspoof model
