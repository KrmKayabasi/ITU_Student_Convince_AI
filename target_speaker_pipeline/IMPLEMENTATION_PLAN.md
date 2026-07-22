# Turkish Target-Speaker Pipeline — Production Implementation Plan

## Root Cause Analysis: Why Speaker Differentiation Failed

The current codebase has a **categorical architectural failure**: it identifies **phonemes** (what sound is being said), not **speakers** (who is saying it).

**Smoking gun evidence:**
- Same speaker saying /i/ ("ee") vs /a/ ("ah"): confidence = **0.86** — REJECTED
- Different speaker saying the same /i/: confidence = **0.998** — ACCEPTED

**Why this happens:** The 16-band spectral envelope at 20ms resolution captures the
formant structure of the current phoneme, not the speaker's vocal tract characteristics.
Speaker-discriminative information lives in:
1. Systematic formant shifts (vocal tract length — affects ALL formants proportionally)
2. Glottal pulse shape (harmonic richness, spectral tilt)
3. Temporal dynamics (how formants MOVE during speech — requires seconds of context)
4. Prosody and rhythm patterns

A 16-dim snapshot cannot encode ANY of this. The "192-dim embedding" is a
nonlinear remapping of 16 degrees of freedom — there is NO additional information.

**The fix:** Replace the hand-crafted spectral feature pipeline with a real,
trained neural speaker embedding model (ECAPA-TDNN/CAM++/ResNet).

---

## Recommended Model: CAM++

| Property | Value |
|----------|-------|
| Parameters | 7.18M |
| Embedding dim | 512 |
| Vox1-O EER | 0.65% |
| ONNX size | 29.3 MB |
| CPU RTF | 0.049 (Xeon) |
| Library | WeSpeaker → ONNX → sherpa-onnx |

**Fallback:** ResNet34-LM (WeSpeaker, 256-dim, 26.5 MB ONNX, 0.72% EER)

---

## Library Stack

```
sherpa-onnx          # ONNX wrapper for speaker embedding + verification
silero-vad           # Neural VAD (replaces webrtcvad)
sounddevice          # Cross-platform real-time audio capture
torchaudio           # Mel filterbank / resampling (only if needed)
numpy + scipy        # Cosine similarity, scoring
onnxruntime-silicon  # Apple Silicon-optimized ONNX inference
```

---

## Architecture: Multi-Threaded Pipeline

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ Audio Thread │    │  VAD Thread  │    │Verify Thread │    │Action Thread │
│              │    │              │    │              │    │              │
│ sounddevice  │    │ silero-vad   │    │ ONNX Runtime │    │ Business     │
│ callback     │    │ frame loop   │    │ inference    │    │ logic        │
│              │    │              │    │              │    │              │
│ RingBuffer   │───▶│ SegmentAccum │───▶│ Embedding    │───▶│ Decision     │
│ (lock-free)  │    │ (queue.Queue)│    │ (queue.Queue)│    │ (queue.Queue)│
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

**Key design decisions:**
- Audio callback NEVER blocks — writes to lock-free ring buffer only
- VAD segments speech into utterances (1.5s min, 300ms silence tail)
- Embedding extraction runs on utterance boundaries (not per-frame — that would be insane at ~30ms per inference)
- Verification is cosine similarity between extracted embedding and enrolled centroid
- Decision callbacks are async — no blocking in the hot path

---

## Implementation Phases (6 Weeks)

### Phase 1: Offline Verification (Week 1)
- Install sherpa-onnx, download CAM++ ONNX model
- Record 10-20 WAV files: you speaking, friend speaking, silence
- Script: extract embeddings → compute cosine similarity → report scores
- **Pass criteria:** target-vs-self > 0.6, target-vs-other < 0.4
- **File:** `offline_verify.py`

### Phase 2: Streaming VAD (Week 1-2)
- sounddevice InputStream at 16kHz, blocksize=320
- silero-vad VADIterator in audio callback
- Segment accumulator with 300ms silence tail
- **Pass criteria:** segments align with spoken utterances
- **File:** `streaming_vad.py`

### Phase 3: Real-Time Verification (Week 2-3)
- Multi-threaded pipeline: audio → VAD → embedding → verify
- EnrollmentManager with JSON persistence
- SpeakerDatabase: centroids, embedding lists, profile modes
- Callbacks: on_target_speaker, on_non_target_speaker
- **Pass criteria:** correct callbacks fire for target vs non-target speakers
- **File:** `pipeline.py` (rewrite)

### Phase 4: Threshold Calibration (Week 3)
- Collect 50 target + 50 non-target utterances
- Plot DET curve, compute EER
- Select operating threshold for desired FAR/FRR
- **Pass criteria:** EER < 5% on Turkish calibration set
- **File:** `calibrate.py`, `threshold_config.json`

### Phase 5: Barge-In Detection (Week 3-4)
- 200ms sustained voice gate before trigger
- TTS flush + LLM abort on non-target detection
- Cooldown after trigger (2 seconds)
- **Pass criteria:** TTS stops within 100ms of non-target speech
- **File:** barge-in logic in `pipeline.py`

### Phase 6: Wake Word Enrollment (Week 4-5)
- Train Turkish wake word model ("Hey Asistan")
- openWakeWord with TTS-generated training data
- Wake word detection triggers enrollment sample capture
- **Pass criteria:** < 5% FAR, < 10% FRR
- **File:** `wakeword.py`

### Phase 7: Turkish Fine-Tuning (Week 5-6)
- Download Common Voice Turkish (~100 hours)
- Fine-tune CAM++ from English checkpoint
- Evaluate EER improvement
- Export fine-tuned model to ONNX
- **Pass criteria:** EER improvement vs zero-shot English model
- **File:** fine-tuned ONNX model

### Phase 8: System Validation (Week 6)
- Latency measurement: P50/P95/P99 per stage
- Accuracy: FAR, FRR, EER on calibration set
- Cross-condition matrix: clean/noisy/far-field
- 24-hour stability test
- **File:** `validation_report.md`

---

## Performance Targets

| Metric | Target |
|--------|--------|
| EER (Turkish, clean) | < 5% |
| FAR at operating threshold | < 2% |
| FRR at operating threshold | < 10% |
| Embedding extraction latency | < 30ms |
| Barge-in latency | < 100ms |
| CPU (idle) | < 2% of one core |
| RAM (steady state) | < 300 MB |
| 24h uptime | No crash, no leak |

---

## Key Risks

1. **Cross-lingual EER degradation** — English CAM++ on Turkish may underperform
   → Mitigated by Phase 1 early validation + Phase 7 fine-tuning
2. **CAM++ ONNX on macOS arm64** — may have ARM-specific issues
   → Fallback: ResNet34-LM via MLX (native Apple Silicon)
3. **Silero VAD false positives in noise** — noise classified as speech
   → Energy gate + raised VAD threshold + WebRTC VAD as second opinion
4. **Embedding drift** — voice changes over time (morning, illness)
   → Rolling enrollment window + EMA adaptation + profile health monitoring
5. **Barge-in false triggers** — TTS cuts out during normal operation
   → 200ms sustained voice gate + cooldown + duration gate
6. **Anti-spoofing absent** — vulnerable to replay attacks
   → Document as known limitation; challenge-response + audio quality check as mitigations
