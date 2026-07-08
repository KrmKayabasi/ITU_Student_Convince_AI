# Turkish Low-Latency Test Pipeline

This repo now has optional Turkish STT/TTS adapters under `services/turkish/`.
They speak the same websocket/msgpack protocol as the Kyutai services, so the normal Unmute backend can use them without changing the browser protocol.

## STT Choice

`faster-whisper` is the inference engine (CTranslate2 backend). **It requires CT2-converted model repos** (the `Systran/faster-*` family), which ship `model.bin`. The original `openai/whisper-*` repos are in safetensors/pytorch format and will fail with `Unable to open file 'model.bin'` — do not use them with faster-whisper.

The default is the CT2-converted large-v3:

```bash
WHISPER_MODEL=Systran/faster-whisper-large-v3 \
WHISPER_DEVICE=cuda \
WHISPER_COMPUTE_TYPE=int8_float16
```

For a smaller/faster model (lower GPU usage, weaker Turkish accuracy), use:

```bash
WHISPER_MODEL=Systran/faster-whisper-small
```

`int8_float16` is usually the best latency/memory setting on NVIDIA GPUs with compute capability >= 7.0. If quality drops, try `float16`.

## TTS Choice

Two local Turkish TTS paths are supported by the adapter:

```bash
TURKISH_TTS_BACKEND=sherpa
```

This uses Sherpa-ONNX/Piper VITS. It is the smallest and most predictable local baseline.

```bash
TURKISH_TTS_BACKEND=supertonic INSTALL_SUPERTONIC=true
```

This uses Supertonic via its Python package. Supertonic supports Turkish and is a good quality/latency candidate, but it is a larger dependency and downloads its own model assets.

## LLM

The default LLM is `google/gemma-4-E2B-it` (overridable via `KYUTAI_LLM_MODEL`). It is run through vLLM with `--gpu-memory-utilization=0.70`, `--max-model-len=2048`, and `--max-num-seqs=64`.

Gemma 4 E2B-it is the smallest Gemma 4 variant (~2B params), chosen so the LLM and the Whisper STT can share a single 16 GB GPU (e.g. RTX 5070 Ti). The larger `google/gemma-4-E4B-it` loads the multimodal/unified checkpoint and will OOM on 16 GB alongside Whisper; use it only on a larger GPU (and consider moving STT to CPU via `WHISPER_DEVICE=cpu`).

### GPU memory budget on 16 GB cards

The STT model (`Systran/faster-whisper-large-v3`) takes ~2–3 GB and vLLM shares the same GPU, so the `--gpu-memory-utilization` fraction must leave room for STT. Empirically on a 16 GB RTX 5070 Ti:

- **`--gpu-memory-utilization=0.70 --max-num-seqs=64`** → vLLM holds ~11.5 GB (weights + KV cache), STT holds ~2.5 GB, ~1.3 GB free. vLLM gets a working KV cache of ~19K tokens (≈9 conversations of 2048 tokens). **This is the recommended starting point.**
- `0.75`+ tends to OOM the vLLM sampler warmup ("CUDA out of memory occurred when warming up sampler with 256 dummy requests") because STT is already resident.
- `--max-num-seqs=64` caps the sampler/batch memory; safe for a single-user voice bot. Lower it further (e.g. `32`) to free more KV cache if you need longer context per conversation.

**Tuning for more context (less forgetting):** if the model forgets early turns, the lever is a larger KV cache. Options: lower `--gpu-memory-utilization` less aggressively is *not* possible (OOM); instead reduce `--max-num-seqs` (frees sampler memory for KV cache), shorten the system prompt, or move STT to CPU (`WHISPER_DEVICE=cpu`) to free ~2.5 GB for KV cache.

## Language (Turkish from the first reply)

The pipeline is wired to speak Turkish from the very first reply, before the browser sends its `session.update` message:

- `UNMUTE_DEFAULT_LANGUAGE=tr` (set on the backend) makes the initial system prompt Turkish.
- `WHISPER_LANGUAGE=tr` forces Turkish transcription.
- The TTS backend uses a Turkish (`tr_TR`) model.

Set `UNMUTE_DEFAULT_LANGUAGE` to another supported code (`en`, `fr`, `en/fr`, `fr/en`, `tr`) to change the startup language; unset it to restore the original English-with-some-French default.

## Run Full Pipeline With Podman

Put secrets such as `HUGGING_FACE_HUB_TOKEN` in `.env`; the Podman helper sources it without printing values.

Start the full pipeline:

```bash
services/turkish/podman_pipeline.sh up
```

Open `http://localhost:3000`, choose the `Türkçe test` character if needed, click `connect`, and talk in Turkish.

Stop everything:

```bash
services/turkish/podman_pipeline.sh down
```

Watch logs:

```bash
services/turkish/podman_pipeline.sh logs backend
services/turkish/podman_pipeline.sh logs stt
services/turkish/podman_pipeline.sh logs tts
services/turkish/podman_pipeline.sh logs llm
```

If Podman cannot find your NVIDIA GPU via CDI, install/configure `nvidia-container-toolkit` for Podman. The helper will generate `/etc/cdi/nvidia.yaml` automatically when it has permission. Otherwise, run this once:

```bash
sudo mkdir -p /etc/cdi
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
```

If CDI does not work but `nvidia-container-runtime` is installed, run with:

```bash
PODMAN_GPU_ARGS="--runtime /usr/bin/nvidia-container-runtime -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all" services/turkish/podman_pipeline.sh up
```

If your setup uses the older OCI hook instead of CDI, run with:

```bash
PODMAN_GPU_ARGS="--hooks-dir=/usr/share/containers/oci/hooks.d" services/turkish/podman_pipeline.sh up
```

For SELinux/AppArmor systems, use:

```bash
PODMAN_GPU_ARGS="--security-opt=label=disable --hooks-dir=/usr/share/containers/oci/hooks.d" services/turkish/podman_pipeline.sh up
```

Set `PODMAN_GPU_ARGS=""` only if you want CPU mode; you must also set `WHISPER_DEVICE=cpu`, and vLLM will not run Gemma 4 locally without a GPU.

STT falls back to CPU by default if CUDA initialization fails. Disable that with `WHISPER_FALLBACK_TO_CPU=false` when debugging GPU setup.

Useful overrides:

```bash
WHISPER_MODEL=Systran/faster-whisper-small \
WHISPER_COMPUTE_TYPE=int8_float16 \
TURKISH_TTS_BACKEND=sherpa \
TTS_CHUNK_WORDS=8 \
services/turkish/podman_pipeline.sh up
```

For Supertonic:

```bash
TURKISH_TTS_BACKEND=supertonic \
INSTALL_SUPERTONIC=true \
SUPERTONIC_STEPS=5 \
SUPERTONIC_SPEED=1.1 \
services/turkish/podman_pipeline.sh up
```

## Run Full Pipeline With Compose

Use this only if you install a Podman Compose provider or if you are on Docker. Do not run compose config if you do not want expanded secrets printed to your terminal.

```bash
podman-compose -f docker-compose.yml -f docker-compose.turkish.yml up --build
```

With Docker Compose, replace `podman-compose` with `docker compose` and open `http://localhost`.

Useful overrides:

```bash
WHISPER_MODEL=Systran/faster-whisper-small \
WHISPER_COMPUTE_TYPE=int8_float16 \
TURKISH_TTS_BACKEND=sherpa \
TTS_CHUNK_WORDS=8 \
podman-compose -f docker-compose.yml -f docker-compose.turkish.yml up --build
```

For Supertonic:

```bash
TURKISH_TTS_BACKEND=supertonic \
INSTALL_SUPERTONIC=true \
SUPERTONIC_STEPS=5 \
SUPERTONIC_SPEED=1.1 \
podman-compose -f docker-compose.yml -f docker-compose.turkish.yml up --build tts backend frontend llm stt traefik
```

## Isolated Service Benchmarks

Run only STT/TTS adapters:

```bash
services/turkish/podman_pipeline.sh up
```

Benchmark TTS first-audio and total RTF:

```bash
uv run services/turkish/benchmark_services.py tts \
  --url ws://localhost:8089/api/tts_streaming \
  --text "Merhaba, bu Türkçe ses sentezi için kısa bir gecikme testidir."
```

Benchmark STT with a Turkish WAV/MP3/OGG file:

```bash
uv run services/turkish/benchmark_services.py stt \
  --url ws://localhost:8090/api/asr-streaming \
  --audio /path/to/turkish-test.wav
```

Add `--realtime` if you want the benchmark to send audio at real microphone speed instead of as fast as possible.

## Full Conversation Load Test

Use Turkish recordings in a folder and pass `--language tr`:

```bash
uv run unmute/loadtest/loadtest_client.py \
  --server-url ws://localhost/api \
  --audio-dir /path/to/turkish-audio-dir \
  --language tr \
  --n-workers 1 \
  --n-conversations 3
```

Use `--server-url ws://localhost:8000` instead if you are running the backend directly with `uv run fastapi dev unmute/main_websocket.py`.

Important metrics in the output:

- `stt_latencies`: user audio start to first transcription
- `vad_latencies`: user audio end to response creation
- `llm_latencies`: response creation to first LLM word
- `tts_start_latencies`: first LLM word to first audio
- `tts_realtime_factors`: generated audio duration divided by wall-clock streaming duration

## Latency Knobs

- `STT_END_SILENCE_SEC`: lower means faster turn-taking, higher means fewer premature cuts. Start with `0.35`.
- `STT_SPEECH_RMS_THRESHOLD`: absolute RMS floor for the speech decision. Effective threshold is `max(STT_SPEECH_RMS_THRESHOLD, noise_floor * STT_SPEECH_FACTOR)`, so in a noisy room the adaptive floor usually dominates. Raise this only if speech is missed in a very quiet environment.
- `STT_NOISE_FLOOR_ALPHA`: EMA weight for background-noise tracking (silence frames only). Smaller = slower, smoother adaptation. Default `0.02`.
- `STT_SPEECH_FACTOR`: multiplier on the noise floor to enter speech. Higher = less sensitive to background noise (fewer false triggers). Default `3.0`.
- `STT_SPEECH_OFF_FACTOR`: multiplier on the noise floor to exit speech (hysteresis). Must be `< STT_SPEECH_FACTOR` so brief dips don't fragment an utterance. Default `1.5`.
- `STT_PREROLL_SEC`: seconds of recent silence frames prepended to each utterance so word onsets reach Whisper intact. `0` disables. Default `0.2`. Does not affect end-of-turn timing.
- `WHISPER_BEAM_SIZE`: keep `1` for latency.
- `TTS_CHUNK_WORDS`: lower gives earlier audio but choppier prosody. Start with `8`; try `4` if first audio is too late.
- `SHERPA_TTS_NUM_THREADS`: for CPU Piper/Sherpa. Try `2`, `4`, `8`.
- `SUPERTONIC_STEPS`: lower is faster; `5` is a good latency test.

Tuning the VAD for a noisy room: first try raising `STT_SPEECH_FACTOR` (e.g. `4.0` or `5.0`) — this raises the effective speech threshold relative to the measured background. If speech is being cut mid-sentence, lower `STT_SPEECH_OFF_FACTOR` (e.g. `1.2`) for more hysteresis. If word beginnings are dropped, increase `STT_PREROLL_SEC` (e.g. `0.3`).

## Turn-taking (end-of-turn gating)

The backend only ends the user's turn (and starts generating a response) when **both** are true:

1. The STT's pause signal indicates the user has stopped speaking (`pause_prediction > 0.6`), AND
2. The user has actually spoken real words in this turn (`words_received_this_turn > 0`).

The second gate is important for utterance-level STT backends like the Turkish Whisper adapter. Unlike the Kyutai streaming STT (which emits a graded, continuous pause-prediction score from a neural head), the Turkish adapter's pause signal is binary and flips to `1.0` the instant RMS energy drops below the exit threshold — i.e. on any natural breath or brief inter-word pause. Without the word-count gate, that instant flip would make the bot respond to a breath before a real utterance was transcribed, fragmenting the user's turn and degrading the conversation history (which in turn causes the LLM to "forget" earlier context). The gate ensures the bot waits until a complete utterance has been transcribed before responding.

## Notes

The Whisper adapter is utterance-level, not true token-streaming STT. It uses an RMS VAD with an adaptive noise floor and hysteresis to decide when to transcribe (see the Latency Knobs above). This is slower than Kyutai's streaming STT in end-of-turn latency, but it is simple and gives a useful Turkish baseline, and the adaptive floor keeps it from feeding Whisper pure background noise.

The Piper/Sherpa and Supertonic adapters are phrase-level TTS. They buffer a few LLM words before synthesis. This is necessary because these TTS engines are not text-token streaming like Kyutai TTS.
