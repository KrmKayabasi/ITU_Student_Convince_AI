# Integrated Speech-to-Speech Server (NVIDIA H200 Multi-GPU)

import os

# Monkey patch transformers to bypass PyTorch 2.6 requirements for vision causal masks
try:
    import transformers.masking_utils
    # Keep the _is_torch_greater_or_equal_than_2_6 check as True to bypass the or_mask/and_mask check
    transformers.masking_utils._is_torch_greater_or_equal_than_2_6 = True
    
    # Save the original sdpa_mask function
    _orig_sdpa_mask = transformers.masking_utils.sdpa_mask
    
    def patched_sdpa_mask(*args, **kwargs):
        try:
            return _orig_sdpa_mask(*args, **kwargs)
        except Exception as e:
            # If vmap fails due to .item() on torch<2.6, retry without vmap
            if "vmap" in str(e) or "TransformGetItemToIndex" in str(e) or "item" in str(e):
                kwargs["use_vmap"] = False
                return _orig_sdpa_mask(*args, **kwargs)
            raise e
            
    transformers.masking_utils.sdpa_mask = patched_sdpa_mask
except ImportError:
    pass
import secrets
import uvicorn
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import numpy as np
import time
import torch
from transformers import pipeline

from config import SERVER_HOST as HOST, SERVER_PORT as PORT, MODEL_ID, QUANTIZATION, DEVICE, ENABLE_THINKING, TTS_MODEL_ID, SYSTEM_INSTRUCTION
from model_handler import GemmaAudioProcessor
from tts_handler import OfflineTTSHandler

app = FastAPI(title="Turkish Speech-to-Speech Server (NVIDIA H200)")

# ── Auth ─────────────────────────────────────────────────────────────────────
# Token-based authentication.  Set SPEECH_SERVER_TOKEN in the environment to
# enable; if unset, the server runs without auth (dev mode).
_AUTH_TOKEN = os.environ.get("SPEECH_SERVER_TOKEN", "")
_AUTH_ENABLED = bool(_AUTH_TOKEN)
_security = HTTPBearer(auto_error=False)


def _verify_auth(credentials: HTTPAuthorizationCredentials | None = Depends(_security)):
    """FastAPI dependency: raises 401 if auth is enabled and the token is missing
    or doesn't match the configured SPEECH_SERVER_TOKEN."""
    if not _AUTH_ENABLED:
        return
    if credentials is None:
        raise HTTPException(status_code=401, detail="Authorization header required")
    if not secrets.compare_digest(credentials.credentials, _AUTH_TOKEN):
        raise HTTPException(status_code=401, detail="Invalid token")


# Global handlers (initialized on startup)
gemma = None
tts = None
asr = None

@app.on_event("startup")
def startup_event():
    global gemma, tts, asr
    print("=" * 60)
    print("STARTING H200 SPEECH-TO-SPEECH SERVER")
    print("=" * 60)
    
    # Load Gemma 4 on CUDA (unquantized for best quality)
    gemma = GemmaAudioProcessor(
        model_id=MODEL_ID,
        quantization=QUANTIZATION,
        device=DEVICE,
        enable_thinking=ENABLE_THINKING
    )
    
    # Load VITS Piper on CUDA/CPU
    tts = OfflineTTSHandler(
        model_id=TTS_MODEL_ID,
        device=DEVICE
    )

    # Load Whisper ASR with hardware acceleration (optimized float16)
    print(f"[Server] Loading Whisper Large v3 Turbo on {DEVICE}...", flush=True)
    asr = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3-turbo",
        torch_dtype=torch.float16 if DEVICE != "cpu" else torch.float32,
        device=DEVICE,
        generate_kwargs={
            "language": "turkish",
            "task": "transcribe",
            "no_repeat_ngram_size": 4
        }
    )
    
    print(f"[Server] All models loaded on {DEVICE} and ready!")

# Conversation history state (sliding window of last 20 turns of text history)
chat_history = []
MAX_HISTORY_TURNS = 20

@app.post("/reset")
def reset_conversation(_auth=Depends(_verify_auth)):
    global chat_history
    chat_history = []
    print("[Server] Conversation history reset!", flush=True)
    return {"status": "reset"}

@app.get("/last_turn")
def get_last_turn(_auth=Depends(_verify_auth)):
    global chat_history
    if len(chat_history) >= 2:
        return {
            "user": chat_history[-2]["content"],
            "assistant": chat_history[-1]["content"]
        }
    return {"user": "", "assistant": ""}

@app.post("/chat_stream")
async def chat_stream(request: Request, _auth=Depends(_verify_auth)):
    """
    Receives raw PCM float32 16kHz mono audio from client,
    transcribes it on the fly using Whisper ASR on GPU,
    runs Gemma 4 (LLM) with conversation history,
    synthesizes audio (24kHz) using Piper VITS, and streams the raw float32 audio bytes back.
    """
    global chat_history, asr

    # 1. Read binary audio data
    audio_bytes = await request.body()
    if not audio_bytes:
        return StreamingResponse(iter([b""]), media_type="application/octet-stream")
        
    # Convert bytes back to float32 numpy array
    audio_data = np.frombuffer(audio_bytes, dtype=np.float32)
    print(f"\n[Server] Received user speech: {len(audio_data)} samples ({len(audio_data)/16000:.2f}s)")
    
    # 2. Run Whisper STT with energy check to prevent silence hallucinations
    rms = np.sqrt(np.mean(audio_data**2)) if len(audio_data) > 0 else 0
    if rms < 0.005:
        print(f"[Server] Silence detected (RMS={rms:.5f} < 0.005). Bypassing Whisper ASR.", flush=True)
        user_transcribed_text = ""
    else:
        print(f"[Server] Running Whisper ASR (RMS={rms:.5f})...", flush=True)
        start_asr = time.time()
        asr_res = asr(audio_data)
        user_transcribed_text = asr_res.get("text", "").strip()
        asr_latency = time.time() - start_asr
        print(f"[Server] User Speech Transcribed Natively: '{user_transcribed_text}' (took {asr_latency:.2f}s)")
    
    # Initialize history if empty
    if not chat_history:
        chat_history = [{"role": "system", "content": SYSTEM_INSTRUCTION}]
        
    # Append the current user text turn
    user_content = user_transcribed_text if user_transcribed_text else "[Kullanıcı anlaşılmayan bir ses gönderdi]"
    chat_history.append({"role": "user", "content": user_content})
    
    # Enforce history limit (sliding window of last 20 turns of text history).
    # pop(1) shifts the list, so we must pop index 1 twice (not 1 then 2),
    # but ONLY when indices [1] and [2] are the user+assistant pair after
    # index 0 (system prompt).  Guard with an explicit check.
    while len(chat_history) > 1 + 2 * MAX_HISTORY_TURNS:
        # Remove the oldest user+assistant pair (indices 1 and 2 after system prompt)
        if len(chat_history) >= 3:
            chat_history.pop(1)  # oldest user  (now index 1 is the old assistant)
            chat_history.pop(1)  # oldest assistant (shifted into index 1)
        else:
            break
        
    # 3. Run Cascaded LLM-to-TTS Streaming
    async def cascaded_audio_generator():
        global chat_history
        full_response_parts = []

        print(f"[Server] Running Cascaded LLM-to-TTS Streaming...", flush=True)
        start_stream_time = time.time()
        first_chunk_sent = False

        # We'll determine the actual sample rate from the first TTS synthesis
        # rather than assuming 24000 (Piper VITS outputs 22050, XTTS outputs 24000).
        actual_sample_rate = 24000  # fallback, updated on first synthesis

        # ── Cross-fade state ───────────────────────────────────────────────
        # Each TTS clause is synthesized independently.  Without smoothing,
        # concatenating two waveforms at arbitrary phase offsets creates an
        # audible "pop" at every clause boundary.  We keep a short tail of the
        # previous clause and apply a linear cross-fade with the new clause.
        _XFADE_SAMPLES = 480  # ~10ms at 48kHz max, ~20ms at 24kHz, scaled below
        _prev_tail = None     # np.ndarray, float32 — trailing samples from last clause

        try:
            # Iterate over text clauses generated in real-time by Gemma
            for clause in gemma.generate_response_stream(chat_history, []):
                clause = clause.strip()
                if not clause:
                    continue

                print(f"[Server Stream] Got clause: '{clause}'", flush=True)
                full_response_parts.append(clause)

                # Synthesize TTS for this clause immediately
                tts_start = time.time()
                tts_audio, sample_rate = tts.synthesize(clause)
                tts_elapsed = time.time() - tts_start

                if tts_audio is None or len(tts_audio) == 0:
                    continue

                # ── Update the actual sample rate from the first synthesis ──
                actual_sample_rate = sample_rate
                fade_len = max(1, int(_XFADE_SAMPLES * sample_rate / 48000))

                # ── Apply cross-fade with previous clause's tail ────────────
                if _prev_tail is not None and len(_prev_tail) > 0 and len(tts_audio) > 0:
                    overlap = min(fade_len, len(_prev_tail), len(tts_audio))
                    # Linear cross-fade: ramp down tail, ramp up head
                    ramp_down = np.linspace(1.0, 0.0, overlap, dtype=np.float32)
                    ramp_up   = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
                    # Yield the non-overlapping part of the previous tail first
                    if len(_prev_tail) > overlap:
                        for i in range(0, len(_prev_tail) - overlap, 1024):
                            yield _prev_tail[i:i + 1024].tobytes()
                    # Yield the cross-faded overlap region
                    crossfaded = _prev_tail[-overlap:] * ramp_down + tts_audio[:overlap] * ramp_up
                    for i in range(0, len(crossfaded), 1024):
                        yield crossfaded[i:i + 1024].tobytes()
                    # Trim the already-yielded head from tts_audio
                    tts_audio = tts_audio[overlap:]
                elif _prev_tail is not None and len(_prev_tail) > 0:
                    # No overlap possible (very short previous chunk) — just yield it
                    for i in range(0, len(_prev_tail), 1024):
                        yield _prev_tail[i:i + 1024].tobytes()

                if not first_chunk_sent:
                    total_first_latency = time.time() - start_stream_time
                    print(f"[Server Stream] First audio chunk ready in {total_first_latency:.2f}s! "
                          f"(TTS took {tts_elapsed:.2f}s, rate={sample_rate}Hz)", flush=True)
                    first_chunk_sent = True

                # Stream the remaining (post-overlap) audio in 1024-sample chunks
                for i in range(0, len(tts_audio), 1024):
                    yield tts_audio[i:i + 1024].tobytes()

                # Save the tail of this clause for cross-fading with the next one
                _prev_tail = tts_audio[-fade_len:] if len(tts_audio) > fade_len else tts_audio.copy()

            # Flush the final tail (no next clause to cross-fade with)
            if _prev_tail is not None and len(_prev_tail) > 0:
                for i in range(0, len(_prev_tail), 1024):
                    yield _prev_tail[i:i + 1024].tobytes()

            # After generation completes, update the assistant chat history
            full_response = " ".join(full_response_parts).strip()
            print(f"[Server Stream] Generation finished. Full response: '{full_response}'", flush=True)

            if full_response:
                chat_history.append({"role": "assistant", "content": full_response})
            else:
                chat_history.pop()

        except Exception as e:
            print(f"[Server Stream Error] Exception during streaming: {e}", flush=True)
            if chat_history and chat_history[-1]["role"] == "user":
                chat_history.pop()

    headers = {
        "X-Sample-Rate": str(tts.sample_rate),
        "X-Streaming": "true"
    }

    return StreamingResponse(
        cascaded_audio_generator(),
        media_type="application/octet-stream",
        headers=headers
    )


@app.get("/health")
async def health():
    """Liveness/readiness probe — intentionally excludes auth so monitoring
    tools and the desktop client's status poller work without a token."""
    return {
        "status": "ok",
        "auth_enabled": _AUTH_ENABLED,
        "model": MODEL_ID,
    }


if __name__ == "__main__":
    print(f"[Server] Auth {'enabled' if _AUTH_ENABLED else 'DISABLED (dev mode)'}", flush=True)
    uvicorn.run("server:app", host=HOST, port=PORT, log_level="info")
