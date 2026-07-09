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

        # ── Inter-clause silence ────────────────────────────────────────────
        # Rather than cross-fading (which duplicates already-streamed audio and
        # creates stutter/pop artifacts), we insert a short silence between
        # independently-synthesized TTS clauses.  VITS models naturally decay
        # to near-zero at utterance boundaries (final phoneme $ maps to silence),
        # so the raw concatenation is already smooth.  The silence just adds a
        # natural pause between sentences.
        _SILENCE_SAMPLES = 0  # will be set on first clause from actual sample rate

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

                # ── Inter-clause silence padding ────────────────────────────
                if _SILENCE_SAMPLES == 0:
                    _SILENCE_SAMPLES = int(sample_rate * 0.15)  # 150ms pause between clauses
                    # Round up to a multiple of 1024 so every chunk is exactly
                    # 4096 bytes (1024 float32) — prevents short final chunks
                    # that trigger micro-fade artifacts on the client.
                    _SILENCE_SAMPLES = ((_SILENCE_SAMPLES + 1023) // 1024) * 1024
                if first_chunk_sent and _SILENCE_SAMPLES > 0:
                    silence = np.zeros(_SILENCE_SAMPLES, dtype=np.float32)
                    for i in range(0, len(silence), 1024):
                        yield silence[i:i + 1024].tobytes()

                if not first_chunk_sent:
                    total_first_latency = time.time() - start_stream_time
                    print(f"[Server Stream] First audio chunk ready in {total_first_latency:.2f}s! "
                          f"(TTS took {tts_elapsed:.2f}s, rate={sample_rate}Hz)", flush=True)
                    first_chunk_sent = True

                # Stream this clause's audio in 1024 float32 sample chunks
                for i in range(0, len(tts_audio), 1024):
                    yield tts_audio[i:i + 1024].tobytes()

            # After generation completes, update the assistant chat history
            full_response = " ".join(full_response_parts).strip()
            print(f"[Server Stream] Generation finished. Full response: '{full_response}'", flush=True)

            if full_response:
                chat_history.append({"role": "assistant", "content": full_response})
            else:
                chat_history.pop()

        except Exception as e:
            print(f"[Server Stream Error] Exception during streaming: {e}", flush=True)
            import traceback
            traceback.print_exc()
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
