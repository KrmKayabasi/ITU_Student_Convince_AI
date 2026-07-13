# Integrated Speech-to-Speech Server

import os
import re

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
import asyncio
import uvicorn
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import numpy as np
import time

try:
    from .config import (
        SERVER_HOST as HOST,
        SERVER_PORT as PORT,
        MODEL_ID,
        QUANTIZATION,
        DEVICE,
        ENABLE_THINKING,
        TTS_MODEL_ID,
        SYSTEM_INSTRUCTION,
        SPEECH_PROVIDER,
        OPENAI_API_KEY,
        OPENAI_REALTIME_MODEL,
        OPENAI_REALTIME_VOICE,
        OPENAI_REALTIME_TRANSCRIPTION_MODEL,
        OPENAI_REALTIME_LANGUAGE,
    )
except ImportError:
    from config import (
        SERVER_HOST as HOST,
        SERVER_PORT as PORT,
        MODEL_ID,
        QUANTIZATION,
        DEVICE,
        ENABLE_THINKING,
        TTS_MODEL_ID,
        SYSTEM_INSTRUCTION,
        SPEECH_PROVIDER,
        OPENAI_API_KEY,
        OPENAI_REALTIME_MODEL,
        OPENAI_REALTIME_VOICE,
        OPENAI_REALTIME_TRANSCRIPTION_MODEL,
        OPENAI_REALTIME_LANGUAGE,
    )

try:
    from .openai_realtime_bridge import OpenAIRealtimeBridge
except ImportError:
    from openai_realtime_bridge import OpenAIRealtimeBridge

app = FastAPI(title="Turkish Speech-to-Speech Server")

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
realtime_sessions = {}


def _speech_session_id(request: Request) -> str:
    session_id = request.headers.get("X-Session-ID", "default").strip()
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", session_id):
        raise HTTPException(status_code=400, detail="Invalid X-Session-ID")
    return session_id


def _new_realtime_bridge() -> OpenAIRealtimeBridge:
    return OpenAIRealtimeBridge(
        api_key=OPENAI_API_KEY,
        model=OPENAI_REALTIME_MODEL,
        voice=OPENAI_REALTIME_VOICE,
        instructions=SYSTEM_INSTRUCTION,
        transcription_model=OPENAI_REALTIME_TRANSCRIPTION_MODEL,
        language=OPENAI_REALTIME_LANGUAGE,
    )


def _get_realtime_bridge(session_id: str) -> OpenAIRealtimeBridge:
    bridge = realtime_sessions.get(session_id)
    if bridge is None:
        bridge = _new_realtime_bridge()
        realtime_sessions[session_id] = bridge
    return bridge

@app.on_event("startup")
def startup_event():
    global gemma, tts, asr
    print("=" * 60)
    print("STARTING SPEECH-TO-SPEECH SERVER")
    print(f"Provider: {SPEECH_PROVIDER}")
    print("=" * 60)

    if SPEECH_PROVIDER == "openai_realtime":
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is required")
        print(
            f"[Server] OpenAI Realtime session manager ready "
            f"(model={OPENAI_REALTIME_MODEL}, voice={OPENAI_REALTIME_VOICE})",
            flush=True,
        )
        return

    if SPEECH_PROVIDER != "cascaded":
        raise RuntimeError(
            "Unsupported SPEECH_PROVIDER. Use 'openai_realtime' or 'cascaded'."
        )

    import torch
    from transformers import pipeline
    try:
        from .model_handler import GemmaAudioProcessor
        from .tts_handler import OfflineTTSHandler
    except ImportError:
        from model_handler import GemmaAudioProcessor
        from tts_handler import OfflineTTSHandler
    
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


@app.on_event("shutdown")
async def shutdown_event():
    bridges = list(realtime_sessions.values())
    realtime_sessions.clear()
    for bridge in bridges:
        await bridge.reset()

# Conversation history state (sliding window of last 20 turns of text history)
chat_history = []
MAX_HISTORY_TURNS = 20

@app.post("/reset")
async def reset_conversation(request: Request, _auth=Depends(_verify_auth)):
    global chat_history
    if SPEECH_PROVIDER == "openai_realtime":
        session_id = _speech_session_id(request)
        bridge = realtime_sessions.pop(session_id, None)
        if bridge is not None:
            await bridge.reset()
        print(f"[Server] Realtime session reset: {session_id}", flush=True)
        return {"status": "reset"}

    chat_history = []
    print("[Server] Conversation history reset!", flush=True)
    return {"status": "reset"}

@app.get("/last_turn")
def get_last_turn(request: Request, _auth=Depends(_verify_auth)):
    global chat_history
    if SPEECH_PROVIDER == "openai_realtime":
        session_id = _speech_session_id(request)
        bridge = realtime_sessions.get(session_id)
        if bridge is None:
            return {"user": "", "assistant": ""}
        return bridge.get_last_turn()

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

    if SPEECH_PROVIDER == "openai_realtime":
        session_id = _speech_session_id(request)
        realtime_bridge = _get_realtime_bridge(session_id)
        speaker_id = request.headers.get("X-Speaker-ID")
        if speaker_id is not None and not speaker_id.isdigit():
            raise HTTPException(status_code=400, detail="Invalid X-Speaker-ID")

        rms = float(np.sqrt(np.mean(audio_data**2))) if len(audio_data) > 0 else 0.0
        if rms < 0.005:
            print(f"[Server] Silence detected (RMS={rms:.5f}). Bypassing OpenAI Realtime.", flush=True)
            return StreamingResponse(
                iter([b""]),
                media_type="application/octet-stream",
                headers={"X-Sample-Rate": str(realtime_bridge.sample_rate)},
            )

        try:
            await realtime_bridge.ensure_ready()
        except Exception as e:
            print(f"[Server Error] OpenAI Realtime connection failed: {e}", flush=True)
            raise HTTPException(status_code=502, detail="OpenAI Realtime connection failed") from e

        async def realtime_audio_generator():
            try:
                async for chunk in realtime_bridge.stream_turn(
                    audio_data,
                    speaker_id=int(speaker_id) if speaker_id is not None else None,
                ):
                    yield chunk
            except Exception as e:
                print(f"[Server Error] OpenAI Realtime streaming failed: {e}", flush=True)
                import traceback
                traceback.print_exc()

        return StreamingResponse(
            realtime_audio_generator(),
            media_type="application/octet-stream",
            headers={
                "X-Sample-Rate": str(realtime_bridge.sample_rate),
                "X-Streaming": "true",
                "X-Speech-Provider": "openai_realtime",
                "X-Session-ID": session_id,
            },
        )
    
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
        
    # 3. Generate full LLM response → synthesize in ONE pass → stream audio
    async def cascaded_audio_generator():
        global chat_history

        print(f"[Server] Running LLM inference...", flush=True)
        llm_start = time.time()

        try:
            # ── Step A: Collect the FULL text response from Gemma ──────────
            # Accumulate ALL clauses from the streamer, then synthesize the
            # complete text at once.  espeak-ng gets full sentence context
            # for natural Turkish prosody — no clause-boundary artifacts.
            full_response_parts = []
            for clause in gemma.generate_response_stream(chat_history, []):
                clause = clause.strip()
                if clause:
                    full_response_parts.append(clause)
                    print(f"[Server] Clause: '{clause[:80]}...'"
                          if len(clause) > 80 else
                          f"[Server] Clause: '{clause}'",
                          flush=True)

            response_text = " ".join(full_response_parts).strip()
            llm_elapsed = time.time() - llm_start

            if not response_text:
                if chat_history and chat_history[-1]["role"] == "user":
                    chat_history.pop()
                return

            print(f"[Server] Full response ({llm_elapsed:.2f}s, {len(response_text)} chars): "
                  f"'{response_text[:150]}...'"
                  if len(response_text) > 150 else
                  f"[Server] Full response ({llm_elapsed:.2f}s): '{response_text}'",
                  flush=True)

            # ── Step B: Synthesize the ENTIRE response at once ─────────────
            print(f"[Server] Synthesizing TTS...", flush=True)
            tts_start = time.time()
            tts_audio, sample_rate = await asyncio.to_thread(tts.synthesize, response_text)
            tts_elapsed = time.time() - tts_start

            if tts_audio is None or len(tts_audio) == 0:
                if chat_history and chat_history[-1]["role"] == "user":
                    chat_history.pop()
                return

            total_latency = time.time() - llm_start
            print(f"[Server] TTS ready in {tts_elapsed:.2f}s "
                  f"(total {total_latency:.2f}s, {len(tts_audio)} samples @ {sample_rate}Hz)",
                  flush=True)

            # ── Step C: Stream audio in 1024-sample chunks ─────────────────
            for i in range(0, len(tts_audio), 1024):
                yield tts_audio[i:i + 1024].tobytes()

            # ── Step D: Update chat history ────────────────────────────────
            chat_history.append({"role": "assistant", "content": response_text})

        except Exception as e:
            print(f"[Server Error] Exception: {e}", flush=True)
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
        "provider": SPEECH_PROVIDER,
        "model": OPENAI_REALTIME_MODEL if SPEECH_PROVIDER == "openai_realtime" else MODEL_ID,
        "active_sessions": len(realtime_sessions) if SPEECH_PROVIDER == "openai_realtime" else 1,
    }


if __name__ == "__main__":
    print(f"[Server] Auth {'enabled' if _AUTH_ENABLED else 'DISABLED (dev mode)'}", flush=True)
    uvicorn.run("server:app", host=HOST, port=PORT, log_level="info")
