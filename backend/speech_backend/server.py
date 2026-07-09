import os
# Force server process to bind exclusively to one GPU (GPU 0)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import numpy as np
import time
import torch
from transformers import pipeline

from server_config import HOST, PORT, MODEL_ID, QUANTIZATION, DEVICE, ENABLE_THINKING, TTS_MODEL_ID, SYSTEM_INSTRUCTION
from model_handler import GemmaAudioProcessor
from tts_handler import OfflineTTSHandler

app = FastAPI(title="Turkish Speech-to-Speech Server (NVIDIA H200)")

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
def reset_conversation():
    global chat_history
    chat_history = []
    print("[Server] Conversation history reset!", flush=True)
    return {"status": "reset"}

@app.get("/last_turn")
def get_last_turn():
    global chat_history
    if len(chat_history) >= 2:
        return {
            "user": chat_history[-2]["content"],
            "assistant": chat_history[-1]["content"]
        }
    return {"user": "", "assistant": ""}

@app.post("/chat_stream")
async def chat_stream(request: Request):
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
    
    # Enforce history limit (sliding window of last 20 turns of text history)
    while len(chat_history) > 1 + 2 * MAX_HISTORY_TURNS:
        chat_history.pop(1)  # Remove oldest user message
        chat_history.pop(1)  # Remove oldest assistant response
        
    # 3. Run Cascaded LLM-to-TTS Streaming
    async def cascaded_audio_generator():
        global chat_history
        full_response_parts = []
        
        print(f"[Server] Running Cascaded LLM-to-TTS Streaming...", flush=True)
        start_stream_time = time.time()
        first_chunk_sent = False
        
        try:
            # Iterate over text clauses generated in real-time by Gemma
            # We pass an empty audio list since the history is now text-only!
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
                
                if tts_audio is not None and len(tts_audio) > 0:
                    if not first_chunk_sent:
                        total_first_latency = time.time() - start_stream_time
                        print(f"[Server Stream] First audio chunk ready in {total_first_latency:.2f}s! (TTS took {tts_elapsed:.2f}s)", flush=True)
                        first_chunk_sent = True
                        
                    # Stream this clause's audio bytes in 1024 float32 sample chunks
                    chunk_size = 1024
                    for i in range(0, len(tts_audio), chunk_size):
                        chunk = tts_audio[i:i+chunk_size]
                        yield chunk.tobytes()
                        
            # After generation completes, update the assistant chat history
            full_response = " ".join(full_response_parts).strip()
            print(f"[Server Stream] Generation finished. Full response: '{full_response}'", flush=True)
            
            if full_response:
                chat_history.append({"role": "assistant", "content": full_response})
            else:
                # Cleanup if empty
                chat_history.pop()
                
        except Exception as e:
            print(f"[Server Stream Error] Exception during streaming: {e}", flush=True)
            # Cleanup history on failure
            if chat_history and chat_history[-1]["role"] == "user":
                chat_history.pop()
                
    headers = {
        "X-Sample-Rate": "24000",
        "X-Streaming": "true"
    }
    
    return StreamingResponse(
        cascaded_audio_generator(),
        media_type="application/octet-stream",
        headers=headers
    )

if __name__ == "__main__":
    uvicorn.run("server:app", host=HOST, port=PORT, log_level="info")
