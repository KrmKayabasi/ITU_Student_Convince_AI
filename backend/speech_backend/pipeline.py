import sys
import time
import numpy as np
from config import (
    MODEL_ID, QUANTIZATION, TTS_MODEL_ID, DEVICE,
    SYSTEM_INSTRUCTION, SAMPLE_RATE, CHANNELS,
    ENERGY_THRESHOLD, SILENCE_DURATION, MIN_SPEECH_DURATION,
    PUSH_TO_TALK, ENABLE_THINKING, PLAYBACK_INTERRUPTION_MODE
)
from audio_handler import AudioHandler
from model_handler import GemmaAudioProcessor
from tts_handler import OfflineTTSHandler

def main():
    print("=" * 60)
    print("      TÜRKÇE SESLİ KONUŞMA HATTI (SPEECH-TO-SPEECH)      ")
    print("=" * 60)
    print(f"Cihaz: {DEVICE}")
    print(f"Gemma 4 Modeli: {MODEL_ID} ({QUANTIZATION})")
    print(f"TTS Modeli: {TTS_MODEL_ID}")
    print("=" * 60)

    # Initialize Audio Handler
    audio_handler = AudioHandler(
        sample_rate=SAMPLE_RATE,
        channels=CHANNELS,
        energy_threshold=ENERGY_THRESHOLD,
        silence_duration=SILENCE_DURATION,
        min_speech_duration=MIN_SPEECH_DURATION,
        interruption_mode=PLAYBACK_INTERRUPTION_MODE
    )

    # Note: Noise calibration is not needed for the robust neural-network Silero VAD

    # Initialize Gemma Model Handler
    try:
        gemma_processor = GemmaAudioProcessor(
            model_id=MODEL_ID,
            quantization=QUANTIZATION,
            device=DEVICE,
            enable_thinking=ENABLE_THINKING
        )
    except Exception as e:
        print(f"[Error] Gemma 4 yüklenirken hata oluştu: {e}")
        print("İpucu: Gereksinimlerin düzgün kurulduğundan ve cihaz hafızasının yeterli olduğundan emin olun.")
        sys.exit(1)

    # TTS Handler disabled by user request
    tts_handler = None

    # Initialize Whisper ASR model for local text-history transcription
    print("[ASR] Çevrimdışı Whisper Large v3 Turbo yükleniyor...", flush=True)
    import torch
    from transformers import pipeline as hf_pipeline
    try:
        asr = hf_pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-large-v3-turbo",
            torch_dtype=torch.float16 if DEVICE != "cpu" else torch.float32,
            device=DEVICE
        )
        print("[ASR] Whisper Large v3 Turbo başarıyla yüklendi.", flush=True)
    except Exception as e:
        print(f"[Error] ASR modeli yüklenirken hata oluştu: {e}")
        sys.exit(1)

    chat_history = []
    MAX_HISTORY_TURNS = 15

    print("\n[Hazır] Sistem hazır! Konuşma başlatılıyor...", flush=True)

    try:
        interrupted = False
        audio_data = None
        
        while True:
            # 1. Listen for user audio (manual push-to-talk or auto VAD, skip if interrupted)
            if not interrupted:
                if PUSH_TO_TALK:
                    audio_data = audio_handler.listen_push_to_talk()
                else:
                    audio_data = audio_handler.listen_until_silence()

            # Reset interruption flag for this turn
            interrupted = False

            if audio_data is None or len(audio_data) == 0:
                continue

            # 2. Gemma 4 Inference
            print("[Düşünüyor] Modeli çalıştırıyorum...", flush=True)

            # Deşifre (Transcribe user audio to update text history).
            # Silence check prevents Whisper from hallucinating Turkish text from
            # near-silent input (matching the guard in server.py).
            rms = float(np.sqrt(np.mean(audio_data**2))) if len(audio_data) > 0 else 0.0
            if rms < 0.008:
                print(f"[Pipeline] Silence detected (RMS={rms:.5f}). Bypassing ASR.", flush=True)
                user_text = ""
            else:
                asr_res = asr(audio_data)
                user_text = asr_res.get("text", "").strip()
                if user_text:
                    print(f"[Kullanıcı]: {user_text}", flush=True)
                else:
                    user_text = "[Kullanıcı anlaşılmayan bir ses çıkardı]"
                    print("[Kullanıcı]: (Ses deşifre edilemedi)", flush=True)
            
            # Build current messages list
            prompt_system = SYSTEM_INSTRUCTION
            if ENABLE_THINKING:
                prompt_system = "<|think|>\n" + prompt_system
                
            messages = [{"role": "system", "content": prompt_system}]
            for turn in chat_history:
                messages.append(turn)
            messages.append({"role": "user", "content": "Kullanıcı ses kaydını gönderdi: <|audio|>"})
            
            thinking, response = gemma_processor.generate_response(
                audio_data=audio_data,
                messages=messages
            )

            # Update chat history
            chat_history.append({"role": "user", "content": user_text})
            chat_history.append({"role": "assistant", "content": response})
            
            # Keep sliding window
            while len(chat_history) > 2 * MAX_HISTORY_TURNS:
                chat_history.pop(0)
                chat_history.pop(0)

            # Print results
            if thinking:
                print(f"\n[Model Düşünce Süreci]:\n{thinking}\n")
            print(f"[Asistan Yanıtı]: {response}\n")

            if not response.strip():
                print("[Sistem] Boş yanıt üretildi, pas geçiliyor.", flush=True)
                continue

            # 3. Text-to-Speech synthesis (Disabled by user request)
            if tts_handler is not None:
                tts_audio, tts_sr = tts_handler.synthesize(response)
                if tts_audio is not None:
                    print("[Konuşuyor] Yanıt seslendiriliyor...", flush=True)
                    interrupted, user_audio = audio_handler.play_audio(tts_audio, sample_rate=tts_sr)
                    if interrupted:
                        print("\n[Söz Kesme] Kullanıcı sözü kesti! Yeni yanıt hazırlanıyor...", flush=True)
                        audio_data = user_audio
                    else:
                        print("[Konuşma Bitti] Yeniden dinlemeye geçiliyor...", flush=True)
            else:
                # Direct local console-only path
                print("[Sistem] Yeniden dinlemeye geçiliyor...", flush=True)
                time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n\n[Çıkış] Program kullanıcı tarafından sonlandırıldı. Hoşçakalın!")
    finally:
        if 'audio_handler' in locals():
            audio_handler.close()

if __name__ == "__main__":
    main()
