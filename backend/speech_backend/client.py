import sys
import time
import httpx
import numpy as np

from client_config import SERVER_URL, SAMPLE_RATE, CHANNELS, BLOCKSIZE, SILENCE_DURATION, MIN_SPEECH_DURATION, INTERRUPTION_MODE
from audio_handler import AudioHandler

def main():
    print("=" * 60)
    print("    TÜRKÇE SESLİ KONUŞMA HATTI (İSTEMCİ - CLIENT)      ")
    print("=" * 60)
    print(f"Sunucu Adresi: {SERVER_URL}")
    print(f"Lokal Kayıt Hızı: {SAMPLE_RATE} Hz")
    print(f"Söz Kesme Modu: {INTERRUPTION_MODE}")
    print("=" * 60)

    # Initialize local Audio Handler
    audio_handler = AudioHandler(
        sample_rate=SAMPLE_RATE,
        channels=CHANNELS,
        energy_threshold=1.5,
        silence_duration=SILENCE_DURATION,
        min_speech_duration=MIN_SPEECH_DURATION,
        interruption_mode=INTERRUPTION_MODE
    )

    # Ensure audio blocksize is aligned with config (320 samples / 20ms)
    audio_handler.blocksize = BLOCKSIZE

    # Reset conversation history on remote server on startup
    reset_url = SERVER_URL.replace("/chat_stream", "/reset")
    try:
        httpx.post(reset_url, timeout=5.0)
        print("[İstemci] Sunucu konuşma geçmişi temizlendi.", flush=True)
    except Exception as e:
        print(f"[İstemci Uyarı] Sunucu geçmişi sıfırlanamadı: {e}", flush=True)

    print("\n[Hazır] Sistem hazır! Konuşma başlatılıyor...", flush=True)

    try:
        interrupted = False
        audio_data = None
        
        while True:
            # 1. Listen for user audio (auto VAD, skip if interrupted)
            if not interrupted:
                audio_data = audio_handler.listen_until_silence()

            # Reset interruption flag for this turn
            interrupted = False

            if audio_data is None or len(audio_data) == 0:
                continue

            # 2. Send audio to remote H200 server and stream back the response
            print("[Düşünüyor] Sunucuya gönderiliyor ve yanıt bekleniyor...", flush=True)
            try:
                # Use httpx to make a streaming POST request with raw float32 bytes
                audio_bytes = audio_data.tobytes()
                sample_rate = 24000  # Default fallback
                
                start_network = time.time()
                response_chunks = []
                ttfa = 0.0
                first_chunk_received = False
                
                with httpx.stream("POST", SERVER_URL, content=audio_bytes, timeout=15.0) as r:
                    r.raise_for_status()
                    
                    # Read metadata from server headers
                    sample_rate = int(r.headers.get("X-Sample-Rate", "24000"))
                    
                    # Accumulate raw float32 chunks as they stream back
                    for chunk in r.iter_bytes():
                        if chunk:
                            if not first_chunk_received:
                                ttfa = time.time() - start_network
                                first_chunk_received = True
                            response_chunks.append(chunk)
                            
                network_latency = time.time() - start_network
                
                if not response_chunks:
                    print("[Sistem] Sunucudan ses gelmedi, pas geçiliyor.", flush=True)
                    continue
                    
                # Concatenate all binary chunks and convert back to float32 numpy array
                full_bytes = b"".join(response_chunks)
                tts_audio = np.frombuffer(full_bytes, dtype=np.float32)
                
                print(f"[Asistan Yanıtı]: Alındı (İlk Ses / TTFA: {ttfa:.2f}s, Toplam Akış: {network_latency:.2f}s)")
                
                # 3. Playback with local hardware barge-in interruption check
                print("[Konuşuyor] Yanıt seslendiriliyor...", flush=True)
                interrupted, user_audio = audio_handler.play_audio(tts_audio, sample_rate=sample_rate)
                
                if interrupted:
                    print("\n[Söz Kesme] Kullanıcı sözü kesti! Yeni yanıt hazırlanıyor...", flush=True)
                    audio_data = user_audio
                else:
                    print("[Konuşma Bitti] Yeniden dinlemeye geçiliyor...", flush=True)
                    
            except httpx.HTTPStatusError as e:
                print(f"\n[HTTP Hata] Sunucu hata kodu döndürdü: {e}")
                time.sleep(1)
            except Exception as e:
                print(f"\n[Bağlantı Hatası] Sunucuya bağlanılamadı: {e}")
                time.sleep(1)

            if not interrupted:
                time.sleep(0.5)  # Small cooldown before next listening turn

    except KeyboardInterrupt:
        print("\n\n[Çıkış] Program kullanıcı tarafından sonlandırıldı. Hoşçakalın!")
    finally:
        if 'audio_handler' in locals():
            audio_handler.close()

if __name__ == "__main__":
    main()
