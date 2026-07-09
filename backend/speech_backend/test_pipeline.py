import numpy as np
import torch
import time
from config import TTS_MODEL_ID, DEVICE
from tts_handler import OfflineTTSHandler

def test_tts():
    print("=" * 40)
    print("TTS MODÜLÜ TEST EDİLİYOR")
    print("=" * 40)
    
    try:
        tts = OfflineTTSHandler(model_id=TTS_MODEL_ID, device=DEVICE)
        
        test_text = "Merhaba, bu çevrimdışı ses sentezi testi başarılı bir şekilde gerçekleştirildi."
        audio, sr = tts.synthesize(test_text)
        
        if audio is not None and len(audio) > 0:
            print(f"[SUCCESS] TTS başarıyla ses üretti. Uzunluk: {len(audio)} örnek, Sample Rate: {sr}Hz")
            
            # Save test file to disk
            import scipy.io.wavfile as wav
            output_file = "test_tts_output.wav"
            wav.write(output_file, sr, (audio * 32767).astype(np.int16))
            print(f"[SUCCESS] Test sesi '{output_file}' dosyasına kaydedildi.")
        else:
            print("[FAILURE] TTS ses üretemedi (boş veya None döndü).")
    except Exception as e:
        print(f"[ERROR] TTS testinde hata: {e}")

def test_gemma_import():
    print("=" * 40)
    print("GEMMA 4 İTHALAT TESTİ")
    print("=" * 40)
    try:
        from transformers import AutoProcessor, AutoModelForImageTextToText
        print("[SUCCESS] AutoProcessor ve AutoModelForImageTextToText başarıyla import edildi.")
        print(f"PyTorch Sürümü: {torch.__version__}")
        print(f"MPS Desteği (Apple Silicon GPU): {torch.backends.mps.is_available()}")
    except Exception as e:
        print(f"[ERROR] Import hatası: {e}")

if __name__ == "__main__":
    test_gemma_import()
    print()
    test_tts()
