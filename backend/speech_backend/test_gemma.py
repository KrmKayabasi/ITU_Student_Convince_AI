import numpy as np
import torch
import time
from model_handler import GemmaAudioProcessor
from config import SYSTEM_INSTRUCTION

def test_gemma_audio():
    print("=" * 40)
    print("GEMMA 4 SES İNFERANS TESTİ")
    print("=" * 40)
    
    # 1. Generate 1 second of dummy audio (16kHz sine wave)
    print("[Test] 1 saniyelik test ses sinyali üretiliyor...")
    sample_rate = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    # 440 Hz tone
    dummy_audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    dummy_audio = dummy_audio.astype(np.float32)
    
    # 2. Load Gemma 4 (Using e2b-it for fast test, with int4 quantization)
    try:
        gemma = GemmaAudioProcessor(
            model_id="google/gemma-4-e2b-it", 
            quantization="none", 
            device="mps",
            enable_thinking=True
        )
        
        # 3. Generate response
        print("[Test] İnferans başlatılıyor (Bu ilk yüklemede birkaç dakika sürebilir)...")
        thinking, response = gemma.generate_response(
            audio_data=dummy_audio,
            system_instruction=SYSTEM_INSTRUCTION
        )
        
        print("\n" + "=" * 40)
        print("[SUCCESS] Gemma 4 Başarıyla Yanıt Üretti:")
        print("=" * 40)
        if thinking:
            print(f"Düşünce Süreci:\n{thinking}\n")
        print(f"Cevap: {response}")
        print("=" * 40)
        
    except Exception as e:
        print(f"[ERROR] Gemma 4 testinde hata oluştu: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_gemma_audio()
