import time
import torch
from transformers import VitsModel, VitsTokenizer

def benchmark_tts():
    model_id = "facebook/mms-tts-tur"
    print(f"Loading tokenizer...")
    tokenizer = VitsTokenizer.from_pretrained(model_id)
    text = "Merhaba! Ben de hoş geldiniz. Size nasıl yardımcı olabilirim?"
    
    print("\n" + "=" * 50)
    print("BENCHMARKING CPU vs MPS FOR VITS TTS")
    print("=" * 50)
    
    # 1. Test CPU
    print("[CPU] Modeli yükleniyor...")
    start_load = time.time()
    model_cpu = VitsModel.from_pretrained(model_id).to("cpu")
    print(f"[CPU] Yükleme süresi: {time.time() - start_load:.2f} saniye")
    
    inputs_cpu = tokenizer(text=text, return_tensors="pt").to("cpu")
    
    # Warmup
    with torch.no_grad():
        _ = model_cpu(**inputs_cpu)
        
    # Benchmark
    start_inf = time.time()
    for _ in range(5):
        with torch.no_grad():
            outputs = model_cpu(**inputs_cpu)
    print(f"[CPU] Ortalama sentez süresi: {(time.time() - start_inf) / 5:.4f} saniye")
    
    # 2. Test MPS
    if torch.backends.mps.is_available():
        print("\n[MPS] Modeli yükleniyor...")
        start_load = time.time()
        model_mps = VitsModel.from_pretrained(model_id).to("mps")
        print(f"[MPS] Yükleme süresi: {time.time() - start_load:.2f} saniye")
        
        inputs_mps = tokenizer(text=text, return_tensors="pt").to("mps")
        
        # Warmup
        with torch.no_grad():
            _ = model_mps(**inputs_mps)
            
        # Benchmark
        start_inf = time.time()
        for _ in range(5):
            with torch.no_grad():
                outputs = model_mps(**inputs_mps)
        print(f"[MPS] Ortalama sentez süresi: {(time.time() - start_inf) / 5:.4f} saniye")
    else:
        print("\nMPS desteklenmiyor.")

if __name__ == "__main__":
    benchmark_tts()
