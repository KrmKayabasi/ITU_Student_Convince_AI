import os
import tarfile
import urllib.request
import time
import sys
import ssl

# Bypass SSL verification for urllib downloads on macOS
ssl._create_default_https_context = ssl._create_unverified_context

def setup_sherpa_tts():
    print("=" * 50)
    print("SETTING UP SHERPA-ONNX OFFLINE TURKISH TTS")
    print("=" * 50)
    
    # 1. Install sherpa-onnx
    try:
        import sherpa_onnx
        print("[Install] sherpa-onnx already installed.")
    except ImportError:
        print("[Install] Installing sherpa-onnx...")
        import subprocess
        subprocess.run(["./venv/bin/pip", "install", "sherpa-onnx"])
        import sherpa_onnx
        print("[Install] Installed successfully.")

    # 2. Download pre-packaged Turkish model (Fahrettin voice, very high quality)
    model_name = "vits-piper-tr_TR-fahrettin-medium"
    tar_file = f"{model_name}.tar.bz2"
    url = f"https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/{tar_file}"
    
    if not os.path.exists(model_name):
        print(f"[Download] Downloading {model_name} (approx. 15MB)...")
        start = time.time()
        urllib.request.urlretrieve(url, tar_file)
        print(f"[Download] Download completed in {time.time() - start:.2f} seconds.")
        
        print(f"[Extract] Extracting {tar_file}...")
        with tarfile.open(tar_file, "r:bz2") as tar:
            tar.extractall()
        print("[Extract] Extracted successfully.")
        
        # Clean up zip
        if os.path.exists(tar_file):
            os.remove(tar_file)
    else:
        print(f"[Model] Model folder {model_name} already exists.")
        
    # Verify files
    model_path = os.path.join(model_name, "tr_TR-fahrettin-medium.onnx")
    tokens_path = os.path.join(model_name, "tokens.txt")
    data_dir = os.path.join(model_name, "espeak-ng-data")
    
    if os.path.exists(model_path) and os.path.exists(tokens_path) and os.path.exists(data_dir):
        print("[Success] All model files verified.")
        
        # 3. Quick benchmark to see quality and speed
        print("\n[Benchmark] Running test synthesis...")
        import numpy as np
        
        tts_config = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                    model=model_path,
                    tokens=tokens_path,
                    data_dir=data_dir
                ),
                num_threads=2,
                debug=False
            )
        )
        
        start_load = time.time()
        tts = sherpa_onnx.OfflineTts(tts_config)
        print(f"[Benchmark] TTS engine loaded in {time.time() - start_load:.4f} seconds.")
        
        text = "Merhaba! Ben Fahrettin. Bu yeni ses sentezi motorunun çevrimdışı sesidir."
        
        start_gen = time.time()
        audio = tts.generate(text)
        duration = time.time() - start_gen
        
        # Save sample output using scipy
        import scipy.io.wavfile as wav
        import numpy as np
        samples = np.array(audio.samples, dtype=np.float32)
        wav.write("fahrettin_test.wav", audio.sample_rate, samples)
        
        print(f"[Benchmark] Sentez süresi: {duration:.4f} saniye!")
        print(f"[Benchmark] Dosya kaydedildi: fahrettin_test.wav")
    else:
        print("[Error] Some model files are missing!")

if __name__ == "__main__":
    setup_sherpa_tts()
