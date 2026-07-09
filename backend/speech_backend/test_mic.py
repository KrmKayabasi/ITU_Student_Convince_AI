import sounddevice as sd
import numpy as np
import time
import sys

def main():
    print("=" * 50)
    # Print device info
    try:
        input_device = sd.query_devices(kind='input')
        print(f"Giriş Cihazı: {input_device.get('name')}")
        print(f"Sample Rate: {input_device.get('default_samplerate')} Hz")
    except Exception as e:
        print(f"Hata (Cihaz sorgulama): {e}")
        
    print("\n10 Saniye boyunca mikrofon seviyesi yazdırılacak.")
    print("Lütfen konuşun ve değerlerin yükselip yükselmediğini izleyin.")
    print("=" * 50)
    time.sleep(1.0)
    
    def callback(indata, frames, time, status):
        rms = np.sqrt(np.mean(indata**2))
        # Draw a simple bar indicator
        bar_length = int(rms * 100)
        bar = "#" * min(bar_length, 40)
        spaces = " " * (40 - len(bar))
        sys.stdout.write(f"\r[{bar}{spaces}] RMS: {rms:.6f}")
        sys.stdout.flush()

    stream = sd.InputStream(samplerate=16000, channels=1, callback=callback)
    with stream:
        try:
            for i in range(100):
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
    print("\n\nTest bitti.")

if __name__ == "__main__":
    main()
