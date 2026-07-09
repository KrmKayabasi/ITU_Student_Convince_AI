import os

# Server endpoint on H200
SERVER_URL = "http://192.168.2.9:8000/chat_stream"

# Local Audio Settings
SAMPLE_RATE = 16000  # VAD and recording rate (Hz)
CHANNELS = 1
BLOCKSIZE = 320  # 20ms frames for optimal WebRTC alignment

# Speech / Silence parameters (Silero VAD)
SILENCE_DURATION = 0.45  # Seconds of silence before user finishes speaking (highly responsive)
MIN_SPEECH_DURATION = 0.15  # Minimum duration to register speech (fast onset)

# Interruption settings
INTERRUPTION_MODE = "both"  # "both", "key_only", "vad_only", "none"
