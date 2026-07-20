import os

# Server endpoint on H200
SERVER_URL = os.environ.get("SPEECH_SERVER_URL", "http://192.168.2.9:8000/chat_stream")

# Auth token — set SPEECH_SERVER_TOKEN in the environment to match the server
AUTH_TOKEN = os.environ.get("SPEECH_SERVER_TOKEN", "")

# Local Audio Settings
SAMPLE_RATE = 16000  # VAD and recording rate (Hz)
CHANNELS = 1
BLOCKSIZE = 320  # 20ms frames for optimal WebRTC alignment

# Speech / Silence parameters (Silero VAD) — tuned for noisy / crowded environments
SILENCE_DURATION = 0.40   # Faster turn detection in noise
MIN_SPEECH_DURATION = 0.35 # Filter short background noise bursts

# Interruption settings
INTERRUPTION_MODE = os.environ.get("INTERRUPTION_MODE", "both")  # "both", "key_only", "vad_only", "none"
