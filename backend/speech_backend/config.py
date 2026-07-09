import os

# --- MODEL CONFIGURATION ---
# Gemma 4 models: "google/gemma-4-e2b-it" or "google/gemma-4-e4b-it"
MODEL_ID = os.environ.get("GEMMA_MODEL_ID", "google/gemma-4-e4b-it")

# Quantization: "none", "int4", or "int8"
# Note: "int4" and "int8" require `optimum-quanto`
QUANTIZATION = os.environ.get("GEMMA_QUANTIZATION", "none")

# Disable chain-of-thought thinking to reduce latency (essential for real-time speech)
ENABLE_THINKING = False

# Text-to-Speech Model Config
TTS_MODEL_ID = os.path.join(os.path.dirname(__file__), "vits-piper-tr_TR-dfki-medium")

# Device configuration: "mps" (Mac GPU), "cuda", or "cpu"
DEVICE = os.environ.get("DEVICE", "mps")

sys_prompt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "SYSTEM_PROMPT.md"))
if os.path.exists(sys_prompt_path):
    with open(sys_prompt_path, 'r', encoding='utf-8') as f:
        SYSTEM_INSTRUCTION = f.read().strip()
else:
    SYSTEM_INSTRUCTION = (
        "Sen yararlı bir Türkçe sesli asistansın. "
        "Kullanıcının sorusuna doğrudan, doğal ve samimi bir şekilde Türkçe cevap ver. "
        "Konuşmanın akıcı olmasını sağla, ancak gerektiğinde detaylı açıklamalar yapmaktan çekinme."
    )

# --- AUDIO CONFIGURATION ---
SAMPLE_RATE = 16000  # Gemma 4 audio and MMS-TTS both work natively/best at 16kHz
CHANNELS = 1         # Mono-channel audio is required

# Chunk size for sounddevice stream reading
CHUNK_SIZE = 1024

# Voice Activity Detection (VAD) config
# ENERGY_THRESHOLD is the RMS multiplier of the background noise level to detect speech
ENERGY_THRESHOLD = 2.5

# Silence duration (seconds) before triggering the generation
SILENCE_DURATION = 0.8

# Pre-speech threshold (how long speech must last to be considered valid, in seconds)
MIN_SPEECH_DURATION = 0.3

# Use manual Press-to-Talk instead of automatic VAD (much more robust)
PUSH_TO_TALK = False

# Interruption mode during assistant playback
# Options: 
#   "both"     -> Can interrupt by either speaking (VAD) or pressing Enter key (requires headphones or low volume for VAD)
#   "key_only" -> Can only interrupt by pressing Enter key (100% immune to speaker echo/feedback)
#   "none"     -> Interruption disabled
PLAYBACK_INTERRUPTION_MODE = "both"
