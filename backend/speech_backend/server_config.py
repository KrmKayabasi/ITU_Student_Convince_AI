import os
import sys

# Server Settings
PORT = 8002
HOST = "0.0.0.0"

# Model Config
MODEL_ID = "google/gemma-4-12B-it"
QUANTIZATION = "none"  # Run unquantized (bf16) on H200 for maximum quality
DEVICE = "mps" if sys.platform == "darwin" else "cuda"
ENABLE_THINKING = False

# TTS Config
TTS_MODEL_ID = os.path.join(os.path.dirname(__file__), "vits-piper-tr_TR-dfki-medium")  # Fast local VITS female voice

# Audio Settings
SAMPLE_RATE = 16000  # Input audio sample rate for LLM

sys_prompt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "SYSTEM_PROMPT.md"))
if os.path.exists(sys_prompt_path):
    with open(sys_prompt_path, 'r', encoding='utf-8') as f:
        SYSTEM_INSTRUCTION = f.read().strip()
else:
    SYSTEM_INSTRUCTION = (
        "Sen son derece yardımsever, kibar ve cana yakın bir Türkçe sesli asistansın. "
        "Kullanıcının konuşmasını dinle ve doğrudan Türkçe yanıt ver. "
        "Yanıtlarını konuşma diline uygun, çok kısa ve öz tut (en fazla 1-2 kısa cümle). "
        "Düşüncelerini içinden geçir ama dışarıya sadece cevabı söyle. "
        "Asla markdown kullanma (*, #, listeler vb. kullanma)."
    )
