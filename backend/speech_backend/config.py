import os
import sys

# --- MODEL CONFIGURATION ---
# Gemma 4 models: "google/gemma-4-e2b-it", "google/gemma-4-e4b-it", "google/gemma-4-12b-it"
# The server deployment uses a larger model; override via GEMMA_MODEL_ID or MODEL_ID env vars.
_DEFAULT_MODEL = "google/gemma-4-12b-it" if os.environ.get("SPEECH_SERVER_MODE") else "google/gemma-4-e4b-it"
MODEL_ID = os.environ.get("GEMMA_MODEL_ID", os.environ.get("MODEL_ID", _DEFAULT_MODEL))

# Quantization: "none", "int4", or "int8"
QUANTIZATION = os.environ.get("GEMMA_QUANTIZATION", "none")

# Disable chain-of-thought thinking to reduce latency (essential for real-time speech)
ENABLE_THINKING = os.environ.get("ENABLE_THINKING", "0") == "1"

# Text-to-Speech Model Config
TTS_MODEL_ID = os.environ.get(
    "TTS_MODEL_ID",
    os.path.join(os.path.dirname(__file__), "vits-piper-tr_TR-fahrettin-medium"),
)

# Device configuration: "mps" (Mac GPU), "cuda", or "cpu".
# Auto-detect CUDA availability for server deployments.
_DEVICE_DEFAULT = "cuda" if (not sys.platform == "darwin") else "mps"
DEVICE = os.environ.get("DEVICE", _DEVICE_DEFAULT)

# System prompt — loaded from file in the local directory (Docker context) or project root (local fallback).
_sys_prompt_path_local = os.path.join(os.path.dirname(__file__), "SYSTEM_PROMPT.md")
_sys_prompt_path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "SYSTEM_PROMPT.md"))
_sys_prompt_path = _sys_prompt_path_local if os.path.exists(_sys_prompt_path_local) else _sys_prompt_path_root

def _strip_markdown_for_llm(text: str) -> str:
    """Strip Markdown formatting from the system prompt before sending to LLM.

    The LLM learns to reproduce formatting it sees in its system prompt.
    Removing Markdown syntax prevents the LLM from outputting *, _, #, `, |
    characters that espeak-ng mispronounces as Turkish words ("yıldız", "alt
    çizgi", etc.).  Content is preserved; only syntax markers are removed.
    """
    import re as _re
    # Strip bold/italic markers
    text = _re.sub(r'\*{1,3}', '', text)
    # Strip underscore emphasis (but not in URLs or words)
    text = _re.sub(r'(?<!\w)_{1,2}(?!\w)', '', text)
    # Strip heading markers (##, ###, etc.)
    text = _re.sub(r'^#{1,6}\s+', '', text, flags=_re.MULTILINE)
    # Strip code backticks
    text = text.replace('`', '')
    # Strip strikethrough
    text = text.replace('~~', '')
    # Strip blockquote markers
    text = _re.sub(r'^>\s?', '', text, flags=_re.MULTILINE)
    # Strip table pipes and formatting
    text = _re.sub(r'\|', ' ', text)
    # Strip horizontal rules
    text = _re.sub(r'^-{3,}$', '', text, flags=_re.MULTILINE)
    # Strip link syntax [text](url) → text
    text = _re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    # Strip HTML/markdown image syntax
    text = _re.sub(r'!\[([^\]]*)\]\([^)]+\)', r'\1', text)
    # Collapse whitespace
    text = _re.sub(r'\n{3,}', '\n\n', text)
    text = _re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()

if os.path.exists(_sys_prompt_path):
    with open(_sys_prompt_path, 'r', encoding='utf-8') as _f:
        SYSTEM_INSTRUCTION = _strip_markdown_for_llm(_f.read().strip())
else:
    SYSTEM_INSTRUCTION = (
        "Sen yararlı bir Türkçe sesli asistansın. "
        "Kullanıcının sorusuna doğrudan, doğal ve samimi bir şekilde Türkçe cevap ver. "
        "Konuşmanın akıcı olmasını sağla, ancak gerektiğinde detaylı açıklamalar yapmaktan çekinme."
    )

# --- AUDIO CONFIGURATION ---
SAMPLE_RATE = 16000  # Gemma 4 audio and TTS both work natively at 16kHz
CHANNELS = 1         # Mono-channel audio is required

# Voice Activity Detection (VAD) config
ENERGY_THRESHOLD = 2.5
SILENCE_DURATION = 0.8
MIN_SPEECH_DURATION = 0.3

# Use manual Press-to-Talk instead of automatic VAD
PUSH_TO_TALK = False

# Interruption mode during assistant playback
#   "both"     -> Can interrupt by either speaking (VAD) or pressing Enter key
#   "key_only" -> Can only interrupt by pressing Enter key (immune to speaker echo)
#   "none"     -> Interruption disabled
PLAYBACK_INTERRUPTION_MODE = os.environ.get("PLAYBACK_INTERRUPTION_MODE", "both")

# --- SERVER ---
SERVER_PORT = int(os.environ.get("SPEECH_SERVER_PORT", "8002"))
SERVER_HOST = os.environ.get("SPEECH_SERVER_HOST", "0.0.0.0")
