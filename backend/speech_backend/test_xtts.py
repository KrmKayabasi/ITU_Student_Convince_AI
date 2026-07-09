import torch
# Patch torch.load for weights_only compatibility in PyTorch 2.6+ at the absolute entrypoint
original_load = torch.load
def custom_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = custom_load

import time
import os

def test_xtts():
    print("=" * 50)
    print("TESTING COQUI XTTS V2 FOR HIGH QUALITY TURKISH TTS")
    print("=" * 50)
    
    # Check if speaker wav exists
    ref_wav = "fahrettin_test.wav"
    if not os.path.exists(ref_wav):
        print(f"[Error] Reference audio {ref_wav} not found. Please run setup_sherpa_tts.py first.")
        return
        
    try:
        import transformers
        from transformers.utils.import_utils import _LazyModule
        
        # Store original getattr and hook in custom resolver
        original_getattr = _LazyModule.__getattr__
        def custom_getattr(self, name):
            if name in ('BeamSearchScorer', 'ConstrainedBeamSearchScorer', 'DisjunctiveConstraint', 'PhrasalConstraint'):
                class Dummy: pass
                return Dummy
            return original_getattr(self, name)
        _LazyModule.__getattr__ = custom_getattr
        
        # Inject generation utils mocks for deprecated classes in v5
        import transformers.generation.utils
        class Dummy: pass
        if not hasattr(transformers.generation.utils, 'SampleOutput'):
            transformers.generation.utils.SampleOutput = Dummy
        if not hasattr(transformers.generation.utils, 'GenerateOutput'):
            transformers.generation.utils.GenerateOutput = Dummy
        
        # Patch GPT2InferenceModel bases for transformers v5 compatibility
        from transformers import GenerationMixin
        import TTS.tts.layers.xtts.gpt_inference as gpt_inf
        if GenerationMixin not in gpt_inf.GPT2InferenceModel.__bases__:
            gpt_inf.GPT2InferenceModel.__bases__ = gpt_inf.GPT2InferenceModel.__bases__ + (GenerationMixin,)
        
        from TTS.api import TTS
    except ImportError as e:
        print(f"[Error] TTS library not found: {e}")
        return
        
    print("[XTTS] Loading XTTS v2 model (this will download weights on the first run, approx 1.8GB)...")
    start = time.time()
    
    # Load model on MPS for GPU acceleration on Mac
    # If MPS fails, it falls back to CPU
    try:
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("mps")
        print(f"[XTTS] Model loaded on MPS in {time.time() - start:.2f} seconds.")
    except Exception as e:
        print(f"[XTTS Warning] MPS loading failed ({e}). Loading on CPU...")
        start = time.time()
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cpu")
        print(f"[XTTS] Model loaded on CPU in {time.time() - start:.2f} seconds.")
        
    text = "Merhaba! Bu, yüksek kaliteli Türk ses sentezi motorunun klonlanmış sesidir. Nasıl yardımcı olabilirim?"
    
    print(f"\n[XTTS] Synthesizing: '{text}'...")
    start_gen = time.time()
    
    try:
        tts.tts_to_file(
            text=text,
            speaker_wav=ref_wav,
            language="tr",
            file_path="xtts_test.wav"
        )
        print(f"[XTTS] Synthesis completed in {time.time() - start_gen:.2f} seconds.")
        print("[XTTS] Saved output to: xtts_test.wav")
    except Exception as e:
        print(f"[XTTS Error] Synthesis failed: {e}")

if __name__ == "__main__":
    test_xtts()
