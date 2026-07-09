import os
import re
import time
import numpy as np

class OfflineTTSHandler:
    def __init__(self, model_id="vits-piper-tr_TR-fahrettin-medium", device="cpu"):
        self.model_id = model_id
        self.device = device
        self.use_sherpa = False
        self.use_xtts = (model_id == "xtts")
        
        if self.use_xtts:
            print("[TTS] Çevrimdışı Coqui XTTS v2 yükleniyor...", flush=True)
            start_time = time.time()

            # Transformers v5 compatibility patches for Coqui TTS.
            # All monkey-patches are scoped: originals are restored after model load
            # to avoid corrupting other libraries that depend on torch.load safety.
            import torch
            _original_torch_load = torch.load

            def _patched_load(*args, **kwargs):
                kwargs.setdefault('weights_only', False)
                return _original_torch_load(*args, **kwargs)

            torch.load = _patched_load

            # Track monkey-patches so we can restore them in finally block
            _restore_ops = []

            try:
                import transformers
                from transformers.utils.import_utils import _LazyModule

                _original_getattr = _LazyModule.__getattr__
                def _patched_getattr(self, name):
                    if name in ('BeamSearchScorer', 'ConstrainedBeamSearchScorer',
                                'DisjunctiveConstraint', 'PhrasalConstraint'):
                        class _Dummy: pass
                        return _Dummy
                    return _original_getattr(self, name)
                _LazyModule.__getattr__ = _patched_getattr
                _restore_ops.append(
                    lambda: setattr(_LazyModule, '__getattr__', _original_getattr)
                )

                # Inject generation utils mocks for deprecated classes in v5
                import transformers.generation.utils
                class _Dummy: pass
                _restored_gen_utils = {}
                if not hasattr(transformers.generation.utils, 'SampleOutput'):
                    transformers.generation.utils.SampleOutput = _Dummy
                    _restored_gen_utils['SampleOutput'] = True
                if not hasattr(transformers.generation.utils, 'GenerateOutput'):
                    transformers.generation.utils.GenerateOutput = _Dummy
                    _restored_gen_utils['GenerateOutput'] = True

                # Patch GPT2InferenceModel bases for transformers v5 compatibility
                from transformers import GenerationMixin
                import TTS.tts.layers.xtts.gpt_inference as gpt_inf
                _original_bases = gpt_inf.GPT2InferenceModel.__bases__
                if GenerationMixin not in _original_bases:
                    gpt_inf.GPT2InferenceModel.__bases__ = _original_bases + (GenerationMixin,)
                    _restore_ops.append(
                        lambda: setattr(gpt_inf.GPT2InferenceModel, '__bases__', _original_bases)
                    )

                from TTS.api import TTS
            except ImportError as e:
                raise ImportError(f"[TTS Error] Coqui TTS veya dependencies yüklü değil: {e}")

            # Load model on MPS if requested and available, else fallback to CPU
            try:
                try:
                    self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)
                    print(f"[TTS] XTTS Model {self.device} üzerinde {time.time() - start_time:.2f} saniyede başarıyla yüklendi.", flush=True)
                except Exception as e:
                    print(f"[TTS Warning] {self.device} üzerinde yükleme başarısız ({e}). CPU'ya geçiliyor...", flush=True)
                    self.device = "cpu"
                    self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cpu")
                    print(f"[TTS] XTTS Model CPU üzerinde {time.time() - start_time:.2f} saniyede yüklendi.", flush=True)
            finally:
                # Restore all monkey-patches so other libraries are unaffected
                torch.load = _original_torch_load
                for op in _restore_ops:
                    try:
                        op()
                    except Exception:
                        pass
                # Clean up injected mock classes
                for attr_name in _restored_gen_utils:
                    try:
                        delattr(transformers.generation.utils, attr_name)
                    except Exception:
                        pass
                
        else:
            # Check if model_id points to a sherpa-onnx local model directory
            if os.path.isdir(model_id):
                onnx_files = [f for f in os.listdir(model_id) if f.endswith(".onnx")]
                if onnx_files:
                    self.model_path = os.path.join(model_id, onnx_files[0])
                    self.tokens_path = os.path.join(model_id, "tokens.txt")
                    self.data_dir = os.path.join(model_id, "espeak-ng-data")
                    
                    if os.path.exists(self.tokens_path) and os.path.exists(self.data_dir):
                        self.use_sherpa = True
                        
            if self.use_sherpa:
                print(f"[TTS] Çevrimdışı ONNX TTS (Sherpa) yükleniyor: {self.model_id}...", flush=True)
                start_time = time.time()
                import sherpa_onnx
                
                self.tts_config = sherpa_onnx.OfflineTtsConfig(
                    model=sherpa_onnx.OfflineTtsModelConfig(
                        vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                            model=self.model_path,
                            tokens=self.tokens_path,
                            data_dir=self.data_dir
                        ),
                        num_threads=2,
                        debug=False
                    )
                )
                self.tts = sherpa_onnx.OfflineTts(self.tts_config)
                print(f"[TTS] ONNX Model {time.time() - start_time:.2f} saniyede başarıyla yüklendi.", flush=True)
            else:
                # Fallback to PyTorch MMS-TTS
                print(f"[TTS] Çevrimdışı PyTorch TTS (MMS) yükleniyor: {self.model_id}...", flush=True)
                import torch
                from transformers import VitsModel, VitsTokenizer
                start_time = time.time()
                
                self.tokenizer = VitsTokenizer.from_pretrained(self.model_id)
                try:
                    self.model = VitsModel.from_pretrained(self.model_id).to(self.device)
                    print(f"[TTS] PyTorch model {self.device} üzerinde yüklendi.", flush=True)
                except Exception as e:
                    print(f"[TTS Warning] {self.device} üzerinde yükleme başarısız, CPU'ya geçiliyor: {e}", flush=True)
                    self.device = "cpu"
                    self.model = VitsModel.from_pretrained(self.model_id).to("cpu")
                    
                print(f"[TTS] PyTorch model {time.time() - start_time:.2f} saniyede başarıyla yüklendi.", flush=True)

    def synthesize(self, text):
        """
        Synthesizes Turkish text into speech offline.
        Returns (audio_data, sample_rate)
        """
        if not text.strip():
            return None, 24000 if self.use_xtts else (22050 if self.use_sherpa else 16000)
            
        print(f"[TTS] Metin sese dönüştürülüyor: '{text}'", flush=True)
        start_time = time.time()
        
        # Clean text
        clean_text = self._clean_text(text)
        
        if self.use_xtts:
            # Generate via Coqui XTTS v2
            try:
                wav = self.tts.tts(
                    text=clean_text,
                    speaker_wav="fahrettin_test.wav",
                    language="tr"
                )
                waveform = np.array(wav, dtype=np.float32)
                sample_rate = 24000
            except Exception as e:
                print(f"[TTS Error] XTTS Sentez Hatası: {e}", flush=True)
                return None, 24000
        elif self.use_sherpa:
            # Generate via sherpa-onnx (ONNX Runtime)
            audio = self.tts.generate(clean_text)
            waveform = np.array(audio.samples, dtype=np.float32)
            sample_rate = audio.sample_rate
        else:
            # Generate via PyTorch
            import torch
            inputs = self.tokenizer(text=clean_text, return_tensors="pt").to(self.device)
            try:
                with torch.no_grad():
                    outputs = self.model(**inputs)
                waveform = outputs.waveform[0].cpu().numpy()
            except Exception as e:
                print(f"[TTS Warning] PyTorch Sentez Hatası, CPU Fallback: {e}", flush=True)
                cpu_inputs = self.tokenizer(text=clean_text, return_tensors="pt").to("cpu")
                cpu_model = self.model.to("cpu")
                with torch.no_grad():
                    outputs = cpu_model(**cpu_inputs)
                waveform = outputs.waveform[0].cpu().numpy()
                if self.device != "cpu":
                    self.model = self.model.to(self.device)
            sample_rate = self.model.config.sampling_rate

        synthesis_time = time.time() - start_time
        print(f"[TTS] Ses sentezi {synthesis_time:.2f} saniyede tamamlandı.", flush=True)
        return waveform, sample_rate

    def _turkish_lowercase(self, text):
        mapping = {
            "İ": "i",
            "I": "ı",
            "Ş": "ş",
            "Ç": "ç",
            "Ğ": "ğ",
            "Ü": "ü",
            "Ö": "ö"
        }
        for upper, lower in mapping.items():
            text = text.replace(upper, lower)
        return text.lower()

    def _clean_text(self, text):
        """Helper to sanitize text for TTS with advanced Turkish normalization."""
        # Convert to Turkish lowercase to prevent capitalization spelling-out bugs
        text = self._turkish_lowercase(text)

        # Clean markdown characters to prevent literal pronunciation (e.g. asterisks as "yıldız")
        text = text.replace("*", "").replace("_", "").replace("#", "").replace("`", "")
        
        # Clean quotes and apostrophes to prevent word splitting or character reading (e.g. Günleri'ne -> Günlerine)
        text = text.replace("'", "").replace('"', "").replace("”", "").replace("“", "").replace("’", "")
        
        # Replace symbols with words
        symbols = {
            "&": " ve ",
            "@": " et ",
            "+": " artı ",
            "=": " eşittir ",
            "₺": " Türk Lirası ",
            "$": " Dolar ",
            "€": " Avro ",
            "/": " veya ",
            "%": " yüzde ",
        }
        for sym, replacement in symbols.items():
            text = text.replace(sym, replacement)
            
        # Phonetic replacements for common abbreviations (in lowercase)
        abbreviations = {
            r"\bitü\b": "itü",
            r"\bai\b": "yapay zeka",
            r"\bllm\b": "el el em",
            r"\btts\b": "ses sentezi",
            r"\basr\b": "ses tanıma",
            r"\bvs\b\.?": "ve saire",
            r"\bvb\b\.?": "ve benzeri",
            r"\byks\b": "ye ke se",
            r"\btl\b": "Türk Lirası",
            r"\bpdf\b": "pe de fe",
            r"\bapi\b": "a pi ay",
            r"\bui\b": "yu ay",
            r"\bcv\b": "si vi",
        }
        for pattern, replacement in abbreviations.items():
            text = re.sub(pattern, replacement, text)
            
        # Replace common symbols or punctuation
        text = text.replace(";", ".").replace(":", ".").replace("-", " ")
        
        # Strip brackets
        text = re.sub(r'[{}\[\]\(\)<>]', '', text)
        
        # Normalize double spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
