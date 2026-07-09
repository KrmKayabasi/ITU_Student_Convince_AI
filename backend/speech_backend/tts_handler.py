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

                # Read the model JSON for inference parameters so they exactly
                # match what the model was trained with.
                _json_path = os.path.join(self.model_id, os.path.basename(self.model_path).replace(".onnx", ".onnx.json"))
                _noise_scale = 0.667
                _noise_w = 0.8
                _length_scale = 1.0
                if os.path.exists(_json_path):
                    import json as _json_mod
                    with open(_json_path, 'r') as _jf:
                        _cfg = _json_mod.load(_jf)
                    _inf = _cfg.get("inference", {})
                    _noise_scale = float(_inf.get("noise_scale", _noise_scale))
                    _noise_w = float(_inf.get("noise_w", _noise_w))
                    _length_scale = float(_inf.get("length_scale", _length_scale))
                    print(f"[TTS] Model inference params: noise_scale={_noise_scale}, "
                          f"noise_w={_noise_w}, length_scale={_length_scale}", flush=True)

                self.tts_config = sherpa_onnx.OfflineTtsConfig(
                    model=sherpa_onnx.OfflineTtsModelConfig(
                        vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                            model=self.model_path,
                            tokens=self.tokens_path,
                            data_dir=self.data_dir,
                            noise_scale=_noise_scale,
                            noise_scale_w=_noise_w,
                            length_scale=_length_scale,
                        ),
                        num_threads=4,   # 4 threads gives stable VITS decoding on CPU
                        debug=False,
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

    @property
    def sample_rate(self) -> int:
        """Return the native sample rate of the loaded TTS backend.

        This is determined at init time from the backend type so the server
        can set the correct X-Sample-Rate header before any synthesis occurs.
        """
        if self.use_xtts:
            return 24000
        if self.use_sherpa:
            return 22050  # Piper VITS (tr_TR-dfki-medium.onnx)
        # MMS PyTorch fallback — try model config, default to 16000
        if hasattr(self, 'model') and hasattr(self.model, 'config'):
            return getattr(self.model.config, 'sampling_rate', 16000)
        return 16000

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
            # Defensive clipping: VITS decoders can occasionally produce
            # samples slightly outside [-1, 1] which cause DAC hard-clipping
            # distortion.  np.clip is vectorized and adds negligible overhead.
            waveform = np.clip(waveform, -1.0, 1.0)
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
        """Helper to sanitize text for TTS with advanced Turkish normalization.

        Processing order matters — each step is documented with its reason:
          1. Symbols -> words (before case changes)
          2. Abbreviations -> expansions (before Turkish lowercase, so regex
             matches uppercase forms like "API", "AI", "ITU")
          3. Turkish lowercase (İ->i, I->ı, etc.) — AFTER abbreviations so
             they can match against the original case
          4. Markdown stripping — AFTER abbreviations so underscores don't
             break compound abbreviations like "LLM_API"
          5. Punctuation normalization
          6. Bracket/URL cleanup
          7. Whitespace normalization
        """
        # -- Step 1: Replace symbols with words (before case changes) ----------
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

        # -- Step 2: Phonetic abbreviations (BEFORE Turkish lowercase!) --------
        # Uppercase abbreviations like "API", "AI", "ITU" must match before
        # Turkish lowercase converts I->ı and İ->i. We run once case-insensitively
        # then again on the lowercased result to catch already-lowercase forms.
        abbreviations_ci = {
            r"\bapi\b": "a pi ay",
            r"\bai\b": "yapay zeka",
            r"\bllm\b": "el el em",
            r"\btts\b": "ses sentezi",
            r"\basr\b": "ses tanıma",
            r"\bui\b": "yu ay",
            r"\bcv\b": "si vi",
            r"\bpdf\b": "pe de fe",
            r"\byks\b": "ye ke se",
            r"\bitü\b": "i tü",       # spelled out: "İTÜ" as letters
            r"\btl\b": "türk lirası",
            r"\bvs\b\.?": "ve saire",
            r"\bvb\b\.?": "ve benzeri",
        }
        for pattern, replacement in abbreviations_ci.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        # -- Step 3: Turkish lowercase (AFTER abbreviations) -------------------
        text = self._turkish_lowercase(text)

        # -- Step 4: Strip markdown characters ---------------------------------
        text = text.replace("*", "").replace("_", "").replace("#", "").replace("`", "")
        text = text.replace("~", "").replace(">", "").replace("|", "")

        # -- Step 5: Quotes and apostrophes -> space (preserves word boundaries) -
        text = text.replace("'", " ").replace('"', " ").replace("“", " ").replace("”", " ").replace("’", " ")

        # -- Step 6: Punctuation normalization ---------------------------------
        # Only replace COLON and SEMICOLON with period.
        # HYPHEN: keep intra-word hyphens (ön-lisans, Türk-Alman) but replace
        # standalone dashes (em-dash, en-dash, or hyphen with spaces around it).
        text = text.replace(";", ".").replace(":", ".")
        text = text.replace("—", " ").replace("–", " ")   # em-dash, en-dash
        text = re.sub(r'\s+-\s+', ' ', text)  # standalone hyphens only

        # -- Step 7: Strip brackets — add space to prevent URL concatenation ---
        # "[İTÜ](https://itu.edu.tr)" -> "İTÜ https://itu.edu.tr"
        text = re.sub(r'[\[\](){}<>]', ' ', text)

        # -- Step 8: Normalize whitespace ──────────────────────────────────────
        text = re.sub(r'\s+', ' ', text)

        return text.strip()
