import sys
import os
import torch
import time
import re
import tempfile
import scipy.io.wavfile as wav
import numpy as np

# Platform Detection
IS_MAC = sys.platform == "darwin"

class GemmaAudioProcessor:
    def __init__(self, model_id="google/gemma-4-e2b-it", quantization="none", device="mps", enable_thinking=False):
        self.model_id = model_id
        self.quantization = quantization
        self.device = device
        self.enable_thinking = enable_thinking
        
        print(f"[Model] Platform: {'macOS (Apple Silicon)' if IS_MAC else 'Linux/Windows (CUDA)'}", flush=True)
        
        if IS_MAC:
            # --- MAC MPS PATH: USE APPLE MLX (mlx-vlm) ---
            print(f"[Model] macOS için MLX (mlx-vlm) backend yükleniyor: {self.model_id}...", flush=True)
            start_time = time.time()
            
            # mlx-vlm is the multimodal framework supporting vision/audio in Gemma 4
            from mlx_vlm import load
            
            # Load model and processor (on MLX, it automatically maps to GPU/MPS)
            # We pass strict=False to handle any redundant weight mappings in checkpoints
            self.model, self.processor = load(self.model_id, strict=False)
            
            print(f"[Model] MLX model {time.time() - start_time:.2f} saniyede başarıyla yüklendi.", flush=True)
            
        else:
            # --- CUDA PATH: USE PYTORCH WITH CUDA ACCELERATION ---
            print(f"[Model] CUDA için PyTorch backend yükleniyor: {self.model_id}...", flush=True)
            start_time = time.time()
            
            from transformers import AutoProcessor, AutoModelForMultimodalLM
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            
            # Setup CUDA optimized quantization
            quantization_config = None
            torch_dtype = torch.bfloat16
            
            if self.quantization == "int4":
                # CUDA-native highly optimized 4-bit loading
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            elif self.quantization == "int8":
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            
            # Load PyTorch model mapping automatically to GPU
            self.model = AutoModelForMultimodalLM.from_pretrained(
                self.model_id,
                quantization_config=quantization_config,
                torch_dtype=torch_dtype,
                device_map="auto"
            )
            
            print(f"[Model] PyTorch CUDA model {time.time() - start_time:.2f} saniyede başarıyla yüklendi.", flush=True)

    def generate_response(self, audio_data, system_instruction=None, messages=None):
        """
        Runs Gemma 4 multimodal inference on the raw audio buffer.
        Returns (thinking_process, final_response)
        """
        if messages is None:
            # Format the system prompt instruction
            prompt_system = system_instruction
            if self.enable_thinking:
                prompt_system = "<|think|>\n" + prompt_system
                
            messages = [
                {"role": "system", "content": prompt_system},
                {"role": "user", "content": "Kullanıcı ses kaydını gönderdi: <|audio|>"}
            ]
            
        # Apply chat template
        prompt = self.processor.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        if IS_MAC:
            # --- MAC MPS PATH: INFERENCE VIA MLX-VLM ---
            # Save raw numpy audio array to a temporary WAV file (near zero latency in-memory write)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
                # Convert float32 [-1, 1] range to int16 PCM
                pcm_audio = (audio_data * 32767).astype(np.int16)
                wav.write(temp_wav.name, 16000, pcm_audio)
                temp_wav_path = temp_wav.name
                
            start_time = time.time()
            from mlx_vlm import generate
            
            try:
                # Run MLX-VLM generation
                # We disable sampling for maximum speed (greedy decode) and set max_tokens=60
                response_text = generate(
                    self.model,
                    self.processor,
                    prompt,
                    audio=[temp_wav_path],
                    max_tokens=1024,
                    verbose=False
                )
            finally:
                # Clean up the temporary file immediately
                try:
                    os.unlink(temp_wav_path)
                except Exception:
                    pass
            
            # Extract plain text if GenerationResult object is returned by mlx_vlm
            if hasattr(response_text, "text"):
                response_text = response_text.text
                    
            generation_time = time.time() - start_time
            print(f"[Model] MLX Yanıtı {generation_time:.2f} saniyede oluşturuldu.", flush=True)
            
        else:
            # --- CUDA PATH: INFERENCE VIA PYTORCH CUDA ---
            inputs = self.processor(
                text=prompt, 
                audio=audio_data, 
                sampling_rate=16000, 
                return_tensors="pt"
            ).to("cuda")
            
            start_time = time.time()
            with torch.no_grad():
                # Greedy decoding (do_sample=False) on CUDA for maximum speed
                outputs = self.model.generate(
                    **inputs, 
                    max_new_tokens=1024,
                    do_sample=False,
                    use_cache=True
                )
            
            generated_ids = outputs[0][inputs.input_ids.shape[1]:]
            response_text = self.processor.decode(generated_ids, skip_special_tokens=False).strip()
            
            generation_time = time.time() - start_time
            print(f"[Model] CUDA Yanıtı {generation_time:.2f} saniyede oluşturuldu.", flush=True)
            
        # Parse the thinking channel output:
        # Format is: <|channel>thought\n[thinking_content]<channel|>[final_response]
        thinking_content = ""
        final_response = response_text
        
        if "<|channel>thought" in response_text:
            parts = response_text.split("<|channel>thought")
            if len(parts) > 1:
                subparts = parts[1].split("<channel|>")
                thinking_content = subparts[0].strip()
                if len(subparts) > 1:
                    final_response = subparts[1].strip()
                else:
                    final_response = ""
        elif "<channel|>" in response_text:
            parts = response_text.split("<channel|>")
            thinking_content = parts[0].strip()
            final_response = parts[1].strip()
            
        # Helper to strip structural tokens
        def clean_special_tokens(text):
            text = re.sub(r'<\|.*?\|>', '', text)
            text = re.sub(r'<[^>]+>', '', text)
            return text.strip()
            
        thinking_content = clean_special_tokens(thinking_content)
        if thinking_content.startswith("thought"):
            thinking_content = thinking_content[7:].strip()
            
        final_response = clean_special_tokens(final_response)
        
        return thinking_content, final_response

    def generate_response_with_history(self, messages, audio_list):
        """
        Runs Gemma 4 multimodal inference on the conversation history (messages) with a list of audio arrays.
        Returns (thinking_process, final_response)
        """
        prompt = self.processor.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        if IS_MAC:
            # MLX fallback for local Mac testing
            print("[Warning] MLX backend does not support multi-audio history yet. Using last audio.", flush=True)
            if len(audio_list) > 0:
                return self.generate_response(audio_list[-1], messages[0]["content"])
            else:
                from mlx_vlm import generate
                response_text = generate(self.model, self.processor, prompt, max_tokens=1024, verbose=False)
                if hasattr(response_text, "text"):
                    response_text = response_text.text
                return "", response_text
            
        else:
            if len(audio_list) > 0:
                inputs = self.processor(
                    text=prompt, 
                    audio=audio_list, 
                    sampling_rate=16000, 
                    return_tensors="pt"
                ).to("cuda")
            else:
                inputs = self.processor(
                    text=prompt, 
                    return_tensors="pt"
                ).to("cuda")
            
            start_time = time.time()
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, 
                    max_new_tokens=1024,
                    do_sample=False,
                    use_cache=True
                )
            
            generated_ids = outputs[0][inputs.input_ids.shape[1]:]
            response_text = self.processor.decode(generated_ids, skip_special_tokens=False).strip()
            
            generation_time = time.time() - start_time
            print(f"[Model] CUDA Yanıtı {generation_time:.2f} saniyede oluşturuldu.", flush=True)
            
        # Parse the thinking channel output:
        thinking_content = ""
        final_response = response_text
        
        if "<|channel>thought" in response_text:
            parts = response_text.split("<|channel>thought")
            if len(parts) > 1:
                subparts = parts[1].split("<channel|>")
                thinking_content = subparts[0].strip()
                if len(subparts) > 1:
                    final_response = subparts[1].strip()
                else:
                    final_response = ""
        elif "<channel|>" in response_text:
            parts = response_text.split("<channel|>")
            thinking_content = parts[0].strip()
            final_response = parts[1].strip()
            
        # Helper to strip structural tokens
        def clean_special_tokens(text):
            text = re.sub(r'<\|.*?\|>', '', text)
            text = re.sub(r'<[^>]+>', '', text)
            return text.strip()
            
        thinking_content = clean_special_tokens(thinking_content)
        if thinking_content.startswith("thought"):
            thinking_content = thinking_content[7:].strip()
            
        final_response = clean_special_tokens(final_response)
        
        return thinking_content, final_response

    def generate_response_stream(self, messages, audio_list):
        """
        Streams Gemma 4 multimodal inference on the conversation history (messages) with a list of audio arrays.
        Yields text clauses (separated by punctuation or newlines) for streaming TTS synthesis.
        """
        prompt = self.processor.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        if IS_MAC:
            # Fallback for Mac (non-streaming for simplicity)
            if len(audio_list) > 0:
                _, response_text = self.generate_response(audio_list[-1], messages[0]["content"])
            else:
                from mlx_vlm import generate
                response_text = generate(self.model, self.processor, prompt, max_tokens=1024, verbose=False)
                if hasattr(response_text, "text"):
                    response_text = response_text.text
            yield response_text
            return
            
        else:
            # --- CUDA PATH: STREAMING INFERENCE VIA PYTORCH CUDA ---
            from transformers import TextIteratorStreamer
            from threading import Thread
            
            if len(audio_list) > 0:
                inputs = self.processor(
                    text=prompt, 
                    audio=audio_list, 
                    sampling_rate=16000, 
                    return_tensors="pt"
                ).to("cuda")
            else:
                inputs = self.processor(
                    text=prompt, 
                    return_tensors="pt"
                ).to("cuda")
            
            streamer = TextIteratorStreamer(self.processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
            
            generation_kwargs = dict(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
                use_cache=True,
                streamer=streamer
            )
            
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()
            
            sentence_buffer = ""
            for new_text in streamer:
                # Filter out any raw channel tokens if they leak through
                clean_chunk = re.sub(r'<\|.*?\|>', '', new_text)
                clean_chunk = re.sub(r'<[^>]+>', '', clean_chunk)
                
                sentence_buffer += clean_chunk
                
                # Check for punctuation markers to yield a clause/sentence
                if any(char in clean_chunk for char in [".", ",", "?", "!", "\n", ";", ":"]):
                    clause = sentence_buffer.strip()
                    # Clean up structural leading/trailing markers if any
                    clause = re.sub(r'^(thought|channel)\s*', '', clause, flags=re.IGNORECASE).strip()
                    if clause:
                        yield clause
                    sentence_buffer = ""
                    
            # Yield any remaining text
            if sentence_buffer.strip():
                clause = sentence_buffer.strip()
                clause = re.sub(r'^(thought|channel)\s*', '', clause, flags=re.IGNORECASE).strip()
                if clause:
                    yield clause


