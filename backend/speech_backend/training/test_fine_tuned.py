import os
import sys
import torch
import librosa
import numpy as np
from transformers import AutoProcessor, AutoModelForMultimodalLM

def test_inference():
    print("[Test] Loading fine-tuned model and processor from checkpoints...", flush=True)
    model_id = "./checkpoints_turkish_gemma/final_merged_lora"
    
    processor = AutoProcessor.from_pretrained(model_id)
    # Load on single GPU (cuda:0)
    model = AutoModelForMultimodalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16, 
        device_map="cuda:0"
    )
    
    # Target audio file
    audio_path = "fahrettin_test.wav"
    if not os.path.exists(audio_path):
        print(f"[Error] Test audio file '{audio_path}' not found! Trying a dataset clip...", flush=True)
        # Fallback to a clip from yodas2 dataset if local test file missing
        audio_path = "/home/yigit/speech-data/yodas2_tr/yodas2_tr/clips/0_0h20wZphy-00000-00000019-00001084.wav"
        
    print(f"[Test] Loading audio file: {audio_path}", flush=True)
    y, sr = librosa.load(audio_path, sr=16000)
    
    # Enforce maximum 30 seconds duration
    y = y[:16000 * 30]
    
    print("[Test] Building instruction message matching training format...", flush=True)
    messages = [
        {'role': 'system', 'content': 'Sen son derece yardımsever, kibar ve cana yakın bir Türkçe sesli asistansın.'},
        {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': 'Lütfen bu ses kaydını deşifre et.'},
                {'type': 'audio'}
            ]
        }
    ]
    
    prompt = processor.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    print("[Test] Preprocessing features...", flush=True)
    inputs = processor(
        text=prompt, 
        audio=y, 
        sampling_rate=16000, 
        return_tensors="pt"
    ).to("cuda:0")
    
    print("[Test] Running model generation directly on audio features...", flush=True)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=128,
            do_sample=False,
            use_cache=True
        )
        
    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    response_text = processor.decode(generated_ids, skip_special_tokens=True).strip()
    
    print("\n============================================================")
    print(f"TEST AUDIO PATH: {audio_path}")
    print(f"MODEL TRANSCRIPTION RESPONSE: '{response_text}'")
    print("============================================================\n")

if __name__ == '__main__':
    test_inference()
