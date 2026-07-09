import os
import sys
import json
import torch
import numpy as np
from transformers import AutoProcessor, AutoModelForMultimodalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from train_multimodal import TurkishSpeechDataset, MultimodalCollator

def test_dry_run():
    print("[Test] Starting dry-run sanity check...", flush=True)
    model_id = "google/gemma-4-12B-it"
    manifest_path = "/home/yigit/speech-data/yodas2_tr/yodas2_tr/manifest_tr000.jsonl"
    
    print("[Test] Initializing processor and model...", flush=True)
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForMultimodalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map='cpu')
    
    print("[Test] Configuring PEFT LoRA...", flush=True)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=['q_proj', 'v_proj'],
        lora_dropout=0.05,
        bias='none',
        task_type='CAUSAL_LM'
    )
    model = get_peft_model(model, lora_config)
    
    print("[Test] Enabling gradients on embed_audio projection...", flush=True)
    base_model = model.base_model.model
    for param in base_model.model.embed_audio.parameters():
        param.requires_grad = True
        
    print("[Test] Trainable parameters:", flush=True)
    model.print_trainable_parameters()
    
    print("[Test] Checking Dataset loading...", flush=True)
    dataset = TurkishSpeechDataset(manifest_path, "yodas")
    # Shorten dataset for dry run
    dataset.items = dataset.items[:4]
    
    print("[Test] Getting a batch from collator...", flush=True)
    collator = MultimodalCollator(processor)
    batch = collator([dataset[0], dataset[1]])
    
    print("[Test] Batch keys:", batch.keys(), flush=True)
    print("[Test] input_ids shape:", batch['input_ids'].shape, flush=True)
    print("[Test] labels shape:", batch['labels'].shape, flush=True)
    
    # Check that labels contain -100 masking
    labels = batch['labels'].tolist()
    masked_count = sum(1 for seq in labels for token in seq if token == -100)
    print(f"[Test] Total tokens: {len(labels)*len(labels[0])}, Masked tokens (-100): {masked_count}", flush=True)
    
    print("[Test] Initializing Trainer...", flush=True)
    training_args = TrainingArguments(
        output_dir="./test_checkpoints",
        max_steps=1,
        per_device_train_batch_size=2,
        bf16=True,
        report_to='none'
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator
    )
    print("[Test] Trainer initialized successfully! Dry-run PASSED.", flush=True)

if __name__ == '__main__':
    test_dry_run()
