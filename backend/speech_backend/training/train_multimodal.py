import os
import sys
import json
import torch
import librosa
import numpy as np
import argparse
from torch.utils.data import Dataset
from transformers import AutoProcessor, AutoModelForMultimodalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

class TurkishSpeechDataset(Dataset):
    def __init__(self, manifest_path, dataset_type='yodas', audio_root=None):
        self.dataset_type = dataset_type
        self.items = []
        with open(manifest_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    text = ''
                    if 'supervisions' in item and len(item['supervisions']) > 0:
                        text = item['supervisions'][0].get('text', '').strip()
                    elif 'text' in item:
                        text = item.get('text', '').strip()
                    if not text:
                        continue
                    audio_path = item['audio']
                    filename = os.path.basename(audio_path)
                    # Resolve audio path: use --audio-root if given, else derive from manifest.
                    if audio_root:
                        true_path = os.path.join(audio_root, filename)
                    elif dataset_type == 'yodas':
                        true_path = os.path.join(
                            os.path.dirname(manifest_path), 'clips', filename
                        )
                    elif dataset_type == 'worldspeech':
                        true_path = os.path.join(
                            os.path.dirname(manifest_path), 'clips', filename
                        )
                    else:
                        true_path = audio_path  # fallback: use path as-is from manifest
                    self.items.append({
                        'id': item['id'],
                        'audio_path': true_path,
                        'text': text
                    })
        print(f'[Dataset] Loaded {len(self.items)} items from {manifest_path}.', flush=True)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        try:
            y, sr = librosa.load(item['audio_path'], sr=16000)
            max_samples = 16000 * 30
            if len(y) > max_samples:
                y = y[:max_samples]
        except Exception as e:
            print(f"[Dataset Error] Failed to load {item['audio_path']}: {e}", flush=True)
            y = np.zeros(16000 * 3, dtype=np.float32)
        transcription = item['text']
        messages = [
            {'role': 'system', 'content': 'Sen son derece yardımsever, kibar ve cana yakın bir Türkçe sesli asistansın.'},
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': 'Lütfen bu ses kaydını deşifre et.'},
                    {'type': 'audio'}
                ]
            },
            {'role': 'assistant', 'content': transcription}
        ]
        return {'audio': y, 'messages': messages}

class MultimodalCollator:
    def __init__(self, processor):
        self.processor = processor
    def __call__(self, batch):
        texts = []
        audios = []
        for item in batch:
            prompt = self.processor.apply_chat_template(item['messages'], tokenize=False, add_generation_prompt=False)
            texts.append(prompt)
            audios.append(item['audio'])
        inputs = self.processor(text=texts, audio=audios, sampling_rate=16000, return_tensors='pt', padding=True)
        labels = inputs.input_ids.clone()
        assistant_token_ids = self.processor.tokenizer.convert_tokens_to_ids(['<|turn>', 'model'])
        for i in range(labels.shape[0]):
            seq = labels[i].tolist()
            assistant_idx = -1
            for j in range(len(seq) - 1):
                if seq[j] == assistant_token_ids[0] and seq[j+1] == assistant_token_ids[1]:
                    assistant_idx = j + 3
                    break
            if assistant_idx != -1 and assistant_idx < len(seq):
                labels[i, :assistant_idx] = -100
                labels[i][inputs.attention_mask[i] == 0] = -100
            else:
                labels[i, :] = -100
        inputs['labels'] = labels
        return inputs

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default='google/gemma-4-12B-it')
    parser.add_argument('--manifest_path', type=str, default='/home/yigit/speech-data/yodas2_tr/yodas2_tr/manifest_tr000.jsonl')
    parser.add_argument('--dataset_type', type=str, default='yodas')
    parser.add_argument('--audio-root', type=str, default=None,
                        help='Root directory containing audio clip files (overrides manifest-relative path)')
    parser.add_argument('--output_dir', type=str, default='./checkpoints_turkish_gemma')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--max_steps', type=int, default=-1)
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(args.model_id)

    # Distributed Training Check: DDP requires passing specific device mapping per process
    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    if local_rank != -1:
        device_map = {'': local_rank}
    else:
        device_map = 'auto'

    model = AutoModelForMultimodalLM.from_pretrained(
        args.model_id, 
        torch_dtype=torch.bfloat16, 
        device_map=device_map
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=['q_proj', 'v_proj', 'k_proj', 'o_proj'],
        lora_dropout=0.05,
        bias='none',
        task_type='CAUSAL_LM'
    )
    model = get_peft_model(model, lora_config)

    base_model = model.base_model.model
    for param in base_model.model.embed_audio.parameters():
        param.requires_grad = True

    model.print_trainable_parameters()
    dataset = TurkishSpeechDataset(args.manifest_path, args.dataset_type, audio_root=args.audio_root)
    collator = MultimodalCollator(processor)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs if args.max_steps == -1 else 1,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        lr_scheduler_type='cosine',
        warmup_ratio=0.03,
        logging_steps=1,
        save_strategy='no' if args.max_steps != -1 else 'epoch',
        save_total_limit=2,
        bf16=True,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        report_to='none'
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator
    )

    print('[Trainer] Starting Gemma 4 Turkish Speech Fine-Tuning...', flush=True)
    trainer.train()
    
    if args.max_steps == -1:
        trainer.save_model(os.path.join(args.output_dir, 'final_merged_lora'))
        processor.save_pretrained(os.path.join(args.output_dir, 'final_merged_lora'))
        print('[Trainer] Fine-Tuning completed and model saved successfully!', flush=True)
    else:
        print('[Trainer] Dry-run completed successfully!', flush=True)

if __name__ == '__main__':
    train()
