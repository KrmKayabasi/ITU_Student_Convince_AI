#!/bin/bash
export CUDA_VISIBLE_DEVICES="0,1"
/home/yigit/Turkish_Speech_to_Speech/venv_3.11/bin/torchrun --nproc_per_node=2 training/train_multimodal.py     --model_id "google/gemma-4-12B-it"     --manifest_path "/home/yigit/speech-data/yodas2_tr/yodas2_tr/manifest_tr000.jsonl"     --dataset_type "yodas"     --output_dir "./checkpoints_turkish_gemma"     --epochs 2     --batch_size 4     --lr 2e-4
