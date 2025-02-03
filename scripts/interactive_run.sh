#!/bin/bash

# Script for interactive SLURM sessions or local debugging

accelerate launch --config_file configs/ds_config.yml /net/scratch/mourad/economic-narratives/src/finetune/phi3_sft_train.py \
   --data_path /net/projects/chai-lab/mourad/narratives-data/sft_data \
   --model_name_or_path microsoft/Phi-3-mini-4k-instruct \
   --bf16 True \
   --learning_rate 1e-4 \
   --log_level "info" \
   --logging_steps 20 \
   --logging_strategy "steps" \
   --lr_scheduler_type "cosine" \
   --max_steps 300 \
   --output_dir /net/projects/chai-lab/mourad/narratives-data/sft_out/phi3 \
   --overwrite_output_dir True \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 4 \
   --evaluation_strategy "steps" \
   --eval_steps 20 \
   --remove_unused_columns True \
   --save_steps 100 \
   --save_total_limit 3 \
   --seed 0 \
   --gradient_accumulation_steps 1 \
   --warmup_ratio 0.2 \
   --gradient_checkpointing True \
   --run_name "phi3_binary_300steps_${1}"