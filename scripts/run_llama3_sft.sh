#!/bin/bash
#SBATCH --mail-user=mourad@uchicago.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/net/scratch/mourad/slurm_output/%A_%a.%N.stdout
#SBATCH --error=/net/scratch/mourad/slurm_output/%A_%a.%N.stderr
#SBATCH --chdir=/net/scratch/mourad/slurm_output
#SBATCH --partition=general
#SBATCH --gres=gpu:a100:4
#SBATCH --job-name=narratives_sft
#SBATCH --nodes=1
#SBATCH --mem=300gb
#SBATCH --time=12:00:00
#SBATCH --signal=SIGUSR1@120

echo $PATH

cd /net/scratch/mourad/economic-narratives/src/finetune
source /net/projects/chai-lab/miniconda3/etc/profile.d/conda.sh
conda activate /net/scratch/mourad/env-py310-a100

accelerate launch --main_process_port 0 --config_file configs/ds_config.yml /net/scratch/mourad/economic-narratives/src/finetune/sft_train.py \
   --model_name_or_path meta-llama/Meta-Llama-3.1-8B \
   --model_max_length 2048 \
   --data_path /net/projects/chai-lab/mourad/narratives-data/sft_data_now_and_proquest \
   --output_dir /net/projects/chai-lab/mourad/narratives-data/sft_out/llama31_now_and_proquest \
   --max_steps 600 \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 4 \
   --gradient_accumulation_steps 1 \
   --evaluation_strategy "steps" \
   --eval_steps 50 \
   --save_strategy "steps" \
   --save_steps 100 \
   --save_total_limit 3 \
   --learning_rate 1e-4 \
   --weight_decay 0. \
   --warmup_ratio 0.03 \
   --lr_scheduler_type "cosine" \
   --logging_steps 5 \
   --bf16 True \
   --run_name "llama31_8b_now_proquest" \
   --gradient_checkpointing True
