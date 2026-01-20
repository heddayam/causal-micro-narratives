#!/bin/bash
#SBATCH --mail-user=mourad@uchicago.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/net/scratch/mourad/legal/slurm_output/%A_%a.%N.stdout
#SBATCH --error=/net/scratch/mourad/legal/slurm_output/%A_%a.%N.stderr
#SBATCH --chdir=/net/scratch/mourad/legal/slurm_output
#SBATCH --partition=general,complementary-ai
#SBATCH --gres=gpu:2
#SBATCH --constraint="a100|h100|h200"
#SBATCH --job-name=fed_predict
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=100gb
#SBATCH --time=12:00:00
#SBATCH --signal=SIGUSR1@120


echo $PATH

cd /home/mourad/causal-micro-narratives
pip install -e .
cd /home/mourad/causal-micro-narratives/src/finetuning

python predict_json.py \
  --model llama31 \
  --split FED_DATA \
  --train_ds now_and_proquest \
  --test_ds fed \
  --ckpt checkpoint-600 \
  --sample ${1:--1} \
  --gpu a100 \
  --reuse \
  ${2:+--start_idx $2} \
  ${3:+--end_idx $3}
