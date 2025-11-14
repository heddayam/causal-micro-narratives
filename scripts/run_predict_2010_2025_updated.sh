#!/bin/bash
#SBATCH --mail-user=mourad@uchicago.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/net/scratch/mourad/legal/slurm_output/%A_%a.%N.stdout
#SBATCH --error=/net/scratch/mourad/legal/slurm_output/%A_%a.%N.stderr
#SBATCH --chdir=/net/scratch/mourad/legal/slurm_output
#SBATCH --partition=general
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=proquest_2010_2025_updated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=100gb
#SBATCH --time=11:00:00
#SBATCH --signal=SIGUSR1@120


echo $PATH

cd /home/mourad/causal-micro-narratives/src/finetuning
source /net/projects/chai-lab/miniconda3/etc/profile.d/conda.sh
conda activate /net/scratch/mourad/env-py310-a100
poetry install

python predict_json.py \
  --model llama31 \
  --split PROQUEST_2010_2025_UPDATED \
  --train_ds now_and_proquest \
  --test_ds proquest \
  --ckpt checkpoint-600 \
  --sample /net/projects/chai-lab/mourad/narratives-data/model_json_preds/proquest/full_proquest/llama31_ft__600s_train-now_and_proquest_sample_0_2010-2025_updated \
  --gpu a100 \
  --reuse

