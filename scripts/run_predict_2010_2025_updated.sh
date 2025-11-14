#!/bin/bash
#SBATCH --mail-user=mourad@uchicago.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/net/scratch/mourad/legal/slurm_output/%A_%a.%N.stdout
#SBATCH --error=/net/scratch/mourad/legal/slurm_output/%A_%a.%N.stderr
#SBATCH --chdir=/net/scratch/mourad/legal/slurm_output
#SBATCH --partition=general
#SBATCH --gres=gpu:4
#SBATCH --constraint="a100|h100|h200"
#SBATCH --job-name=proquest_2010_2025_updated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=100gb
#SBATCH --time=12:00:00
#SBATCH --signal=SIGUSR1@120


echo $PATH


source /net/projects/chai-lab/miniconda3/etc/profile.d/conda.sh
conda activate /net/scratch/mourad/env-py310-a40
pip install -e .
cd /home/mourad/causal-micro-narratives/src/finetuning

python predict_json.py \
  --model llama31 \
  --split PROQUEST_2010_2025_UPDATED \
  --train_ds now_and_proquest \
  --test_ds proquest \
  --ckpt checkpoint-600 \
  --sample $1 \
  --gpu a100 \
  --reuse

