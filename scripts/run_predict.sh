#!/bin/bash
#SBATCH --mail-user=mourad@uchicago.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/net/scratch/mourad/legal/slurm_output/%A_%a.%N.stdout
#SBATCH --error=/net/scratch/mourad/legal/slurm_output/%A_%a.%N.stderr
#SBATCH --chdir=/net/scratch/mourad/legal/slurm_output
#SBATCH --partition=general
#SBATCH --gres=gpu:a40:4
#SBATCH --job-name=run_NOW_predict
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=300gb
#SBATCH --time=11:00:00
#SBATCH --signal=SIGUSR1@120


echo $PATH

cd /net/scratch/mourad/economic-narratives/src/finetune
source /net/projects/chai-lab/miniconda3/etc/profile.d/conda.sh
conda activate /net/scratch/mourad/env-py310-a40

if [ "$1" == "llama31" ]; then
    python predict_proquest_llama31.py --model llama31 --gpu a100 --ckpt=checkpoint-600 --split PROQUEST_filtered --train_ds now_and_proquest --test_ds proquest --debug --reuse --sample $2
else
    python predict.py --model phi2 --gpu a40 --ckpt= --split NOW_filtered --reuse --debug --sample $2
fi

