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

# An example to use SLURM_ARRAY_TASK_ID
# when you submit jobs in sbatch -a 0-7, SLURM will create 8 jobs with SLURM_ARRAY_TASK_ID set to be 0,1,2,3,4,5,6,7
# root_dir="/net/scratch/mourad/legal/"
# readarray -t domains < <(find "${root_dir}" -maxdepth 1 -type d ! -path "${root_dir}" -exec basename {} \;)

# localhost=${CUDA_VISIBLE_DEVICES}
# domain=${domains[${SLURM_ARRAY_TASK_ID}]}
# echo "domain-${domain}"
# echo "localhost-${localhost}"

# Now you can write your own bash codes
# python predict.py --model phi2 --ckpt= 
python predict.py --model phi2 --gpu a40 --ckpt= --split NOW_filtered --reuse --debug --sample $1

# 84022
