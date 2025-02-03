#!/bin/bash
#SBATCH --mail-user=mourad@uchicago.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/net/scratch/mourad/slurm_output/%A_%a.%N.stdout
#SBATCH --error=/net/scratch/mourad/slurm_output/%A_%a.%N.stderr
#SBATCH --chdir=/net/scratch/mourad/slurm_output
#SBATCH --partition=general
#SBATCH --gres=gpu:a100:2
#SBATCH --job-name=narratives_sft
#SBATCH --nodes=1
#SBATCH --mem=300gb
#SBATCH --exclude=h[002],g[006,009],i[001],j002-ds
#SBATCH --time=11:00:00
#SBATCH --signal=SIGUSR1@120

#g[006,009]

echo $PATH

cd /net/scratch/mourad/economic-narratives/src/finetune
source /net/projects/chai-lab/miniconda3/etc/profile.d/conda.sh
conda activate /net/scratch/mourad/env-py38-$1

# An example to use SLURM_ARRAY_TASK_ID
# when you submit jobs in sbatch -a 0-7, SLURM will create 8 jobs with SLURM_ARRAY_TASK_ID set to be 0,1,2,3,4,5,6,7
# root_dir="/net/scratch/mourad/legal/"
# readarray -t domains < <(find "${root_dir}" -maxdepth 1 -type d ! -path "${root_dir}" -exec basename {} \;)

# localhost=${CUDA_VISIBLE_DEVICES}
# domain=${domains[${SLURM_ARRAY_TASK_ID}]}
# echo "domain-${domain}"
# echo "localhost-${localhost}"

# Now you can write your own bash codes

torchrun --nproc_per_node=2 --master_port=45969 /net/scratch/mourad/economic-narratives/src/finetune/mistral_sft_train.py \
   --model_name_or_path mistralai/Mistral-7B-Instruct-v0.2 \
    --model_max_length 8096 \
    --data_path /net/projects/chai-lab/mourad/narratives-data/sft_data \
    --output_dir /net/projects/chai-lab/mourad/narratives-data/sft_out/mistral_$1 \
    --max_steps 500 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "steps" \
    --eval_steps 20 \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 3 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 5 \
    --deepspeed "configs/ds_config.json" \
    --tf32 True \
    --run_name "mistral_500step64128lora_${1}" \
   #  --gradient_checkpointing True
    # --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    # --num_train_epochs 3 \

