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
#SBATCH --exclude=h[002],g[006,009],i[001],j002-ds
#SBATCH --time=11:00:00
#SBATCH --signal=SIGUSR1@120

#g[006,009]

echo $PATH

cd /net/scratch/mourad/economic-narratives/src/finetune
source /net/projects/chai-lab/miniconda3/etc/profile.d/conda.sh
conda activate /net/scratch/mourad/env-py310-$1
chmod 777 /net/scratch/mourad/economic-narratives/scripts/interactive_run.sh

# An example to use SLURM_ARRAY_TASK_ID
# when you submit jobs in sbatch -a 0-7, SLURM will create 8 jobs with SLURM_ARRAY_TASK_ID set to be 0,1,2,3,4,5,6,7
# root_dir="/net/scratch/mourad/legal/"
# readarray -t domains < <(find "${root_dir}" -maxdepth 1 -type d ! -path "${root_dir}" -exec basename {} \;)

# localhost=${CUDA_VISIBLE_DEVICES}
# domain=${domains[${SLURM_ARRAY_TASK_ID}]}
# echo "domain-${domain}"
# echo "localhost-${localhost}"

# Now you can write your own bash codes

torchrun --nproc_per_node=4 --master_port=45283 /net/scratch/mourad/economic-narratives/src/finetune/phi2_sft_train.py \
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
   --remove_unused_columns True \
   --save_steps 100 \
   --save_total_limit 3 \
   --seed 0 \
   --gradient_accumulation_steps 1 \
   --warmup_ratio 0.2 \
   --gradient_checkpointing True \
   --use_reentrant False \
   --deepspeed "configs/ds_config.yml" \
   --run_name "phi3_binary_300steps_${1}"
    
   #  --model_max_length 4000 \
   #  --data_path /net/projects/chai-lab/mourad/narratives-data/sft_data \
   #  --output_dir /net/projects/chai-lab/mourad/narratives-data/sft_out/phi3 \
   #  --max_steps 300 \
   #  --per_device_train_batch_size 2 \
   #  --per_device_eval_batch_size 2 \
   #  --gradient_accumulation_steps 1 \
   #  --evaluation_strategy "steps" \
   #  --eval_steps 20 \
   #  --save_strategy "steps" \
   #  --save_steps 160 \
   #  --save_total_limit 3 \
   #  --learning_rate 1e-4 \
   #  --weight_decay 0. \
   #  --warmup_ratio 0.03 \
   #  --lr_scheduler_type "cosine" \
   #  --logging_steps 5 \
   #  --deepspeed "configs/ds_config.yml" \
   #  --tf32 True \
   #  --run_name "phi3_binary_300steps_${1}" \
   #  --gradient_checkpointing True
    # --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    # --num_train_epochs 3 \

