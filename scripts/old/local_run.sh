python sft_train.py \
    --model_name_or_path microsoft/phi-2 \
    --data_path /data/mourad/narratives/ft_data/ \
    --bf16 True \
    --output_dir /data/mourad/narratives/ft_out \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --tf32 True


conda create -n sft python=3.8