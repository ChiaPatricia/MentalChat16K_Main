#!/bin/bash
main_dir=/cbica/home/xjia/qlora
cd $main_dir
echo "Alpaca prompt format. modified 'unk_token' = 0 in qlora.py (originally 2 because pad_token_id = 2 for zephyr). model-phase2-date \
Finetune base models on gpt generated data and gpt paraphrased data."

source /cbica/software/external/python/anaconda/3/bin/activate

conda activate textlearning

# mistral-instruct-7b-v0.2
python qlora.py \
    --model_name_or_path mistralai/Mistral-7B-Instruct-v0.2 \
    --output_dir ./output/Mistral-7B-Instruct-v0.2-phase2-1223 \
    --logging_steps 10 \
    --save_strategy steps \
    --data_seed 42 \
    --save_steps 150 \
    --save_total_limit 40 \
    --evaluation_strategy steps \
    --eval_dataset_size 1024 \
    --max_eval_samples 1000 \
    --per_device_eval_batch_size 8 \
    --dataloader_num_workers 3 \
    --group_by_length \
    --logging_strategy steps \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --do_mmlu_eval \
    --max_new_tokens 256 \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_modules all \
    --double_quant \
    --quant_type nf4 \
    --bf16 \
    --bits 4 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type constant \
    --gradient_checkpointing \
    --dataset /cbica/home/xjia/qlora/data/phase2/gpt_online.csv \
    --dataset_format alpaca \
    --source_max_len 256 \
    --target_max_len 512 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 8 \
    --max_steps 650 \
    --eval_steps 64 \
    --learning_rate 0.0002 \
    --adam_beta2 0.999 \
    --max_grad_norm 0.3 \
    --lora_dropout 0.1 \
    --weight_decay 0.0 \
    --seed 0

# mixtral-7b-instruct-v0.1
python qlora.py \
    --model_name_or_path mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --output_dir ./output/Mixtral-8x7B-Instruct-v0.1-phase2-1223 \
    --logging_steps 10 \
    --save_strategy steps \
    --data_seed 42 \
    --save_steps 150 \
    --save_total_limit 40 \
    --evaluation_strategy steps \
    --eval_dataset_size 1024 \
    --max_eval_samples 1000 \
    --per_device_eval_batch_size 8 \
    --dataloader_num_workers 3 \
    --group_by_length \
    --logging_strategy steps \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --do_mmlu_eval \
    --max_new_tokens 256 \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_modules all \
    --double_quant \
    --quant_type nf4 \
    --bf16 \
    --bits 4 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type constant \
    --gradient_checkpointing \
    --dataset /cbica/home/xjia/qlora/data/phase2/gpt_online.csv \
    --dataset_format alpaca \
    --source_max_len 256 \
    --target_max_len 512 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 8 \
    --max_steps 650 \
    --eval_steps 64 \
    --learning_rate 0.0002 \
    --adam_beta2 0.999 \
    --max_grad_norm 0.3 \
    --lora_dropout 0.1 \
    --weight_decay 0.0 \
    --seed 0