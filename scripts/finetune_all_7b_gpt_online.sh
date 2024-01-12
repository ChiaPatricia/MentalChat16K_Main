#!/bin/bash
main_dir=/cbica/home/xjia/qlora
cd $main_dir
echo "Alpaca prompt format. modified 'unk_token' = 0 in qlora.py (originally 2 because pad_token_id = 2 for zephyr). model-gpt-online-date \
Continue finetuning models finetuned on gpt generated data on gpt paraphrased online data. ~3 epochs"

source /cbica/software/external/python/anaconda/3/bin/activate

conda activate textlearning

# vicuna-v1.5-7b
python qlora.py \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --output_dir ./output/vicuna-7b-gpt-online-0104 \
    --logging_steps 10 \
    --save_strategy steps \
    --data_seed 42 \
    --save_steps 275 \
    --save_total_limit 40 \
    --evaluation_strategy steps \
    --eval_dataset_size 1024 \
    --max_eval_samples 1000 \
    --per_device_eval_batch_size 1 \
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
    --dataset /cbica/home/xjia/qlora/data/phase2/public_qa_pairs_paraphrased.csv \
    --dataset_format alpaca \
    --source_max_len 256 \
    --target_max_len 512 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --max_steps 1110 \
    --eval_steps 100 \
    --learning_rate 0.0002 \
    --adam_beta2 0.999 \
    --max_grad_norm 0.3 \
    --lora_dropout 0.1 \
    --weight_decay 0.0 \
    --seed 0

# Samantha-v1.11-7b
python qlora.py \
    --model_name_or_path ehartford/Samantha-1.11-7b \
    --output_dir ./output/samantha111-7b-gpt-online-0104 \
    --logging_steps 10 \
    --save_strategy steps \
    --data_seed 42 \
    --save_steps 275 \
    --save_total_limit 40 \
    --evaluation_strategy steps \
    --eval_dataset_size 1024 \
    --max_eval_samples 1000 \
    --per_device_eval_batch_size 1 \
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
    --dataset /cbica/home/xjia/qlora/data/phase2/public_qa_pairs_paraphrased.csv \
    --dataset_format alpaca \
    --source_max_len 256 \
    --target_max_len 512 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --max_steps 1110 \
    --eval_steps 100 \
    --learning_rate 0.0002 \
    --adam_beta2 0.999 \
    --max_grad_norm 0.3 \
    --lora_dropout 0.1 \
    --weight_decay 0.0 \
    --seed 0

# Samantha-v1.2-7b
python qlora.py \
    --model_name_or_path ehartford/samantha-1.2-mistral-7b \
    --output_dir ./output/samantha12-7b-gpt-online-0104 \
    --logging_steps 10 \
    --save_strategy steps \
    --data_seed 42 \
    --save_steps 275 \
    --save_total_limit 40 \
    --evaluation_strategy steps \
    --eval_dataset_size 1024 \
    --max_eval_samples 1000 \
    --per_device_eval_batch_size 1 \
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
    --dataset /cbica/home/xjia/qlora/data/phase2/public_qa_pairs_paraphrased.csv \
    --dataset_format alpaca \
    --source_max_len 256 \
    --target_max_len 512 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --max_steps 1110 \
    --eval_steps 100 \
    --learning_rate 0.0002 \
    --adam_beta2 0.999 \
    --max_grad_norm 0.3 \
    --lora_dropout 0.1 \
    --weight_decay 0.0 \
    --seed 0

# Zephyr-7b
python qlora.py \
    --model_name_or_path HuggingFaceH4/zephyr-7b-alpha \
    --output_dir ./output/zephyr-7b-gpt-online-0104 \
    --logging_steps 10 \
    --save_strategy steps \
    --data_seed 42 \
    --save_steps 275 \
    --save_total_limit 40 \
    --evaluation_strategy steps \
    --eval_dataset_size 1024 \
    --max_eval_samples 1000 \
    --per_device_eval_batch_size 1 \
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
    --dataset /cbica/home/xjia/qlora/data/phase2/public_qa_pairs_paraphrased.csv \
    --dataset_format alpaca \
    --source_max_len 256 \
    --target_max_len 512 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --max_steps 1110 \
    --eval_steps 100 \
    --learning_rate 0.0002 \
    --adam_beta2 0.999 \
    --max_grad_norm 0.3 \
    --lora_dropout 0.1 \
    --weight_decay 0.0 \
    --seed 0
