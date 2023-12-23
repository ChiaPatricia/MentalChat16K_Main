#!/bin/bash
main_dir=/cbica/home/xjia/qlora
cd $main_dir
echo "Alpaca prompt format. samantha12-7b-newCombined-1022"

source /cbica/software/external/python/anaconda/3/bin/activate

conda activate textlearning

python qlora.py \
    --model_name_or_path ehartford/samantha-1.2-mistral-7b \
    --output_dir ./output/samantha12-7b-newCombined-1022 \
    --logging_steps 10 \
    --save_strategy steps \
    --data_seed 42 \
    --save_steps 450 \
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
    --dataset /cbica/home/xjia/qlora/data/lab/new_combined_qa_pairs_instruction.csv \
    --dataset_format alpaca \
    --source_max_len 256 \
    --target_max_len 512 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --max_steps 1400 \
    --eval_steps 187 \
    --learning_rate 0.0002 \
    --adam_beta2 0.999 \
    --max_grad_norm 0.3 \
    --lora_dropout 0.1 \
    --weight_decay 0.0 \
    --seed 0

#   --max_new_tokens 32 \ Use default 256
#   --source_max_len 16 \ Use default 1024
#   --max_steps 1875 \ Change to 1400, 1400 is for ~ 3 epochs on new combined data
