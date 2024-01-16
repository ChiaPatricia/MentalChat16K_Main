#!/bin/bash
#SBATCH --job-name=job_name
#SBATCH --time=20:00:00
#SBATCH --account=plgllm1-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --gres=gpu

source /net/pr2/projects/plgrid/plggllm/anaconda3/bin/activate
conda activate llm

cd /net/pr2/projects/plgrid/plggllm/MentalGPT

srun python qlora.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --use_auth \
    --output_dir ./output/llama2-7b-gpt-0113 \
    --logging_steps 10 \
    --save_strategy steps \
    --data_seed 42 \
    --save_steps 120 \
    --save_total_limit 40 \
    --evaluation_strategy steps \
    --eval_dataset_size 1024 \
    --max_eval_samples 1000 \
    --per_device_eval_batch_size 8 \
    --max_new_tokens 256 \
    --dataloader_num_workers 3 \
    --group_by_length \
    --logging_strategy steps \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --do_mmlu_eval \
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
    --dataset /net/pr2/projects/plgrid/plggllm/MentalGPT/data/lab/self_instruct_gpt3.5_instruction.csv \
    --dataset_format alpaca \
    --source_max_len 256 \
    --target_max_len 512 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 8 \
    --max_steps 480 \
    --eval_steps 48 \
    --learning_rate 0.0002 \
    --adam_beta2 0.999 \
    --max_grad_norm 0.3 \
    --lora_dropout 0.1 \
    --weight_decay 0.0 \
    --seed 0 \