import os
from os.path import exists, join, isdir
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
from peft import PeftModel
from peft.tuners.lora import LoraLayer
import pandas as pd
import time
import json

def get_last_checkpoint(checkpoint_dir):
    if isdir(checkpoint_dir):
        # is_completed = exists(join(checkpoint_dir, 'completed'))
        # if is_completed: return None, True # already finished
        max_step = 0
        for filename in os.listdir(checkpoint_dir):
            if isdir(join(checkpoint_dir, filename)) and filename.startswith('checkpoint'):
                max_step = max(max_step, int(filename.replace('checkpoint-', '')))
        if max_step == 0: return None # training started, but no checkpoint
        checkpoint_dir = join(checkpoint_dir, f'checkpoint-{max_step}')
        print(f"Found a previous checkpoint at: {checkpoint_dir}")
        return checkpoint_dir # checkpoint found!
    return None # first training

max_new_tokens = 512
top_p = 0.5
temperature=0.7

def generate(model, instruction, input, prompt, max_new_tokens=max_new_tokens, top_p=top_p, temperature=temperature):
    inputs = tokenizer(prompt.format(instruction=instruction, input=input), return_tensors="pt").to('cuda')

    outputs = model.generate(
        **inputs, 
        generation_config=GenerationConfig(
            do_sample=True,
            max_new_tokens=max_new_tokens,
            top_p=0.5, #originally 0.9
            temperature=temperature,
            pad_token_id = 0,
        )
    )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    text = text.split('### Response: ')
    # print(text)
    return text[0], text[1]

user_dir = '/cbica/home/xjia/qlora/'
base_model_names = ['lmsys/vicuna-7b-v1.5', 
                    'ehartford/Samantha-1.2-mistral-7b', 
                    'ehartford/Samantha-1.11-7b', 
                    'HuggingFaceH4/zephyr-7b-alpha', 
                    'EmoCareAI/ChatPsychiatrist']
adapter_dirs = ['vicuna-7b-gpt-1018', 
                 'samantha12-7b-gpt-1022',
                 'samantha111-7b-gpt-1022',
                 'zephyr-7b-gpt-1025', 
                 '']

prompt = (
    "Below is an instruction that describes a task, paired with an input that provides further context. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response: "
)

instruction = (
    "You are a helpful mental health counselling assistant, please answer the mental health questions based on the patient's description. "
    "The assistant gives helpful, comprehensive, and appropriate answers to the user's questions."
)

# questions = pd.read_json('/cbica/home/xjia/qlora/data/lab/newCombined_bench.jsonl', lines=True)
questions = pd.read_json('/cbica/home/xjia/qlora/data/lab/newCombined_bench.jsonl', lines=True)
result_df = pd.DataFrame({})
input_prompt = []
base_responses = []
responses = []

for base_model_name, adapter_dir in zip(base_model_names, adapter_dirs):
    adapter_path = get_last_checkpoint(join(user_dir, 'output', adapter_dir))

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    # Fixing some of the early LLaMA HF conversion issues.
    tokenizer.bos_token_id = 1

    # Load the model (use bf16 for faster inference)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        load_in_4bit=True,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
        )
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        load_in_4bit=True,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
        )
    )
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, os.path.join(adapter_path, "adapter_model"))

    base_model.eval()
    model.eval()

    for index, question in questions.iterrows():
        # print("="*40+f"Question {i+1}"+"="*40)
        # print("Question: ", questions['turns'][0])
        # print("="*40+"Base Vicuna"+"="*40)
        print("="*40+f"Question {index+1}"+"="*40)
        print(question['turns'][0])
        _, base_response = generate(base_model, instruction, question["turns"][0], prompt)
        print("="*40+"Base model"+"="*40)
        print(base_response)
        print("="*40+"Finetuned model"+"="*40)
        _, response = generate(model, instruction, question["turns"][0], prompt)
        print(response)

        choices_base = [{"index": 0, "turns": [base_response]}]

        with open(join(user_dir, f"data/eval_jsonl/newCombined_bench/{base_model_name.split('/')[1]}.jsonl"), "a") as fout:
            ans_json = {
                "question_id": question["question_id"],
                "answer_id": question["question_id"],
                "model_id": base_model_name,
                "choices": choices_base,
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json) + "\n")

        choices = [{"index": 0, "turns": [response]}]
        with open(join(user_dir, f"data/eval_jsonl/newCombined_bench/{adapter_dir}.jsonl"), "a") as fout:
            ans_json = {
                "question_id": question["question_id"],
                "answer_id": question["question_id"],
                "model_id": adapter_dir,
                "choices": choices,
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json) + "\n")
       