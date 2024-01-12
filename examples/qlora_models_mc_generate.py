import os
from os.path import exists, join, isdir
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
from peft import PeftModel
from peft.tuners.lora import LoraLayer
import pandas as pd
import time
import json
import re

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
top_p = 0.9
temperature=0.9

def generate(model, input, prompt, max_new_tokens=max_new_tokens, top_p=top_p, temperature=temperature, pad0=True):
    inputs = tokenizer(prompt.format(input=input), return_tensors="pt").to('cuda')

    if pad0:
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
    else:
        outputs = model.generate(
            **inputs, 
            generation_config=GenerationConfig(
                do_sample=True,
                max_new_tokens=max_new_tokens,
                top_p=0.5, #originally 0.9
                temperature=temperature,
                pad_token_id = 2,
            )
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    match = re.search(r'### Answer: ([ABCDabcd])', text)
    # text = text.split('### Answer: ')
    # print(text[1])
    # match = re.search(r'\[\[(\w)\]\]', text[1])
    # if not match:
    #     match = re.search(r"\[(\w)\]", text[1])
    # if not match:
    #     match = re.search(r"([ABCD]).", text[1])
    
    # if not match:
    #     match = re.search(r"(\w).", text[1])
    # try:
    #     mc_answer = re.search(r'Answer: (\w)', text[1]).group(1)
    # except:
    #     return generate(model, input, prompt, max_new_tokens=max_new_tokens, top_p=top_p, temperature=temperature)
    return text[1], match.group(1)

user_dir = '/cbica/home/xjia/qlora/'
base_model_names = ['lmsys/vicuna-7b-v1.5', 
                    'ehartford/Samantha-1.2-mistral-7b', 
                    'ehartford/Samantha-1.11-7b', 
                    'HuggingFaceH4/zephyr-7b-alpha', 
                    'lmsys/vicuna-7b-v1.5', 
                    'ehartford/Samantha-1.11-7b',
                    'ehartford/Samantha-1.2-mistral-7b',
                    'HuggingFaceH4/zephyr-7b-alpha',
                    'mistralai/Mixtral-8x7B-v0.1',
                    'mistralai/Mistral-7B-Instruct-v0.2',
                    'mistralai/Mixtral-8x7B-Instruct-v0.1',
                    'mistralai/Mistral-7B-v0.1',
                    'EmoCareAI/ChatPsychiatrist']
adapter_dirs = ['vicuna-7b-gpt-1018', 
                'samantha12-7b-gpt-1022',
                'samantha111-7b-gpt-1022',
                'zephyr-7b-gpt-1025',
                'vicuna-7b-v1.5-phase2-1223',
                'samantha-v1.1-7b-phase2-1226',
                'samantha-v1.2-mistral-7b-phase2-1223', 
                'zephyr-7b-alpha-phase2-1223',
                'Mixtral-8x7B-v0.1-phase2-1223',
                'Mistral-7B-Instruct-v0.2-phase2-1223',
                'Mixtral-8x7B-Instruct-v0.1-phase2-1223',
                'Mistral-7B-v0.1-phase2-1223',
                '']

# prompt = (
#     "Below is an instruction that describes a task, paired with an input that provides further context. "
#     "Write a response that appropriately completes the request.\n\n"
#     "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response: "
# )

# prompt = (
#     "Below is a multiple choice question with four choices A, B, C, D. Please output the most correct answer choice with its corresponding letter and an explanation."
#     "Output your answer choice by strictly following this format:\n"
#     "\"Answer: [answer choice]\", for example, \"Answer: A\"\n\n"
#     "\"Explanation: [explanation]\"\n\n"
#     "Below is an example question and response. Please output your answer following exactly the response format in the example:\n"
#     "[Question]\nThe adverse effect of clozapine is: A. Hypertension. B. Sialorrhea. C. Extrapyramidal S/E. D. Neuroleptic malignant syndrome.\n\n"
#     "[Response]\nAnswer: B\n\nExplanation: Side effects of clozapine include Agranulocytosis, Urinary incontinence, Unstable BP & Tachycardia, Hypersalivation (sialorrhoea), Worsening of diabetes, Weight gain, Seizures, Sedation.\n\n"
#     "### Question:\n{input}\n\n### Response: "
# )

# prompt = (
#     "Below is a multiple choice question with four choices A, B, C, D (or a, b, c, d). Which choice is the most correct?"
#     "Output your answer choice by strictly following this format: "
#     "\"[[A]]\" if choice A (or choice a) is the most correct, \"[[B]]\" if choice B (or choice b) is the most correct. \"[[C]]\" if choice C (or choice c) is the most correct. and \"[[D]]\" if choice D (or choice d) is the most correct."
#     "Please output exactly one letter choice. \n\n"
#     "Below is an example question and response. Please output your answer following exactly the response format in the example:\n"
#     "Question:\nThe adverse effect of clozapine is: A. Hypertension. B. Sialorrhea. C. Extrapyramidal S/E. D. Neuroleptic malignant syndrome.\n\n"
#     "Answer: B\n\n"
#     "### Question:\n{input}\n\n### Answer: "
# )

prompt = (
    "The following are multiple choice questions (with answers)\n"  
    "Question: A patient who was admitted yesterday with an adjustment disorder and depressed mood has not left his or her room. The psychiatric-mental health nurse's most appropriate approach at meal time today is to respond: A. 'I will bring your tray to your room, if it will make you more comfortable.' B.'I will walk with you to the dining room and sit with you while you eat.' C.'Where would you like to eat your meal this noon?' D.'You will feel better if you go to the dining room and eat with the others.'\n"
    "Answer: B\n\n"
    "Question: A 17-year-old, female patient with anorexia nervosa has just been released from the hospital. To facilitate recovery at home, the psychiatric-mental health nurse instructs the family to: A.discourage the patient from sneaking food between meals, by unobtrusively reducing her access to the kitchen. B. encourage the patient's interest in menu planning, food magazines, and cooking lessons, by leaving information and materials around the house. C. permit the patient to eat her meals privately in her bedroom to discourage family preoccupation with meals. D. recommend that the patient joins in routine family meals and clears the dishes after dinner, even if she does not eat.\n"
    "Answer: D\n\n"
    "Question: A supervisor observes inconsistency in the psychiatric-mental health nurse's behavior toward a patient; the nurse is unreasonably concerned, overly kind, or irrationally hostile. The most appropriate explanation is that the nurse is displaying: A.countertransference. B.empathic resonance. C.splitting behavior. D.transference.\n"
    "Answer: A\n\n"
    "Please provide the answer (A, B, C or D) for the following Question:\n"
    "### Question: {input}\n### Answer: "
)

# instruction = (
#     "You are a helpful mental health counselling assistant, please answer the mental health questions based on the patient's description. "
#     "The assistant gives helpful, comprehensive, and appropriate answers to the user's questions."
# )

# questions = pd.read_json('/cbica/home/xjia/qlora/data/lab/newCombined_bench.jsonl', lines=True)
questions = pd.read_json('/cbica/home/xjia/qlora/data/lab/JT_bench_new.jsonl', lines=True)
result_df = pd.DataFrame({})
input_prompt = []
base_responses = []
responses = []

for base_model_name, adapter_dir in zip(base_model_names, adapter_dirs):
    # Specify pad token
    pad0 = True
    if "zephyr" in base_model_name:
        pad0 = False

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
        print("="*40+f"{base_model_name}"+"="*40)
        print("="*40+f"Question {index+1}"+"="*40)
        print(question['turns'][0])
        
        try:
            base_response, base_answer = generate(base_model, question["turns"][0], prompt, pad0=pad0)
        except Exception as error:
            print(error)
            continue

        print("="*40+"Base model"+"="*40)
        print(base_response)
        print("="*40+"Finetuned model"+"="*40)

        try:
            response, answer = generate(model, question["turns"][0], prompt, pad0=pad0)
        except Exception as error:
            print(error)
            continue 
        
        print(response)

        choices_base = [{"index": 0, "turns": [base_response]}]

        with open(join(user_dir, f"data/eval_jsonl/JT_new_bench/model_answer/{base_model_name.split('/')[1]}.jsonl"), "a") as fout:
            ans_json = {
                "question_id": question["question_id"],
                "answer_id": question["question_id"],
                "model_id": base_model_name,
                "choices": choices_base,
                "predicted_answer": base_answer,
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json) + "\n")

        choices = [{"index": 0, "turns": [response]}]
        with open(join(user_dir, f"data/eval_jsonl/JT_new_bench/model_answer/{adapter_dir}.jsonl"), "a") as fout:
            ans_json = {
                "question_id": question["question_id"],
                "answer_id": question["question_id"],
                "model_id": adapter_dir,
                "choices": choices,
                "predicted_answer": answer,
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json) + "\n")
       