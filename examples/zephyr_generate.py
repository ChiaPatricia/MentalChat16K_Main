import os
from os.path import exists, join, isdir
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
from peft import PeftModel
from peft.tuners.lora import LoraLayer
import pandas as pd

def get_last_checkpoint(checkpoint_dir):
    if isdir(checkpoint_dir):
        is_completed = exists(join(checkpoint_dir, 'completed'))
        if is_completed: return None, True # already finished
        max_step = 0
        for filename in os.listdir(checkpoint_dir):
            if isdir(join(checkpoint_dir, filename)) and filename.startswith('checkpoint'):
                max_step = max(max_step, int(filename.replace('checkpoint-', '')))
        if max_step == 0: return None, is_completed # training started, but no checkpoint
        checkpoint_dir = join(checkpoint_dir, f'checkpoint-{max_step}')
        print(f"Found a previous checkpoint at: {checkpoint_dir}")
        return checkpoint_dir, is_completed # checkpoint found!
    return None, False # first training


# TODO: Update variables
max_new_tokens = 512
top_p = 0.9
temperature=0.7
user_question = '''Lately, I've been feeling trapped in a never-ending cycle of darkness. It's hard to describe, but everything feels so overwhelming and exhausting. My main goal in seeking counseling is to find ways to break free from this constant state of sadness and regain control over my life.
My mind feels clouded with negative thoughts and self-doubt. I keep questioning my worth and abilities, which makes it challenging to take any positive steps forward. It almost feels as if happiness is unattainable and that this darkness will consume me forever.
One situation that particularly triggers my depression is social gatherings. The fear of judgment and not fitting in can be paralyzing. I often isolate myself from these events, as they tend to exacerbate my feelings of loneliness and worthlessness.
These depressive symptoms have been a part of my life for the past year. I wake up feeling down every day and the heaviness stays with me throughout. Some days it's more debilitating than others, making even simple tasks a struggle.
In terms of coping mechanisms, I've tried engaging in creative activities like painting and writing to channel my emotions, but it hasn't provided sustained relief. I would love to explore additional strategies that could help improve my mental well-being.
I have some questions about the therapeutic process. How will you approach our sessions in order to address my depression? Are there any specific techniques or interventions you recommend for someone in my situation? Lastly, how can I work on building a support system outside of therapy?'''

# Base model
model_name_or_path = 'HuggingFaceH4/zephyr-7b-alpha'
# Adapter name on HF hub or local checkpoint path.
# import pdb; pdb.set_trace()
# adapter_path_replicate, _ = get_last_checkpoint('output/guanaco-7b-A40')
# adapter_path_replicate = "output/MentalGPT-7b-newCombined-1016/checkpoint-1400/adapter_model"
adapter_path = '/cbica/home/xjia/qlora/output/zephyr-7b-SelfInstruct-1022/checkpoint-1850/adapter_model' #'timdettmers/guanaco-7b'

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
# Fixing some of the early LLaMA HF conversion issues.
tokenizer.bos_token_id = 1
tokenizer.pad_token_id = 2

# Load the model (use bf16 for faster inference)
base_model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
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
base_model.config.pad_token_id = 2

model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
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
model.config.pad_token_id = 2
model = PeftModel.from_pretrained(model, adapter_path)

base_model.eval()
model.eval()

# prompt = (
#     "You are a helpful mental health counselling assistant, your language is English, please answer the mental health questions based on the patient's description."
#     "The assistant gives helpful, comprehensive, and appropriate answers to the user's questions."
#     "### User: {user_question}"
#     "### Assistant: "
# )

prompt = (
    "Below is an instruction that describes a task, paired with an input that provides further context. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response: "
)
# prompt = (
#     "A chat between a user with mental illness concern and a professional, helpful mental health counseling assitant. "
#     "The assistant gives helpful, comprehensive, and appropriate answers to the user's questions. "
#     "### User: {user_question}"
#     "### Assistant: "
# )

# def generate(model, user_question, prompt, max_new_tokens=max_new_tokens, top_p=top_p, temperature=temperature):
#     inputs = tokenizer(prompt.format(user_question=user_question), return_tensors="pt").to('cuda')

#     outputs = model.generate(
#         **inputs, 
#         generation_config=GenerationConfig(
#             do_sample=True,
#             max_new_tokens=max_new_tokens,
#             top_p=0.5, #originally 0.9
#             temperature=temperature,
#         )
#     )

#     text = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     text = text.split('### Assistant: ')
#     # print(text)
#     return text[0], text[1]

def generate(model, instruction, input, prompt, max_new_tokens=max_new_tokens, top_p=top_p, temperature=temperature):
    inputs = tokenizer(prompt.format(instruction=instruction, input=input), return_tensors="pt").to('cuda')

    try:
        outputs = model.generate(
            **inputs, 
            generation_config=GenerationConfig(
                do_sample=True,
                max_new_tokens=max_new_tokens,
                top_p=0.5, #originally 0.9
                temperature=temperature,
                pad_token_id = 2
            )
        )
    except:
        outputs = model.generate(
            **inputs, 
            generation_config=GenerationConfig(
                do_sample=True,
                max_new_tokens=max_new_tokens,
                top_p=0.5, #originally 0.9
                temperature=temperature,
                pad_token_id = 0
            )
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    text = text.split('### Response: ')
    # print(text)
    return text[0], text[1]

test_df = pd.read_csv('/cbica/home/xjia/qlora/data/lab/new_combined_instr_test_mini.csv')
result_df = pd.DataFrame({})
questions = []
input_prompt = []
base_responses = []
responses = []

for i in range(len(test_df)):
    print("="*40+f"Question {i+1}"+"="*40)
    print("Question: ", test_df['instruction'][i])
    print("="*40+"Base Zephyr"+"="*40)
    pt, base_response = generate(base_model, test_df['instruction'][i], test_df["input"][i], prompt)
    print(base_response)
    print("="*40+"Zephyr qlora"+"="*40)
    _, response = generate(model, test_df['instruction'][i],  test_df["input"][i], prompt)
    print(response)
    questions.append(test_df['input'][i])
    input_prompt.append(pt)
    base_responses.append(base_response)
    responses.append(response)

result_df['question'] = questions
result_df['input_prompt'] = input_prompt
result_df['Zephyr_7b'] = base_responses
result_df['Zephyr_qlora'] = responses

result_df.to_csv('/cbica/home/xjia/qlora/data/eval/Zephyr-SelfInstruct-1022_newcombined_instr_test_mini_tp05.csv', index=False)

# base_response = generate(base_model, user_question, prompt)
# print("\n")
# print("="*40+"Vicuna qlora"+"="*40)
# response = generate(model, user_question, prompt)
# import pdb; pdb.set_trace()