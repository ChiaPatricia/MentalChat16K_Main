import os
from os.path import exists, join, isdir
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
from peft import PeftModel
from peft.tuners.lora import LoraLayer
import tqdm
import json
import fire

def get_json_list(file_path):
    file_path = os.path.expanduser(file_path)
    with open(file_path, "r") as f:
        json_list = []
        for line in f:
            json_list.append(json.loads(line))
        return json_list

def main(
    load_in_4bit: bool = False,
    model_name_or_path: str = "",
    adapter_path: str = "",
    prompt_path: str = "/cbica/home/xjia/qlora/templates/guanaco.txt",
    questions: str = "vicuna",
    max_new_tokens: int = 512,
    top_p: float = 0.9,
    temperature: float = 0.7,
    output_path: str = "/cbica/home/xjia/qlora/eval/replicate/guanaco_7b_vicuna.jsonl",
):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    # Fixing some of the early LLaMA HF conversion issues.
    tokenizer.bos_token_id = 1
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        load_in_4bit=True,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
        )
    )
    
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    
    with open(prompt_path, 'r') as f:
        prompt = f.read()
        
    if questions == "vicuna":
        question_ls = get_json_list("/cbica/home/xjia/qlora/eval/prompts/vicuna_questions.jsonl")
        
    for idx in range(len(question_ls)):
        user_question = question_ls[idx]["text"]
        inputs = tokenizer(prompt.format(user_question=user_question), return_tensors="pt").to('cuda')
        outputs = model.generate(
            **inputs, 
            generation_config=GenerationConfig(
                do_sample=True,
                max_new_tokens=max_new_tokens,
                top_p=top_p,
                temperature=temperature,
            )
        )
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        text_trunc = text.split("###")[0]
        question_ls[idx]["generation"] = text
        question_ls[idx]["generation_truncated"] = text_trunc
        
    with open(output_path, "w") as f:
        table = [json.dumps(ans) for ans in question_ls]
        f.write("\n".join(table))
        
if __name__ == "__main__":
    fire.Fire(main)
    
    