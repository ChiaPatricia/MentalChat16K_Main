#!/bin/bash
main_dir=/cbica/home/xjia/qlora
cd $main_dir
echo "Generate reponse for the finetuned models: 'vicuna-7b-gpt-online-0104', 'samantha12-7b-gpt-online-0104', \
'samantha111-7b-gpt-online-0104', 'zephyr-7b-gpt-online-0104', 'Mixtral-8x7B-v0.1-gpt-0104', \
'Mistral-7B-Instruct-v0.2-gpt-0104', 'Mixtral-8x7B-Instruct-v0.1-gpt-0104', 'Mistral-7B-v0.1-gpt-0104'"

source /cbica/software/external/python/anaconda/3/bin/activate

conda activate textlearning

python examples/qlora_models_generate.py 