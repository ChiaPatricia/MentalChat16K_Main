#!/bin/bash
main_dir=/cbica/home/xjia/qlora
cd $main_dir

source /cbica/software/external/python/anaconda/3/bin/activate

conda activate textlearning

python examples/qlora_models_mc_generate.py 