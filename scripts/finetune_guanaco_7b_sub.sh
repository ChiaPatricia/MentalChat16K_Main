#!/bin/bash

echo -e "\n"
echo -e "Running commands on              : `hostname`"
echo -e "Start time                       : `date +%F-%H:%M:%S`"

main_dir=/cbica/home/xjia/qlora
cd $main_dir

source /cbica/software/external/python/anaconda/3/bin/activate

conda activate textlearning


jid=$(qsub \
      -terse \
      -M "jiaxu7@upenn.edu,tywei@seas.upenn.edu" \
      -m bea \
      -l A40 \
      -l h_vmem=100G \
      -o ${main_dir}/logs/replicate/guanaco_7b_\$JOB_ID.stdout \
      -e ${main_dir}/logs/replicate/guanaco_7b_\$JOB_ID.stderr \
      ${main_dir}/scripts/finetune_guanaco_7b.sh \
      )

echo -e "Job ID                           : $jid\n"

#      -l h_rt=12:00:00 \
      # -q all.q@211affn017\


# hostname=211affn017