#!/bin/sh
# Grid Engine options (lines prefixed with #$)
# $ -N hello              
# $ -cwd                  
# $ -l h_rss=32G
# $ -l h_vmem=32G
# $ -pe sharedmem 16

#  These options are:
#  job name: -N
#  use the current working directory: -cwd
#  runtime limit of 5 minutes: -l h_rt
#  virtual memory limit of 32 Gbyte: -l h_vmem
#  resident set size memory limie of 32 Gbyte: -l h_rss

# Initialise the environment modules
. /etc/profile.d/modules.sh
module load python/3.12.9

cd /exports/eddie/scratch/s2859622/

# # Load Python

# python -m venv venv

source venv/bin/activate

# pip lists
# # Run the program
# pip list

# pip freeze | xargs pip uninstall -y

# pip list
# ./venv/bin/python  ./scripts/img_to_tn.py 
cd ./QuLIP-fork
# python  ./scripts/img_to_tn.py 
python  ./scripts/caption_to_einsum_aro.py 
# python  ./scripts/run_circuits.py 
# python  ./scripts/run_model.py 
# python  ./scripts/run_model_emb.py

# python ./scripts/caption_to_tn_aro.py