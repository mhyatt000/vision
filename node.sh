#!/bin/bash
#
# This is for execution on the GPU node... usually called by multinode.sh

whoami && hostname;
source ~/.bashrc
source ~/.zshrc
conda activate vision;
cd ~/cs/vision;
git pull;

clear;

export OMP_NUM_THREADS=10;
torchrun --nproc_per_node=$2 --nnodes=2 --node_rank=$1 \
    --master_addr=$3 --master_port=12581 \
    ~/cs/vision/general/master.py --config-name $4

    # --rdzv_id=456 --rdzv_backend=c10d --rdzv_endpoint=$3 \
# torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0general/tools/train.py --config-name $1
