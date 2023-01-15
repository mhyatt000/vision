#!/bin/bash

# echo $PATH
# cd ~/cs/vision;
# git pull:
# conda activate vision;

if [$NODE1 == hostname]; then
    GPUS=4
else
    GPUS=2
fi

clear;
torchrun --standalone --nproc_per_node=$GPUS general/master.py --config-name $1

# torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=12581 general/tools/train.py --config-name $1
