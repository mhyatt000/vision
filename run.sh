#!/bin/bash

export OMP_NUM_THREADS=62;
torchrun --standalone --nproc_per_node gpu ~/cs/vision/general/master.py $@

# torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=12581 general/tools/train.py --config-file $1
