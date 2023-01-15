#!/bin/bash
#
# This is for execution on the GPU node... usually called by multinode.sh

whoami && hostname;
conda activate vision;
cd cs/vision;
git pull;

# export OMP_NUM_THREADS=10;
# torchrun --nproc_per_node=2 --nnodes=2 --node_rank=0 \
    # --rdzv_id=456 --rdzv_backend=c10d --rdzv_endpoint=$ENVVAR \
    # general/tools/train.py --config-name $1

# torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=12581 general/tools/train.py --config-name $1
