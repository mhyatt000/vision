#!/bin/bash

# Read the lines of the file into an array
mapfile -t NODES < $PBS_NODEFILE
IDXS=("${!NODES[@]}")

RZV=${NODES[0]} # rendezvous

# Print the array to the console
printf '%s\n' "${NODES[@]}"

for i in "${IDXS[@]}"; do
  (ssh "${NODES[$i]}"  torchrun --nproc_per_node=4 --nnodes=2 --node_rank=$i \
    --rdzv_id=456 --rdzv_backend=c10d --rdzv_endpoint=$RZV:29500 \
    ~/cs/vision/general/master.py --config-name $1 ) &
  done

wait
exit 0

# git add -A ; git commit -m 'deploy' ; git push ;

