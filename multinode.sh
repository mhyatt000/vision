#!/bin/bash

# Read the lines of the file into an array
mapfile -t NODES < $PBS_NODEFILE
IDXS=("${!NODES[@]}")

RZV=${NODES[0]} # rendezvous

# Print the array to the console
printf '%s\n' "${NODES[@]}"

for i in "${IDXS[@]}"; do
  (ssh "${NODES[$i]}"  torchrun --nproc_per_node=4 --nnodes=2 --node_rank=$i \
     --master_addr=$RZV --master_port=29500 \
    ~/cs/vision/general/master.py --config-name $1 ) &
  done

wait
exit 0

mpirun \
--hostfile $PBS_NODEFILE \
--ppn 4 \
-x MASTER_ADDR= "$(hostname)" \
-x MASTER_PORT= 29500 \
-x $PATH \
-bind-to none -map-by slot \
-mca pml ob1 -mca btl ^openib \
python3 ~/cs/vision/general/master.py --config-name $1

exit 0

