#!/bin/bash

# clear
echo running multinode
echo $PBS_JOBID

mapfile -t NODES < $PBS_NODEFILE # Read the lines of the file into an array

SCRIPT='~/cs/vision/_multinode.sh'
CFG='$1'

rm -rf ~/node_alive/*  # Remove existing files and directories
mkdir -p ~/node_alive  # Create directory if it doesn't exist

for((rank=0;rank<${#NODES[*]};rank++));
do
    ssh ${NODES[rank]}  $SCRIPT $1 ${#NODES[*]} $rank ${NODES[0]} $PBS_NODEFILE &
done
#
# staying alive...
file_paths=()
for NODE in "${NODES[@]}"; do
    file_paths+=("$HOME/node_alive/${PBS_JOBID}_node_${NODE}")
done


for file_path in "${file_paths[@]}"; do
    while [ ! -f "$file_path" ]; do
        sleep 60  # Adjust the delay time as needed
    done
done

echo "All files found!"
