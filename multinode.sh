#!/bin/bash

# clear
echo running multinode
mapfile -t NODES < $PBS_NODEFILE # Read the lines of the file into an array

SCRIPT='~/cs/vision/_multinode.sh'
CFG='$1'

for((rank=0;rank<${#NODES[*]};rank++));
do
    ssh ${NODES[rank]}  $SCRIPT $1 ${#NODES[*]} $rank ${NODES[0]} $PBS_NODEFILE &
done

rm -rf ~/nodefile_return/*  # Remove existing files and directories
mkdir -p ~/nodefile_return  # Create directory if it doesn't exist

# staying alive...

file_paths=()
for NODE in "${NODES[@]}"; do
    file_paths+=("$HOME/nodefile_return/${PBS_JOBID_node}_node${NODE}")
done


for file_path in "${file_paths[@]}"; do
    while [ ! -f "$file_path" ]; do
        sleep 60  # Adjust the delay time as needed
    done
done


# num_files=$(find ~/nodefile_return -maxdepth 1 -type f | wc -l)
# num_nodes=${#NODES[@]}

# while [ "$num_files" -lt "$num_nodes" ]; do
    # sleep 60  # Adjust the delay time as needed
    # num_files=$(find ~/nodefile_return -maxdepth 1 -type f | wc -l)
# done

echo "All files found!"






