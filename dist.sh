$COBALT_NODEFILE

# Read the lines of the file into an array
mapfile -t NODES < $PBS_NODEFILE

# Print the array to the console
printf '%s\n' "${NODES[@]}"

for((rank=0;rank<${#NODES[*]};rank++));
do 
    ssh ${NODES[rank]} " CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    torchrun \
    --nproc_per_node=8 \
    --nnodes=${#NODES[*]} \
    --node_rank=$rank \
    --master_addr=${NODES[0]} \
    --master_port=22345 / 
    ~/cs/vision/general/master.py --config-name $1" &
done
