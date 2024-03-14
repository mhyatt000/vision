#  used to send a lots of 1 node job to many nodes

clear
# Read the lines of the file into an array
mapfile -t NODES < $PBS_NODEFILE

# Print the array to the console
# printf '%s\n' "${NODES[@]}"

for((rank=0;rank<${#NODES[*]};rank++));
do
    ssh ${NODES[rank]} ~/cs/vision/_send.sh $1 $PBS_NODEFILE &
done
