cd ~/cs/vision;
git pull:

clear;
export OMP_NUM_THREADS=10;
torchrun --standalone --nproc_per_node=-1 general/master.py --config-name $1

# torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=12581 general/tools/train.py --config-name $1
