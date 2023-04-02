# source ~/.bashrc
# echo $1 $2 $3 $4
# exit 0

# which torchrun
# conda env list
# python --version
# CUDA_VISIBLE_DEVICES=0,1,2,3
MASTER_ADDR=$4
MASTER_PORT=29400
NCCL_DEBUG=INFO
NCCL_NET_GDR_LEVEL=PHB
NCCL_COLLNET_ENABLE=1
NCCL_SOCKET_IFNAME=eno1

torchrun \
--nproc_per_node=4 \
--nnodes=$2 \
--node_rank=$3 \
--rdzv_id=456 \
--rdzv_backend=c10d \
--rdzv_endpoint=$4:29400 \
~/cs/vision/general/master.py --config-name $1 

# --master_addr=$4 \
# --master_port=2345 \ 

# torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 --rdzv_id=456 --rdzv_backend=c10d --rdzv_endpoint=x3003c0s13b0n0 ~/cs/vision/general/master.py --config-name configs/multi_ffc_arc64.yaml
