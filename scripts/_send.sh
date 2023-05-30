export PBS_NODEFILE=$2
export OMP_NUM_THREADS=62;
torchrun --standalone --nproc_per_node gpu ~/cs/vision/general/master.py --config-name $1
