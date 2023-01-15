clear;
export OMP_NUM_THREADS=10;
torchrun --standalone --nproc_per_node=2 ../master.py --config-name symia
# torchrun --standalone --nproc_per_node=2 train_v2.py configs/wblot_config.py 
