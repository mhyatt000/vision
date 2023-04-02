# runs deep speed on a model

deepspeed --hostfile=$PBS_NODEFILE ~/cs/vision/general/master.py --config-name $1 --deepspeed
