NPROC=$(( 4*$(wc -l < ${PBS_NODEFILE}) ))
echo $NPROC

mpiexec -n ${NPROC} --ppn 4 --hostfile ${PBS_NODEFILE} ~/cs/vision/set_affinity.sh python ~/cs/vision/general/master.py --config-name $1
