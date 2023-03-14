NPROC=$(( 4*$(wc -l < ${PBS_NODEFILE}) ))
echo $NPROC

mpirun -n ${NPROC} --ppn 4 --hostfile ${PBS_NODEFILE} python --version
