#!/bin/bash
#
##SBATCH --partition=short
##SBATCH --nodes=1
##SBATCH --ntasks-per-node=1
##SBATCH --time=00:10:00
##SBATCH --job-name=example-job

# Run your command
echo "Hello, Slurm!"

