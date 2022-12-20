#!/bin/bash

#SBATCH --nodes=1             # Total # of nodes 
#SBATCH --ntasks-per-node=1   # Number of MPI ranks per node (one rank per GPU)
#SBATCH --gpus-per-node=1     # Number of GPUs per node
#SBATCH --time=48:00:00        # Total run time limit (hh:mm:ss)
#SBATCH -J myjobname          # Job name
#SBATCH -o myjob.o%j          # Name of stdout output file
#SBATCH -e myjob.e%j          # Name of stderr error file
#SBATCH -p gpu                # Queue (partition) name


module load anaconda

source activate mujoco_gym

python run_hierIL_baselines.py --env_name KitchenMetaEnv-v0 --n_traj 1000 --algo DI_gail --seed 4
