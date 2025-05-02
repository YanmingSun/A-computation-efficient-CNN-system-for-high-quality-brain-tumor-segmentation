#!/bin/bash
#SBATCH --nodes 1
#SBATCH --gres=gpu:p100:2          # Request 2 GPU "generic resources”.
#SBATCH --tasks-per-node=2 # This is the number of model replicas we will place on the GPU. Change this to 10,12,14,... to see the effect on performance
#SBATCH --cpus-per-task=8 # increase this parameter and increase "--num_workers" accordingly to see the effect on performance
#SBATCH --mem=124G
#SBATCH --time=0-12:00
#SBATCH --output=%1-%2.out

module load cuda cudnn
module load python/3.8
source /home/sym/pytorch/bin/activate

export NCCL_BLOCKING_WAIT=1  #Set this environment variable if you wish to use the NCCL backend for inter-GPU communication.
export MASTER_ADDR=$(hostname) #Store the master node’s IP address in the MASTER_ADDR environment variable.

echo "r$SLURM_NODEID master: $MASTER_ADDR"
echo "r$SLURM_NODEID Launching python script"

# The SLURM_NTASKS variable tells the script how many processes are available for this execution. “srun” executes the script <tasks-per-node * nodes> times

srun python3.8 CV_TC1.py --init_method tcp://$MASTER_ADDR:3456 --world_size $SLURM_NTASKS