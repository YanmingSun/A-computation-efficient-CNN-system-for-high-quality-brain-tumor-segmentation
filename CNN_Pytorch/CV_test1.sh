#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:p100:2             # request GPU "generic resource", 4 on Cedar, 2 on Graham
#SBATCH --cpus-per-task=32      # CPU cores/threads
#SBATCH --mem=124G              # memory per node
#SBATCH --time=0-03:00            # time (DD-HH:MM)
#SBATCH --output=%1-%0.out  # %N for node name, %j for jobID

module load cuda cudnn
module load python/3.8
source /home/sym/pytorch/bin/activate
python ./CV_test1.py