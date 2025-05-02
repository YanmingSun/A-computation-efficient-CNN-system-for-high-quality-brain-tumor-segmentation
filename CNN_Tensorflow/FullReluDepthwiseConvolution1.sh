#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:2             # request GPU "generic resource", 4 on Cedar, 2 on Graham
#SBATCH --ntasks-per-node=32      # CPU cores/threads
#SBATCH --mem=124G              # memory per node
#SBATCH --time=0-12:00            # time (DD-HH:MM)
#SBATCH --output=%1-%1.out  # %N for node name, %j for jobID

module load cuda cudnn
module load python/3.7
source /home/sym/tensorflow/bin/activate
python ./FullReluDepthwiseConvolution1.py