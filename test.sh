#!/bin/bash
#SBATCH --gres=gpu:4
echo "Environment Test:"
echo "SLURM_GPUS_ON_NODE: $SLURM_GPUS_ON_NODE"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi