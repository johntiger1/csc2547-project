#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH -p gpu
#SBATCH -c 2
#SBATCH -n 1

#module load cuda-8.0
#source ~/anaconda3/bin/activate pytorch

module load pytorch1.0-cuda9.0-python3.6
python main.py --cuda
