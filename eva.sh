#!/bin/bash
#SBATCH -w node003
#SBATCH -q ttx2
#SBATCH -J gcnet
#SBATCH -p ttxp
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -t 600
#SBATCH --output=log.%j.out
#SBATCH --error=log.%j.err
#SBATCH --gres=gpu:ttxp:1
module load cuda10.2
module load cudnn7.6-cuda10.2
source activate gcnet
python evaluation.py
