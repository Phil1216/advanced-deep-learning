#!/usr/bin/env bash

#SBATCH --job-name=rectified-linear
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:08:00
#SBATCH --qos=dw87

uv run main.py 4