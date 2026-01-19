#!/bin/bash

#SBATCH --job-name=train_independent       # Job name
#SBATCH --output=./logs/independent/train_independent_%j.out   # Output file (stdout and stderr)
#SBATCH --time=0-6           # Time limit (DD-HH:MM)
#SBATCH --partition=gpu_a100      # Partition name
#SBATCH --gres=gpu:a100:1         # Request 1 GPU of type a100
#SBATCH --qos=default             # Quality of Service


# Initialize Conda and activate the environment
conda init
conda activate act


# Run the Python script directly
python directopt.py --model google/gemma-2-2b-it --n_inits 5 --train_multiple_independent
